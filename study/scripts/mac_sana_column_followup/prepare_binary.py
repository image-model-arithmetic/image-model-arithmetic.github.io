"""Encode answer-free binary input images and supervised raster targets."""
import argparse,json,os
from pathlib import Path
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
import numpy as np
import torch
from diffusers import AutoencoderDC
from common import atomic_json,digest,render,pixels,pil,answer_templates,read_answer
from binary_protocol import training_rows,audit_corpus

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
config=json.loads((root/'source/mac_sana_pixel_v1.json').read_text())
out=root/'data';out.mkdir(exist_ok=True)
audit=audit_corpus(config['evaluation_seeds']['program'])
audit['protocol_sha256']=digest(root/'source/mac_sana_pixel_v1.json')
atomic_json(root/'program-split-audit.json',audit)
torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.65)
snapshot=Path(json.loads((root/'model.json').read_text())['snapshot'])
vae=AutoencoderDC.from_pretrained(snapshot,subfolder='vae',variant='fp16',torch_dtype=torch.float32).eval().requires_grad_(False).to('mps')
font=root/'fonts/train.ttf';bank=answer_templates(font);rows=training_rows();checks=[]
for row in rows:
    path=out/(row['id']+'.npz')
    ref=render(row['a'],row['b'],row['c'],font,row['variant'],row['copy'])
    target=render(row['a'],row['b'],row['c'],font,row['variant'],row['copy'],row['answer'])
    assert np.all(np.asarray(ref)[288:]==255),row['id']
    with torch.no_grad():
        z=vae.encode(torch.from_numpy(np.stack([pixels(ref),pixels(target)])).to('mps'),return_dict=False)[0]*vae.config.scaling_factor
        hybrid=z[1:2].clone();hybrid[:,:,:9,:]=z[:1,:,:9,:]
        decoded=pil(vae.decode(hybrid/vae.config.scaling_factor,return_dict=False)[0][0].cpu().numpy())
    values=z.cpu().numpy()
    if path.exists():
        existing=np.load(path)
        assert set(existing.files)=={'context','target'}
        assert np.array_equal(existing['context'],values[0]) and np.array_equal(existing['target'],values[1]),row['id']
    else:
        temporary=path.with_name(path.stem+'.tmp.npz')
        np.savez_compressed(temporary,context=values[0],target=values[1]);assert temporary.is_file();temporary.replace(path)
    reading=read_answer(decoded,bank)
    checks.append({'id':row['id'],'expected':row['answer'],**reading,'correct':reading['valid'] and reading['prediction']==row['answer']})
    row.update(file=path.name,sha256=digest(path))
    print(json.dumps({'encoded':len(checks),'total':40,'roundtrip_correct':checks[-1]['correct']}),flush=True)
atomic_json(root/'data/manifest.json',{'rows':rows,'font_sha256':digest(font),
            'transfer_font_sha256':digest(root/'fonts/transfer.ttf'),'schema':'binary-column-latents-v1'})
failures=[r for r in checks if not r['correct']]
atomic_json(root/'binary-vae-renderability.json',{'passed':not failures,'count':len(checks),'correct':sum(r['correct'] for r in checks),
            'checks':checks,'failures':failures,'protocol_sha256':digest(root/'source/mac_sana_pixel_v1.json')})
assert not failures,failures
