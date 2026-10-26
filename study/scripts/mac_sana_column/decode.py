import argparse,json,os
from pathlib import Path
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
import numpy as np,torch
from diffusers import AutoencoderDC
from common import *
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--input',type=Path,required=True);a=p.parse_args();root=a.root
out=a.input.with_suffix('');out.mkdir(exist_ok=True)
rows=json.loads((root/'data/manifest.json').read_text())['rows'];z=np.load(a.input);assert set(z.files)=={'latents','indices'}
snapshot=Path(json.loads((root/'model.json').read_text())['snapshot']);torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.65)
vae=AutoencoderDC.from_pretrained(snapshot,subfolder='vae',variant='fp16',torch_dtype=torch.float32).eval().requires_grad_(False).to('mps')
templates=answer_templates(root/'fonts/train.ttf');records=[]
for i,(latent,index) in enumerate(zip(z['latents'],z['indices'])):
 with torch.no_grad():x=vae.decode(torch.from_numpy(latent[None]).to('mps')/vae.config.scaling_factor,return_dict=False)[0][0].cpu().numpy()
 im=pil(x);r=rows[int(index)];im.save(out/(r['id']+'.png'));reading=read_answer(im,templates)
 records.append({'id':r['id'],'copy':r['copy'],'expected':r['answer'],**reading,'correct':reading['valid'] and reading['prediction']==r['answer']})
summary={}
for tag,subset in [('copy',[r for r in records if r['copy']]),('add',[r for r in records if not r['copy']])]:
 summary[tag]={'correct':sum(r['correct'] for r in subset),'count':len(subset),'valid':sum(r['valid'] for r in subset)}
atomic_json(out/'summary.json',{'metrics':summary,'records':records,'latent_sha256':digest(a.input)});print(json.dumps(summary),flush=True)
