import argparse,gc,json,time,os,shutil
from pathlib import Path
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
import numpy as np, torch
from huggingface_hub import snapshot_download
from diffusers import AutoencoderDC
from common import *
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
root.mkdir(exist_ok=True);(root/'data').mkdir(exist_ok=True)
assert shutil.disk_usage(root).free/2**30>43
snapshot=Path(snapshot_download(MODEL,revision=REVISION,cache_dir=str(root/'hf-cache'),allow_patterns=['transformer/config.json','transformer/diffusion_pytorch_model.fp16.safetensors','vae/config.json','vae/diffusion_pytorch_model.fp16.safetensors']))
assert digest(snapshot/'transformer/diffusion_pytorch_model.fp16.safetensors')=='4616f5f26161c4f072083edcd9b771b8a840057602e928dca00c8a05175b22a3'
atomic_json(root/'model.json',{'snapshot':str(snapshot),'revision':REVISION,'files':[{'path':str(x.relative_to(snapshot)),'bytes':x.stat().st_size,'sha256':digest(x)} for x in sorted(snapshot.rglob('*')) if x.is_file()]})
print('MODEL_VERIFIED',flush=True)
torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.65)
vae=AutoencoderDC.from_pretrained(snapshot,subfolder='vae',variant='fp16',torch_dtype=torch.float32).eval().requires_grad_(False).to('mps')
font=root/'fonts/train.ttf';other=root/'fonts/transfer.ttf';templates=answer_templates(font)
cal=[]
for f in [font,other]:
 for y in range(20):
  im=render(0,0,0,f,answer=y);r=read_answer(im,templates);cal.append({'font':f.name,'answer':y,**r})
assert all(r['valid'] and r['prediction']==r['answer'] for r in cal if r['font']=='train.ttf')
# Readout for unfamiliar style uses its own equally frozen templates.
other_templates=answer_templates(other)
assert all(read_answer(render(0,0,0,other,answer=y),other_templates)['prediction']==y for y in range(20))
atomic_json(root/'readout-calibration.json',cal)
rows=[]
for copy in [True,False]:
 for variant in range(4):
  for x in range(10):
   for b,c in ([(0,0)] if copy else [(b,c) for b in range(10) for c in range(2)]):
    rows.append({'id':f'{"copy" if copy else "add"}-{x}-{b}-{c}-v{variant}','a':x,'b':b,'c':c,'variant':variant,'copy':copy,'answer':x if copy else x+b+c})
for i,r in enumerate(rows):
 path=root/'data'/(r['id']+'.npz')
 if path.exists():continue
 ref=render(r['a'],r['b'],r['c'],font,r['variant'],r['copy'])
 target=render(r['a'],r['b'],r['c'],font,r['variant'],r['copy'],r['answer'])
 assert np.all(np.asarray(ref)[288:]==255)
 with torch.no_grad():
  z=vae.encode(torch.from_numpy(np.stack([pixels(ref),pixels(target)])).to('mps'),return_dict=False)[0]*vae.config.scaling_factor
 z=z.cpu().numpy();np.savez_compressed(path,context=z[0],target=z[1])
 if i%50==0:print(json.dumps({'encoded':i,'total':len(rows),'mps_gib':torch.mps.driver_allocated_memory()/2**30}),flush=True)
 # Explicit VAE renderability audit on all twenty possible answer values.
 if not r['copy'] and r['variant']==0 and r['a']==9:
  with torch.no_grad():dec=vae.decode(torch.from_numpy(z[1:]).to('mps')/vae.config.scaling_factor,return_dict=False)[0].cpu().numpy()[0]
  im=pil(dec);im.save(root/'data'/(r['id']+'-roundtrip.png'))
  result=read_answer(im,templates)
  print(json.dumps({'vae_check':r['id'],'answer':r['answer'],'read':result}),flush=True)
  if not result['valid'] or result['prediction']!=r['answer']:raise RuntimeError('VAE readout failed '+r['id'])
for r in rows:r.update(file=r['id']+'.npz',sha256=digest(root/'data'/(r['id']+'.npz')))
atomic_json(root/'data/manifest.json',{'rows':rows,'font_sha256':digest(font),'transfer_font_sha256':digest(other),'schema':'column-latents-v1'})
print('PREPARE_COMPLETE',flush=True)
