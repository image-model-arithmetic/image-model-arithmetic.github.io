"""Independent answer-free evaluation and raw-raster carry composition."""
import argparse,json,os,time,shutil
from pathlib import Path
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
import numpy as np,torch
from diffusers import AutoencoderDC
from common import *
from train import load_model,sample
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
out=root/'evaluation';out.mkdir(exist_ok=False);config=json.loads((root/'source/mac_sana_column_v1.json').read_text())
torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.65)
model,_=load_model(root);state=torch.load(root/'training/best.pt',map_location='cpu',weights_only=False)
assert state['protocol_sha256']==digest(root/'source/mac_sana_column_v1.json')
assert set(state['trainable'])=={n for n,p in model.named_parameters() if p.requires_grad}
model.load_state_dict(state['trainable'],strict=False);model.eval();del state
snapshot=Path(json.loads((root/'model.json').read_text())['snapshot'])
vae=AutoencoderDC.from_pretrained(snapshot,subfolder='vae',variant='fp16',torch_dtype=torch.float32).eval().requires_grad_(False).to('mps')
font=root/'fonts/train.ttf';other=root/'fonts/transfer.ttf';templates=answer_templates(font);other_templates=answer_templates(other)

def independent_read(im):
 candidates=[read_answer(im,bank) for bank in [templates,other_templates]]
 return min(candidates,key=lambda r:r['carry']['mse']+r['digit']['mse'])

def generate(images,seed):
 with torch.no_grad():
  z=vae.encode(torch.from_numpy(np.stack([pixels(im) for im in images])).to('mps'),return_dict=False)[0]*vae.config.scaling_factor
  ys=sample(model,z.cpu(),seed,12)
  xs=vae.decode(ys.to('mps')/vae.config.scaling_factor,return_dict=False)[0].cpu().numpy()
 return [pil(x) for x in xs]

allrecords=[]
for style,f,ts in [('familiar',font,templates),('unfamiliar',other,other_templates)]:
 folder=out/style;folder.mkdir()
 for a0 in range(10):
  for b in range(10):
   # Same noise for the original and carry intervention. Full generation each time.
   for c in range(2):
    identifier=f'{style}-{a0}-{b}-{c}'
    im=generate([render(a0,b,c,f)],2200910+a0*10+b)[0];im.save(folder/(identifier+'.png'));r=independent_read(im)
    record={'id':identifier,'style':style,'a':a0,'b':b,'c':c,'expected':a0+b+c,**r,'correct':r['valid'] and r['prediction']==a0+b+c}
    allrecords.append(record)
    with (out/'local-records.jsonl').open('a') as h:h.write(json.dumps(record)+'\n')
  print(json.dumps({'style':style,'completed_left_operands':a0+1}),flush=True)
summary={}
for style in ['familiar','unfamiliar']:
 rs=[r for r in allrecords if r['style']==style]
 by={(r['a'],r['b'],r['c']):r for r in rs}
 summary[style]={'correct':sum(r['correct'] for r in rs),'count':len(rs),'accuracy':sum(r['correct'] for r in rs)/len(rs),'valid':sum(r['valid'] for r in rs),'carry_pairs_joint_exact':sum(by[(a,b,0)]['correct'] and by[(a,b,1)]['correct'] for a in range(10) for b in range(10)),'carry_pairs':100}
atomic_json(out/'local-summary.json',summary)
# Programs are new compositions of supervised local facts. Only generated pixels carry state.
rng=np.random.default_rng(3300910);programs=[]
for length in [2,4,8]:
 for k in range(32):
  aa=[int(rng.integers(0,10)) for _ in range(length)];bb=[int(rng.integers(0,10)) for _ in range(length)]
  aa[-1]=int(rng.integers(1,10));bb[-1]=int(rng.integers(1,10))
  left=int(''.join(map(str,reversed(aa))));right=int(''.join(map(str,reversed(bb))))
  identifier=f'program-{length}-{k:03d}';folder=out/identifier;folder.mkdir()
  carry=None;images=[]
  for j in range(length):
   ref=render(aa[j],bb[j],0,font,carry_pixels=carry)
   im=generate([ref],4400910+length*10000+k*100+j)[0]
   im.save(folder/f'step-{j:02d}.png');images.append(im);carry=im.crop(CARRY_OUT)
  # No classification is executed until the last generated stage is complete.
  readings=[independent_read(im) for im in images]
  predicted=''.join([str(readings[-1]['carry']['prediction'])]+[str(r['digit']['prediction']) for r in readings[::-1]])
  valid=all(r['valid'] for r in readings)
  expected_carry=0;all_stages_correct=True
  for j,r in enumerate(readings):
   expected_subtotal=aa[j]+bb[j]+expected_carry
   all_stages_correct &= r['valid'] and r['prediction']==expected_subtotal
   expected_carry=expected_subtotal//10
  record={'id':identifier,'length':length,'left':left,'right':right,'expected':left+right,'prediction':int(predicted),'valid':valid,'correct':valid and all_stages_correct and int(predicted)==left+right,'final_answer_correct':valid and int(predicted)==left+right,'all_stages_correct':all_stages_correct,'stages':readings}
  programs.append(record)
  with (out/'program-records.jsonl').open('a') as h:h.write(json.dumps(record)+'\n')
  print(json.dumps({k:v for k,v in record.items() if k!='stages'}),flush=True)
program_summary={str(n):{'correct':sum(r['correct'] for r in programs if r['length']==n),'count':sum(r['length']==n for r in programs),'accuracy':sum(r['correct'] for r in programs if r['length']==n)/sum(r['length']==n for r in programs)} for n in [2,4,8]}
passed=summary['familiar']['accuracy']>=.95 and summary['unfamiliar']['accuracy']>=.8 and all(v['accuracy']>=.9 for v in program_summary.values())
atomic_json(out/'summary.json',{'local':summary,'programs':program_summary,'passed':passed,'checkpoint_sha256':digest(root/'training/best.pt'),'claim':config['evaluation']['claim_boundary']})
print('EVALUATION_COMPLETE',json.dumps({'local':summary,'programs':program_summary,'passed':passed}),flush=True)
