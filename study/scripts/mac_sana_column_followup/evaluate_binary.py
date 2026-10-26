"""Frozen binary addition evaluation; raw generated carry is the only state."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
import torch
from diffusers import AutoencoderDC
from common import *
from binary_protocol import programs,binary_read
from train_pixel import load_pixel_model,sample_pixel
from serial_image_inference import generate_serial

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
out=root/'evaluation';out.mkdir(exist_ok=False)
config=json.loads((root/'source/mac_sana_pixel_v1.json').read_text())
subprocess.run([sys.executable,str(root/'source/serial_inference_control.py'),'--root',str(root)],check=True)
torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.65)
model,_=load_pixel_model(root)
state=torch.load(root/'training/best.pt',map_location='cpu',weights_only=False)
assert state['pixel_protocol_sha256']==digest(root/'source/mac_sana_pixel_v1.json')
assert state['data_sha256']==digest(root/'data/manifest.json')
assert set(state['trainable'])=={n for n,p in model.named_parameters() if p.requires_grad}
assert not model.load_state_dict(state['trainable'],strict=False).unexpected_keys
model.eval();del state
snapshot=Path(json.loads((root/'model.json').read_text())['snapshot'])
vae=AutoencoderDC.from_pretrained(snapshot,subfolder='vae',variant='fp16',torch_dtype=torch.float32).eval().requires_grad_(False)
font=root/'fonts/train.ttf';other=root/'fonts/transfer.ttf'
banks=[answer_templates(font),answer_templates(other)]
def read(im):return binary_read(im,banks,read_answer)
def generate(im,seed):return generate_serial(model,vae,[im],seed,sample_pixel,pixels,pil,1)[0]
def append(name,row):
 with (out/name).open('a') as stream:stream.write(json.dumps(row)+'\n')
locals=[]
for style,f in [('familiar',font),('unfamiliar',other)]:
 folder=out/style;folder.mkdir()
 for repetition in range(25):
  for aa in range(2):
   for bb in range(2):
    for cc in range(2):
     identifier=f'{style}-{repetition:02d}-{aa}-{bb}-{cc}'
     seed=config['evaluation_seeds']['local']+repetition*4+aa*2+bb
     ref=render(aa,bb,cc,f);im=generate(ref,seed);path=folder/(identifier+'.png');im.save(path)
     reading=read(im)
     row={'id':identifier,'style':style,'repetition':repetition,'a':aa,'b':bb,'c':cc,'seed':seed,
          'expected':aa+bb+cc,**reading,'correct':reading['valid'] and reading['numeric_prediction']==aa+bb+cc,
          'input_rgb_sha256':hashlib.sha256(ref.tobytes()).hexdigest(),'image_sha256':digest(path)}
     locals.append(row);append('local-records.jsonl',row)
  if (repetition+1)%5==0:print(json.dumps({'style':style,'completed_repetitions':repetition+1}),flush=True)
local_summary={}
for style in ['familiar','unfamiliar']:
 rows=[r for r in locals if r['style']==style]
 local_summary[style]={'correct':sum(r['correct'] for r in rows),'count':len(rows),
                       'accuracy':sum(r['correct'] for r in rows)/len(rows),'valid':sum(r['valid'] for r in rows)}
atomic_json(out/'local-summary.json',local_summary)
program_rows=[]
for spec in programs(config['evaluation_seeds']['program']):
 folder=out/spec['id'];folder.mkdir();carry=None;images=[];input_digests=[]
 for j,(aa,bb) in enumerate(zip(spec['a_bits'],spec['b_bits'])):
  ref=render(aa,bb,0,font,carry_pixels=carry)
  input_digests.append(hashlib.sha256(ref.tobytes()).hexdigest())
  seed=config['evaluation_seeds']['program_noise']+spec['length']*10000+int(spec['id'].rsplit('-',1)[1])*100+j
  im=generate(ref,seed);im.save(folder/f'step-{j:02d}.png');images.append(im);carry=im.crop(CARRY_OUT)
 # Only after the complete program has been generated may answers be read/scored.
 readings=[read(im) for im in images]
 valid=all(r['valid'] for r in readings)
 prediction_bits=''.join([str(readings[-1]['carry']['prediction'])]+[str(r['digit']['prediction']) for r in readings[::-1]])
 prediction=int(prediction_bits,2) if valid else None
 correct_stages=True;expected_carry=0
 for j,r in enumerate(readings):
  expected=spec['a_bits'][j]+spec['b_bits'][j]+expected_carry
  correct_stages &= r['valid'] and r['numeric_prediction']==expected
  expected_carry=expected//2
  r.update(input_rgb_sha256=input_digests[j],image_sha256=digest(folder/f'step-{j:02d}.png'))
 row={**spec,'expected':spec['left']+spec['right'],'prediction_bits':prediction_bits,'prediction':prediction,
      'valid':valid,'all_stages_correct':bool(correct_stages),'stages':readings,
      'correct':valid and correct_stages and prediction==spec['left']+spec['right']}
 program_rows.append(row);append('program-records.jsonl',row)
 print(json.dumps({'program':spec['id'],'complete':True}),flush=True)
program_summary={}
for length in [4,8,16]:
 rows=[r for r in program_rows if r['length']==length]
 program_summary[str(length)]={'correct':sum(r['correct'] for r in rows),'count':len(rows),
                                'accuracy':sum(r['correct'] for r in rows)/len(rows),'valid':sum(r['valid'] for r in rows)}
success=(local_summary['familiar']['accuracy']>=config['gates']['local_exact_minimum'] and
         local_summary['unfamiliar']['accuracy']>=config['gates']['unfamiliar_style_exact_minimum'] and
         all(r['accuracy']>=config['gates']['each_program_length_exact_minimum'] for r in program_summary.values()))
atomic_json(out/'summary.json',{'local':local_summary,'programs':program_summary,'success':success,
    'checkpoint_sha256':digest(root/'training/best.pt'),'protocol_sha256':digest(root/'source/mac_sana_pixel_v1.json'),
    'data_sha256':digest(root/'data/manifest.json'),'claim_boundary':config['claim_boundary']})
