"""Resumable MPS adapter training on clean-context conditional image diffusion."""
import argparse, json, math, os, shutil, time
from pathlib import Path
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
import numpy as np,torch
from torch import nn
from diffusers import SanaTransformer2DModel
from common import atomic_json,digest

class Adapter(nn.Module):
 def __init__(self,base,rank=32):
  super().__init__();self.base=base
  self.down=nn.Linear(base.in_features,rank,bias=False)
  self.up=nn.Linear(rank,base.out_features,bias=False)
  nn.init.normal_(self.down.weight,std=.02);nn.init.zeros_(self.up.weight)
 @property
 def weight(self):return self.base.weight
 @property
 def bias(self):return self.base.bias
 def forward(self,x):return self.base(x)+self.up(self.down(x))

def load_model(root):
 snapshot=Path(json.loads((root/'model.json').read_text())['snapshot'])
 model=SanaTransformer2DModel.from_pretrained(snapshot,subfolder='transformer',variant='fp16',torch_dtype=torch.float32).requires_grad_(False)
 chosen=[]
 for name,mod in list(model.named_modules()):
  if isinstance(mod,nn.Linear) and ('.attn1.' in name or '.attn2.' in name):
   parent,leaf=name.rsplit('.',1);setattr(model.get_submodule(parent),leaf,Adapter(mod));chosen.append(name)
 model.proj_out.requires_grad_(True)
 assert len(chosen)>100
 model.enable_gradient_checkpointing()
 return model.to('mps'),chosen

def predict(model,x,s):
 b=x.shape[0]
 with torch.autocast('mps',dtype=torch.bfloat16):
  return model(hidden_states=x,encoder_hidden_states=torch.zeros(b,1,2304,device='mps'),encoder_attention_mask=torch.ones(b,1,device='mps'),timestep=s*1000,return_dict=False)[0].float()

def sample(model,contexts,seed,steps):
 generator=torch.Generator().manual_seed(seed)
 mask=torch.zeros(1,1,16,16,device='mps');mask[:,:,9:,:]=1
 outputs=[];was_training=model.training;model.eval()
 with torch.no_grad():
  for start in range(0,len(contexts),4):
   c=contexts[start:start+4].to('mps');x=torch.randn(c.shape,generator=generator).to('mps');x=x*mask+c*(1-mask)
   for k in range(steps):
    sigma=1-k/steps;v=predict(model,x,torch.full((len(c),),sigma,device='mps'))
    x=(x-v/steps)*mask+c*(1-mask)
   outputs.append(x.cpu())
 model.train(was_training)
 return torch.cat(outputs)

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--end',type=int,required=True);p.add_argument('--pilot',action='store_true');p.add_argument('--resume',type=Path);p.add_argument('--batch',type=int,default=4);p.add_argument('--skip-development',action='store_true');a=p.parse_args();root=a.root
 protocol=root/'source/mac_sana_column_v1.json';config=json.loads(protocol.read_text())
 assert shutil.disk_usage(root).free/2**30>40
 seed=config['training']['seed']
 torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.65);torch.manual_seed(seed)
 out=root/('pilot' if a.pilot else 'training');out.mkdir(exist_ok=True)
 manifest=json.loads((root/'data/manifest.json').read_text());rows=manifest['rows']
 contexts=[];targets=[]
 for r in rows:
  f=root/'data'/r['file'];assert digest(f)==r['sha256'],r['id'];z=np.load(f);assert set(z.files)=={'context','target'}
  contexts.append(z['context']);targets.append(z['target'])
 contexts=torch.tensor(np.stack(contexts));targets=torch.tensor(np.stack(targets))
 copy_ids=[i for i,r in enumerate(rows) if r['copy']];add_ids=[i for i,r in enumerate(rows) if not r['copy']]
 model,chosen=load_model(root);params=[p for p in model.parameters() if p.requires_grad]
 opt=torch.optim.AdamW(params,lr=config['training']['learning_rate'],weight_decay=.01,foreach=False)
 rng=np.random.default_rng(seed);gen=torch.Generator().manual_seed(seed);start=0
 if a.resume:
  state=torch.load(a.resume,map_location='cpu',weights_only=False)
  assert state['protocol_sha256']==digest(protocol) and state['data_sha256']==digest(root/'data/manifest.json')
  expected={n for n,p in model.named_parameters() if p.requires_grad}
  assert set(state['trainable'])==expected
  model.load_state_dict(state['trainable'],strict=False)
  opt.load_state_dict(state['optimizer']);rng.bit_generator.state=state['numpy_rng'];gen.set_state(state['torch_rng']);start=state['step']
 atomic_json(out/'model-audit.json',{'total_parameters_with_adapters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in params),'adapter_modules':chosen,'torch':torch.__version__,'protocol_sha256':digest(protocol),'data_sha256':digest(root/'data/manifest.json'),'batch':a.batch})
 mask=torch.zeros(1,1,16,16,device='mps');mask[:,:,9:,:]=1
 began=time.monotonic();times=[]
 def save(step):
  state={'step':step,'protocol_sha256':digest(protocol),'data_sha256':digest(root/'data/manifest.json'),'trainable':{n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad},'optimizer':opt.state_dict(),'numpy_rng':rng.bit_generator.state,'torch_rng':gen.get_state()}
  tmp=out/'resume.tmp.pt';torch.save(state,tmp);tmp.replace(out/'resume.pt')
  atomic_json(out/'checkpoint.json',{'step':step,'sha256':digest(out/'resume.pt'),'bytes':(out/'resume.pt').stat().st_size})
 model.train()
 for step in range(start+1,a.end+1):
  tick=time.monotonic()
  ids=rng.choice(copy_ids if step<=200 else add_ids,size=a.batch).tolist()
  c=contexts[ids].to('mps');y=targets[ids].to('mps');noise=torch.randn(y.shape,generator=gen).to('mps')
  s=torch.tensor(np.where(rng.random(a.batch)<.5,1.,rng.uniform(.1,.99,a.batch)),dtype=torch.float32,device='mps')
  ss=s[:,None,None,None];x=((1-ss)*y+ss*noise)*mask+c*(1-mask)
  pred=predict(model,x,s);error=(pred-(noise-y)).square()
  # Target-derived weights affect supervision only, never conditioning or sampling.
  weight=1+4*(y-c).abs().mean(1,keepdim=True).clamp(0,1)
  loss=(error*mask*weight).sum()/(mask.expand_as(error)*weight).sum()
  assert torch.isfinite(loss).item(),f'nonfinite loss at {step}'
  loss.backward();norm=nn.utils.clip_grad_norm_(params,1.0);assert math.isfinite(float(norm)),step
  opt.step();opt.zero_grad(set_to_none=True);torch.mps.synchronize();elapsed=time.monotonic()-tick;times.append(elapsed)
  row={'step':step,'loss':float(loss.detach()),'gradient_norm':float(norm),'seconds':elapsed,'mps_gib':torch.mps.driver_allocated_memory()/2**30,'elapsed_seconds':time.monotonic()-began,'phase':'copy' if step<=200 else 'add'}
  with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush()
  if step%10==0 or step==start+1:print(json.dumps(row),flush=True)
  if step%100==0 and not a.pilot:save(step)
  if shutil.disk_usage(root).free/2**30<40:save(step);raise RuntimeError('disk reserve reached')
  if time.monotonic()-began>18000:save(step);raise RuntimeError('session time budget reached')
 if a.pilot:
  atomic_json(out/'summary.json',{'steps':len(times),'median_seconds':float(np.median(times[2:])),'peak_mps_gib':torch.mps.driver_allocated_memory()/2**30,'passed':float(np.median(times[2:]))<20})
 else:
  save(a.end)
  if a.skip_development:return
  # Fixed development panels, fully generated from noise; no teacher-forced scoring.
  ids=[i for i,r in enumerate(rows) if r['variant']==0 and (r['copy'] or (r['a'] in [2,9]))]
  generated=sample(model,contexts[ids],1100910,12)
  np.savez_compressed(out/f'development-{a.end:05d}.npz',latents=generated.numpy(),indices=np.array(ids))
  print('SEGMENT_COMPLETE',a.end,flush=True)
if __name__=='__main__':main()
