"""Train and freeze an output-digit critic before any arithmetic fine-tuning."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
import torch
from torch.nn import functional as F
from safetensors.torch import save_file, load_file

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();root=args.root
sys.path.insert(0,str(root/'source'))
from common import answer_templates, atomic_json, digest, pixels, render
from ocr_critic import DigitCritic, augment, critic_loss
protocol=root/'source/mac_sana_pixel_v1.json';config=json.loads(protocol.read_text())['critic']
torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.65);torch.manual_seed(config['seed'])
output=root/'critic';output.mkdir(exist_ok=False)
font=root/'fonts/train.ttf'
bank=torch.from_numpy(np.stack([1-x for x in answer_templates(font)]))[:,None]
bank=F.interpolate(bank.contiguous(),size=(32,32),mode='bilinear',align_corners=False)
generator=torch.Generator().manual_seed(config['seed'])
model=DigitCritic().to('mps').train()
optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=.01)
started=time.monotonic()
for step in range(1,config['updates']+1):
    labels=torch.randint(0,10,(config['batch_size'],),generator=generator)
    images=augment(bank,labels,generator).to('mps')
    loss=F.cross_entropy(model(images),labels.to('mps'))
    assert torch.isfinite(loss)
    loss.backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
    if step%100==0:
        record={'step':step,'loss':float(loss.detach()),'elapsed_seconds':time.monotonic()-started}
        with (output/'training.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
model.eval().requires_grad_(False)
tensors={name:value.detach().cpu().contiguous() for name,value in model.state_dict().items()}
assert all(torch.isfinite(value).all() for value in tensors.values())
save_file(tensors,str(output/'model.safetensors'),metadata={'protocol_sha256':digest(protocol),'training_font_sha256':digest(font),'role':'frozen training-only output-digit recognizer; no operands or arithmetic'})
restored=load_file(str(output/'model.safetensors'))
assert set(restored)==set(tensors) and all(torch.equal(value,restored[name]) for name,value in tensors.items())
# Score the exact frozen, input-gradient-enabled path that the image loss will use.
canonical=model(bank.to('mps').requires_grad_(True)).detach().argmax(-1).cpu()
validation=torch.Generator().manual_seed(config['validation_seed'])
failures=[];correct=0;counts=[0]*10
for start in range(0,config['validation_cases'],100):
    labels=torch.arange(start,min(start+100,config['validation_cases']))%10
    batch=augment(bank,labels,validation).to('mps').requires_grad_(True)
    predicted=model(batch).detach().argmax(-1).cpu()
    for j,(truth,guess) in enumerate(zip(labels.tolist(),predicted.tolist())):
        counts[truth]+=1;correct+=truth==guess
        if truth!=guess:failures.append({'id':f'critic-validation-{start+j:04d}','expected':truth,'prediction':guess})
del model,optimizer,images,loss,batch
torch.mps.empty_cache()
# Wrong rendered outputs ensure a non-trivial derivative, rather than a saturated
# already-correct loss that could conceal backend gradient disagreements.
source=torch.from_numpy(np.stack([pixels(render(0,0,0,font,answer=y))[:,288:,128:384] for y in [3,18]]))
targets=torch.tensor([7,12],dtype=torch.long)
comparisons={}
for device in ['cpu','mps']:
    critic=DigitCritic();critic.load_state_dict(tensors);critic=critic.to(device).eval().requires_grad_(False)
    point=source.detach().clone().to(device).requires_grad_(True)
    value=critic_loss(point,targets.to(device),critic);value.backward()
    comparisons[device]={'loss':float(value.detach()),'gradient':point.grad.detach().cpu().clone()}
    if device=='cpu':
        trial=(point.detach()-config['control_gradient_descent_step']*point.grad.detach()).requires_grad_(True)
        after=float(critic_loss(trial,targets,critic).detach())
    del point,value,critic
delta=comparisons['mps']['gradient']-comparisons['cpu']['gradient']
norm=float(comparisons['cpu']['gradient'].norm())
backend={'cpu_loss':comparisons['cpu']['loss'],'mps_loss':comparisons['mps']['loss'],
         'loss_absolute_difference':abs(comparisons['cpu']['loss']-comparisons['mps']['loss']),
         'gradient_relative_l2_difference':float(delta.norm()/comparisons['cpu']['gradient'].norm().clamp_min(1e-12)),
         'gradient_maximum_absolute_difference':float(delta.abs().max()),'gradient_norm':norm,
         'loss_after_cpu_gradient_step':after,'gradient_step':config['control_gradient_descent_step'],
         'loss_tolerance':config['cpu_mps_loss_absolute_tolerance'],
         'gradient_tolerance':config['cpu_mps_gradient_relative_l2_tolerance']}
backend['passed']=(backend['loss_absolute_difference']<=backend['loss_tolerance'] and
                   backend['gradient_relative_l2_difference']<=backend['gradient_tolerance'] and
                   norm>config['minimum_control_gradient_norm'] and after<backend['cpu_loss'])
canonical_correct=int((canonical==torch.arange(10)).sum())
passed=canonical_correct==config['canonical_digits_correct_required'] and correct/config['validation_cases']>=config['minimum_validation_accuracy'] and backend['passed']
receipt={'passed':passed,'canonical_correct':canonical_correct,'canonical_count':10,
         'validation_correct':correct,'validation_count':config['validation_cases'],
         'validation_accuracy':correct/config['validation_cases'],'validation_label_counts':counts,
         'validation_failures':failures,'backend_equivalence':backend,
         'critic_sha256':digest(output/'model.safetensors'),'critic_bytes':(output/'model.safetensors').stat().st_size,
         'critic_parameters':sum(t.numel() for t in tensors.values()),'protocol_sha256':digest(protocol),
         'font_sha256':digest(font),'script_sha256':digest(Path(__file__)),
         'scope':'Recognition of output digits only; the critic sees no operands, performs no arithmetic, and is absent from inference'}
atomic_json(output/'validation.json',receipt)
print(json.dumps({k:v for k,v in receipt.items() if k!='validation_failures'}),flush=True)
assert passed, 'Frozen digit critic did not satisfy its prospective calibration gates'
