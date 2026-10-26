"""Generate paired carry interventions; answer reading is absent from this process."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
sys.dont_write_bytecode=True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
from generate_long_binary import digest,write,load_committed_stage,preserve_uncommitted
from carry_intervention_protocol import cases,audit_corpus


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--parent',type=Path,required=True);args=parser.parse_args()
    root,parent=args.root.resolve(),args.parent.resolve()
    config_path=root/'source/protocol.json';cfg=json.loads(config_path.read_text())
    admission=json.loads((root/'admission.json').read_text())
    assert admission['passed'] and admission['protocol_sha256']==digest(config_path)
    assert audit_corpus(cfg)['passed']
    for base in [root,parent]:
        for row in json.loads((base/'source-manifest.json').read_text())['files']:
            p=base/row['path'];assert p.stat().st_size==row['bytes'] and digest(p)==row['sha256'],row['path']
    assert digest(parent/'source/mac_sana_pixel_v1.json')==cfg['parent_protocol_sha256']
    checkpoint=parent/'training/best.pt';assert digest(checkpoint)==admission['checkpoint_sha256']
    specs_path=root/'data/cases.json';specs=json.loads(specs_path.read_text())
    assert specs==cases(cfg) and digest(specs_path)==admission['cases_sha256']
    sys.path.insert(0,str(parent/'source'))
    from common import CARRY_OUT,render,pixels,pil
    from train_pixel import load_pixel_model,sample_pixel
    from serial_image_inference import generate_serial
    import torch
    from diffusers import AutoencoderDC
    torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.65)
    model,_=load_pixel_model(parent)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert state['pixel_protocol_sha256']==cfg['parent_protocol_sha256']
    assert set(state['trainable'])=={n for n,p in model.named_parameters() if p.requires_grad}
    assert not model.load_state_dict(state['trainable'],strict=False).unexpected_keys
    model.eval();del state
    base=json.loads((parent/'model.json').read_text());snapshot=Path(base['snapshot'])
    assert base['revision']==cfg['base_revision']
    for row in base['files']:
        p=snapshot/row['path'];assert p.stat().st_size==row['bytes'] and digest(p)==row['sha256']
    vae=AutoencoderDC.from_pretrained(snapshot,subfolder='vae',variant='fp16',torch_dtype=torch.float32).eval().requires_grad_(False)
    out=root/'evaluation';out.mkdir(exist_ok=True)
    identity={'protocol_sha256':digest(config_path),'cases_sha256':digest(specs_path),
        'checkpoint_sha256':admission['checkpoint_sha256'],'script_sha256':digest(Path(__file__)),
        'parent_source_manifest_sha256':digest(parent/'source-manifest.json'),
        'font_sha256':digest(parent/'fonts/train.ttf')}
    if (out/'identity.json').exists():assert json.loads((out/'identity.json').read_text())==identity
    else:
        assert not list(out.iterdir());write(out/'identity.json',identity)
    started=time.monotonic();completed=0
    def generate(identifier,reference,seed,donor):
        nonlocal completed
        assert time.monotonic()-started<cfg['resources']['maximum_controller_hours']*3600
        path=out/(identifier+'.png');receipt=out/(identifier+'.json')
        expected={'id':identifier,'seed':seed,'incoming_donor':donor,
                  'input_rgb_sha256':hashlib.sha256(reference.tobytes()).hexdigest(),'image_path':path.name}
        if receipt.exists():image,saved=load_committed_stage(path,receipt,expected)
        else:
            preserve_uncommitted(root,identifier,[path,path.with_name(path.stem+'.tmp.png'),receipt.with_name(receipt.stem+'.tmp.json')],expected)
            image=generate_serial(model,vae,[reference],seed,sample_pixel,pixels,pil,1)[0]
            temporary=path.with_name(path.stem+'.tmp.png');image.save(temporary);assert temporary.is_file();temporary.replace(path)
            write(receipt,{**expected,'image_sha256':digest(path),'image_bytes':path.stat().st_size})
        completed+=1
        write(root/'progress.json',{'completed_images':completed,'total_images':184,'last_case':identifier,'updated_unix':time.time()})
        return image
    donors={}
    font=parent/'fonts/train.ttf'
    for spec in specs['donors']:
        donors[spec['id']]=generate(spec['id'],render(spec['a'],spec['b'],spec['c'],font),spec['seed'],None)
    for spec in specs['pairs']:
        for carry,key in [(0,'zero_donor'),(1,'one_donor')]:
            donor=spec[key]
            reference=render(spec['a'],spec['b'],0,font,carry_pixels=donors[donor].crop(CARRY_OUT))
            generate(spec['id']+f'-carry{carry}',reference,spec['seed'],donor)
        if spec['repeat_zero']:
            donor=spec['zero_donor'];reference=render(spec['a'],spec['b'],0,font,carry_pixels=donors[donor].crop(CARRY_OUT))
            generate(spec['id']+'-repeat0',reference,spec['seed'],donor)
        print(json.dumps({'pair':spec['id'],'completed_images':completed,'total_images':184}),flush=True)
    assert completed==184
    write(out/'generation-summary.json',{**identity,'complete':True,'image_count':completed,
        'pair_count':80,'answer_scoring_performed':False,'finished_unix':time.time()})


if __name__=='__main__':main()
