"""Independently verify all paired inputs, outputs and exact-repeat controls."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'mac_sana_column'))
from common import CARRY_IN,CARRY_OUT,answer_templates,digest,read_answer,render
from audit_binary import ocr
from carry_intervention_protocol import cases


def audit(directory,cfg,specs,fonts,parent_report,protocol_sha,ocr_reader=ocr,expected_identity=None,
          require_parent_success=True):
    errors=[];inventory={'identity.json','generation-summary.json'}
    def require(condition,identifier,invariant):
        if not condition:errors.append({'id':identifier,'invariant':invariant})
    require(specs==cases(cfg),'cases','exact prospective corpus')
    require(type(require_parent_success) is bool,'protocol','explicit parent mathematical gate')
    require(parent_report['complete'] and parent_report['integrity_passed'] and
            (parent_report['audited_success'] or not require_parent_success),
            'parent','complete independently audited parent and declared mathematical gate')
    require(parent_report['checkpoint']['pixel_protocol_sha256']==cfg['parent_protocol_sha256'],'parent','protocol binding')
    identity=json.loads((directory/'identity.json').read_text())
    generation=json.loads((directory/'generation-summary.json').read_text())
    require(identity['protocol_sha256']==protocol_sha,'identity','protocol binding')
    require(identity['checkpoint_sha256']==parent_report['checkpoint']['full_checkpoint_sha256'],'identity','checkpoint binding')
    require(identity['font_sha256']==digest(fonts[0]),'identity','input font binding')
    if expected_identity:
        for key,value in expected_identity.items():require(identity.get(key)==value,'identity',key)
    for key,value in identity.items():require(generation.get(key)==value,'generation',key)
    require(generation.get('complete') is True and generation.get('image_count')==184 and
        generation.get('pair_count')==80 and generation.get('answer_scoring_performed') is False,
        'generation','complete unscored generation')
    require(cfg['gates']=={'donors_correct_required':8,'paired_consensus_minimum':76,
        'pairs_required':80,'each_operand_pair_minimum':19,'each_donor_bank_minimum':19,
        'exact_repeat_controls_required':16},'protocol','unchanged causal gates')
    banks=[answer_templates(font) for font in fonts]
    def load(identifier,reference,seed,donor,expected):
        record_path=directory/(identifier+'.json');image_path=directory/(identifier+'.png')
        inventory.update([record_path.name,image_path.name])
        if not record_path.is_file() or not image_path.is_file():
            require(False,identifier,'image and commit exist');return None,None
        record=json.loads(record_path.read_text())
        required={'id':identifier,'seed':seed,'incoming_donor':donor,
            'input_rgb_sha256':hashlib.sha256(reference.tobytes()).hexdigest(),'image_path':image_path.name}
        require(set(record)==set(required)|{'image_sha256','image_bytes'},identifier,'exact commit schema')
        for key,value in required.items():require(record.get(key)==value,identifier,key)
        require(record.get('image_sha256')==digest(image_path),identifier,'image SHA-256')
        require(record.get('image_bytes')==image_path.stat().st_size,identifier,'image size')
        image=Image.open(image_path)
        require(image.mode=='RGB' and image.size==(512,512),identifier,'image format')
        readings=[read_answer(image,bank) for bank in banks]
        result=min(readings,key=lambda r:r['carry']['mse']+r['digit']['mse'])
        valid=result['valid'] and result['digit']['prediction'] in [0,1]
        prediction=2*result['carry']['prediction']+result['digit']['prediction'] if valid else None
        text,other=ocr_reader(image)
        detail={'id':identifier,'expected':expected,'template_prediction':prediction,'ocr_text':text,
            'ocr_prediction':other,'template_correct':prediction==expected,'ocr_correct':other==expected,
            'consensus_correct':prediction==expected and other==expected}
        return image,detail
    donor_images={};donor_details=[]
    for spec in specs['donors']:
        reference=render(spec['a'],spec['b'],spec['c'],fonts[0])
        image,detail=load(spec['id'],reference,spec['seed'],None,spec['a']+spec['b']+spec['c'])
        donor_images[spec['id']]=image
        if detail is not None:donor_details.append(detail)
    pairs=[];repeats=[]
    for spec in specs['pairs']:
        zero=donor_images[spec['zero_donor']];one=donor_images[spec['one_donor']]
        if zero is None or one is None:
            require(False,spec['id'],'both donor images available');continue
        references=[render(spec['a'],spec['b'],0,fonts[0],carry_pixels=im.crop(CARRY_OUT)) for im in [zero,one]]
        outside=np.ones((512,512),dtype=bool);x0,y0,x1,y1=CARRY_IN;outside[y0:y1,x0:x1]=False
        require(np.array_equal(np.asarray(references[0])[outside],np.asarray(references[1])[outside]),
                spec['id'],'paired inputs differ only inside carry crop')
        images=[];readings=[]
        for carry,reference in enumerate(references):
            donor=spec['one_donor'] if carry else spec['zero_donor']
            im,detail=load(spec['id']+f'-carry{carry}',reference,spec['seed'],donor,spec['a']+spec['b']+carry)
            images.append(im);readings.append(detail)
        if any(r is None for r in readings):continue
        def increments(key):
            values=[r[key] for r in readings]
            return all(v is not None for v in values) and values[1]-values[0]==1
        pair={'id':spec['id'],'bank':spec['bank'],'operands':f"{spec['a']}{spec['b']}",
            'seed':spec['seed'],'readings':readings,
            'template_correct':all(r['template_correct'] for r in readings),
            'ocr_correct':all(r['ocr_correct'] for r in readings),
            'consensus_correct':all(r['consensus_correct'] for r in readings),
            'template_increment_one':increments('template_prediction'),'ocr_increment_one':increments('ocr_prediction')}
        pairs.append(pair)
        if spec['repeat_zero']:
            image,detail=load(spec['id']+'-repeat0',references[0],spec['seed'],spec['zero_donor'],spec['a']+spec['b'])
            if image is not None:
                difference=np.abs(np.asarray(image,dtype=np.int16)-np.asarray(images[0],dtype=np.int16))
                repeats.append({'id':spec['id']+'-repeat0','maximum_pixel_difference':int(difference.max()),
                                'exact':not bool(np.any(difference)),'reading':detail})
    actual={p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}
    require(actual==inventory,'inventory','exact 370-file evaluation inventory')
    require(len(actual)==370,'inventory','370 files')
    def metrics(rows):
        return {'count':len(rows),**{key:sum(r[key] for r in rows) for key in ['template_correct','ocr_correct','consensus_correct']}}
    paired=metrics(pairs);donor_metrics=metrics(donor_details)
    by_operand={key:metrics([r for r in pairs if r['operands']==key]) for key in ['00','01','10','11']}
    by_bank={str(bank):metrics([r for r in pairs if r['bank']==bank]) for bank in range(4)}
    deterministic=len(repeats)==16 and all(r['exact'] for r in repeats)
    math_passes=(len(donor_details)==8 and donor_metrics['consensus_correct']==8 and
        paired['count']==80 and paired['consensus_correct']>=76 and
        all(r['count']==20 and r['consensus_correct']>=19 for r in [*by_operand.values(),*by_bank.values()]))
    return {'integrity_passed':not errors,'audited_success':not errors and deterministic and math_passes,
        'parent_mathematical_success_required':require_parent_success,
        'determinism_passed':deterministic,'donors':donor_metrics,'pairs':paired,'by_operand':by_operand,
        'by_donor_bank':by_bank,'repeat_controls':repeats,'donor_records':donor_details,'pair_records':pairs,
        'failures':errors,'checkpoint_sha256':identity['checkpoint_sha256'],'protocol_sha256':protocol_sha,
        'template_increment_one_count':sum(r['template_increment_one'] for r in pairs),
        'ocr_increment_one_count':sum(r['ocr_increment_one'] for r in pairs),'claim_boundary':cfg['claim_boundary']}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,required=True);parser.add_argument('--cases',type=Path,required=True)
    parser.add_argument('--fonts',nargs=2,type=Path,required=True);parser.add_argument('--parent-report',type=Path,required=True)
    parser.add_argument('--generation-source',type=Path,required=True);parser.add_argument('--parent-source-manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert subprocess.check_output(['tesseract','--version'],text=True).splitlines()[0]=='tesseract 5.5.3'
    result=audit(args.directory,json.loads(args.protocol.read_text()),json.loads(args.cases.read_text()),args.fonts,
        json.loads(args.parent_report.read_text()),digest(args.protocol),expected_identity={
            'cases_sha256':digest(args.cases),'script_sha256':digest(args.generation_source),
            'parent_source_manifest_sha256':digest(args.parent_source_manifest)})
    result.update(script_sha256=digest(Path(__file__)),parent_report_sha256=digest(args.parent_report),
                  font_sha256=[digest(p) for p in args.fonts])
    args.output.parent.mkdir(parents=True,exist_ok=True);assert not args.output.exists()
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:result[k] for k in ['integrity_passed','audited_success','determinism_passed','donors','pairs']}))
    assert result['integrity_passed'],result['failures']
