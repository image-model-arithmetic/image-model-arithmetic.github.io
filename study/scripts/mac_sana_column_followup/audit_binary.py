"""Independent PNG/OCR recomputation of the registered binary experiment."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'mac_sana_column'))
from common import CARRY_OUT,answer_templates,digest,read_answer,render
from binary_protocol import programs


def ocr(im):
    stream=io.BytesIO();im.crop((128,300,384,488)).save(stream,format='PNG')
    result=subprocess.run(['tesseract','stdin','stdout','--psm','7','-c','tessedit_char_whitelist=0123456789'],
                          input=stream.getvalue(),capture_output=True,check=True,timeout=20)
    text=re.sub(r'\s+','',result.stdout.decode())
    return text,int(text,2) if re.fullmatch('[01]{2}',text) else None


def audit(root,fonts,config,ocr_reader=ocr,expected_protocol_sha256=None):
    summary=json.loads((root/'summary.json').read_text())
    local=[json.loads(x) for x in (root/'local-records.jsonl').read_text().splitlines()]
    recorded_programs=[json.loads(x) for x in (root/'program-records.jsonl').read_text().splitlines()]
    errors=[];details=[];program_details=[];inventory=set();banks=[answer_templates(f) for f in fonts]
    def require(condition,identifier,invariant):
        if not condition:errors.append({'id':identifier,'invariant':invariant})
    require(config['gates']=={'local_exact_minimum':.95,'unfamiliar_style_exact_minimum':.8,
                            'each_program_length_exact_minimum':.9},'protocol','unchanged binary success gates')
    if expected_protocol_sha256 is not None:
        require(summary['protocol_sha256']==expected_protocol_sha256,'protocol','evaluated protocol identity')
    def image_read(path,identifier,saved):
        require(path.is_file(),identifier,'image exists')
        if not path.is_file():return None,None,None
        im=Image.open(path).convert('RGB');inventory.add(path.relative_to(root).as_posix())
        require(im.size==(512,512),identifier,'image dimensions')
        require(digest(path)==saved['image_sha256'],identifier,'image SHA-256')
        candidates=[read_answer(im,b) for b in banks]
        fresh=min(candidates,key=lambda r:r['carry']['mse']+r['digit']['mse'])
        valid=fresh['valid'] and fresh['digit']['prediction'] in [0,1]
        number=2*fresh['carry']['prediction']+fresh['digit']['prediction']
        require(saved['valid']==valid,identifier,'binary validity recomputation')
        require(saved['prediction']==fresh['prediction'],identifier,'display-label recomputation')
        require(saved['numeric_prediction']==number,identifier,'binary numeric recomputation')
        for component in ['carry','digit']:
            for key in ['prediction','valid','mse','ink']:
                old=saved[component][key];new=fresh[component][key]
                require(abs(old-new)<=1e-6 if key in ['mse','ink'] else old==new,identifier,component+'.'+key)
        text,other=ocr_reader(im)
        return im,number if valid else None,{'text':text,'prediction':other}
    expected_ids={f'{style}-{repeat:02d}-{a}-{b}-{c}' for style in ['familiar','unfamiliar']
                  for repeat in range(25) for a in range(2) for b in range(2) for c in range(2)}
    counts=Counter(row['id'] for row in local)
    require(set(counts)==expected_ids,'local-inventory','exact 400-case identifier set')
    for identifier,count in counts.items():require(count==1,identifier,'unique identifier')
    for row in local:
        identifier=row['id'];style=row['style'];a,b,c=row['a'],row['b'],row['c'];repeat=row['repetition']
        require(identifier==f'{style}-{repeat:02d}-{a}-{b}-{c}',identifier,'identifier fields')
        require(row['seed']==config['evaluation_seeds']['local']+repeat*4+a*2+b,identifier,'fixed seed')
        expected=a+b+c;require(row['expected']==expected,identifier,'expected subtotal')
        reference=render(a,b,c,fonts[['familiar','unfamiliar'].index(style)])
        require(row['input_rgb_sha256']==hashlib.sha256(reference.tobytes()).hexdigest(),identifier,'answer-free raster input')
        im,pred,other=image_read(root/style/(identifier+'.png'),identifier,row)
        if im is None:continue
        correct=pred==expected;ocr_correct=other['prediction']==expected
        require(row['correct']==correct,identifier,'local exactness')
        details.append({'id':identifier,'style':style,'expected':expected,'prediction':pred,'correct':correct,
                        'ocr':other,'ocr_correct':ocr_correct,'consensus_correct':correct and ocr_correct})
    specs=programs(config['evaluation_seeds']['program']);expected={s['id']:s for s in specs}
    counts=Counter(row['id'] for row in recorded_programs)
    require(set(counts)==set(expected),'program-inventory','exact 96-program identifier set')
    for identifier,count in counts.items():require(count==1,identifier,'unique identifier')
    for row in recorded_programs:
        identifier=row['id'];spec=expected.get(identifier)
        if spec is None:continue
        for key in ['length','left','right','a_bits','b_bits']:require(row[key]==spec[key],identifier,'prospective '+key)
        stages=row['stages'];require(len(stages)==spec['length'],identifier,'stage count')
        expected_sum=spec['left']+spec['right'];require(row['expected']==expected_sum,identifier,'full sum')
        carry=None;expected_carry=0;correct_stages=True;ocr_stages=True;predicted=[];ocr_predicted=[]
        for j,stage in enumerate(stages):
            stage_id=f'{identifier}/step-{j:02d}'
            aa=(spec['left']>>j)&1;bb=(spec['right']>>j)&1
            ref=render(aa,bb,0,fonts[0],carry_pixels=carry)
            require(stage['input_rgb_sha256']==hashlib.sha256(ref.tobytes()).hexdigest(),stage_id,'raw generated carry transport')
            im,pred,other=image_read(root/identifier/f'step-{j:02d}.png',stage_id,stage)
            if im is None:
                correct_stages=ocr_stages=False;continue
            carry=im.crop(CARRY_OUT)
            subtotal=aa+bb+expected_carry;expected_carry=subtotal//2
            correct_stages &= pred==subtotal;ocr_stages &= other['prediction']==subtotal
            predicted.append(pred);ocr_predicted.append(other['prediction'])
        def assemble(values):
            if len(values)!=spec['length'] or any(v is None for v in values):return None
            return sum((v%2)<<j for j,v in enumerate(values))+((values[-1]//2)<<len(values))
        final=assemble(predicted);other_final=assemble(ocr_predicted)
        all_valid=final is not None
        correct=bool(all_valid and correct_stages and final==expected_sum)
        ocr_correct=bool(other_final is not None and ocr_stages and other_final==expected_sum)
        require(row['prediction']==final,identifier,'assembled numeric answer')
        require(row['valid']==all_valid,identifier,'all-stage validity')
        require(row['all_stages_correct']==correct_stages,identifier,'all-stage exactness')
        require(row['correct']==correct,identifier,'complete program exactness')
        bits=''.join([str(stages[-1]['carry']['prediction'])]+[str(s['digit']['prediction']) for s in stages[::-1]]) if stages else ''
        require(row['prediction_bits']==bits,identifier,'recorded output digit order')
        program_details.append({'id':identifier,'length':spec['length'],'expected':expected_sum,'prediction':final,
            'correct':correct,'valid':all_valid,'ocr_prediction':other_final,'ocr_correct':ocr_correct,
            'consensus_correct':correct and ocr_correct})
    require({str(p.relative_to(root)) for p in root.rglob('*.png')}==inventory,'image-inventory','exact generated PNG set')
    local_metrics={};program_metrics={}
    for style in ['familiar','unfamiliar']:
        rows=[r for r in details if r['style']==style]
        local_metrics[style]={'count':len(rows),'correct':sum(r['correct'] for r in rows),
          'valid':sum(r['prediction'] is not None for r in rows),'ocr_correct':sum(r['ocr_correct'] for r in rows),
          'consensus_correct':sum(r['consensus_correct'] for r in rows)}
    for length in [4,8,16]:
        rows=[r for r in program_details if r['length']==length]
        program_metrics[str(length)]={'count':len(rows),'correct':sum(r['correct'] for r in rows),
          'valid':sum(r['valid'] for r in rows),'ocr_correct':sum(r['ocr_correct'] for r in rows),
          'consensus_correct':sum(r['consensus_correct'] for r in rows)}
    for name,fresh in [('local',local_metrics),('programs',program_metrics)]:
        for key,metrics in fresh.items():
            primary={k:metrics[k] for k in ['correct','count','valid']}
            primary['accuracy']=metrics['correct']/metrics['count'] if metrics['count'] else 0
            require(summary[name][key]==primary,f'{name}/{key}','reported aggregate recomputation')
    def passes(key):
        return (local_metrics['familiar']['count']==200 and local_metrics['unfamiliar']['count']==200 and
            local_metrics['familiar'][key]>=190 and local_metrics['unfamiliar'][key]>=160 and
            all(r['count']==32 and r[key]>=29 for r in program_metrics.values()))
    require(summary['success']==passes('correct'),'summary','registered primary success')
    return {'integrity_passed':not errors,'audited_success':not errors and passes('correct') and passes('consensus_correct'),
            'local':local_metrics,'programs':program_metrics,'failures':errors,'local_records':details,
            'independent_local':{k:{n:v[n] for n in ['count','ocr_correct','consensus_correct']} for k,v in local_metrics.items()},
            'independent_programs':{k:{n:v[n] for n in ['count','ocr_correct','consensus_correct']} for k,v in program_metrics.items()},
            'program_records':program_details,'reported_summary_sha256':digest(root/'summary.json'),
            'scope':config['claim_boundary']}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--fonts',nargs=2,type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    version=subprocess.check_output(['tesseract','--version'],text=True).splitlines()[0]
    assert version=='tesseract 5.5.3',version
    result=audit(a.directory,a.fonts,json.loads(a.protocol.read_text()),expected_protocol_sha256=digest(a.protocol))
    result.update(protocol_sha256=digest(a.protocol),script_sha256=digest(Path(__file__)),
                  tesseract_version=version,font_inventory=[{'path':str(f),'sha256':digest(f)} for f in a.fonts],
                  reader_common_sha256=digest(Path(__file__).resolve().parent.parent/'mac_sana_column/common.py'))
    a.output.parent.mkdir(parents=True,exist_ok=True);assert not a.output.exists()
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:result[k] for k in ['integrity_passed','audited_success','local','programs']}))
    assert result['integrity_passed'],result['failures']
