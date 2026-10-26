"""Independent local Tesseract audit; never used in generation or training."""
import argparse,io,json,re,subprocess
from pathlib import Path
from PIL import Image
from common import atomic_json,digest
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
summary=json.loads((a.directory/'summary.json').read_text());records=summary['records'];results=[]
for r in records:
 path=a.directory/(r['id']+'.png');im=Image.open(path).convert('RGB').crop((128,300,384,488));b=io.BytesIO();im.save(b,format='PNG')
 process=subprocess.run(['tesseract','stdin','stdout','--psm','7','-c','tessedit_char_whitelist=0123456789'],input=b.getvalue(),capture_output=True,check=True,timeout=20)
 text=re.sub(r'\s+','',process.stdout.decode());valid=bool(re.fullmatch(r'\d{2}',text));prediction=int(text) if valid else None
 results.append({'id':r['id'],'copy':r['copy'],'expected':r['expected'],'template_correct':r['correct'],'ocr_text':text,'ocr_valid':valid,'ocr_prediction':prediction,'ocr_correct':valid and prediction==r['expected'],'consensus_correct':r['correct'] and valid and prediction==r['expected'],'image_sha256':digest(path)})
metrics={}
for name,rows in [('copy',[r for r in results if r['copy']]),('add',[r for r in results if not r['copy']])]:
 metrics[name]={'count':len(rows),'template_correct':sum(r['template_correct'] for r in rows),'ocr_correct':sum(r['ocr_correct'] for r in rows),'consensus_correct':sum(r['consensus_correct'] for r in rows)}
atomic_json(a.output,{'tesseract_version':subprocess.check_output(['tesseract','--version'],text=True).splitlines()[0],'command':'tesseract stdin stdout --psm 7 -c tessedit_char_whitelist=0123456789','crop':[128,300,384,488],'parser':'exactly two decimal digits after removing whitespace','metrics':metrics,'records':results});print(json.dumps(metrics))
