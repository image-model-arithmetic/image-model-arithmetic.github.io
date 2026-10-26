"""One-job-at-a-time Mac runner with durable status and measured time budget."""
import argparse,json,os,shutil,signal,subprocess,time
from pathlib import Path
from common import atomic_json,digest
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root;python=root/'.venv/bin/python';start=time.time()
status={'phase':'waiting-for-data','started_unix':start,'complete':False};atomic_json(root/'status.json',status)
def run(name,args):
 status.update(phase=name,updated_unix=time.time());atomic_json(root/'status.json',status)
 with (root/(name+'.log')).open('ab') as log:
  child=subprocess.Popen([str(python),'-u',*map(str,args)],cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  atomic_json(root/'active-process.json',{'pid':child.pid,'phase':name,'started_unix':time.time()})
  low=0
  while child.poll() is None:
   time.sleep(5)
   disk=shutil.disk_usage(root).free/2**30
   pressure=subprocess.check_output(['memory_pressure'],text=True)
   import re
   match=re.search(r'System-wide memory free percentage: (\d+)%',pressure)
   free=int(match[1]) if match else 0
   low=low+1 if free<15 else 0
   with (root/'resource-monitor.jsonl').open('a') as f:f.write(json.dumps({'unix':time.time(),'phase':name,'pid':child.pid,'disk_free_gib':disk,'memory_free_percent':free})+'\n')
   if disk<40 or low>=2:
    os.killpg(child.pid,signal.SIGINT)
    try:child.wait(timeout=30)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
    raise RuntimeError(f'Resource guard: disk={disk:.2f} GiB, memory free={free}%')
  if child.returncode:raise RuntimeError(f'{name} exited {child.returncode}; see {name}.log')
try:
 while not (root/'data/manifest.json').exists():
  if time.time()-start>2400:raise RuntimeError('data preparation timeout')
  time.sleep(10)
 run('pilot-run',['source/train.py','--root',root,'--end',10,'--pilot'])
 pilot=json.loads((root/'pilot/summary.json').read_text());assert pilot['passed'],pilot
 step_seconds=pilot['median_seconds'];endpoint=min(4000,max(200,int(12000/step_seconds)//100*100))
 atomic_json(root/'schedule.json',{'pilot':pilot,'max_updates':endpoint,'estimated_training_seconds':endpoint*step_seconds,'development_endpoints':[e for e in [200,800,2000,endpoint] if e<=endpoint]})
 best_score=(-1,-1);best_step=None
 for end in sorted(set(e for e in [200,800,2000,endpoint] if e<=endpoint)):
  args=['source/train.py','--root',root,'--end',end]
  if (root/'training/resume.pt').exists():args+=['--resume',root/'training/resume.pt']
  run(f'train-{end}',args)
  latent=root/f'training/development-{end:05d}.npz'
  run(f'decode-{end}',['source/decode.py','--root',root,'--input',latent])
  result=json.loads(latent.with_suffix('').joinpath('summary.json').read_text())['metrics']
  score=(result['add']['correct'],result['copy']['correct'])
  if score>best_score:
   best_score=score;best_step=end
   shutil.copy2(root/'training/resume.pt',root/'training/best.pt')
   atomic_json(root/'training/selection.json',{'step':end,'metrics':result,'sha256':digest(root/'training/best.pt'),'selection':'fixed local development panel only'})
  status.update(last_step=end,last_development=result,best_step=best_step);atomic_json(root/'status.json',status)
 status.update(phase='training-and-development-complete',complete=True,finished_unix=time.time(),best_step=best_step)
 atomic_json(root/'status.json',status)
 # Independent evaluation is invoked only by a checked evaluator present in the sealed source.
 if (root/'source/evaluate.py').is_file():
  status['complete']=False
  run('independent-evaluation',['source/evaluate.py','--root',root])
  status.update(phase='complete',complete=True,finished_unix=time.time());atomic_json(root/'status.json',status)
except BaseException as e:
 status.update(phase='stopped',error=repr(e),finished_unix=time.time());atomic_json(root/'status.json',status);raise
