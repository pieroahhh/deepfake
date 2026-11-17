"""Sequential GPU runner for the explicitly requested short scaling benchmark."""
import json,os,subprocess,sys,time
from pathlib import Path
W=next(p for p in Path(__file__).resolve().parents if (p/'workspace-layout.json').is_file())
sys.path.insert(0,str(W/'06_Tools/workspace'))
from workspace_paths import workspace_path
sys.path.insert(0,str(workspace_path('main')/'src'))
from deepfake_detection.gpu_queue import gpu_slot
OUT=workspace_path('main_results')/'training-scaling-001';OUT.mkdir(parents=True,exist_ok=True)
SCRIPT=Path(__file__).with_name('benchmark_training_scaling.py');PYTHON=workspace_path('runtime')/'environments/main/.venv-gpu/Scripts/python.exe'
STAGES=['prepare','ours','avf-visual','avf-audio','avt-audio','avt-visual','avt-audiovisual','avt-fusion']
def progress(stage,done,status='RUNNING',**extra):
 d={'status':status,'stage':stage,'processed':done,'total':len(STAGES),**extra};tmp=OUT/'progress.tmp';tmp.write_text(json.dumps(d,indent=2),encoding='utf-8');os.replace(tmp,OUT/'progress.json');print(json.dumps(d),flush=True)
def run():
 with gpu_slot():
  for i,stage in enumerate(STAGES):
   marker=OUT/('protocol.json' if stage=='prepare' else stage+'.json')
   if marker.exists():continue
   progress(stage,i)
   if stage=='avt-audiovisual':
    docker='C:/Program Files/Docker/Docker/resources/bin/docker.exe'
    cmd=[docker,'-H','npipe:////./pipe/dockerDesktopLinuxEngine','run','--rm','--gpus','all','--network','none','--shm-size','1g','-e','OMP_NUM_THREADS=2','-e','OPENBLAS_NUM_THREADS=2',
         '--mount',f'type=bind,source={W},target=/inputs,readonly','--mount',f'type=bind,source={workspace_path("avhubert")},target=/workspace,readonly',
         '--mount',f'type=bind,source={OUT},target=/study','--entrypoint','python','fk-avhubert:cuda117-v1','-B','/inputs/06_Tools/workspace-index/benchmark_training_scaling.py','--workspace','/inputs','--stage',stage,'--output','/study']
   else:cmd=[str(PYTHON),'-B','-u',str(SCRIPT),'--workspace',str(W),'--stage',stage,'--output',str(OUT)]
   with (OUT/(stage+'.log')).open('a',encoding='utf-8') as log:
    p=subprocess.Popen(cmd,cwd=str(W),env={**os.environ,'PYTHONIOENCODING':'utf-8'},stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    if p.wait():raise RuntimeError(stage+' failed; see '+str(OUT/(stage+'.log')))
  progress('complete',len(STAGES),'FINISHED')
if __name__=='__main__':
 try:run()
 except BaseException as e:progress('failed',0,'FAILED',error=str(e));raise
