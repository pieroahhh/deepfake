"""Read-only forward workload timing after the authorized training benchmark."""
import json,os,subprocess,sys
from pathlib import Path
W=next(p for p in Path(__file__).resolve().parents if (p/'workspace-layout.json').is_file())
sys.path.insert(0,str(W/'06_Tools/workspace'))
from workspace_paths import workspace_path
sys.path.insert(0,str(workspace_path('main')/'src'))
from deepfake_detection.gpu_queue import gpu_slot
OUT=workspace_path('main_results')/'training-scaling-001';SCRIPT=Path(__file__).with_name('benchmark_training_scaling.py');PYTHON=workspace_path('runtime')/'environments/main/.venv-gpu/Scripts/python.exe'
STAGES=['ours','avf-visual','avf-audio','avt-audio','avt-visual','avt-audiovisual']
def write_status(stage,count,status='RUNNING',**extra):
 d={'stage':stage,'processed':count,'total':len(STAGES),'status':status,**extra};temp=OUT/'extraction-progress.tmp';temp.write_text(json.dumps(d,indent=2));os.replace(temp,OUT/'extraction-progress.json');print(json.dumps(d),flush=True)
if __name__=='__main__':
 try:
  with gpu_slot():
   assert json.loads((OUT/'progress.json').read_text())['status']=='FINISHED'
   for i,stage in enumerate(STAGES):
    if (OUT/('extract-'+stage+'.json')).exists():continue
    write_status(stage,i)
    if stage=='avt-audiovisual':
     cmd=['C:/Program Files/Docker/Docker/resources/bin/docker.exe','-H','npipe:////./pipe/dockerDesktopLinuxEngine','run','--rm','--gpus','all','--network','none','--shm-size','1g','-e','OMP_NUM_THREADS=2','-e','OPENBLAS_NUM_THREADS=2',
          '--mount',f'type=bind,source={W},target=/inputs,readonly','--mount',f'type=bind,source={workspace_path("avhubert")},target=/workspace,readonly','--mount',f'type=bind,source={OUT},target=/study',
          '--entrypoint','python','fk-avhubert:cuda117-v1','-B','/inputs/06_Tools/workspace-index/benchmark_training_scaling.py','--workspace','/inputs','--stage',stage,'--output','/study','--extract-only']
    else:cmd=[str(PYTHON),'-B','-u',str(SCRIPT),'--workspace',str(W),'--stage',stage,'--output',str(OUT),'--extract-only']
    with (OUT/('extract-'+stage+'.log')).open('a',encoding='utf-8') as log:
     p=subprocess.Popen(cmd,cwd=str(W),env={**os.environ,'PYTHONIOENCODING':'utf-8'},stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
     if p.wait():raise RuntimeError(stage+' extraction failed')
   write_status('complete',len(STAGES),'FINISHED')
 except BaseException as e:write_status('failed',0,'FAILED',error=str(e));raise
