"""Run explicitly approved HF-joint-ViViT AVTENet adaptation on the shared subset.

AST/ViViT run on the host CUDA environment. AV-HuBERT runs in its pinned CUDA
container. Every CUDA stage participates in the same host GPU queue as FK and
AVFakeNet, including across the Docker boundary.
"""

# Workspace relocation: portable paths, historical metadata stays immutable.
import sys as _fk_sys
from pathlib import Path as _FkBootstrapPath
_fk_workspace = next((p for p in _FkBootstrapPath(__file__).resolve().parents
                      if (p / 'workspace-layout.json').is_file()), None)
if _fk_workspace is None:
    _fk_hint = _FkBootstrapPath(__import__('os').environ.get('FK_WORKSPACE_ROOT', '/inputs'))
    if (_fk_hint / 'workspace-layout.json').is_file():
        _fk_workspace = _fk_hint
if _fk_workspace is None:
    raise RuntimeError('workspace-layout.json is missing')
if str(_fk_workspace / '06_Tools/workspace') not in _fk_sys.path:
    _fk_sys.path.insert(0, str(_fk_workspace / '06_Tools/workspace'))
import workspace_paths as _fk_paths

import argparse
from pathlib import Path
import os
import subprocess
import sys
import time
from datetime import datetime,timezone

ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(_fk_paths.relocate((ROOT/'src'), __file__))]
from baselines.common.io import read,write,digest
from deepfake_detection.quality_fitting import exclusive_run
from deepfake_detection.gpu_queue import gpu_slot


class WorkflowPaused(Exception):pass


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('manifest','dataset-root','assets','docker-assets','output'):parser.add_argument('--'+flag,required=True)
    parser.add_argument('--config',default=str(_fk_paths.relocate((ROOT/'baselines/avtenet/configs/avtenet.hf-joint-adaptation-v1.json'), __file__)))
    parser.add_argument('--epochs',type=int,default=1)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--checkpoint-seconds',type=float,default=120)
    parser.add_argument('--progress-seconds',type=float,default=10)
    parser.add_argument('--image',default='fk-avhubert:cuda117-v1')
    parser.add_argument('--compare-with',help='FK report for matched-ID metrics when available')
    parser.add_argument('--integration-max-steps',type=int,help='Only for software checks; recorded in all checkpoints')
    parser.add_argument('--stages',default='audio,visual,audiovisual,fusion',help='Comma-separated experts/fusion assigned to this machine')
    parser.add_argument('--index',help='Shared, byte-identical AVTENet index copied between machines')
    parser.add_argument('--audiovisual-features',help='Verified returned AV-HuBERT cache when fusion runs on another machine')
    parser.add_argument('--reuse-video-decoder',action='store_true',help='Opt-in equivalent sequential decoder reuse')
    parser.add_argument('--prefetch-cached-inputs',type=int,choices=range(5),default=0)
    parser.add_argument('--cache-face-detections',action='store_true')
    parser.add_argument('--fast-vivit-processor',action='store_true')
    parser.add_argument('--prefetch-processed-inputs',type=int,choices=range(5),default=0)
    args=parser.parse_args(argv)
    stages=args.stages.split(',')
    if len(set(stages))!=len(stages) or not set(stages)<= {'audio','visual','audiovisual','fusion'}:
        raise ValueError('invalid or duplicate AVTENet stage')
    if not 1<=args.epochs<=50:raise ValueError('epochs must be between 1 and 50')
    root=_fk_paths.relocate((Path(args.output)), __file__).resolve();root.mkdir(parents=True,exist_ok=True)
    manifest=_fk_paths.relocate((Path(args.manifest)), __file__).resolve();config=_fk_paths.relocate((Path(args.config)), __file__).resolve()
    assets=_fk_paths.relocate((Path(args.assets)), __file__).resolve();docker_assets=_fk_paths.relocate((Path(args.docker_assets)), __file__).resolve()
    # All experiment/config/weight paths must be on the mounted repository;
    # dataset itself has its independent read-only mount.
    def linux(path):return '/workspace/'+_fk_paths.relocate((Path(path)), __file__).resolve().relative_to(ROOT).as_posix()
    linux(root);linux(config);linux(docker_assets)
    identity={'model':'AVTENet-adaptation-HF-joint-ViViT','manifest_sha256':digest(manifest),
              'assets_sha256':digest(assets),'docker_assets_sha256':digest(docker_assets),
              'config_sha256':digest(config),'seed':7,'integration_max_steps':args.integration_max_steps}
    os.environ.setdefault('HF_HOME',str(_fk_paths.relocate((ROOT/'.cache/huggingface-avtenet'), __file__)))
    logs=_fk_paths.relocate((root/'logs'), __file__);logs.mkdir(exist_ok=True)
    with exclusive_run(_fk_paths.relocate((root/'workflow.lock'), __file__)):
        header=_fk_paths.relocate((root/'workflow.json'), __file__)
        if header.exists():
            if read(header)!=identity:raise ValueError('AVTENet workflow inputs changed')
        else:write(header,identity)
        def run(name,arguments,*,docker=False,gpu=True):
            log=_fk_paths.relocate((logs/(name+'.log')), __file__)
            write(_fk_paths.relocate((root/'status.json'), __file__),{'stage':name,'status':'WAITING_FOR_GPU' if gpu else 'RUNNING','log':str(log)},replace=True)
            with gpu_slot(gpu):
                write(_fk_paths.relocate((root/'status.json'), __file__),{'stage':name,'status':'RUNNING','log':str(log),'utc':datetime.now(timezone.utc).isoformat()},replace=True)
                environment=os.environ.copy();environment['FK_GPU_QUEUE_MANAGED']='1';environment['OMP_NUM_THREADS']=str(args.threads)
                if docker:
                    command=['docker','run','--rm','--gpus','all','-e','FK_GPU_QUEUE_MANAGED=1',
                             '-e','PYTHONUNBUFFERED=1','-e',f'OMP_NUM_THREADS={args.threads}',
                             '-e',f'MKL_NUM_THREADS={args.threads}','--shm-size','1g',
                             '-e','OPENBLAS_NUM_THREADS=1',
                             '--mount',f'type=bind,source={ROOT},target=/workspace',
                             '--mount',f'type=bind,source={_fk_paths.relocate((Path(args.dataset_root)), __file__).resolve()},target=/dataset,readonly',
                             args.image,*map(str,arguments)]
                else:command=[sys.executable,'-u','-m','baselines.avtenet',*map(str,arguments)]
                started=time.monotonic();print('START',name,log,flush=True)
                with log.open('a',encoding='utf-8') as stream:
                    stream.write('\nATTEMPT '+datetime.now(timezone.utc).isoformat()+'\n');stream.flush()
                    result=subprocess.run(command,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT)
                if result.returncode:
                    if result.returncode==75:
                        write(_fk_paths.relocate((root/'status.json'), __file__),{'stage':name,'status':'PAUSED','log':str(log)},replace=True)
                        raise WorkflowPaused()
                    write(_fk_paths.relocate((root/'status.json'), __file__),{'stage':name,'status':'FAILED','log':str(log),'exit_code':result.returncode},replace=True)
                    raise RuntimeError(name+' failed; inspect '+str(log))
                write(_fk_paths.relocate((logs/(name+'.timing.json')), __file__),{'seconds':time.monotonic()-started},replace=True)
                print('DONE',name,flush=True)
        index=_fk_paths.relocate((Path(args.index)), __file__).resolve() if args.index else _fk_paths.relocate((root/'index'), __file__)
        if not (_fk_paths.relocate((index/'index.json'), __file__)).exists():run('index',['prepare','--manifest',manifest,'--output',index],gpu=False)
        from baselines.common.training import load_checkpoint
        checkpoints={}
        for stage in ('audio','visual','audiovisual','fusion'):
            if stage not in stages:continue
            if stage=='fusion' and not (_fk_paths.relocate((root/'fusion-features/cache.json'), __file__)).exists():
                run('merge',['merge-features','--visual',_fk_paths.relocate((root/'visual-features'), __file__),'--audio',_fk_paths.relocate((root/'audio-features'), __file__),
                    '--audiovisual',args.audiovisual_features or _fk_paths.relocate((root/'audiovisual-features'), __file__),
                    '--output',_fk_paths.relocate((root/'fusion-features'), __file__)],gpu=False)
            container=stage=='audiovisual'
            stage_root=_fk_paths.relocate((root/stage), __file__);stage_root.mkdir(exist_ok=True)
            attempts=sorted(stage_root.glob('attempt-*'))
            resume=None;complete=False
            resumable=next((p for p in reversed(attempts) if (_fk_paths.relocate((p/'last.pt'), __file__)).exists()),None)
            if resumable:
                checkpoint=load_checkpoint(_fk_paths.relocate((resumable/'last.pt'), __file__))
                complete=checkpoint['epoch']>=args.epochs or checkpoint['stale']>=5
                if not complete:resume=_fk_paths.relocate((resumable/'last.pt'), __file__)
                del checkpoint
            if complete:destination=resumable
            else:
                destination=_fk_paths.relocate((stage_root/f'attempt-{len(attempts)+1:03d}'), __file__)
                convert=linux if container else lambda p:str(p)
                command=['train','--index',convert(index),'--dataset-root','/dataset' if container else args.dataset_root,
                         '--config',convert(config),'--stage',stage,'--epochs',str(args.epochs),'--seed','7',
                         '--device','cuda','--threads',str(args.threads),'--output',convert(destination)]
                command+=['--checkpoint-seconds',str(args.checkpoint_seconds),'--progress-seconds',str(args.progress_seconds)]
                command+=['--pause-file',convert(_fk_paths.relocate((root/'pause.request'), __file__))]
                if stage=='fusion':command+=['--feature-cache',str(_fk_paths.relocate((root/'fusion-features'), __file__))]
                else:command+=['--assets',convert(docker_assets if container else assets),'--prepared-cache',convert(_fk_paths.relocate((root/'prepared-inputs'), __file__))]
                if stage!='fusion':
                    if args.reuse_video_decoder:command+=['--reuse-video-decoder']
                    if args.prefetch_cached_inputs:command+=['--prefetch-cached-inputs',str(args.prefetch_cached_inputs)]
                    if args.cache_face_detections:command+=['--cache-face-detections']
                    if args.fast_vivit_processor and stage=='visual':command+=['--fast-vivit-processor']
                if resume:command+=['--resume',convert(resume)]
                if args.integration_max_steps:command+=['--max-steps',str(args.integration_max_steps)]
                run(stage+'-train',command,docker=container)
            checkpoints[stage]=_fk_paths.relocate((destination/'best.pt'), __file__)
            if stage!='fusion':
                cache=_fk_paths.relocate((root/(stage+'-features')), __file__)
                if not (_fk_paths.relocate((cache/'cache.json'), __file__)).exists():
                    feature_resume=None
                    if cache.exists():
                        from baselines.common.population import Population
                        from baselines.avtenet.feature_resume import snapshot_owned
                        suffix=datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
                        archived=_fk_paths.relocate((root/(stage+'-features-interrupted-'+suffix)), __file__)
                        feature_resume=_fk_paths.relocate((root/(stage+'-features-resume-'+suffix)), __file__)
                        snapshot_owned(cache,feature_resume,Population(index,args.dataset_root,'avtenet'),checkpoints[stage],stage)
                        cache.rename(archived)
                    convert=linux if container else lambda p:str(p)
                    feature_command=['cache-features','--index',convert(index),'--dataset-root','/dataset' if container else args.dataset_root,
                        '--config',convert(config),'--assets',convert(docker_assets if container else assets),
                        '--stage',stage,'--checkpoint',convert(checkpoints[stage]),'--device','cuda','--threads',str(args.threads),
                        '--prepared-cache',convert(_fk_paths.relocate((root/'prepared-inputs'), __file__)),'--output',convert(cache)]
                    feature_command+=['--pause-file',convert(_fk_paths.relocate((root/'pause.request'), __file__)),'--progress-seconds',str(args.progress_seconds)]
                    if feature_resume:feature_command+=['--resume-feature-cache',convert(feature_resume)]
                    if args.reuse_video_decoder:feature_command+=['--reuse-video-decoder']
                    if args.prefetch_cached_inputs:feature_command+=['--prefetch-cached-inputs',str(args.prefetch_cached_inputs)]
                    if args.cache_face_detections:feature_command+=['--cache-face-detections']
                    if args.fast_vivit_processor and stage=='visual':feature_command+=['--fast-vivit-processor']
                    if args.prefetch_processed_inputs and stage in ('visual','audiovisual'):feature_command+=['--prefetch-processed-inputs',str(args.prefetch_processed_inputs)]
                    run(stage+'-features',feature_command,docker=container)
        if 'fusion' not in stages:
            write(_fk_paths.relocate((root/'status.json'), __file__),{'stage':'assigned_stages_complete','status':'FINISHED',
                'method':identity['model'],'stages':stages,'next':'transfer expert feature caches for fusion'},replace=True)
            print('Assigned stages finished:',stages,flush=True)
            return
        if not (_fk_paths.relocate((root/'evaluation/report.json'), __file__)).exists():
            if (_fk_paths.relocate((root/'evaluation'), __file__)).exists():raise RuntimeError('partial evaluation exists; preserve it before retry')
            command=['evaluate','--index',index,'--dataset-root',args.dataset_root,'--config',config,
                     '--checkpoint',checkpoints['fusion'],'--feature-cache',_fk_paths.relocate((root/'fusion-features'), __file__),'--split','validation',
                     '--device','cuda','--output',_fk_paths.relocate((root/'evaluation'), __file__)]
            if args.compare_with and _fk_paths.relocate((Path(args.compare_with)), __file__).exists():command+=['--compare-with',args.compare_with]
            run('evaluate',command)
        write(_fk_paths.relocate((root/'status.json'), __file__),{'stage':'complete','status':'FINISHED','method':identity['model'],
                                'report':str(_fk_paths.relocate((root/'evaluation/report.json'), __file__))},replace=True)
        print('FINISHED',_fk_paths.relocate((root/'evaluation/report.json'), __file__),flush=True)


if __name__=='__main__':
    try:main()
    except WorkflowPaused:print('PAUSED safely; relaunch to resume the newest checkpoint.',flush=True)
