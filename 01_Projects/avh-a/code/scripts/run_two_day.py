"""Sequential GPU FK/AVFakeNet subset comparison with logs and restartable stages.

Run with the separate CUDA environment. A completed stage is verified before
reuse. Training resume uses new attempt directories; never silently restarts an
interrupted training run from random weights.
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
from datetime import datetime,timezone
from pathlib import Path
import os
import shutil
import subprocess
import sys
import time

ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(_fk_paths.relocate((ROOT/'src'), __file__))]
from baselines.common.io import read,write,digest,fingerprint
from deepfake_detection.quality_fitting import exclusive_run


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--dataset-root',required=True)
    parser.add_argument('--assets',help='FK asset registry; required when running FK')
    parser.add_argument('--methods',choices=('fk','avfakenet','both'),default='both')
    parser.add_argument('--compare-with',help='Optional FK evaluation report for AVFakeNet-only worker')
    parser.add_argument('--output',required=True)
    parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args(argv)
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required; use .venv-gpu')
    torch.set_num_threads(args.threads)
    root=_fk_paths.relocate((Path(args.output)), __file__).resolve();root.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('HF_HOME',str(_fk_paths.relocate((ROOT/'.cache/huggingface-two-day'), __file__)))
    os.environ.setdefault('OMP_NUM_THREADS',str(args.threads))
    manifest=_fk_paths.relocate((Path(args.manifest)), __file__).resolve();assets=_fk_paths.relocate((Path(args.assets)), __file__).resolve() if args.assets else None
    if args.methods in ('fk','both') and assets is None:raise ValueError('--assets is required for FK')
    identity={'manifest_sha256':digest(manifest),'assets_sha256':digest(assets) if assets else None,
              'dataset_root':str(_fk_paths.relocate((Path(args.dataset_root)), __file__).resolve()),'device':'cuda',
              'torch':str(torch.__version__),'gpu':torch.cuda.get_device_name(0),
              'FK_epochs':5,'AVFakeNet_epochs':3,'AVFakeNet_batch':8,'seed':7}
    with exclusive_run(_fk_paths.relocate((root/'workflow.lock'), __file__)):
        if (_fk_paths.relocate((root/'workflow.json'), __file__)).exists():
            if read(_fk_paths.relocate((root/'workflow.json'), __file__))!=identity:raise ValueError('workflow inputs changed; use a new output')
        else:write(_fk_paths.relocate((root/'workflow.json'), __file__),identity)
        logs=_fk_paths.relocate((root/'logs'), __file__);logs.mkdir(exist_ok=True)
        # Separate profile freezes only the declared training batch/budget choices.
        config=_fk_paths.relocate((root/'avfakenet.profile.json'), __file__)
        if not config.exists():
            profile=read(_fk_paths.relocate((ROOT/'baselines/avfakenet/configs/avfakenet.reconstruction-v1.json'), __file__))
            profile['training']['batch_size']=8
            write(config,profile)

        def run(name,command):
            started=time.monotonic();log=_fk_paths.relocate((logs/(name+'.log')), __file__)
            write(_fk_paths.relocate((root/'status.json'), __file__),{'stage':name,'status':'RUNNING','log':str(log),
                                     'utc':datetime.now(timezone.utc).isoformat()},replace=True)
            print('START',name,'log:',log,flush=True)
            with log.open('a',encoding='utf-8') as stream:
                stream.write('\nATTEMPT '+datetime.now(timezone.utc).isoformat()+'\n');stream.flush()
                result=subprocess.run([sys.executable,'-u',*map(str,command)],cwd=ROOT,
                                      stdout=stream,stderr=subprocess.STDOUT)
            if result.returncode:
                write(_fk_paths.relocate((root/'status.json'), __file__),{'stage':name,'status':'FAILED','exit_code':result.returncode,
                                         'log':str(log)},replace=True)
                raise RuntimeError(name+' failed; read '+str(log)+' and rerun this same workflow after fixing it')
            write(_fk_paths.relocate((logs/(name+'.timing.json')), __file__),{'seconds':time.monotonic()-started},replace=True)
            print('DONE',name,flush=True)

        fk_report=_fk_paths.relocate((root/'fk/validation.json'), __file__)
        if args.methods in ('fk','both'):
            fk=_fk_paths.relocate((root/'fk'), __file__);fk.mkdir(exist_ok=True)
            quality=_fk_paths.relocate((fk/'quality.json'), __file__)
            if not quality.exists():
                cmd=['-m','deepfake_detection.cli','fit-quality','--manifest',manifest,'--dataset-root',args.dataset_root,
                     '--assets',assets,'--output',quality]
                if quality.with_name(quality.name+'.work').exists():cmd+=['--resume']
                run('fk-quality',cmd)
            else:
                # Always verify a completed fit against its recorded work state.
                run('fk-quality-verify',['-m','deepfake_detection.cli','fit-quality','--manifest',manifest,
                    '--dataset-root',args.dataset_root,'--assets',assets,'--output',quality,'--resume'])
            cache=_fk_paths.relocate((fk/'cache'), __file__)
            if not (_fk_paths.relocate((cache/'training-manifest.json'), __file__)).exists():
                if shutil.disk_usage(root).free<30*1024**3:raise RuntimeError('need at least 30 GiB free before FK cache')
                cmd=['-m','deepfake_detection.cli','cache-fk','--manifest',manifest,'--dataset-root',args.dataset_root,
                     '--assets',assets,'--quality-normalization',quality,'--output',cache,'--device','cuda']
                if cache.exists():cmd+=['--resume']
                run('fk-cache',cmd)

        def training_attempt(directory,complete):
            directory.mkdir(exist_ok=True)
            attempts=sorted(directory.glob('attempt-*'))
            if attempts and complete(attempts[-1]):return attempts[-1],None,True
            if attempts and not (_fk_paths.relocate((attempts[-1]/'last.pt'), __file__)).exists():
                # No complete epoch was saved; retain diagnostics and start anew.
                resume=None
            else:resume=_fk_paths.relocate((attempts[-1]/'last.pt'), __file__) if attempts else None
            return directory/f'attempt-{len(attempts)+1:03d}',resume,False

        if args.methods in ('fk','both'):
            destination,resume,complete=training_attempt(_fk_paths.relocate((fk/'training'), __file__),lambda p:(_fk_paths.relocate((p/'selected.pt'), __file__)).exists())
            if not complete:
                if resume:
                    previous=torch.load(_fk_paths.relocate_value((resume), __file__),map_location='cpu',weights_only=True)
                    if previous['epoch']>=5 or previous['stale_epochs']>=previous['config']['patience']:
                        raise RuntimeError('FK training finished but calibration did not; inspect logs for recovery')
                cmd=['-m','deepfake_detection.cli','train-fk','--manifest',_fk_paths.relocate((cache/'training-manifest.json'), __file__),
                     '--model-config',_fk_paths.relocate((ROOT/'configs/ours.provisional.json'), __file__),'--training-config',_fk_paths.relocate((ROOT/'configs/fk.training-v1.json'), __file__),
                     '--epochs','5','--calibrate','--output',destination,'--device','cuda']
                if resume:cmd+=['--resume',resume]
                run('fk-train',cmd)
            fk_report=_fk_paths.relocate((fk/'validation.json'), __file__)
            if not fk_report.exists():run('fk-evaluate',['-m','deepfake_detection.cli','evaluate-fk',
                '--manifest',_fk_paths.relocate((cache/'training-manifest.json'), __file__),'--checkpoint',_fk_paths.relocate((destination/'selected.pt'), __file__),
                '--split','validation','--output',fk_report,'--device','cuda'])

        if args.methods in ('avfakenet','both'):
            av=_fk_paths.relocate((root/'avfakenet'), __file__);av.mkdir(exist_ok=True)
            if not (_fk_paths.relocate((av/'index/index.json'), __file__)).exists():run('avfakenet-index',['-m','baselines.avfakenet','prepare',
                '--manifest',manifest,'--output',_fk_paths.relocate((av/'index'), __file__)])
            checkpoints={}
            from baselines.common.training import load_checkpoint
            for stage in ('visual','audio'):
                destination,resume,complete=training_attempt(_fk_paths.relocate((av/stage), __file__),lambda p:(_fk_paths.relocate((p/'metrics.json'), __file__)).exists() and
                    (read(_fk_paths.relocate((p/'metrics.json'), __file__))['history'][-1]['epoch']>=3 or load_checkpoint(_fk_paths.relocate((p/'last.pt'), __file__))['stale']>=5))
                if not complete:
                    if shutil.disk_usage(root).free<15*1024**3:raise RuntimeError('insufficient free space for prepared inputs')
                    cmd=['-m','baselines.avfakenet','train','--index',_fk_paths.relocate((av/'index'), __file__),'--dataset-root',args.dataset_root,
                         '--stage',stage,'--config',config,'--prepared-cache',_fk_paths.relocate((av/'prepared-cache'), __file__),'--epochs','3',
                         '--output',destination,'--device','cuda','--threads',str(args.threads),'--seed','7']
                    if resume:cmd+=['--resume',resume]
                    run('avfakenet-'+stage,cmd)
                checkpoints[stage]=_fk_paths.relocate((destination/'best.pt'), __file__)
            evaluation=_fk_paths.relocate((av/'evaluation'), __file__)
            if not (_fk_paths.relocate((evaluation/'report.json'), __file__)).exists():
                if evaluation.exists():raise RuntimeError('partial AVFakeNet evaluation exists; preserve it under another name before retry')
                run('avfakenet-evaluate',['-m','baselines.avfakenet','evaluate','--index',_fk_paths.relocate((av/'index'), __file__),
                    '--dataset-root',args.dataset_root,'--visual-checkpoint',checkpoints['visual'],
                    '--audio-checkpoint',checkpoints['audio'],'--config',config,'--split','validation',
                    '--output',evaluation,'--device','cuda','--threads',str(args.threads),
                    *(['--compare-with',args.compare_with or fk_report] if (args.compare_with or fk_report.exists()) else [])])
        summary={'scope':'shared subset; validation development results; one seed; limited epochs',
                 'population_sha256':digest(manifest),'methods':args.methods,
                 'AVTENet':{'status':'SEPARATE_WORKFLOW','method':'AVTENet adaptation; Google joint-attention ViViT'}}
        if args.methods in ('fk','both'):summary['FK']=read(fk_report)['video_metrics']
        if args.methods in ('avfakenet','both'):
            report=read(_fk_paths.relocate((evaluation/'report.json'), __file__))
            summary['AVFakeNet']=report['metrics']
            if 'comparison' in report:summary['matched_video_comparison']=report['comparison']
        write(_fk_paths.relocate((root/'comparison.json'), __file__),summary,replace=True)
        write(_fk_paths.relocate((root/'status.json'), __file__),{'stage':'complete','status':'FINISHED','comparison':str(_fk_paths.relocate((root/'comparison.json'), __file__))},replace=True)
        print('FINISHED:',_fk_paths.relocate((root/'comparison.json'), __file__),flush=True)


if __name__=='__main__':main()
