"""Shared command grammar with comparator-specific, lazy imports."""
import argparse
from pathlib import Path
import json

from .io import read, write
from .exceptions import TrainingPaused


def _main(model, argv=None):
    parser=argparse.ArgumentParser(description=model+' paper-guided AV-Deepfake1M comparator')
    commands=parser.add_subparsers(dest='command',required=True)
    check=commands.add_parser('check',help='Verify imports, assets and runtime; no training')
    check.add_argument('--assets');check.add_argument('--device',default='cpu');check.add_argument('--output')
    prepare=commands.add_parser('prepare',help='Create a portable metadata index, without decoding media')
    prepare.add_argument('--manifest',required=True);prepare.add_argument('--output',required=True)
    prepare.add_argument('--limit-per-split',type=int);prepare.add_argument('--seed',type=int,default=7)
    for name in ('train','evaluate'):
        sub=commands.add_parser(name)
        sub.add_argument('--index',required=True);sub.add_argument('--dataset-root',required=True)
        sub.add_argument('--output',required=True);sub.add_argument('--config');sub.add_argument('--assets')
        sub.add_argument('--device',default='cpu');sub.add_argument('--threads',type=int,default=2)
        if name=='train':
            choices=('visual','audio') if model=='avfakenet' else ('visual','audio','audiovisual','fusion')
            sub.add_argument('--stage',choices=choices,required=True)
            sub.add_argument('--epochs',type=int);sub.add_argument('--seed',type=int,default=7)
            sub.add_argument('--resume');sub.add_argument('--max-steps',type=int,help='Explicit integration-only cap per epoch/split')
            sub.add_argument('--prepared-cache',help='Lossless prepared media cache reused across epochs')
            if model=='avtenet':
                sub.add_argument('--schedule');sub.add_argument('--feature-cache')
                sub.add_argument('--checkpoint-seconds',type=float,default=0,help='Atomic mid-epoch checkpoint interval; 0 disables')
                sub.add_argument('--progress-seconds',type=float,default=10)
                sub.add_argument('--pause-file',help='Save and stop when this file exists')
                sub.add_argument('--reuse-video-decoder',action='store_true',help='Reuse sequential decoded frames; preserve reference frame selection')
                sub.add_argument('--prefetch-cached-inputs',type=int,choices=range(5),default=0,help='Read ahead 0-4 existing CPU caches; no background GPU preparation')
                sub.add_argument('--cache-face-detections',action='store_true',help='Reuse exact RGB frame detections within each clip; tracking unchanged')
                sub.add_argument('--fast-vivit-processor',action='store_true',help='Same image processor, faster equivalent tensor packing')
        else:
            sub.add_argument('--split',choices=('validation','test'),default='validation')
            sub.add_argument('--compare-with',help='Comparator predictions.jsonl or FK evaluation JSON')
            if model=='avfakenet':
                sub.add_argument('--visual-checkpoint',required=True);sub.add_argument('--audio-checkpoint',required=True)
            else:
                sub.add_argument('--checkpoint',required=True);sub.add_argument('--feature-cache',required=True)
    if model=='avtenet':
        cache=commands.add_parser('cache-features',help='Extract one selected expert; merge verifies aligned IDs')
        for flag in ('index','dataset-root','output','assets','checkpoint'):
            cache.add_argument('--'+flag,required=True)
        cache.add_argument('--stage',choices=('visual','audio','audiovisual'),required=True)
        cache.add_argument('--device',default='cpu');cache.add_argument('--threads',type=int,default=2)
        cache.add_argument('--config')
        cache.add_argument('--prepared-cache')
        cache.add_argument('--pause-file')
        cache.add_argument('--progress-seconds',type=float,default=10)
        cache.add_argument('--reuse-video-decoder',action='store_true')
        cache.add_argument('--prefetch-cached-inputs',type=int,choices=range(5),default=0)
        cache.add_argument('--cache-face-detections',action='store_true')
        cache.add_argument('--fast-vivit-processor',action='store_true')
        cache.add_argument('--prefetch-processed-inputs',type=int,choices=range(5),default=0,
                           help='AV-HuBERT / HF ViViT: CPU cache read and preprocessing ahead of inference')
        cache.add_argument('--resume-feature-cache',help='New output from an attested, checksummed partial-cache snapshot; never overwrite in place')
        merge=commands.add_parser('merge-features')
        for flag in ('visual','audio','audiovisual','output'):merge.add_argument('--'+flag,required=True)
    args=parser.parse_args(argv)
    try:
        if args.command=='prepare':
            from .population import prepare as prepare_index
            result=prepare_index(args.manifest,args.output,model,limit=args.limit_per_split,seed=args.seed)
        elif args.command=='check':
            from .readiness import check_runtime
            result=check_runtime(model,args.assets,args.device)
            if args.output:write(args.output,result)
            print(json.dumps(result,indent=2))
            return 0 if result['ready'] else 2
        else:
            import torch
            if hasattr(args,'threads'):
                if args.threads<1:raise ValueError('threads must be positive')
                torch.set_num_threads(args.threads)
            from .population import Population
            if args.command!='merge-features':population=Population(args.index,args.dataset_root,model)
            if model=='avfakenet':
                from .media import detector
                from baselines.avfakenet.pipeline import train,evaluate,PROFILE
                detect,provenance=detector(args.device)
                common={'profile_path':args.config or PROFILE,'device':args.device}
                if args.command=='train':
                    result=train(population,args.output,detect,provenance,stage=args.stage,seed=args.seed,
                                 epochs=args.epochs,resume=args.resume,max_steps=args.max_steps,prepared_cache=args.prepared_cache,**common)
                else:
                    result=evaluate(population,args.output,detect,provenance,visual_checkpoint=args.visual_checkpoint,
                                    audio_checkpoint=args.audio_checkpoint,split=args.split,compare_with=args.compare_with,**common)
            else:
                from baselines.avtenet.pipeline import dispatch
                result=dispatch(args,None if args.command=='merge-features' else population)
        print(json.dumps(result,indent=2))
        return 0
    except TrainingPaused as error:
        print(str(error),flush=True)
        return 75
    except (ValueError,RuntimeError,FileNotFoundError,ImportError) as error:
        parser.exit(2,'error: '+str(error)+'\n')


def main(model,argv=None):
    import sys
    from deepfake_detection.gpu_queue import gpu_slot,explicit_cuda
    values=sys.argv[1:] if argv is None else argv
    with gpu_slot(explicit_cuda(values)):
        return _main(model,values)
