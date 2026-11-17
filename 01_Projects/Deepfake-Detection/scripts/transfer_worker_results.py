"""Transfer selected worker results; keep media and prepared caches on their worker."""

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
import shutil
import sys
ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(_fk_paths.relocate((ROOT/'src'), __file__))]
from baselines.common.io import read,write,digest,checked_path


def export(worker_root,output,*,role='legacy'):
    worker=_fk_paths.relocate((Path(worker_root)), __file__).resolve();output=_fk_paths.relocate((Path(output)), __file__).resolve()
    if output.exists():raise ValueError('return output already exists')
    if role not in ('B','C','legacy'):raise ValueError('unknown worker role')
    folders=('avfakenet',) if role=='B' else ('avtenet',) if role=='C' else ('avfakenet','avtenet')
    copies=[];population=None
    for folder in folders:
        experiment=_fk_paths.relocate((worker/folder), __file__)
        status=read(_fk_paths.relocate((experiment/'status.json'), __file__))
        if status['status']!='FINISHED':raise ValueError(folder+' has not finished')
        workflow=read(_fk_paths.relocate((experiment/'workflow.json'), __file__))
        current=workflow['manifest_sha256']
        if population is not None and population!=current:raise ValueError('worker populations differ')
        population=current
        # run_two_day nests the model under its workflow output directory.
        model=_fk_paths.relocate((experiment/'avfakenet'), __file__) if folder=='avfakenet' else experiment
        evaluation=_fk_paths.relocate((model/'evaluation'), __file__)
        if role=='legacy' and folder=='avtenet':
            copies.append((_fk_paths.relocate((model/'audiovisual-features'), __file__),_fk_paths.relocate((Path(folder)/'audiovisual-features'), __file__)))
        else:
            for name in ('predictions.jsonl','report.json'):
                if not (evaluation/name).is_file():raise ValueError('missing return artifact '+str(evaluation/name))
            copies.append((evaluation,_fk_paths.relocate((Path(folder)/'evaluation'), __file__)))
        stages=('visual','audio') if folder=='avfakenet' else ('audiovisual',) if role=='legacy' else ('audio','visual','audiovisual','fusion')
        if folder=='avtenet' and role=='C' and workflow.get('integration_max_steps'):
            raise ValueError('integration smoke results cannot be exported as a completed worker run')
        for stage in stages:
            attempts=sorted((model/stage).glob('attempt-*'))
            if not attempts or not (_fk_paths.relocate((attempts[-1]/'best.pt'), __file__)).is_file():
                raise ValueError('missing selected checkpoint for '+folder+'/'+stage)
            attempt=attempts[-1]
            for name in ('best.pt','run.json','metrics.json'):
                if not (attempt/name).is_file():raise ValueError('missing checkpoint evidence '+str(attempt/name))
                copies.append((attempt/name,_fk_paths.relocate((Path(folder)/stage/attempt.name/name), __file__)))
        copies.extend((_fk_paths.relocate((experiment/name), __file__),_fk_paths.relocate((Path(folder)/name), __file__)) for name in ('status.json','workflow.json'))
    for source,_ in copies:
        if not source.exists():raise ValueError('missing return artifact '+str(source))
    output.mkdir(parents=True)
    for source,relative in copies:
        target=_fk_paths.relocate((output/relative), __file__);target.parent.mkdir(parents=True,exist_ok=True)
        if source.is_dir():shutil.copytree(source,target)
        else:shutil.copy2(source,target)
    metadata={'schema':'worker-results-v2','role':role,'population_sha256':population,
              'files':{p.relative_to(output).as_posix():digest(p) for p in output.rglob('*') if p.is_file()}}
    write(_fk_paths.relocate((output/'return.json'), __file__),metadata)
    return metadata


def receive(bundle,output,*,expected_role=None,manifest=None):
    bundle=_fk_paths.relocate((Path(bundle)), __file__).resolve();output=_fk_paths.relocate((Path(output)), __file__).resolve()
    metadata=read(_fk_paths.relocate((bundle/'return.json'), __file__))
    if metadata['schema'] not in ('two-machine-results-v1','worker-results-v2'):raise ValueError('invalid result bundle')
    if expected_role and metadata.get('role')!=expected_role:raise ValueError('return worker role mismatch')
    if manifest and metadata.get('population_sha256')!=digest(manifest):raise ValueError('return population mismatch')
    if not metadata.get('files'):raise ValueError('empty return bundle')
    for relative,sha in metadata['files'].items():
        if digest(checked_path(bundle,relative))!=sha:raise ValueError('return checksum mismatch: '+relative)
    if output.exists():
        if not (_fk_paths.relocate((output/'return.json'), __file__)).is_file() or read(_fk_paths.relocate((output/'return.json'), __file__))!=metadata:
            raise ValueError('import output already exists with different or partial results')
        for relative,sha in metadata['files'].items():
            if digest(checked_path(output,relative))!=sha:raise ValueError('imported checksum mismatch: '+relative)
        return metadata
    output.mkdir(parents=True)
    for relative,sha in metadata['files'].items():
        target=checked_path(output,relative);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(checked_path(bundle,relative),target)
        if digest(target)!=sha:raise ValueError('copied result checksum mismatch')
    write(_fk_paths.relocate((output/'return.json'), __file__),metadata)
    return metadata


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    e=sub.add_parser('export');e.add_argument('--worker-root',required=True);e.add_argument('--output',required=True)
    e.add_argument('--role',choices=('B','C','legacy'),default='legacy')
    i=sub.add_parser('import');i.add_argument('--bundle',required=True);i.add_argument('--output',required=True)
    i.add_argument('--expected-role',choices=('B','C'));i.add_argument('--manifest')
    args=p.parse_args()
    result=(export(args.worker_root,args.output,role=args.role) if args.command=='export' else
            receive(args.bundle,args.output,expected_role=args.expected_role,manifest=args.manifest))
    print('Verified result artifacts:',len(result['files']))
