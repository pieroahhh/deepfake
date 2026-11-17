"""Export the exact current code snapshot and selected media, never the full dataset."""

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
import subprocess
ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(_fk_paths.relocate((ROOT/'src'), __file__))]
from baselines.common.io import read,write,digest,checked_path


def export(manifest,dataset_root,index,output,*,include_checkpoint=True,include_container=False,role='legacy',container_archive=None):
    if role not in ('B','C','legacy'):raise ValueError('unknown worker role')
    if role=='C' and not include_checkpoint:raise ValueError('machine C requires all pretrained checkpoints')
    if role=='B':include_checkpoint=False;include_container=False
    manifest=_fk_paths.relocate((Path(manifest)), __file__);dataset=_fk_paths.relocate((Path(dataset_root)), __file__).resolve();output=_fk_paths.relocate((Path(output)), __file__).resolve()
    if output.exists():raise ValueError('export output must be a new directory')
    rows=read(manifest);audit=read(_fk_paths.relocate((manifest.parent/'audit.json'), __file__))
    if digest(manifest)!=audit['manifest_sha256']:raise ValueError('population audit mismatch')
    total=sum(checked_path(dataset,row['path']).stat().st_size for row in rows)
    output.parent.mkdir(parents=True,exist_ok=True)
    extra=0
    if include_checkpoint:
        extra+=(_fk_paths.relocate((ROOT/'checkpoints/comparators-native-001/avhubert_native/base_lrs3_iter5.pt'), __file__)).stat().st_size
    if role=='C':
        for relative in ('checkpoints/vivit-joint-adaptation-002','checkpoints/comparators-ast-001/ast'):
            directory=_fk_paths.relocate((ROOT/relative), __file__)
            extra+=sum(checked_path(directory,name).stat().st_size for name in read(_fk_paths.relocate((directory/'artifact.json'), __file__))['files'])
    if include_container:
        extra+=_fk_paths.relocate((Path(container_archive)), __file__).stat().st_size if container_archive else 3*1024**3
    if shutil.disk_usage(output.parent).free<total+extra+2*1024**3:raise ValueError('insufficient space to export subset and assets')
    output.mkdir();code=_fk_paths.relocate((output/'code'), __file__);code.mkdir()
    guide_name='START_HERE_THIRD_MACHINE.md' if role=='C' else 'START_HERE_SECOND_MACHINE.md'
    start_guide=_fk_paths.relocate((ROOT/'docs'/guide_name), __file__)
    if start_guide.is_file():shutil.copy2(start_guide,_fk_paths.relocate((output/guide_name), __file__))
    for folder in ('src','baselines','scripts','configs','tests','docs'):
        shutil.copytree(_fk_paths.relocate((ROOT/folder), __file__),_fk_paths.relocate((code/folder), __file__),ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.local.json'))
    for name in ('pyproject.toml','README.md','HANDOFF.md','.gitignore'):
        shutil.copy2(_fk_paths.relocate((ROOT/name), __file__),_fk_paths.relocate((code/name), __file__))
    destination=_fk_paths.relocate((code/'runs/two-day-001/population'), __file__);destination.mkdir(parents=True)
    for name in ('manifest.json','audit.json'):shutil.copy2(_fk_paths.relocate((manifest.parent/name), __file__),_fk_paths.relocate((destination/name), __file__))
    shutil.copytree(index,_fk_paths.relocate((code/'runs/two-day-001/avtenet/shared-index'), __file__))
    if include_checkpoint:
        target=_fk_paths.relocate((code/'checkpoints/comparators-native-001/avhubert_native'), __file__);target.mkdir(parents=True)
        source=_fk_paths.relocate((ROOT/'checkpoints/comparators-native-001/avhubert_native'), __file__)
        if digest(_fk_paths.relocate((source/'base_lrs3_iter5.pt'), __file__))!=read(_fk_paths.relocate((source/'artifact.json'), __file__))['files']['base_lrs3_iter5.pt']:
            raise ValueError('AV-HuBERT checkpoint checksum mismatch')
        for name in ('base_lrs3_iter5.pt','artifact.json','LICENSE'):shutil.copy2(_fk_paths.relocate((source/name), __file__),_fk_paths.relocate((target/name), __file__))
        (_fk_paths.relocate((target/'NOTICE'), __file__)).write_text('AV-HuBERT is licensed under the AV-HuBERT license, Copyright (c) Meta Platforms, Inc. All Rights Reserved.\n',encoding='utf-8')
    if role=='C':
        for relative in ('checkpoints/vivit-joint-adaptation-002','checkpoints/comparators-ast-001/ast'):
            source=_fk_paths.relocate((ROOT/relative), __file__);target=_fk_paths.relocate((code/relative), __file__);target.mkdir(parents=True)
            metadata=read(_fk_paths.relocate((source/'artifact.json'), __file__))
            for name,expected in metadata['files'].items():
                original=checked_path(source,name)
                if digest(original)!=expected:raise ValueError('pretrained asset checksum mismatch: '+str(original))
                destination=checked_path(target,name);destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(original,destination)
            shutil.copy2(_fk_paths.relocate((source/'artifact.json'), __file__),_fk_paths.relocate((target/'artifact.json'), __file__))
    data=_fk_paths.relocate((output/'dataset'), __file__);data.mkdir()
    for number,row in enumerate(rows,1):
        source=checked_path(dataset,row['path']);target=checked_path(data,row['path'])
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        if digest(target)!=row['source_sha256']:raise ValueError('exported source mismatch: '+row['video_id'])
        if number%100==0:print(f'Copied {number:,}/{len(rows):,} clips',flush=True)
    if include_container:
        print('Exporting AV-HuBERT CUDA image...',flush=True)
        if container_archive:
            shutil.copy2(container_archive,_fk_paths.relocate((output/'avhubert-cuda117-v1.tar'), __file__))
        else:subprocess.run(['docker','save','--output',str(_fk_paths.relocate((output/'avhubert-cuda117-v1.tar'), __file__)),'fk-avhubert:cuda117-v1'],check=True)
    inventory={p.relative_to(output).as_posix():digest(p) for p in output.rglob('*') if p.is_file()}
    report={'schema':'second-machine-bundle-v1','clips':len(rows),'media_bytes':total,
            'population_sha256':digest(manifest),'index_sha256':digest(_fk_paths.relocate((Path(index)/'index.sqlite'), __file__)),
            'files':inventory,'code_snapshot':'current workspace, including uncommitted changes; no .git or environments',
            'worker_role':role,'roles':{'A':'FK and final comparison','B':'AVFakeNet only','C':'Complete AVTENet adaptation'}}
    write(_fk_paths.relocate((output/'bundle.json'), __file__),report)
    print('READY:',output,'media GiB:',round(total/1024**3,3),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('manifest','dataset-root','index','output'):p.add_argument('--'+flag,required=True)
    p.add_argument('--include-container-image',action='store_true')
    p.add_argument('--role',choices=('B','C','legacy'),default='B')
    p.add_argument('--container-archive',help='Copy a previously exported CUDA image instead of docker save')
    args=p.parse_args();export(args.manifest,args.dataset_root,args.index,args.output,
        include_container=args.include_container_image,role=args.role,container_archive=args.container_archive)
