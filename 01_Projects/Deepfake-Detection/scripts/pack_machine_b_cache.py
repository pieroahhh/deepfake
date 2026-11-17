"""Pack a verified, homogeneous B cache without recomputing any pixels."""

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

import os,sys,time,concurrent.futures
from pathlib import Path
ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(_fk_paths.relocate((ROOT/'src'), __file__)),str(_fk_paths.relocate((ROOT/'scripts'), __file__))]
import numpy as np
from baselines.common.io import read,digest,fingerprint
from avfakenet_progress_io import write_json
from import_avfakenet_machine_b_cache import converted

SOURCE=_fk_paths.relocate((Path(r'C:\Users\teeku\Downloads\LANDrop\prepared-cache-machb')), __file__)
RUN=_fk_paths.relocate((ROOT/'runs/avfakenet-normalized-pilot-001'), __file__)
OUT=_fk_paths.relocate((ROOT/'runs/avfakenet-machine-b-prepared-001'), __file__)


def pack(row):
    key=fingerprint(row);head=_fk_paths.relocate((SOURCE/(key+'.json')), __file__);payload=_fk_paths.relocate((SOURCE/(key+'.npz')), __file__)
    rec=read(head)
    if rec['source_sha256']!=row['source_sha256'] or digest(payload)!=rec['payload_sha256']:raise ValueError('Source/checksum mismatch')
    directory=_fk_paths.relocate((OUT/key), __file__);directory.mkdir(exist_ok=True);header=_fk_paths.relocate((directory/'record.json'), __file__)
    if header.exists():
        old=read(header)
        if old['source_sha256']!=row['source_sha256'] or any(digest(_fk_paths.relocate((directory/name), __file__))!=sha for name,sha in old['files'].items()):raise ValueError('Existing packed payload differs')
        return
    with np.load(_fk_paths.relocate_value((payload), __file__),allow_pickle=False) as arrays:values=converted(row,arrays,rec['reasons'])
    files={}
    for name,value in values.items():
        temporary=_fk_paths.relocate((directory/(name+'.tmp')), __file__)
        with temporary.open('wb') as stream:np.save(stream,value,allow_pickle=False)
        temporary.replace(_fk_paths.relocate((directory/name), __file__));files[name]=digest(_fk_paths.relocate((directory/name), __file__))
    write_json(header,{'video_id':row['video_id'],'key':key,'split':row['split'],'source_sha256':row['source_sha256'],
        'label':row['label'],'audio_label':row['audio_label'],'reasons':rec['reasons'],'files':files})


def main():
    audits=sorted(RUN.glob('machine-b-cache-import-*/report.json'))
    if not audits:raise ValueError('Run complete cache verification first')
    audit=read(audits[-1])
    if audit['verified']!=3500:raise ValueError('Full cache audit not yet complete')
    identity=read(_fk_paths.relocate((SOURCE/'identity.json'), __file__));pilot=read(_fk_paths.relocate((RUN/'identity.json'), __file__))
    manifest_path=_fk_paths.relocate((ROOT/'runs/two-day-001/population/manifest.json'), __file__);manifest=read(manifest_path)
    if digest(manifest_path)!=identity['population_sha256'] or identity['detector']!=pilot['detector'] or identity['profile']['preprocessing']!=pilot['profile']['preprocessing']:raise ValueError('Identity differs')
    OUT.mkdir(exist_ok=True)
    import msvcrt
    lock=(_fk_paths.relocate((OUT/'worker.lock'), __file__)).open('a+b');lock.write(b'0');lock.flush();lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    info={'source':str(SOURCE),'source_identity_sha256':digest(_fk_paths.relocate((SOURCE/'identity.json'), __file__)),'source_identity':identity,
          'population_sha256':digest(manifest_path),'policy':'Use B pixels for every clip, no mixing A arrays',
          'overlap_comparison_report':str(audits[-1]),'overlap_comparison_sha256':digest(audits[-1]),
          'packer_sha256':digest(_fk_paths.relocate((Path(__file__)), __file__)),'status':'PACKING'}
    if (_fk_paths.relocate((OUT/'identity.json'), __file__)).exists():
        old=read(_fk_paths.relocate((OUT/'identity.json'), __file__))
        if old['source_identity_sha256']!=info['source_identity_sha256']:raise ValueError('Different source')
    write_json(_fk_paths.relocate((OUT/'identity.json'), __file__),info);started=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for done,_ in enumerate(pool.map(pack,manifest),1):
            if done%100==0:
                elapsed=time.monotonic()-started
                write_json(_fk_paths.relocate((OUT/'progress.json'), __file__),{'status':'PACKING','processed':done,'total':3500,'eta_seconds':elapsed/done*(3500-done)})
                print(f'Packed {done}/3500',flush=True)
    if any(digest(_fk_paths.relocate((SOURCE/name), __file__))!=sha for name,sha in audit['source_files'].items()):raise ValueError('Original B cache changed')
    info.update(status='COMPLETE',clips=3500,source_files_unchanged=True)
    write_json(_fk_paths.relocate((OUT/'identity.json'), __file__),info)
    write_json(_fk_paths.relocate((OUT/'progress.json'), __file__),{'status':'FINISHED','processed':3500,'total':3500})
    print('Complete homogeneous Machine B cache ready',flush=True)
    lock.close()
    if '--then-train' in sys.argv:
        import subprocess
        target=_fk_paths.relocate((ROOT/'runs/avfakenet-one-epoch-001'), __file__)
        write_json(_fk_paths.relocate((RUN/'next-run.json'), __file__),{'output':str(target),'epochs_per_branch':1,'prepared_cache':str(OUT),
            'reason':'Verified homogeneous Machine B cache','automatic_heldout':False})
        subprocess.run([sys.executable,'-u',str(_fk_paths.relocate((ROOT/'scripts/run_avfakenet_one_epoch.py'), __file__)),
            '--sanity',str(_fk_paths.relocate((ROOT/'runs/avfakenet-normalized-full-tiny-004'), __file__)),
            '--bundle',r'C:\Users\teeku\Downloads\LANDrop\return-b-recovery-probe-inputs-001',
            '--dataset-root',str(_fk_paths.relocate((ROOT.parent/'avh-a/dataset'), __file__)),'--output',str(target),
            '--prepared-cache',str(OUT),'--resume'],cwd=ROOT,check=True)


if __name__=='__main__':main()
