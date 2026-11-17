"""Verify the complete B cache, compare overlapping outputs, import missing rows.

Original cache is read-only. Run only while preparation is safely paused.
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

import os,sys,argparse,time
from pathlib import Path
ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(_fk_paths.relocate((ROOT/'src'), __file__)),str(_fk_paths.relocate((ROOT/'scripts'), __file__))]
import numpy as np
from baselines.common.io import read,digest,fingerprint
from baselines.common.population import target
from avfakenet_progress_io import write_json


def converted(row,arrays,reasons):
    result={}
    if 'visual' not in reasons:
        faces=arrays['faces'];times=arrays['source_times']
        if faces.dtype!=np.uint8 or faces.ndim!=4 or faces.shape[1:]!=(128,128,3):raise ValueError('Invalid face arrays')
        if len(faces)!=len(times) or not np.isfinite(times).all() or len(times)==0:raise ValueError('Invalid timestamps')
        if (times<-.06).any() or (times>row['duration']+.06).any() or (np.diff(times)<0).any():raise ValueError('Invalid source time range')
        if len(times)!=max(1,int(np.ceil(row['duration']*25-1e-9))):raise ValueError('Frame count mismatch')
        labels=[target(row,'visual',float(t)) for t in times]
        result['visual.npy']=faces
        result['visual-labels.npy']=np.array([-1 if v is None else v for v in labels],dtype=np.int8)
    if 'audio' not in reasons:
        mel=arrays['mel']
        if mel.dtype!=np.float32 or mel.shape!=(3,175,175) or not np.isfinite(mel).all() or mel.min()<0 or mel.max()>1:raise ValueError('Invalid Mel array')
        result['audio.npy']=mel
    if ('visual' in reasons and 'faces' in arrays) or ('audio' in reasons and 'mel' in arrays):raise ValueError('Contradictory cache entry')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    source=args.source.resolve();run=args.run.resolve();cache=_fk_paths.relocate((run/'prepared'), __file__)
    if read(_fk_paths.relocate((run/'progress.json'), __file__))['status']!='PAUSED':raise ValueError('Wait for safe pause')
    lock=(_fk_paths.relocate((run/'worker.lock'), __file__)).open('r+b')
    import msvcrt
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    try:
        identity=read(_fk_paths.relocate((source/'identity.json'), __file__));pilot=read(_fk_paths.relocate((run/'identity.json'), __file__))
        manifest_path=_fk_paths.relocate((ROOT/'runs/two-day-001/population/manifest.json'), __file__);manifest=read(manifest_path)
        from baselines.avfakenet.pipeline import PREPARATION
        if identity['model']!='avfakenet' or identity['stage']!='prepared-inputs' or identity['preparation']!=PREPARATION:raise ValueError('Preparation differs')
        if identity['population_sha256']!=digest(manifest_path) or identity['population_sha256']!=pilot['population_sha256']:raise ValueError('Population differs')
        for part in ('architecture','preprocessing','decision'):
            if identity['profile'][part]!=pilot['profile'][part]:raise ValueError('Profile differs: '+part)
        if identity['detector']!=pilot['detector']:raise ValueError('Detector differs')
        audit=run/('machine-b-cache-import-'+time.strftime('%Y%m%d-%H%M%S'));audit.mkdir()
        report={'source':str(source),'source_identity_sha256':digest(_fk_paths.relocate((source/'identity.json'), __file__)),
            'population_sha256':digest(manifest_path),'importer_sha256':digest(_fk_paths.relocate((Path(__file__)), __file__)),
            'verified':0,'overlap':0,'imported':0,'differences':[],'source_files':{},'protected_existing':{}}
        # Pass 1: verify every payload and compare all existing arrays. Nothing is
        # imported until the full cache and all overlapping records pass.
        for number,row in enumerate(manifest,1):
            key=fingerprint(row);header=source/(key+'.json');payload=source/(key+'.npz');record=read(header)
            if record['source_sha256']!=row['source_sha256'] or digest(payload)!=record['payload_sha256']:raise ValueError('Source/checksum mismatch: '+row['video_id'])
            report['source_files'][header.name]=digest(header);report['source_files'][payload.name]=record['payload_sha256']
            with np.load(_fk_paths.relocate_value((payload), __file__),allow_pickle=False) as arrays:values=converted(row,arrays,record['reasons'])
            existing=_fk_paths.relocate((cache/key/'record.json'), __file__)
            if existing.exists():
                old=read(existing);report['overlap']+=1
                if old['source_sha256']!=record['source_sha256'] or old['reasons']!=record['reasons'] or set(old['files'])!=set(values):
                    report['differences'].append({'video_id':row['video_id'],'type':'eligibility_or_files'})
                else:
                    for name,array in values.items():
                        path=existing.parent/name
                        if digest(path)!=old['files'][name]:raise ValueError('Existing cache corrupt')
                        if not np.array_equal(np.load(_fk_paths.relocate_value((path), __file__),allow_pickle=False),array):report['differences'].append({'video_id':row['video_id'],'array':name})
                        report['protected_existing'][str(path)]=old['files'][name]
                report['protected_existing'][str(existing)]=digest(existing)
            report['verified']=number
            if number%100==0:
                write_json(_fk_paths.relocate((audit/'report.json'), __file__),report);print(f'Verified {number}/3500; overlap {report["overlap"]}; differences {len(report["differences"])}',flush=True)
        write_json(_fk_paths.relocate((audit/'report.json'), __file__),report)
        if report['differences']:raise ValueError('Caches differ; no missing entries imported. Review report.json')
        for number,row in enumerate(manifest,1):
            key=fingerprint(row);directory=cache/key;header=_fk_paths.relocate((directory/'record.json'), __file__)
            if header.exists():continue
            record=read(source/(key+'.json'));payload=source/(key+'.npz')
            if digest(payload)!=report['source_files'][payload.name]:raise ValueError('Transferred cache changed during import')
            with np.load(_fk_paths.relocate_value((payload), __file__),allow_pickle=False) as arrays:values=converted(row,arrays,record['reasons'])
            directory.mkdir(exist_ok=True);files={}
            for name,array in values.items():
                temp=directory/(name+'.import.tmp')
                with temp.open('wb') as stream:np.save(stream,array,allow_pickle=False)
                temp.replace(directory/name);files[name]=digest(directory/name)
            write_json(header,{'video_id':row['video_id'],'key':key,'split':row['split'],'source_sha256':row['source_sha256'],
                'label':row['label'],'audio_label':row['audio_label'],'reasons':record['reasons'],'files':files})
            report['imported']+=1
            if report['imported']%100==0:print(f'Imported {report["imported"]} missing clips',flush=True)
        if any(digest(_fk_paths.relocate((Path(path)), __file__))!=sha for path,sha in report['protected_existing'].items()):raise ValueError('Original completed files changed')
        if any(digest(source/name)!=sha for name,sha in report['source_files'].items()) or digest(_fk_paths.relocate((source/'identity.json'), __file__))!=report['source_identity_sha256']:raise ValueError('Machine B input changed')
        report.update(status='COMPLETE',all_existing_files_unchanged=True,all_source_files_unchanged=True)
        write_json(_fk_paths.relocate((audit/'report.json'), __file__),report)
        print(f'COMPLETE: {report["imported"]} imported, {report["overlap"]} preserved',flush=True)
    finally:lock.close()


if __name__=='__main__':main()
