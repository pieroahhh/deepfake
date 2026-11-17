"""Deterministic bounded-memory sampling from the already audited population."""

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

from collections import Counter, defaultdict
import hashlib
import heapq
import json
from pathlib import Path

from .io import read,write,digest


def stream_rows(path):
    decoder=json.JSONDecoder()
    with _fk_paths.relocate((Path(path)), __file__).open(encoding='utf-8') as stream:
        buffer='';ended=False;started=False
        while True:
            if not ended:
                chunk=stream.read(1024*1024);ended=not chunk;buffer+=chunk
            offset=0
            while True:
                while offset<len(buffer) and buffer[offset] in ' \n\r\t,':offset+=1
                if not started and offset<len(buffer):
                    if buffer[offset]!='[':raise ValueError('population must be a JSON array')
                    offset+=1;started=True;continue
                if offset<len(buffer) and buffer[offset]==']':return
                try:value,end=decoder.raw_decode(buffer,offset)
                except json.JSONDecodeError:break
                yield value;offset=end
            buffer=buffer[offset:]
            if ended:raise ValueError('incomplete population JSON')


def quotas(counts,total):
    if type(total) is not int or total<len(counts) or total>sum(counts.values()):
        raise ValueError('subset size must cover each stratum and fit within available clips')
    exact={key:total*value/sum(counts.values()) for key,value in counts.items()}
    result={key:max(1,int(value)) for key,value in exact.items()}
    while sum(result.values())<total:
        key=max((k for k in counts if result[k]<counts[k]),key=lambda k:(exact[k]-result[k],k))
        result[key]+=1
    while sum(result.values())>total:
        key=max((k for k in counts if result[k]>1),key=lambda k:(result[k]-exact[k],k))
        result[key]-=1
    return result


def make_subset(manifest,output,*,train=3000,validation=500,seed=7):
    from deepfake_detection.data import audit
    manifest=_fk_paths.relocate((Path(manifest)), __file__);source_hash=digest(manifest)
    source_audit=read(_fk_paths.relocate((manifest.parent/'audit.json'), __file__))
    if source_audit.get('manifest_sha256')!=source_hash:
        raise ValueError('source manifest does not match its audit')
    counts=defaultdict(Counter)
    for row in stream_rows(manifest):
        if row['split'] in ('train','validation'):
            counts[row['split']][row['source_modify_type']]+=1
    targets={split:quotas(counts[split],size) for split,size in (('train',train),('validation',validation))}
    heaps=defaultdict(list)
    for row in stream_rows(manifest):
        split=row['split']
        if split not in targets:continue
        group=row['source_modify_type'];limit=targets[split][group]
        score=int.from_bytes(hashlib.sha256(f'{seed}\0{row["video_id"]}'.encode()).digest(),'big')
        item=(-score,row['video_id'],row)
        heap=heaps[split,group]
        if len(heap)<limit:heapq.heappush(heap,item)
        elif item>heap[0]:heapq.heapreplace(heap,item)
    rows=sorted((item[2] for heap in heaps.values() for item in heap),key=lambda r:(r['split'],r['video_id']))
    subset_audit=audit(rows)
    destination=_fk_paths.relocate((Path(output)), __file__);destination.mkdir(parents=True,exist_ok=False)
    write(_fk_paths.relocate((destination/'manifest.json'), __file__),rows)
    report={'schema':'two-day-subset-v1','manifest_sha256':digest(_fk_paths.relocate((destination/'manifest.json'), __file__)),
            'source_manifest_sha256':source_hash,'seed':seed,'study_audit':subset_audit,
            'source_counts':{s:dict(c) for s,c in counts.items()},'selected_counts':targets,
            'selection':'proportional stratification by official split and source_modify_type; lowest seeded SHA256 per clip',
            'group_policy':'original official split retained; clip-level sample; families not excluded or reassigned',
            'test_access':'none','result_scope':'subset development comparison; not full benchmark or independent test'}
    write(_fk_paths.relocate((destination/'audit.json'), __file__),report)
    return report


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--train',type=int,default=3000);p.add_argument('--validation',type=int,default=500)
    p.add_argument('--seed',type=int,default=7)
    args=p.parse_args()
    print(json.dumps(make_subset(args.manifest,args.output,train=args.train,validation=args.validation,seed=args.seed),indent=2))
