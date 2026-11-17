"""Strict, read-only validation of an attested partial feature-cache snapshot."""

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

from contextlib import closing
from pathlib import Path
import json
import sqlite3
import numpy as np
from baselines.common.io import read,digest
from baselines.common.population import windows


def snapshot_owned(source,output,population,checkpoint,stage):
    """Recover only our identified, stopped producer; preserve original bytes."""
    import shutil,tempfile
    from baselines.common.io import write,fingerprint
    from baselines.common.training import load_checkpoint
    source,output=_fk_paths.relocate((Path(source)), __file__),_fk_paths.relocate((Path(output)), __file__)
    producer=read(_fk_paths.relocate((source/'producer.json'), __file__))
    selected=load_checkpoint(checkpoint);identity=selected['identity']
    expected={'stage':stage,'index_sha256':population.metadata['index_sha256'],
        'profile_sha256':fingerprint(identity['profile']),
        'preparation_sha256':fingerprint(identity['preparation']),
        'checkpoint_sha256':digest(checkpoint),'identity':identity}
    if producer.get('schema')!='avtenet-owned-feature-producer-v1' or any(producer.get(k)!=v for k,v in expected.items()):
        raise ValueError('partial feature producer differs from selected checkpoint/population')
    output.mkdir(exist_ok=False)
    with tempfile.TemporaryDirectory(dir=output) as scratch:
        recovery=_fk_paths.relocate((Path(scratch)/'features.sqlite'), __file__)
        shutil.copy2(_fk_paths.relocate((source/'features.sqlite'), __file__),recovery)
        journal=_fk_paths.relocate((source/'features.sqlite-journal'), __file__)
        if journal.exists():shutil.copy2(journal,recovery.with_name(recovery.name+'-journal'))
        with closing(sqlite3.connect(recovery)) as original,closing(sqlite3.connect(_fk_paths.relocate((output/'features.sqlite'), __file__))) as copied:
            original.backup(copied)
            keys,_,_=validate_rows(copied,population)
    # A killed process can leave one incomplete final JSONL write. It represents
    # no complete failure decision and will be prepared again, never fabricated.
    lines=(_fk_paths.relocate((source/'preparation-events.jsonl'), __file__)).read_text(encoding='utf-8').splitlines()
    valid=[]
    for i,line in enumerate(lines):
        try:valid.append(json.loads(line))
        except json.JSONDecodeError:
            if i==len(lines)-1:break
            raise
    (_fk_paths.relocate((output/'preparation-events.jsonl'), __file__)).write_text(''.join(json.dumps(v)+'\n' for v in valid),encoding='utf-8')
    metadata={'schema':'avtenet-attested-partial-features-v1',**expected,'windows':len(keys),
        'source':str(source),'producer_sha256':digest(_fk_paths.relocate((source/'producer.json'), __file__)),
        'payload_sha256':digest(_fk_paths.relocate((output/'features.sqlite'), __file__)),'events_sha256':digest(_fk_paths.relocate((output/'preparation-events.jsonl'), __file__))}
    write(_fk_paths.relocate((output/'resume.json'), __file__),metadata)
    validate_snapshot(output,expected,population)
    return output


FIELDS=('stage','index_sha256','profile_sha256','preparation_sha256','checkpoint_sha256','identity')


def validate_rows(connection,population):
    rows={row['video_id']:row for split in ('train','validation','test') for row in population.rows(split)}
    intervals={vid:set(windows(row['duration'])) for vid,row in rows.items()}
    keys=set()
    for vid,start,end,split,sha,blob in connection.execute('SELECT video_id,start,end,split,source_sha256,features FROM features'):
        row=rows.get(vid)
        if (row is None or split!=row['split'] or sha!=row['source_sha256'] or
                (start,end) not in intervals[vid] or len(blob)!=768*4 or
                not np.isfinite(np.frombuffer(blob,dtype='<f4')).all()):
            raise ValueError('partial feature row differs from the frozen population')
        key=(vid,start,end)
        if key in keys:raise ValueError('duplicate partial feature row')
        keys.add(key)
    return keys,rows,intervals


def validate_snapshot(directory,expected,population):
    root=_fk_paths.relocate((Path(directory)), __file__)
    metadata=read(_fk_paths.relocate((root/'resume.json'), __file__))
    if metadata.get('schema')!='avtenet-attested-partial-features-v1':raise ValueError('unattested partial feature cache')
    if any(metadata.get(key)!=expected[key] for key in FIELDS):
        raise ValueError('resume selected checkpoint/population/profile mismatch')
    if digest(_fk_paths.relocate((root/'features.sqlite'), __file__))!=metadata['payload_sha256']:raise ValueError('partial feature payload checksum mismatch')
    events=_fk_paths.relocate((root/'preparation-events.jsonl'), __file__)
    if digest(events)!=metadata['events_sha256']:raise ValueError('partial feature events checksum mismatch')
    with closing(sqlite3.connect((_fk_paths.relocate((root/'features.sqlite'), __file__)).resolve().as_uri()+'?mode=ro',uri=True)) as conn:
        if conn.execute('PRAGMA quick_check').fetchone()[0]!='ok':raise ValueError('invalid partial feature database')
        keys,rows,intervals=validate_rows(conn,population)
    if len(keys)!=metadata['windows']:raise ValueError('partial feature count mismatch')
    failures=set()
    for line in events.read_text(encoding='utf-8').splitlines():
        event=json.loads(line);row=rows.get(event.get('video_id'))
        if row is None or event.get('split')!=row['split']:raise ValueError('partial failure event population mismatch')
        if 'start' in event and 'end' in event:
            pair=(event['start'],event['end'])
            if pair not in intervals[row['video_id']]:raise ValueError('partial failure event window mismatch')
            failures.add((row['video_id'],*pair))
    if keys & failures:raise ValueError('partial window cannot be both cached and failed')
    return metadata,keys,failures
