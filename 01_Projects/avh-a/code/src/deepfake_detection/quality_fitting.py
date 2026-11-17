"""Durable training-only quality measurement. SQLite commits one window at a time."""

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

from contextlib import contextmanager, closing
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import importlib.metadata
from time import monotonic

import numpy as np

from .cli import load_json, save
from .forensics import canonical, sha256_file
from .quality import QUALITY_V2, profile_hash

KEYS = ('mean_face_laplacian_variance', 'estimated_snr_db')


@contextmanager
def exclusive_run(path):
    """OS lock releases even after a crash; never remove another process's lock."""
    with _fk_paths.relocate((Path(path)), __file__).open('a+b') as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b'0'); stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError('another fit-quality process is using this output') from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _payload(value):
    raw = canonical(value).decode('utf-8')
    return raw, hashlib.sha256(raw.encode()).hexdigest()


def _verified(raw, expected):
    if hashlib.sha256(raw.encode()).hexdigest() != expected:
        raise ValueError('quality resume measurement checksum mismatch')
    return json.loads(raw)


def _time(seconds):
    if seconds is None:
        return 'warming up'
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours // 24}d {hours % 24:02}:{minutes:02}:{sec:02}' if hours >= 24 else f'{hours:02}:{minutes:02}:{sec:02}'


class Progress:
    def __init__(self, clips, windows, saved, completed, interval):
        self.clips, self.windows, self.saved = clips, windows, saved
        self.done, self.completed = saved, completed
        self.interval, self.started, self.last = interval, monotonic(), -math.inf

    def show(self, message='', *, force=False):
        now = monotonic()
        if force or now - self.last >= self.interval:
            elapsed = now - self.started
            speed = (self.done-self.saved)/elapsed if elapsed else 0
            eta = (self.windows-self.done)/speed if speed else None
            print(f'fit-quality: clips {self.completed:,}/{self.clips:,}; '
                  f'windows {self.done:,}/{self.windows:,} ({100*self.done/self.windows:.2f}%); '
                  f'{speed:.2f} windows/s; elapsed {_time(elapsed)}; ETA {_time(eta)}; {message}',
                  file=sys.stderr, flush=True)
            self.last = now


def _profile(connection, training, asset_hashes):
    count = connection.execute('SELECT COUNT(*) FROM windows WHERE usable=1').fetchone()[0]
    if not count:
        raise ValueError('no usable training windows for quality normalization')
    bounds = {}
    for column in ('blur', 'snr'):
        number = connection.execute(f'SELECT COUNT({column}) FROM windows WHERE usable=1').fetchone()[0]
        values = np.fromiter((item[0] for item in connection.execute(
            f'SELECT {column} FROM windows WHERE usable=1 AND {column} IS NOT NULL')), dtype=np.float64, count=number)
        bounds[KEYS[0 if column == 'blur' else 1]] = np.percentile(values, [5,95]).tolist() if number else None
    return {'schema': QUALITY_V2, 'fit_split': 'train', 'bounds': bounds,
            'training_population_sha256': profile_hash([
                {'video_id':r['video_id'], 'source_sha256':r['source_sha256']} for r in training]),
            'usable_windows': count, 'missing_or_constant_policy': 'neutral_0.5',
            'snr_method': 'speech_power_minus_nonspeech_power_v1', 'asset_sha256': asset_hashes}


def _measure(connection, training, root, detector, vad, asset_hashes, interval):
    from .preparation import prepare_fk_window, fk_source_speech
    total = sum(math.ceil(row['duration']) for row in training)
    saved = connection.execute('SELECT COUNT(*) FROM windows').fetchone()[0]
    completed = connection.execute('SELECT COUNT(*) FROM clips WHERE complete=1').fetchone()[0]
    progress = Progress(len(training), total, saved, completed, interval)
    progress.show(f'restored {saved:,} saved windows', force=True)
    for row in training:
        vid, duration = row['video_id'], row['duration']
        source = (root / row['path']).resolve()
        source.relative_to(root)
        progress.show('checking source: '+vid)
        if sha256_file(source) != row['source_sha256']:
            raise ValueError('quality fitting source hash mismatch: '+vid)
        clip = connection.execute('SELECT speech,speech_hash,complete FROM clips WHERE video_id=?', (vid,)).fetchone()
        if clip:
            speech = _verified(clip[0], clip[1])
            if clip[2]:
                progress.show('verified completed source: '+vid)
                continue
        else:
            progress.show('VAD: '+vid)
            try:
                speech = fk_source_speech(source,duration,vad)
            except ValueError:
                speech = []
            raw, digest = _payload(speech)
            with connection:
                connection.execute('INSERT INTO clips VALUES (?,?,?,0)',(vid,raw,digest))
        for start in range(math.ceil(duration)):
            if connection.execute('SELECT 1 FROM windows WHERE video_id=? AND start=?',(vid,start)).fetchone():
                continue
            progress.show(f'{vid} window {start+1}/{math.ceil(duration)}')
            prepared = prepare_fk_window(source,float(start),duration,detector,speech,fit_to_frame=True)
            item = {key:prepared[key] for key in ('status','quality_measurements')}
            raw,digest = _payload(item)
            usable = item['status']=='USABLE'
            measures = item['quality_measurements']
            values = [measures.get(key) if usable else None for key in KEYS]
            if any(v is not None and (isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v)) for v in values):
                raise ValueError('nonfinite quality measurement')
            with connection:
                connection.execute('INSERT INTO windows VALUES (?,?,?,?,?,?,?,?)',
                                   (vid,start,raw,digest,int(usable),*values, row['source_sha256']))
            progress.done += 1
            progress.show(f'saved {vid} window {start+1}/{math.ceil(duration)}')
        with connection:
            connection.execute('UPDATE clips SET complete=1 WHERE video_id=?',(vid,))
        progress.completed += 1
        progress.show('completed '+vid)
    progress.show('computing exact 5th/95th percentiles',force=True)
    profile = _profile(connection,training,asset_hashes)
    progress.show('finished',force=True)
    return profile


def _validate_saved(connection, training):
    expected = {r['video_id']:r for r in training}
    for vid,raw,digest,complete in connection.execute('SELECT video_id,speech,speech_hash,complete FROM clips'):
        if vid not in expected or complete not in (0,1):
            raise ValueError('unexpected clip in quality resume state')
        _verified(raw,digest)
        number = connection.execute('SELECT COUNT(*) FROM windows WHERE video_id=?',(vid,)).fetchone()[0]
        if complete and number != math.ceil(expected[vid]['duration']):
            raise ValueError('completed quality clip has missing windows')
    for vid,start,raw,digest,usable,blur,snr,sha in connection.execute('SELECT * FROM windows'):
        if vid not in expected or sha!=expected[vid]['source_sha256'] or not 0<=start<math.ceil(expected[vid]['duration']):
            raise ValueError('quality resume window population mismatch')
        item=_verified(raw,digest)
        values=[item['quality_measurements'].get(key) if item['status']=='USABLE' else None for key in KEYS]
        if int(item['status']=='USABLE')!=usable or values!=[blur,snr]:
            raise ValueError('quality resume measurement columns mismatch')


def fit_quality_manifest(manifest_path,dataset_root,assets_path,output,*,resume=False,progress_seconds=10):
    from .commands import load_fk_assets, fk_asset_hashes
    from .data import audit
    if not math.isfinite(progress_seconds) or progress_seconds<=0:
        raise ValueError('progress-seconds must be positive and finite')
    output=_fk_paths.relocate((Path(output)), __file__)
    state=output.with_name(output.name+'.work')
    if output.exists() and not resume:
        raise ValueError('quality output already exists; use a new path')
    if state.exists() and not resume:
        raise ValueError('quality work state already exists; add --resume with the same output')
    if resume and not (_fk_paths.relocate((state/'state.sqlite'), __file__)).is_file():
        raise ValueError('no quality work state exists for --resume; run without --resume first')
    state.mkdir(parents=True,exist_ok=True)
    with exclusive_run(_fk_paths.relocate((state/'run.lock'), __file__)):
        print('fit-quality: loading/auditing manifest...',file=sys.stderr,flush=True)
        manifest_hash=sha256_file(manifest_path)
        rows=load_json(manifest_path);audit(rows)
        training=[row for row in rows if row['split']=='train']
        del rows
        if not training:raise ValueError('quality normalization requires training rows only')
        print(f'fit-quality: {len(training):,} training clips; loading verified assets...',file=sys.stderr,flush=True)
        assets=load_json(assets_path)
        encoders,detector,vad=load_fk_assets(assets)
        hashes=fk_asset_hashes(assets,encoders)
        del encoders
        identity={'schema':'quality-fit-state-v1','manifest_sha256':manifest_hash,'asset_sha256':hashes,
                  'asset_configuration_sha256':profile_hash(assets),
                  'implementation_sha256':{name:sha256_file(_fk_paths.relocate((Path(__file__)), __file__).with_name(name+'.py'))
                    for name in ('quality','quality_fitting','preparation','faces','media','vad')},
                  'sampling':'integer seconds until ceil(duration); fit_to_frame=True',
                  'runtime':{name:importlib.metadata.version(name) for name in
                             ('numpy','torch','av','opencv-python-headless')}}
        with closing(sqlite3.connect(_fk_paths.relocate((state/'state.sqlite'), __file__))) as conn:
            conn.execute('PRAGMA foreign_keys=ON')
            conn.execute('PRAGMA synchronous=FULL')
            conn.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY,value TEXT NOT NULL)')
            previous=conn.execute("SELECT value FROM meta WHERE key='identity'").fetchone()
            if previous:
                if not resume:raise ValueError('quality state exists; use --resume')
                if json.loads(previous[0])!=identity:raise ValueError('quality resume manifest/assets/implementation mismatch')
                print('fit-quality: validating saved measurements...',file=sys.stderr,flush=True)
                _validate_saved(conn,training)
            else:
                if resume:raise ValueError('quality work state has no valid identity; use a new output path')
                with conn:
                    conn.execute('INSERT INTO meta VALUES (?,?)',('identity',canonical(identity).decode()))
                    conn.execute('CREATE TABLE clips (video_id TEXT PRIMARY KEY,speech TEXT NOT NULL,speech_hash TEXT NOT NULL,complete INTEGER NOT NULL)')
                    conn.execute('CREATE TABLE windows (video_id TEXT REFERENCES clips(video_id),start INTEGER, payload TEXT NOT NULL,payload_hash TEXT NOT NULL,usable INTEGER NOT NULL,blur REAL,snr REAL,source_sha256 TEXT NOT NULL,PRIMARY KEY(video_id,start))')
            if output.exists():
                recorded=conn.execute("SELECT value FROM meta WHERE key='output_sha256'").fetchone()
                if not recorded or sha256_file(output)!=recorded[0]:raise ValueError('existing quality output is not the verified completed result')
                print('fit-quality: already complete; output verified',file=sys.stderr,flush=True)
                return load_json(output)
            profile=_measure(conn,training,_fk_paths.relocate((Path(dataset_root)), __file__).resolve(strict=True),detector,vad,hashes,progress_seconds)
            # Persist the intended hash before exclusive output creation so resume
            # also recovers a crash immediately after the final JSON was written.
            with conn:
                conn.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',('output_sha256',hashlib.sha256(canonical(profile)+b'\n').hexdigest()))
            save(output,profile)
            return profile
