"""Isolated, frozen-checkpoint Table III inference instrumentation. No training."""

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
import contextlib
import functools
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import threading
import time
from pathlib import Path


def read(path):
    return json.loads(_fk_paths.relocate((Path(path)), __file__).read_text(encoding='utf-8'))


def write(path, value):
    path = _fk_paths.relocate((Path(path)), __file__)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(str(temporary), str(path))


def digest(path):
    value = hashlib.sha256()
    with _fk_paths.relocate((Path(path)), __file__).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


class Timer:
    """Nested synchronized wall timers; self times prevent double counting."""
    def __init__(self, torch):
        self.torch = torch
        self.local = threading.local()
        self.values = {}
        self.undo = []

    def reset(self):
        self.values = {}

    def sync(self):
        self.torch.cuda.synchronize()

    def add(self, name, elapsed, child=0.):
        entry = self.values.setdefault(name, {'inclusive_seconds': 0., 'self_seconds': 0., 'calls': 0})
        entry['inclusive_seconds'] += elapsed
        entry['self_seconds'] += max(0., elapsed - child)
        entry['calls'] += 1

    @contextlib.contextmanager
    def phase(self, name, gpu=False):
        if gpu:
            self.sync()
        stack = getattr(self.local, 'stack', None)
        if stack is None:
            stack = self.local.stack = []
        frame = [time.perf_counter(), 0.]
        stack.append(frame)
        try:
            yield
        finally:
            if gpu:
                self.sync()
            elapsed = time.perf_counter() - frame[0]
            stack.pop()
            self.add(name, elapsed, frame[1])
            if stack:
                stack[-1][1] += elapsed

    def patch(self, obj, name, label, gpu=False):
        original = getattr(obj, name)
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            with self.phase(label, gpu):
                return original(*args, **kwargs)
        setattr(obj, name, wrapped)
        self.undo.append((obj, name, original))

    def module(self, module, name):
        self.patch(module, 'forward', name, True)


class PeakRSS:
    def __enter__(self):
        try:
            import psutil
            process = psutil.Process()
            self.current = lambda: process.memory_info().rss
        except ImportError:
            self.current = lambda: int(_fk_paths.relocate((Path('/proc/self/statm')), __file__).read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
        self.peak = self.current()
        self.samples = 1
        self.stop = threading.Event()
        def sample():
            while not self.stop.wait(.02):
                self.peak = max(self.peak, self.current())
                self.samples += 1
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.peak = max(self.peak, self.current())


class TimedDetector:
    def __init__(self, original, timer):
        self.original, self.timer = original, timer

    def __getattr__(self, name):
        return getattr(self.original, name)

    def __call__(self, rgb):
        with self.timer.phase('pre.face_detector', True):
            return self.original(rgb)


def media_timers(timer):
    import deepfake_detection.faces as faces
    import deepfake_detection.media as media
    import deepfake_detection.preparation as prep
    from baselines.common.video_reader import VideoWindowReader
    timer.patch(media, 'aligned_audio', 'io.audio_decode_resample')
    timer.patch(prep, 'aligned_audio', 'io.audio_decode_resample')
    timer.patch(media, 'sample_video', 'io.video_decode_sample')
    timer.patch(prep, 'sample_video', 'io.video_decode_sample')
    timer.patch(VideoWindowReader, 'sample', 'io.video_decode_sample')
    timer.patch(faces, 'track_and_crop', 'pre.tracking_and_frame_bookkeeping')
    timer.patch(prep, 'track_and_crop', 'pre.tracking_and_frame_bookkeeping')
    original = faces.aligned_crops
    def alignment(*args, **kwargs):
        starts = {}
        def crop_profile(frame, event, arg):
            if frame.f_code.co_name != 'crop' or frame.f_code.co_filename != original.__code__.co_filename:
                return
            if event == 'call':
                starts[id(frame)] = time.perf_counter()
            elif event == 'return' and id(frame) in starts:
                elapsed = time.perf_counter() - starts.pop(id(frame))
                label = 'pre.lip_crop' if frame.f_locals.get('pixels') == 96 else 'pre.face_crop'
                timer.add(label, elapsed)
                timer.local.stack[-1][1] += elapsed
        with timer.phase('pre.shared_alignment_geometry'):
            old = sys.getprofile()
            sys.setprofile(crop_profile)
            try:
                return original(*args, **kwargs)
            finally:
                sys.setprofile(old)
    faces.aligned_crops = alignment


def fk_timers(timer, fk):
    import deepfake_detection.preparation as prep
    import deepfake_detection.inference as inference
    import deepfake_detection.forensics as forensic
    for module in (prep, inference, forensic):
        timer.patch(module, 'sha256_file', 'record.file_hash')
    timer.patch(inference, 'preserve', 'record.source_copy')
    timer.patch(inference, 'prepare_fk_window', 'pre.quality_and_window_residual')
    timer.patch(inference, 'fk_source_speech', 'pre.audio_speech_preparation')
    timer.patch(inference, 'encode_fk_window', 'feature.encoder_normalization_transfer')
    timer.patch(inference, 'save_prepared_window', 'record.derived_persistence')
    timer.patch(inference, 'record_from_outputs', 'record.canonical_construction')
    timer.patch(inference, 'build_record', 'record.build')
    timer.patch(inference, 'commitment', 'record.commitment')
    timer.patch(forensic, 'commitment', 'record.commitment')
    timer.patch(forensic, 'canonical', 'record.serialization')
    timer.patch(inference, 'verify', 'record.verify')
    timer.patch(forensic, 'verify', 'record.verify')
    timer.patch(forensic, 'hybrid', 'aggregation.bin_to_window')
    timer.patch(forensic, 'video_score', 'aggregation.window_to_video')
    timer.patch(forensic, 'coverage', 'aggregation.coverage')
    timer.patch(inference, 'save', 'record.json_persistence')
    timer.patch(prep, 'save', 'record.json_persistence')
    timer.patch(fk.vad, '__call__', 'pre.vad')  # Special lookup is timed by proxy below.
    vad = fk.vad
    class VAD:
        def __call__(self, samples):
            with timer.phase('pre.vad', True):
                return type(vad).__call__(vad, samples)
    fk.vad = VAD()
    fk.detector = TimedDetector(fk.detector, timer)
    timer.module(fk.encoders.visual_model.features, 'feature.efficientnet')
    timer.module(fk.encoders.audio_model, 'feature.wav2vec2')
    timer.patch(fk.model, 'forward', 'evidence.attention_tensorops_validation', True)
    timer.module(fk.model.gru, 'feature.gru')
    timer.module(fk.model.visual, 'feature.visual_projection')
    timer.module(fk.model.audio, 'feature.audio_projection')
    timer.patch(fk.model, 'pool', 'feature.bin_pool')
    for name in ('vq', 'vk', 'vo', 'aq', 'ak', 'ao', 'av', 'dv', 'da', 'temporal'):
        timer.module(getattr(fk.model, name), 'evidence.' + name)
    timer.module(fk.model.summary, 'evidence.summary')
    timer.module(fk.model.reliability, 'evidence.reliability')
    for i, head in enumerate(fk.model.heads):
        timer.module(head, 'evidence.head_' + str(i))
    original_hybrid = fk.model._hybrid
    def hybrid(*args, **kwargs):
        with timer.phase('aggregation.tensor_topk', True):
            return original_hybrid(*args, **kwargs)
    fk.model._hybrid = hybrid
    timer.patch(fk.model, 'aggregate_video', 'aggregation.video_tensor', True)


def parameters(model):
    return {'parameters': sum(p.numel() for p in model.parameters()),
            'requires_grad_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)}


def check_prediction(actual, expected):
    if actual['status'] != expected['status'] or actual['prediction'] != expected['prediction']:
        raise ValueError('Frozen assessment/decision parity failed')
    a, b = actual.get('score'), expected.get('score')
    if (a is None) != (b is None) or (a is not None and abs(a - b) > 5e-6):
        raise ValueError('Frozen score parity failed (unchanged 5e-6 gate)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--code-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--stage', choices=('fk', 'avfakenet', 'audio', 'visual', 'audiovisual', 'fusion'), required=True)
    parser.add_argument('--assets')
    args = parser.parse_args()
    workspace, root, output = _fk_paths.relocate((Path(args.workspace)), __file__), _fk_paths.relocate((Path(args.code_root)), __file__), _fk_paths.relocate((Path(args.output)), __file__)
    repo = _fk_paths.relocate((workspace / 'Deepfake-Detection'), __file__)
    protocol = read(_fk_paths.relocate((output / 'protocol.json'), __file__))
    if digest(_fk_paths.relocate((Path(__file__)), __file__)) != protocol['profiler_sha256']:
        raise ValueError('Profiler changed after protocol freeze')
    sys.path[:0] = [str(root), str(_fk_paths.relocate((root / 'src'), __file__)), str(_fk_paths.relocate((root / 'scripts'), __file__))]
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '2'
    os.environ['HF_HOME'] = str(_fk_paths.relocate((root / '.cache/huggingface'), __file__))
    os.environ['MPLCONFIGDIR'] = str(_fk_paths.relocate((output / 'matplotlib'), __file__))
    # The old workload must finish, including fusion, before this measurement.
    if not (_fk_paths.relocate((repo / 'runs/remaining-supplement-002/complete.json'), __file__)).exists():
        raise RuntimeError('Original workload is not complete; do not compete for GPU')
    import numpy as np
    import torch
    from baselines.common.io import fingerprint
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = args.stage == 'fk'
    torch.backends.cudnn.benchmark = False
    rows = read(_fk_paths.relocate((repo / 'experiments/remaining-001/profiling-manifest.json'), __file__))
    if digest(_fk_paths.relocate((repo / 'experiments/remaining-001/profiling-manifest.json'), __file__)) != protocol['manifest_sha256']:
        raise ValueError('Profiling population changed')
    timer = Timer(torch)
    dest = output / args.stage
    dest.mkdir(exist_ok=True)
    baseline = None
    metadata = {'stage': args.stage, 'profiler_sha256': protocol['profiler_sha256'],
                'torch': str(torch.__version__), 'python': platform.python_version(),
                'device': torch.cuda.get_device_name(), 'tf32_cudnn': args.stage == 'fk',
                'execution': 'frozen checkpoints; synchronized nested instrumentation; no fitting',
                'original_uninstrumented_timing': 'remaining-supplement-002 retained separately'}
    if args.stage == 'fk':
        from run_remaining_supplement import FrozenFK
        model = FrozenFK()
        metadata['checkpoint_sha256'] = model.sha
        metadata['model'] = parameters(model.model)
        metadata['encoders'] = parameters(model.encoders)
        media_timers(timer)
        fk_timers(timer, model)
        baseline = {x['video_id']: x for x in read(_fk_paths.relocate((workspace / 'fk-heldout-copy.json'), __file__))['videos']}
    elif args.stage == 'avfakenet':
        from remaining_avfakenet import FrozenAVFakeNet
        model = FrozenAVFakeNet(repo)
        metadata['training_identity'] = model.sha
        metadata['model'] = {k: parameters(v) for k, v in model.models.items()}
        media_timers(timer)
        model.detect = TimedDetector(model.detect, timer)
        for branch, network in model.models.items():
            timer.module(network, 'feature.dst_' + branch)
        import remaining_avfakenet as adapter
        original_faces = adapter.faces
        def serial_faces(*a, **kw):
            kw['prefetch_decode'] = False
            return original_faces(*a, **kw)
        adapter.faces = serial_faces
        timer.patch(adapter, 'prepare_audio', 'pre.mel_spectrogram')
        timer.patch(adapter, 'avfakenet_decision', 'aggregation.majority_and_or')
        baseline = {x['row']['video_id']: x for x in read(_fk_paths.relocate((repo / 'runs/remaining-supplement-002/profiling-avfakenet/summary.json'), __file__))['clips']}
    else:
        from baselines.avtenet.model import load_component_profile, FeatureFusion
        from baselines.common.training import load_checkpoint, move
        plan = read(_fk_paths.relocate((root / 'runs/auto-next-001/test-plan.json'), __file__))
        config = load_component_profile(root / plan['config'])
        checkpoint = root / plan['checkpoints'][args.stage]
        if digest(checkpoint) != plan['files'][plan['checkpoints'][args.stage]]:
            raise ValueError('Expert/fusion checkpoint hash changed')
        selected = load_checkpoint(checkpoint)
        metadata['checkpoint_sha256'] = digest(checkpoint)
        if args.stage == 'fusion':
            model = FeatureFusion(config).cuda().eval()
            model.load_state_dict(selected['model_state'], strict=True)
            timer.module(model, 'evidence.feature_fusion')
        else:
            from baselines.avtenet.assets import registry
            from baselines.avtenet.pipeline import Prepared, stage_model
            from baselines.common.media import detector
            assets = registry(args.assets)
            model, backend, proof = stage_model(args.stage, config, assets)
            model.load_state_dict(selected['model_state'], strict=True)
            model.cuda().eval().requires_grad_(False)
            detect, provenance = detector('cuda')
            media_timers(timer)
            detect = TimedDetector(detect, timer)
            timer.module(model.backend, 'feature.' + {'audio': 'ast', 'visual': 'vivit', 'audiovisual': 'avhubert'}[args.stage])
            if args.stage == 'audiovisual':
                timer.module(model.temporal_head, 'feature.ms_tcn_and_pool')
                import python_speech_features
                timer.patch(python_speech_features, 'logfbank', 'pre.log_filterbank')
        metadata['model'] = parameters(model)
        del selected
    write(_fk_paths.relocate((dest / 'identity.json'), __file__), metadata)
    for index, row in enumerate(rows):
        target = dest / ('%02d.json' % index)
        if target.exists():
            continue
        if time.time() > protocol['deadline_unix'] - 120:
            write(_fk_paths.relocate((dest / 'stopped-at-deadline.json'), __file__), {'next_clip': index, 'completed_clips': index})
            break
        write(_fk_paths.relocate((output / 'progress.json'), __file__), {'stage': args.stage, 'clip': index + 1, 'clips': len(rows), 'updated_unix': time.time()})
        print('%s clip %d/%d %s' % (args.stage, index + 1, len(rows), row['video_id']), flush=True)
        source = _fk_paths.relocate((workspace / 'Heldout-1000/dataset'), __file__) / row['path']
        if digest(source) != row['source_sha256']:
            raise ValueError('Profiling source changed')
        samples = []
        for repeat in range(4):
            timer.reset()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            with PeakRSS() as rss:
                started = time.perf_counter()
                with torch.inference_mode():
                    if args.stage == 'fk':
                        artifact = dest / ('artifacts-%02d-%d' % (index, repeat))
                        model.record(source, row, artifact)
                        torch.cuda.synchronize()
                        measured_elapsed = time.perf_counter() - started
                        rss.__exit__()
                        record = read(_fk_paths.relocate((artifact / 'record.json'), __file__))
                        result = {k: record[k] for k in ('status', 'prediction', 'score')}
                        # Check the same retained prediction, while adding full persistence.
                        expected = baseline[row['video_id']]
                        check_prediction(result, expected)
                        extra = {'windows': len(record['windows']), 'usable_windows': sum(w['status'] == 'USABLE' for w in record['windows']),
                                 'derived_items': len(record['derived_evidence']), 'record_bytes': (_fk_paths.relocate((artifact / 'record.json'), __file__)).stat().st_size,
                                 'verification': read(_fk_paths.relocate((artifact / 'verification.json'), __file__))}
                    elif args.stage == 'avfakenet':
                        result = model.predict(source, row)
                        torch.cuda.synchronize()
                        measured_elapsed = time.perf_counter() - started
                        rss.__exit__()
                        check_prediction(result, baseline[row['video_id']]['samples'][min(repeat, 3)])
                        extra = {}
                    elif args.stage == 'fusion':
                        streams = {stage: read(output / stage / ('%02d.json' % index))['samples'][repeat]['vectors'] for stage in ('audio', 'visual', 'audiovisual')}
                        lookups = {stage: {(x['start'], x['end']): x['vector'] for x in values} for stage, values in streams.items()}
                        common = sorted(set.intersection(*(set(v) for v in lookups.values())))
                        scores = []
                        for pair in common:
                            inputs = {stage: torch.tensor(values[pair], dtype=torch.float32, device='cuda').reshape(1, 768) for stage, values in lookups.items()}
                            scores.append(float((-model(inputs)).sigmoid().item()))
                        with timer.phase('aggregation.mean_windows'):
                            score = sum(scores) / len(scores) if scores else None
                            result = {'status': 'ASSESSED' if score is not None else 'UNASSESSED', 'prediction': None if score is None else int(score >= .5), 'score': score}
                        torch.cuda.synchronize()
                        measured_elapsed = time.perf_counter() - started
                        rss.__exit__()
                        original = _fk_paths.relocate((repo / 'runs/remaining-supplement-002/avtenet/fusion'), __file__) / (fingerprint({'purpose': 'profiling', 'row': dict(row, media_path='Heldout-1000/dataset/' + row['path'])}) + '.json')
                        check_prediction(result, read(original)['samples'][repeat]['prediction'])
                        extra = {'usable_windows': len(common)}
                    else:
                        class Single:
                            def rows(self, split):
                                return [row]
                            def path(self, item):
                                return source
                        prepared = Prepared(Single(), args.stage, backend, assets, detect, _fk_paths.relocate((dest / 'failures.jsonl'), __file__), None,
                                            reuse_video_decoder=True, cache_face_detections=True,
                                            fast_vivit_processor=args.stage == 'visual', prefetch_video_decode=False)
                        timer.patch(prepared, 'transform', 'pre.model_input_transform')
                        if args.stage == 'audio':
                            extractor = prepared.extractor
                            class Extractor:
                                def __call__(self, *a, **kw):
                                    with timer.phase('pre.mel_spectrogram'):
                                        return extractor(*a, **kw)
                            prepared.extractor = Extractor()
                        vectors = []
                        for _, start, end, values, _ in prepared.records('test'):
                            vector = model.extract_features(move(values, 'cuda')).cpu().numpy().astype('<f4').reshape(-1)
                            vectors.append({'start': start, 'end': end, 'vector': vector.tolist()})
                        torch.cuda.synchronize()
                        measured_elapsed = time.perf_counter() - started
                        rss.__exit__()
                        original_item = {'purpose': 'profiling', 'row': dict(row, media_path='Heldout-1000/dataset/' + row['path'])}
                        original = read(_fk_paths.relocate((repo / 'runs/remaining-supplement-002/avtenet'), __file__) / args.stage / (fingerprint(original_item) + '.json'))['samples'][repeat]['vectors']
                        if len(original) != len(vectors):
                            raise ValueError('Expert coverage parity failed')
                        error = 0.
                        for a, b in zip(vectors, original):
                            if (a['start'], a['end']) != (b['start'], b['end']):
                                raise ValueError('Expert source-time interval parity failed')
                            error = max(error, float(np.max(np.abs(np.asarray(a['vector']) - np.asarray(b['vector'])))))
                        if error > 5e-6:
                            raise ValueError('Expert feature parity failed: ' + str(error))
                        result = {'status': 'ASSESSED' if vectors else 'UNASSESSED', 'prediction': None, 'score': None}
                        extra = {'vectors': vectors, 'usable_windows': len(vectors), 'maximum_feature_error': error}
                torch.cuda.synchronize()
                elapsed = measured_elapsed
            samples.append({'repeat': repeat, 'warmup': repeat == 0, 'seconds': elapsed,
                            'timers': dict(timer.values), 'result': result, **extra,
                            'sampled_peak_rss_bytes': rss.peak, 'rss_sampling_interval_seconds': .02, 'rss_samples': rss.samples,
                            'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                            'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved()})
        write(target, {'row': row, 'samples': samples, 'parity': 'PASSED; original assessment and score/vector gate <=5e-6',
                       'execution_note': 'Synchronized instrumented wall times; expert decode prefetch disabled for additive main-thread accounting; weights/inputs/intervals unchanged.'})
    records = [read(dest / ('%02d.json' % i)) for i in range(len(rows)) if (dest / ('%02d.json' % i)).exists()]
    write(_fk_paths.relocate((dest / 'summary.json'), __file__), {'metadata': metadata, 'attempted': len(records), 'planned': len(rows),
                                 'clips': [{'video_id': x['row']['video_id'], 'status': x['samples'][1]['result']['status'],
                                            'median_seconds': statistics.median(s['seconds'] for s in x['samples'][1:])} for x in records]})
    if len(records) == len(rows):
        write(_fk_paths.relocate((dest / 'complete.json'), __file__), {'clips': len(rows), 'summary_sha256': digest(_fk_paths.relocate((dest / 'summary.json'), __file__)), 'no_training': True})
    print('Stage finished: ' + args.stage, flush=True)


if __name__ == '__main__':
    main()
