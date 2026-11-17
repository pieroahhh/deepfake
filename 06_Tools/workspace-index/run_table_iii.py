"""Run an isolated one-hour inference study and publish actual measured coverage."""

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
import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = _fk_paths.workspace_root(__file__)
REPO = _fk_paths.relocate((ROOT / 'Deepfake-Detection'), __file__)
OUT = _fk_paths.relocate((REPO / 'runs/table-iii-cost-001'), __file__)
PROFILER = _fk_paths.relocate((Path(__file__)), __file__).with_name('table_iii_profiler.py')
spec = importlib.util.spec_from_file_location('table_iii_tools', PROFILER)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
read, write, digest = helpers.read, helpers.write, helpers.digest


def prepare():
    OUT.mkdir(exist_ok=False)
    paths = [_fk_paths.relocate((REPO / 'scripts/run_remaining_avtenet.py'), __file__), _fk_paths.relocate((REPO / 'scripts/run_remaining_supplement.py'), __file__),
             _fk_paths.relocate((REPO / 'scripts/continue_remaining_avtenet.py'), __file__), _fk_paths.relocate((ROOT / 'fk-heldout-copy.json'), __file__),
             _fk_paths.relocate((ROOT / 'Heldout-1000/population/manifest.json'), __file__),
             _fk_paths.relocate((REPO / 'runs/three-model-heldout-002/comparison.json'), __file__)]
    write(_fk_paths.relocate((OUT / 'protocol.json'), __file__), {
        'scope': 'Frozen-model, 12 original metadata-selected profiling clips; one warmup plus three measured repeats. No training.',
        'deadline_unix': datetime.fromisoformat('2026-11-17T13:31:00+07:00').timestamp(),
        'created_unix': time.time(), 'profiler_sha256': digest(PROFILER),
        'manifest_sha256': digest(_fk_paths.relocate((REPO / 'experiments/remaining-001/profiling-manifest.json'), __file__)),
        'protected': {str(p): digest(p) for p in paths},
        'precision': 'Original method-specific TF32 setting retained; assessment/prediction and feature parity <=5e-6',
        'limitations': ['Synchronized instrumented sequential replay; original optimized uninstrumented timings retained separately.',
                        'Decode prefetch disabled only in isolated profiler; raw inputs/weights/sampling/thresholds unchanged.',
                        'RSS peak is sampled every 20ms; not an exact instantaneous high-water mark.',
                        'AVTENet expert costs measured in separate native/Linux processes; sum is not monolithic end-to-end latency.',
                        'No new optimizer step, seed, threshold selection, training or checkpoint update.']})
    write(_fk_paths.relocate((OUT / 'progress.json'), __file__), {'stage': 'prepared_waiting_for_original', 'updated_unix': time.time()})
    print(str(OUT), flush=True)


def historical_training():
    paths = [_fk_paths.relocate((REPO / 'runs/two-day-001/experiment/fk/training/attempt-001/run.json'), __file__),
             _fk_paths.relocate((REPO / 'runs/two-day-001/experiment/fk/training/attempt-001/metrics.json'), __file__),
             _fk_paths.relocate((REPO / 'runs/avfakenet-one-epoch-001/complete.json'), __file__),
             _fk_paths.relocate((REPO / 'runs/avfakenet-one-epoch-001/identity.json'), __file__)]
    avroot = _fk_paths.relocate((ROOT / 'avh-a/code'), __file__)
    plan = read(_fk_paths.relocate((avroot / 'runs/auto-next-001/test-plan.json'), __file__))
    for stage in ('audio', 'visual', 'audiovisual', 'fusion'):
        parent = (_fk_paths.relocate((avroot / plan['checkpoints'][stage]), __file__)).parent
        paths += [_fk_paths.relocate((parent / 'run.json'), __file__), _fk_paths.relocate((parent / 'metrics.json'), __file__)]
    result = {}
    for p in paths:
        if p.exists():
            result[str(p.relative_to(ROOT))] = {'sha256': digest(p), 'contents': read(p)}
    write(_fk_paths.relocate((OUT / 'historical-training.json'), __file__), {'scope': 'Descriptive retained historical logs only; unequal budgets; not a new controlled training benchmark', 'sources': result})


def report():
    import statistics
    protocol = read(_fk_paths.relocate((OUT / 'protocol.json'), __file__))
    records = {}
    summaries = {}
    for stage in ('fk', 'avfakenet', 'audio', 'visual', 'audiovisual', 'fusion'):
        records[stage] = [read(p) for p in sorted((_fk_paths.relocate((OUT / stage), __file__)).glob('[0-9][0-9].json'))] if (_fk_paths.relocate((OUT / stage), __file__)).exists() else []
        summaries[stage] = {'clips': len(records[stage]), 'complete': (_fk_paths.relocate((OUT / stage / 'complete.json'), __file__)).exists()}
    category = {'pre': 'preprocessing', 'feature': 'feature_extraction', 'evidence': 'fusion_evidence', 'aggregation': 'aggregation', 'record': 'forensic_record', 'io': 'decode_io'}
    data = {}
    for stage, rows in records.items():
        clips = []
        for row in rows:
            measured = row['samples'][1:]
            values = {}
            for group, title in category.items():
                values[title] = statistics.median(sum(v['self_seconds'] for k, v in sample['timers'].items() if k.startswith(group + '.')) for sample in measured)
            clips.append({'video_id': row['row']['video_id'], 'status': measured[0]['result']['status'],
                          'seconds': statistics.median(s['seconds'] for s in measured),
                          'seconds_per_source_second': statistics.median(s['seconds'] for s in measured) / row['row']['duration'],
                          **values, 'sampled_peak_rss_bytes': max(s['sampled_peak_rss_bytes'] for s in measured),
                          'cuda_peak_allocated_bytes': max(s['cuda_peak_allocated_bytes'] for s in measured),
                          'cuda_peak_reserved_bytes': max(s['cuda_peak_reserved_bytes'] for s in measured), 'parity': row['parity']})
        data[stage] = clips
    changed = [p for p, sha in protocol['protected'].items() if digest(p) != sha]
    if changed:
        raise RuntimeError('Protected original changed: ' + str(changed))
    write(_fk_paths.relocate((OUT / 'results.json'), __file__), {'protocol': protocol, 'stages': summaries, 'per_clip': data,
                               'protected_originals_unchanged': True, 'training': 'historical logs only; no new training'})
    def med(rows, key):
        return 'N/A' if not rows else '%.4f' % statistics.median(r[key] for r in rows)
    lines = ['# Table III: measured inference cost supplement', '',
             'Frozen checkpoints; original 12 metadata-selected clips; 1 warmup + 3 timed repeats. No training, threshold selection or prediction replacement.', '',
             '## Execution coverage', '', '| Stage | Completed clips | Complete marker |', '|---|---:|---|']
    for stage, state in summaries.items():
        lines.append('| %s | %d/12 | %s |' % (stage, state['clips'], state['complete']))
    lines += ['', '## Median instrumented self time by Table III column (seconds per clip)', '',
              'Medians across all attempted clips in each row; failed/unassessed attempts are retained. Columns use nested self time to avoid double counting, but the median of column medians need not sum to the median total.', '',
              '| Stage | Assessed / attempted | Preprocessing | Feature | Fusion / evidence | Aggregation | Forensic | Decode / I/O | Measured total |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for stage, rows in data.items():
        if not rows:
            continue
        forensic = med(rows, 'forensic_record') if stage == 'fk' else 'N/A'
        lines.append('| %s | %d/%d | %s | %s | %s | %s | %s | %s | %s |' %
                     (stage, sum(r['status'] == 'ASSESSED' for r in rows), len(rows), med(rows, 'preprocessing'),
                      med(rows, 'feature_extraction'), med(rows, 'fusion_evidence'), med(rows, 'aggregation'),
                      forensic, med(rows, 'decode_io'), med(rows, 'seconds')))
    model_rows = {'FK': data['fk'], 'AVFakeNet adaptation': data['avfakenet'], 'AVTENet fusion head only': data['fusion']}
    available = [set(r['video_id'] for r in rows if r['status'] == 'ASSESSED') for rows in model_rows.values()]
    common = set.intersection(*available) if all(available) else set()
    lines += ['', '## Memory and common assessed sample', '',
              'Common assessed IDs across FK, AVFakeNet and AVTENet final fusion: %d. AVTENet fusion-head timing is not total three-expert pipeline timing.' % len(common), '',
              '| Stage | Maximum sampled RSS (MiB) | Maximum allocated VRAM (MiB) | Maximum reserved VRAM (MiB) |', '|---|---:|---:|---:|']
    for stage, rows in data.items():
        if rows:
            lines.append('| %s | %.1f | %.1f | %.1f |' % (stage, max(r['sampled_peak_rss_bytes'] for r in rows) / 2**20,
                         max(r['cuda_peak_allocated_bytes'] for r in rows) / 2**20, max(r['cuda_peak_reserved_bytes'] for r in rows) / 2**20))
    lines += ['', '## What this adds', '',
              '- FK: EfficientNet, Wav2Vec2, GRU, bin pooling/projections and evidence submodules; aggregation; full source-to-persisted-record replay, source/derived hashes, serialization, commitment and verification.',
              '- AVFakeNet: face detector, shared geometry, face/lip crops, Mel preparation and separate DST branches; voting/OR helper remains one combined aggregation timer.',
              '- AVTENet: per-expert detector/crops and input transform, AST/ViViT/AV-HuBERT, separate MS-TCN with pooling, fusion head and clip-level mean aggregation.',
              '- Actual call counts and self/inclusive times are retained per repeat in stage JSON. Input/feature/decision parity must pass before a clip result is published.', '',
              '## Remaining boundaries and limits', '',
              '- Source-file hashing inside FK preparation is actually repeated per window; record.file_hash reports all measured calls. This is not automatically the analytical 2-source-hash expression.',
              '- Shared face/lip rotation/geometry is timed separately; the cropped lip rectangle is separately timed. Shared tracking and preparation residuals are not claimed as pure detector or pure quality-estimation cost.',
              '- Functional attention/tensor operations and input validation remain combined in the FK evidence residual; named projection, discrepancy and reliability modules are separately timed.',
              '- Timed calls synchronize CUDA and Python crop tracing adds overhead. Use these for component accounting; keep earlier uninstrumented optimized profiling as a separate result.',
              '- AVTENet uses separate Windows/Linux expert runtimes. Do not add their RSS/VRAM peaks into a claimed simultaneous peak, or call their timing sum observed monolithic latency.',
              '- RSS sampled every 20ms is an observed sampled maximum, not an exact instantaneous peak. Model loading excluded; OS cache uncontrolled; no cold-cache claim.',
              '- Baseline-native localization and forensic records are not provided (N/A), not unfinished zero-time tasks.',
              '- Models remain local adaptations; unequal original training budgets and low AVFakeNet coverage still limit scientific comparison.', '',
              '## Training', '',
              'Only historical run/metrics/provenance are retained in historical-training.json. FK used five head-training epochs, AVTENet expert/fusion stages one epoch, AVFakeNet one recovery pilot epoch plus documented tiny diagnostics. These are actual past budgets, not matched training-cost estimates.', '',
              'A controlled online-vs-cached training benchmark is NOT executed. It requires a separately frozen training protocol and new measured training steps/runs. The analytical factor 3 is an approximation, not an empirically verified constant.', '',
              '## Integrity', '', 'Protected original reports, predictions, population and bound runners match their pre-study hashes. No original checkpoint, prediction or active runner was edited.', '']
    (_fk_paths.relocate((OUT / 'REPORT.md'), __file__)).write_text('\n'.join(lines), encoding='utf-8')
    if all(v['complete'] for v in summaries.values()):
        write(_fk_paths.relocate((OUT / 'complete.json'), __file__), {'results_sha256': digest(_fk_paths.relocate((OUT / 'results.json'), __file__)), 'report_sha256': digest(_fk_paths.relocate((OUT / 'REPORT.md'), __file__)), 'no_training': True})
    print('Report: ' + str(_fk_paths.relocate((OUT / 'REPORT.md'), __file__)), flush=True)


def run(native_only=False):
    protocol = read(_fk_paths.relocate((OUT / 'protocol.json'), __file__))
    while not (_fk_paths.relocate((REPO / 'runs/remaining-supplement-002/complete.json'), __file__)).exists():
        state = read(_fk_paths.relocate((REPO / 'runs/remaining-supplement-002/progress.json'), __file__))
        if state.get('status') in ('FAILED', 'PAUSED'):
            raise RuntimeError('Original workload needs attention: ' + str(state))
        if time.time() > protocol['deadline_unix'] - 600:
            raise RuntimeError('Original workload not complete within measurement window')
        print('Waiting for original: %s %s/%s' % (state.get('stage'), state.get('processed'), state.get('total')), flush=True)
        time.sleep(10)
    sys.path[:0] = [str(_fk_paths.relocate((REPO / 'src'), __file__))]
    from deepfake_detection.gpu_queue import gpu_slot
    from deepfake_detection.quality_fitting import exclusive_run
    python = str(_fk_paths.relocate((REPO / '.venv-gpu/Scripts/python.exe'), __file__))
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    avroot = _fk_paths.relocate((ROOT / 'avh-a/code'), __file__)
    with exclusive_run(_fk_paths.relocate((OUT / 'worker.lock'), __file__)), gpu_slot():
        historical_training()
        for stage in ('fk', 'avfakenet', 'audio', 'visual', 'audiovisual', 'fusion'):
            if native_only and stage == 'audiovisual':
                break
            if (_fk_paths.relocate((OUT / stage / 'complete.json'), __file__)).exists():
                continue
            if time.time() > protocol['deadline_unix'] - 120:
                break
            if stage == 'audiovisual':
                command = ['C:/Program Files/Docker/Docker/resources/bin/docker.exe', '-H', 'npipe:////./pipe/dockerDesktopLinuxEngine',
                           'run', '--rm', '--gpus', 'all', '--shm-size', '1g', '-e', 'OMP_NUM_THREADS=2', '-e', 'OPENBLAS_NUM_THREADS=2',
                           '--mount', 'type=bind,source=%s,target=/workspace,readonly' % avroot,
                           '--mount', 'type=bind,source=%s,target=/inputs,readonly' % ROOT,
                           '--mount', 'type=bind,source=%s,target=/study' % OUT,
                           '--mount', 'type=bind,source=%s,target=/profiler.py,readonly' % PROFILER,
                           '--mount', 'type=bind,source=%s,target=/detector_compat.py,readonly' % PROFILER.with_name('table_iii_detector_compat.py'),
                           '--entrypoint', 'python', 'fk-avhubert:cuda117-v1', '/detector_compat.py', '--profiler-source', '/profiler.py',
                           '--workspace', '/inputs', '--code-root', '/workspace', '--output', '/study', '--stage', stage,
                           '--assets', '/workspace/runs/worker-c/assets.docker.local.json']
            else:
                code = REPO if stage in ('fk', 'avfakenet') else avroot
                command = [python, '-u', str(PROFILER), '--workspace', str(ROOT), '--code-root', str(code), '--output', str(OUT), '--stage', stage]
                if stage == 'visual':
                    command = [python, '-u', str(PROFILER.with_name('table_iii_detector_compat.py')), '--profiler-source', str(PROFILER)] + command[3:]
                if stage in ('audio', 'visual'):
                    command += ['--assets', str(_fk_paths.relocate((avroot / 'runs/worker-c/assets.local.json'), __file__))]
            print('START ' + stage, flush=True)
            with (_fk_paths.relocate((OUT / (stage + '.log')), __file__)).open('a', encoding='utf-8') as log:
                process = subprocess.Popen(command, cwd=str(REPO), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, encoding='utf-8', errors='replace', creationflags=subprocess.CREATE_NO_WINDOW)
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line.rstrip(), flush=True)
                if process.wait():
                    raise RuntimeError('Profiling stage failed: ' + stage)
            report()
        report()
        write(_fk_paths.relocate((OUT / 'progress.json'), __file__), {'stage': 'complete' if (_fk_paths.relocate((OUT / 'complete.json'), __file__)).exists() else ('awaiting_linux_stage' if native_only else 'deadline_partial'), 'updated_unix': time.time()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--native-only', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.report_only:
        historical_training()
        report()
    else:
        try:
            run(native_only=args.native_only)
        except Exception as error:
            write(_fk_paths.relocate((OUT / 'progress.json'), __file__), {'stage': 'FAILED', 'error': str(error), 'updated_unix': time.time()})
            report()
            raise
