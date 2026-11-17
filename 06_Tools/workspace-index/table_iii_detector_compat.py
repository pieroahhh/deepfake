"""Keep the original stateless detector type while timing its unchanged call."""

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
import functools
import importlib.util
import sys
from pathlib import Path

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--profiler-source', required=True)
options, remaining = parser.parse_known_args()
source = _fk_paths.relocate((Path(options.profiler_source)), __file__)
spec = importlib.util.spec_from_file_location('frozen_table_iii_profiler', source)
profiler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profiler)


def original_detector_type(detector, timer):
    detector_type = type(detector)
    original_call = detector_type.__call__
    @functools.wraps(original_call)
    def timed_call(self, rgb):
        with timer.phase('pre.face_detector', True):
            return original_call(self, rgb)
    detector_type.__call__ = timed_call
    return detector


profiler.TimedDetector = original_detector_type
args = argparse.ArgumentParser(add_help=False)
args.add_argument('--output', required=True)
args.add_argument('--stage', required=True)
settings, _ = args.parse_known_args(remaining)
dest = _fk_paths.relocate((Path(settings.output) / settings.stage), __file__)
dest.mkdir(exist_ok=True)
profiler.write(_fk_paths.relocate((dest / 'execution-overlay.json'), __file__), {
    'wrapper_sha256': profiler.digest(_fk_paths.relocate((Path(__file__)), __file__)), 'original_profiler_sha256': profiler.digest(source),
    'scope': 'Instrument original MTCNNDetector.__call__ in place instead of changing its type. Original detector, caching, mathematical output and 5e-6 parity gate retained.',
    'reason': 'Prepared permits detection reuse only for the original stateless MTCNNDetector type; no completed clip from this stage existed before applying the overlay.',
    'no_training': True})
sys.argv = [str(source)] + remaining
profiler.main()
