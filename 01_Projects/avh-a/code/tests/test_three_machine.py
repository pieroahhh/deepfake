"""Role isolation, portable expert assets and complete-model result transfer."""

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

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = _fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, _fk_paths.relocate((ROOT / 'scripts' / (name + '.py')), __file__))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ThreeMachineTests(unittest.TestCase):
    def fixture(self, root):
        from baselines.common.io import write, digest
        source = _fk_paths.relocate((root / 'source'), __file__)
        for name in ('src', 'baselines', 'scripts', 'configs', 'tests', 'docs'):
            (_fk_paths.relocate((source / name), __file__)).mkdir(parents=True)
        for name in ('pyproject.toml', 'README.md', 'HANDOFF.md', '.gitignore'):
            (_fk_paths.relocate((source / name), __file__)).write_text('fixture')
        for name in ('START_HERE_SECOND_MACHINE.md', 'START_HERE_THIRD_MACHINE.md'):
            (_fk_paths.relocate((source / 'docs' / name), __file__)).write_text('Read ' + name)
        (_fk_paths.relocate((source / 'configs/secrets.local.json'), __file__)).write_text('DO_NOT_COPY')
        media = _fk_paths.relocate((root / 'media'), __file__)
        media.mkdir()
        (_fk_paths.relocate((media / 'clip.mp4'), __file__)).write_bytes(b'media')
        population = _fk_paths.relocate((root / 'population'), __file__)
        write(_fk_paths.relocate((population / 'manifest.json'), __file__), [{'path': 'clip.mp4', 'source_sha256': digest(_fk_paths.relocate((media / 'clip.mp4'), __file__))}])
        write(_fk_paths.relocate((population / 'audit.json'), __file__), {'manifest_sha256': digest(_fk_paths.relocate((population / 'manifest.json'), __file__))})
        index = _fk_paths.relocate((root / 'index'), __file__)
        index.mkdir()
        (_fk_paths.relocate((index / 'index.sqlite'), __file__)).write_bytes(b'index')
        for relative, filename in (
            ('checkpoints/vivit-joint-adaptation-002', 'model.safetensors'),
            ('checkpoints/comparators-ast-001/ast', 'model.safetensors'),
            ('checkpoints/comparators-native-001/avhubert_native', 'base_lrs3_iter5.pt'),
        ):
            directory = _fk_paths.relocate((source / relative), __file__)
            directory.mkdir(parents=True)
            (_fk_paths.relocate((directory / filename), __file__)).write_bytes(b'weights')
            (_fk_paths.relocate((directory / 'LICENSE'), __file__)).write_text('license')
            write(_fk_paths.relocate((directory / 'artifact.json'), __file__), {'files': {filename: digest(_fk_paths.relocate((directory / filename), __file__)), 'LICENSE': digest(_fk_paths.relocate((directory / 'LICENSE'), __file__))}})
            (_fk_paths.relocate((directory / '.download-cache'), __file__)).mkdir()
            (_fk_paths.relocate((directory / '.download-cache/extra.bin'), __file__)).write_bytes(b'do not transfer')
        template = _fk_paths.relocate((ROOT / 'baselines/avtenet/configs/assets.example.json'), __file__)
        destination = _fk_paths.relocate((source / 'baselines/avtenet/configs/assets.example.json'), __file__)
        destination.parent.mkdir(parents=True)
        destination.write_bytes(template.read_bytes())
        return source, media, population, index

    def test_role_b_omits_models_and_role_c_loads_relocated_experts(self):
        from baselines.common.io import read
        exporter = script('export_second_machine')
        verifier = script('verify_second_machine')
        configure = script('configure_worker_c')
        with tempfile.TemporaryDirectory() as folder:
            root = _fk_paths.relocate((Path(folder)), __file__)
            source, media, population, index = self.fixture(root)
            image = _fk_paths.relocate((root / 'cuda.tar'), __file__)
            image.write_bytes(b'cuda container')
            with patch.object(exporter, 'ROOT', source):
                exporter.export(_fk_paths.relocate((population / 'manifest.json'), __file__), media, index, _fk_paths.relocate((root / 'B'), __file__), role='B', include_container=True)
                exporter.export(_fk_paths.relocate((population / 'manifest.json'), __file__), media, index, _fk_paths.relocate((root / 'C'), __file__), role='C', include_container=True, container_archive=image)
            self.assertFalse((_fk_paths.relocate((root / 'B/code/checkpoints'), __file__)).exists())
            self.assertFalse((_fk_paths.relocate((root / 'B/avhubert-cuda117-v1.tar'), __file__)).exists())
            self.assertTrue((_fk_paths.relocate((root / 'B/START_HERE_SECOND_MACHINE.md'), __file__)).is_file())
            self.assertTrue((_fk_paths.relocate((root / 'C/START_HERE_THIRD_MACHINE.md'), __file__)).is_file())
            self.assertFalse(list((_fk_paths.relocate((root / 'C'), __file__)).rglob('.download-cache')))
            verifier.verify(_fk_paths.relocate((root / 'B'), __file__), 'B')
            verifier.verify(_fk_paths.relocate((root / 'C'), __file__), 'C')
            with self.assertRaisesRegex(ValueError, 'role mismatch'):
                verifier.verify(_fk_paths.relocate((root / 'B'), __file__), 'C')
            host, docker = configure.configure(_fk_paths.relocate((root / 'C/code'), __file__))
            for stage in ('audio', 'visual'):
                self.assertTrue(_fk_paths.relocate((Path(host['experts'][stage]['directory'])), __file__).is_dir())
                self.assertTrue(docker['experts'][stage]['directory'].startswith('/workspace/checkpoints/'))
            self.assertEqual(docker['experts']['audiovisual']['repository'], '/opt/av_hubert')
            self.assertNotIn('secrets.local.json', str(read(_fk_paths.relocate((root / 'C/bundle.json'), __file__))['files']))
            checkpoint = _fk_paths.relocate((Path(host['experts']['visual']['directory']) / 'model.safetensors'), __file__)
            checkpoint.write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                configure.configure(_fk_paths.relocate((root / 'C/code'), __file__))

    def worker_result(self, root, role, population_hash):
        from baselines.common.io import write
        name = 'avfakenet' if role == 'B' else 'avtenet'
        workflow = _fk_paths.relocate((root / name), __file__)
        model = _fk_paths.relocate((workflow / 'avfakenet'), __file__) if role == 'B' else workflow
        write(_fk_paths.relocate((workflow / 'status.json'), __file__), {'status': 'FINISHED'})
        write(_fk_paths.relocate((workflow / 'workflow.json'), __file__), {'manifest_sha256': population_hash, 'integration_max_steps': None})
        write(_fk_paths.relocate((model / 'evaluation/report.json'), __file__), {'complete': True})
        (_fk_paths.relocate((model / 'evaluation/predictions.jsonl'), __file__)).write_text('{"video_id":"fixture"}\n')
        stages = ('visual', 'audio') if role == 'B' else ('audio', 'visual', 'audiovisual', 'fusion')
        for stage in stages:
            for attempt in ('attempt-001', 'attempt-002'):
                directory = model / stage / attempt
                directory.mkdir(parents=True)
                (_fk_paths.relocate((directory / 'best.pt'), __file__)).write_bytes((stage + attempt).encode())
                (_fk_paths.relocate((directory / 'last.pt'), __file__)).write_bytes(b'optimizer state is not returned')
                write(_fk_paths.relocate((directory / 'run.json'), __file__), {'stage': stage})
                write(_fk_paths.relocate((directory / 'metrics.json'), __file__), {'epoch': 1})
        (_fk_paths.relocate((model / 'prepared-inputs'), __file__)).mkdir()
        (_fk_paths.relocate((model / 'prepared-inputs/cache.bin'), __file__)).write_bytes(b'not returned')
        return workflow, model

    def test_separate_returns_import_roles_population_and_latest_checkpoint(self):
        from baselines.common.io import digest
        transfer = script('transfer_worker_results')
        with tempfile.TemporaryDirectory() as folder:
            root = _fk_paths.relocate((Path(folder)), __file__)
            manifest = _fk_paths.relocate((root / 'manifest.json'), __file__)
            manifest.write_text('[]')
            for role in ('B', 'C'):
                worker = _fk_paths.relocate((root / ('worker-' + role)), __file__)
                self.worker_result(worker, role, digest(manifest))
                bundle = _fk_paths.relocate((root / ('return-' + role)), __file__)
                result = transfer.export(worker, bundle, role=role)
                names = set(result['files'])
                model = 'avfakenet' if role == 'B' else 'avtenet'
                self.assertIn(model + '/evaluation/predictions.jsonl', names)
                self.assertFalse(any('last.pt' in p or 'prepared-inputs' in p or 'attempt-001' in p for p in names))
                self.assertTrue(any('attempt-002/best.pt' in p for p in names))
                with self.assertRaisesRegex(ValueError, 'role mismatch'):
                    transfer.receive(bundle, _fk_paths.relocate((root / 'wrong'), __file__), expected_role='C' if role == 'B' else 'B', manifest=manifest)
                self.assertFalse((_fk_paths.relocate((root / 'wrong'), __file__)).exists())
                imported = _fk_paths.relocate((root / ('incoming-' + role)), __file__)
                transfer.receive(bundle, imported, expected_role=role, manifest=manifest)
                transfer.receive(bundle, imported, expected_role=role, manifest=manifest)
                payload = _fk_paths.relocate((imported / model / 'evaluation/predictions.jsonl'), __file__)
                payload.write_text('corrupted')
                with self.assertRaisesRegex(ValueError, 'imported checksum'):
                    transfer.receive(bundle, imported, expected_role=role, manifest=manifest)
            manifest.write_text('[1]')
            with self.assertRaisesRegex(ValueError, 'population mismatch'):
                transfer.receive(_fk_paths.relocate((root / 'return-C'), __file__), _fk_paths.relocate((root / 'changed'), __file__), expected_role='C', manifest=manifest)
            self.assertFalse((_fk_paths.relocate((root / 'changed'), __file__)).exists())

    def test_export_refuses_incomplete_and_smoke_runs_before_creating_return(self):
        from baselines.common.io import write
        transfer = script('transfer_worker_results')
        with tempfile.TemporaryDirectory() as folder:
            root = _fk_paths.relocate((Path(folder)), __file__)
            workflow, model = self.worker_result(_fk_paths.relocate((root / 'worker'), __file__), 'C', 'a' * 64)
            (_fk_paths.relocate((model / 'fusion/attempt-002/best.pt'), __file__)).unlink()
            with self.assertRaisesRegex(ValueError, 'checkpoint'):
                transfer.export(_fk_paths.relocate((root / 'worker'), __file__), _fk_paths.relocate((root / 'missing'), __file__), role='C')
            self.assertFalse((_fk_paths.relocate((root / 'missing'), __file__)).exists())
            write(_fk_paths.relocate((workflow / 'workflow.json'), __file__), {'manifest_sha256': 'a' * 64, 'integration_max_steps': 1}, replace=True)
            with self.assertRaisesRegex(ValueError, 'smoke'):
                transfer.export(_fk_paths.relocate((root / 'worker'), __file__), _fk_paths.relocate((root / 'smoke'), __file__), role='C')
            self.assertFalse((_fk_paths.relocate((root / 'smoke'), __file__)).exists())


if __name__ == '__main__':
    unittest.main()
