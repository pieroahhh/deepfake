
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

import contextlib
import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

SCRIPT = _fk_paths.relocate((Path(__file__).resolve().parents[1] / 'scripts' / 'prepare_heldout.py'), __file__)
spec = importlib.util.spec_from_file_location('prepare_heldout', SCRIPT)
heldout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(heldout)


class HeldoutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = _fk_paths.relocate((Path(self.temp.name)), __file__)
        self.media = _fk_paths.relocate((self.root / 'media'), __file__)
        self.media.mkdir()
        self.source = _fk_paths.relocate((self.root / 'source' / 'manifest.json'), __file__)
        self.used = _fk_paths.relocate((self.root / 'used' / 'manifest.json'), __file__)
        self.output = _fk_paths.relocate((self.root / 'output'), __file__)
        self.used_rows = [self.row('used-train', 'real', 'train'), self.row('used-val', 'real')]
        self.rows = self.used_rows + [self.row(f'{kind}-{i}', kind) for kind in heldout.KINDS for i in range(3)]
        self.save_inputs()

    def row(self, name, kind, split='validation'):
        payload = name.encode()
        path = 'val/' + name + '.mp4'
        file = self.media / path
        file.parent.mkdir(exist_ok=True)
        file.write_bytes(payload)
        visual = int(kind in ('visual_modified', 'both_modified'))
        audio = int(kind in ('audio_modified', 'both_modified'))
        return dict(video_id=name, source_family=name, source_sha256=hashlib.sha256(payload).hexdigest(),
                    dataset='AV-Deepfake1M', release='2024', split=split, path=path, duration=2,
                    source_modify_type=kind, label=int(visual or audio), visual_label=visual,
                    audio_label=audio, temporal_complete=True,
                    manipulated_intervals=[[0, 1]] if visual or audio else [], source_overlap_policy='keep-official')

    def save_inputs(self):
        heldout.write(self.source, self.rows, replace=True)
        heldout.write(_fk_paths.relocate((self.source.parent / 'audit.json'), __file__), {'manifest_sha256': heldout.digest(self.source)}, replace=True)
        heldout.write(self.used, self.used_rows, replace=True)
        heldout.write(_fk_paths.relocate((self.used.parent / 'audit.json'), __file__), {
            'manifest_sha256': heldout.digest(self.used), 'source_manifest_sha256': heldout.digest(self.source)
        }, replace=True)

    def prepare(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return heldout.prepare(self.source, self.used, self.media, self.output, per_type=1)

    def test_round_trip_and_preserved_annotations(self):
        report = self.prepare()
        self.assertEqual(report['selected_counts'], dict.fromkeys(heldout.KINDS, 1))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(report, heldout.verify(self.output, self.media))
        original = {r['video_id']: r for r in self.rows}
        for row in heldout.read(_fk_paths.relocate((self.output / 'manifest.json'), __file__)):
            self.assertEqual(row, dict(original[row['video_id']], split='test', official_split='validation'))
        self.assertEqual(self.used.read_bytes(), (_fk_paths.relocate((self.output / 'used-manifest.json'), __file__)).read_bytes())

    def test_order_independent_selection_and_overlap_exclusion(self):
        for key in ('video_id', 'source_family', 'source_sha256'):
            row = self.row('overlap-' + key, 'real')
            row[key] = self.used_rows[0][key]
            self.rows.append(row)
        duplicate = self.row('duplicate-family', 'real')
        duplicate['source_family'] = self.rows[2]['source_family']
        self.rows.append(duplicate)
        duplicate = self.row('duplicate-content', 'real')
        duplicate['source_sha256'] = self.rows[2]['source_sha256']
        self.rows.append(duplicate)
        self.save_inputs()
        first, counts, _ = heldout.select_rows(self.source, self.used_rows, 1, 17)
        self.rows.reverse()
        self.save_inputs()
        second, _, _ = heldout.select_rows(self.source, self.used_rows, 1, 17)
        self.assertEqual(first, second)
        self.assertEqual(counts['excluded_overlap'], 4)  # Three additions plus original validation row.
        heldout.check_separation(first, self.used_rows)

    def test_insufficient_candidates(self):
        with self.assertRaisesRegex(ValueError, 'insufficient'):
            heldout.select_rows(self.source, self.used_rows, 4, 17)

    def test_bad_source_audit_and_existing_output(self):
        self.source.write_text('[]', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.prepare()
        self.assertFalse(self.output.exists())
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_missing_or_corrupted_media_never_publishes_output(self):
        chosen, _, _ = heldout.select_rows(self.source, self.used_rows, 1, 17)
        path = self.media / chosen[0]['path']
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.prepare()
        self.assertFalse(self.output.exists())
        path.write_bytes(b'corrupted')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_verification_detects_modified_metadata_and_used_manifest(self):
        self.prepare()
        used_copy = _fk_paths.relocate((self.output / 'used-manifest.json'), __file__)
        used_copy.write_text('[]', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'used manifest checksum'):
            heldout.verify(self.output, self.media)
        used_copy.write_bytes(self.used.read_bytes())
        manifest = _fk_paths.relocate((self.output / 'manifest.json'), __file__)
        manifest.write_text('[]', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            heldout.verify(self.output, self.media)

    def test_rejects_escaping_media_paths(self):
        row = dict(self.rows[2], path='../outside.mp4')
        with self.assertRaises(ValueError):
            heldout.verify_media([row], self.media)


if __name__ == '__main__':
    unittest.main()
