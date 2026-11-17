
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

import json
from pathlib import Path
import tempfile
import unittest

from deepfake_detection.lavdf import normalize_lavdf, prepare_lavdf_pilot


def family(split):
    original = f"{split}/original.mp4"
    return [
        {"file": f"{split}/{name}.mp4", "split": split, "duration": 3.0,
         "original": original if audio or visual else None,
         "modify_audio": audio, "modify_video": visual,
         "n_fakes": int(audio or visual), "fake_periods": [[1.2, 1.5]] if audio or visual else []}
        for name, audio, visual in [("original", False, False), ("audio", True, False),
                                    ("visual", False, True), ("both", True, True)]
    ]


class LavdfTests(unittest.TestCase):
    def test_pilot_keeps_official_splits_and_modality_labels(self):
        metadata = family("train") + family("dev") + family("test")
        metadata.append({**metadata[1], "file": "train/invalid.mp4", "fake_periods": [[2.0, 3.1]]})
        with tempfile.TemporaryDirectory() as tmp:
            root = _fk_paths.relocate((Path(tmp)), __file__)
            (_fk_paths.relocate((root / "metadata.min.json"), __file__)).write_text(json.dumps(metadata), encoding="utf-8")
            # Held-out media is absent, so the command can only read train and dev clips.
            for item in metadata[:8]:
                clip = _fk_paths.relocate((root / item["file"]), __file__)
                clip.parent.mkdir(exist_ok=True)
                clip.write_text(item["file"], encoding="utf-8")
            output = _fk_paths.relocate((root / "pilot"), __file__)
            report = prepare_lavdf_pilot(root, output, train_families=1, validation_families=1)
            manifest = json.loads((_fk_paths.relocate((output / "manifest.json"), __file__)).read_text(encoding="utf-8"))

        self.assertEqual(report["pilot_audit"]["counts"], {"train": 4, "validation": 4})
        self.assertEqual(report["pilot_audit"]["missing_source_hashes"], 0)
        self.assertEqual([row["path"] for row in report["excluded_annotations"]], ["train/invalid.mp4"])
        by_path = {row["path"]: row for row in manifest}
        audio = by_path["train/audio.mp4"]
        self.assertEqual((audio["label"], audio["audio_label"], audio["visual_label"]), (1, 1, 0))
        self.assertEqual(audio["source_family"], "train/original.mp4")
        self.assertEqual(audio["audio_manipulated_intervals"], [[1.2, 1.5]])
        self.assertEqual(audio["visual_manipulated_intervals"], [])
        both = by_path["dev/both.mp4"]
        self.assertEqual((both["audio_label"], both["visual_label"]), (1, 1))
        self.assertEqual(both["manipulated_intervals"], [[1.2, 1.5]])
        self.assertIsNone(both["audio_manipulated_intervals"])
        self.assertIsNone(both["visual_manipulated_intervals"])
        self.assertEqual(both["split"], "validation")

    def test_cross_split_original_is_rejected(self):
        metadata = family("train") + family("dev")
        metadata[1]["original"] = "dev/original.mp4"
        with self.assertRaisesRegex(ValueError, "genuine clip in the same split"):
            normalize_lavdf(metadata, "fixture")
