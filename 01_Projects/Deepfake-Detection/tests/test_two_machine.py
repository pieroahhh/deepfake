"""Two-machine portability, integrity and matched-population report contracts."""

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

ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]


def script(name):
    spec=importlib.util.spec_from_file_location(name,_fk_paths.relocate((ROOT/'scripts'/f'{name}.py'), __file__))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


class TwoMachineTests(unittest.TestCase):
    def test_snapshot_and_subset_export_hashes_and_no_local_secrets(self):
        from baselines.common.io import write,digest
        exporter=script('export_second_machine');verifier=script('verify_second_machine')
        with tempfile.TemporaryDirectory() as folder:
            root=_fk_paths.relocate((Path(folder)), __file__);source=_fk_paths.relocate((root/'source'), __file__);source.mkdir()
            for name in ('src','baselines','scripts','configs','tests','docs'):(_fk_paths.relocate((source/name), __file__)).mkdir()
            for name in ('pyproject.toml','README.md','HANDOFF.md','.gitignore'):(_fk_paths.relocate((source/name), __file__)).write_text('fixture')
            (_fk_paths.relocate((source/'configs/secret.local.json'), __file__)).write_text('DO_NOT_EXPORT')
            media=_fk_paths.relocate((root/'media'), __file__);media.mkdir();(_fk_paths.relocate((media/'clip.mp4'), __file__)).write_bytes(b'video')
            population=_fk_paths.relocate((root/'population'), __file__);write(_fk_paths.relocate((population/'manifest.json'), __file__),[{'path':'clip.mp4','source_sha256':digest(_fk_paths.relocate((media/'clip.mp4'), __file__))}])
            write(_fk_paths.relocate((population/'audit.json'), __file__),{'manifest_sha256':digest(_fk_paths.relocate((population/'manifest.json'), __file__))})
            index=_fk_paths.relocate((root/'index'), __file__);index.mkdir();(_fk_paths.relocate((index/'index.sqlite'), __file__)).write_bytes(b'index');write(_fk_paths.relocate((index/'index.json'), __file__),{'schema':'fixture'})
            with patch.object(exporter,'ROOT',source):
                result=exporter.export(_fk_paths.relocate((population/'manifest.json'), __file__),media,index,_fk_paths.relocate((root/'bundle'), __file__),include_checkpoint=False)
            self.assertEqual(result['clips'],1)
            self.assertFalse((_fk_paths.relocate((root/'bundle/code/configs/secret.local.json'), __file__)).exists())
            verifier.verify(_fk_paths.relocate((root/'bundle'), __file__))
            (_fk_paths.relocate((root/'bundle/dataset/clip.mp4'), __file__)).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'checksum'):verifier.verify(_fk_paths.relocate((root/'bundle'), __file__))

    def test_return_import_checks_before_writing(self):
        from baselines.common.io import write,digest
        transfer=script('transfer_worker_results')
        with tempfile.TemporaryDirectory() as folder:
            root=_fk_paths.relocate((Path(folder)), __file__);bundle=_fk_paths.relocate((root/'return'), __file__);bundle.mkdir();(_fk_paths.relocate((bundle/'result.json'), __file__)).write_text('ok')
            write(_fk_paths.relocate((bundle/'return.json'), __file__),{'schema':'two-machine-results-v1','files':{'result.json':digest(_fk_paths.relocate((bundle/'result.json'), __file__))}})
            transfer.receive(bundle,_fk_paths.relocate((root/'received'), __file__))
            self.assertEqual((_fk_paths.relocate((root/'received/result.json'), __file__)).read_text(),'ok')
            (_fk_paths.relocate((bundle/'result.json'), __file__)).write_text('changed')
            with self.assertRaisesRegex(ValueError,'checksum'):transfer.receive(bundle,_fk_paths.relocate((root/'bad'), __file__))
            self.assertFalse((_fk_paths.relocate((root/'bad'), __file__)).exists())

    def test_three_way_common_population_and_missing_id_rejection(self):
        from baselines.common.io import write
        comparator=script('compare_three_models')
        with tempfile.TemporaryDirectory() as folder:
            root=_fk_paths.relocate((Path(folder)), __file__)
            write(_fk_paths.relocate((root/'manifest.json'), __file__),[{'video_id':'a','split':'validation','label':0},{'video_id':'b','split':'validation','label':1}])
            predictions=[{'video_id':'a','label':0,'prediction':0,'score':.1},{'video_id':'b','label':1,'prediction':1,'score':.9}]
            for name in ('FK','AVF','AVT'):write(_fk_paths.relocate((root/(name+'.json')), __file__),predictions)
            write(_fk_paths.relocate((root/'AVT.json'), __file__),[predictions[0],{**predictions[1],'prediction':None,'score':None}],replace=True)
            paths={name:_fk_paths.relocate((root/(name+'.json')), __file__) for name in ('FK','AVF','AVT')}
            report=comparator.compare_models(_fk_paths.relocate((root/'manifest.json'), __file__),paths)
            self.assertEqual(report['all_three_assessed']['videos'],1)
            self.assertEqual(report['full_population']['AVT']['coverage'],.5)
            write(_fk_paths.relocate((root/'AVT.json'), __file__),[predictions[0]],replace=True)
            with self.assertRaisesRegex(ValueError,'same validation'):comparator.compare_models(_fk_paths.relocate((root/'manifest.json'), __file__),paths)


if __name__=='__main__':unittest.main()
