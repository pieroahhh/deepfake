"""Interrupted/resumed quality fitting must equal an uninterrupted exact fit."""

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

from contextlib import redirect_stderr
import copy
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from deepfake_detection.cli import save
from deepfake_detection.forensics import sha256_file
from deepfake_detection.quality import fit_profile,fit_quality_manifest
from deepfake_detection.quality_fitting import exclusive_run


class QualityResumeTests(unittest.TestCase):
    def fixture(self,root):
        rows=[]
        for index,split in enumerate(('train','train','validation')):
            path=_fk_paths.relocate((root/f'{index}.mp4'), __file__);path.write_bytes(str(index).encode())
            rows.append({'video_id':str(index),'path':path.name,'source_sha256':sha256_file(path),
                         'split':split,'duration':2.1,'dataset':'AV-Deepfake1M','release':'fixture',
                         'source_family':str(index),'label':0,'visual_label':0,'audio_label':0,
                         'temporal_complete':True,'manipulated_intervals':[]})
        save(_fk_paths.relocate((root/'manifest.json'), __file__),rows);save(_fk_paths.relocate((root/'assets.json'), __file__),{'fixture':True})
        return rows

    def run_fit(self,root,name='quality.json',resume=False,provider=None):
        with patch('deepfake_detection.commands.load_fk_assets',return_value=(None,None,None)), \
             patch('deepfake_detection.commands.fk_asset_hashes',return_value={'fixture':'a'*64}), \
             patch('deepfake_detection.preparation.fk_source_speech',return_value=[[0,1]]), \
             patch('deepfake_detection.preparation.prepare_fk_window',side_effect=provider or self.measure):
            return fit_quality_manifest(_fk_paths.relocate((root/'manifest.json'), __file__),root,_fk_paths.relocate((root/'assets.json'), __file__),_fk_paths.relocate((root/name), __file__),
                                        resume=resume,progress_seconds=.001)

    def measure(self,path,start,*args,**kwargs):
        # Include unavailable windows and an unknown SNR to test exact filtering.
        return {'status':'USABLE' if start<2 else 'UNASSESSED',
                'quality_measurements':{'mean_face_laplacian_variance':float(path.stem)*10+start,
                                        'estimated_snr_db':None if start==0 else start+4}}

    def test_resume_mid_clip_equals_original_and_only_reads_remaining(self):
        with tempfile.TemporaryDirectory() as directory:
            root=_fk_paths.relocate((Path(directory)), __file__);rows=self.fixture(root);calls=[]
            def interrupted(path,start,*args,**kwargs):
                calls.append((path.stem,start))
                if len(calls)==3:raise KeyboardInterrupt()
                return self.measure(path,start)
            with redirect_stderr(io.StringIO()),self.assertRaises(KeyboardInterrupt):
                self.run_fit(root,provider=interrupted)
            self.assertFalse((_fk_paths.relocate((root/'quality.json'), __file__)).exists())
            resumed=[]
            def remaining(path,start,*args,**kwargs):
                resumed.append((path.stem,start));return self.measure(path,start)
            output=io.StringIO()
            with redirect_stderr(output):profile=self.run_fit(root,resume=True,provider=remaining)
            self.assertEqual(resumed,[('0',2.),('1',0.),('1',1.),('1',2.)])
            expected=fit_profile([{**row,'windows':[self.measure(_fk_paths.relocate((root/row['path']), __file__),float(s)) for s in range(3)]}
                                  for row in rows if row['split']=='train'])
            self.assertEqual(profile,{**expected,'asset_sha256':{'fixture':'a'*64}})
            self.assertIn('restored 2 saved windows',output.getvalue())
            self.assertIn('ETA',output.getvalue())
            with redirect_stderr(io.StringIO()):self.assertEqual(self.run_fit(root,resume=True),profile)

    def test_refuses_changed_inputs_corruption_and_accidental_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root=_fk_paths.relocate((Path(directory)), __file__);self.fixture(root)
            with redirect_stderr(io.StringIO()):self.run_fit(root)
            with self.assertRaisesRegex(ValueError,'already exists'):self.run_fit(root)
            with (_fk_paths.relocate((root/'manifest.json'), __file__)).open('a') as stream:stream.write(' ')
            with redirect_stderr(io.StringIO()),self.assertRaisesRegex(ValueError,'mismatch'):self.run_fit(root,resume=True)
        with tempfile.TemporaryDirectory() as directory:
            root=_fk_paths.relocate((Path(directory)), __file__);self.fixture(root)
            with redirect_stderr(io.StringIO()):self.run_fit(root)
            conn=sqlite3.connect(_fk_paths.relocate((root/'quality.json.work/state.sqlite'), __file__))
            with conn:conn.execute('UPDATE windows SET blur=999')
            conn.close()
            with redirect_stderr(io.StringIO()),self.assertRaisesRegex(ValueError,'columns mismatch'):self.run_fit(root,resume=True)

    def test_source_change_fails_after_interruption_and_lock_is_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            root=_fk_paths.relocate((Path(directory)), __file__);self.fixture(root)
            with redirect_stderr(io.StringIO()),self.assertRaises(KeyboardInterrupt):
                self.run_fit(root,provider=lambda *a,**kw:(_ for _ in ()).throw(KeyboardInterrupt()))
            (_fk_paths.relocate((root/'0.mp4'), __file__)).write_bytes(b'changed')
            with redirect_stderr(io.StringIO()),self.assertRaisesRegex(ValueError,'source hash mismatch'):
                self.run_fit(root,resume=True)
            with exclusive_run(_fk_paths.relocate((root/'lock'), __file__)):
                with self.assertRaisesRegex(ValueError,'another fit-quality'):
                    with exclusive_run(_fk_paths.relocate((root/'lock'), __file__)):pass


if __name__=='__main__':unittest.main()
