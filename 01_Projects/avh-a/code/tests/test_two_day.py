"""Shared subset reproducibility and lossless prepared-input cache behavior."""

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

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from baselines.common.io import write,digest,read
from baselines.common.subset import quotas,make_subset


class TwoDayTests(unittest.TestCase):
    def test_quota_and_deterministic_stratified_selection(self):
        self.assertEqual(sum(quotas({'real':40,'fake':60},17).values()),17)
        with tempfile.TemporaryDirectory() as folder:
            root=_fk_paths.relocate((Path(folder)), __file__);rows=[]
            for split in ('train','validation'):
                for index in range(20):
                    rows.append({'video_id':split+str(index),'path':split+str(index)+'.mp4','split':split,
                        'dataset':'AV-Deepfake1M','release':'fixture','duration':3.,'source_family':split+str(index),
                        'source_sha256':f'{len(rows):064x}','source_overlap_policy':'keep-official',
                        'source_modify_type':'real' if index<8 else 'audio_modified',
                        'label':int(index>=8),'audio_label':int(index>=8),'visual_label':0,
                        'temporal_complete':True,'manipulated_intervals':[[1.,2.]] if index>=8 else []})
            write(_fk_paths.relocate((root/'manifest.json'), __file__),rows);write(_fk_paths.relocate((root/'audit.json'), __file__),{'manifest_sha256':digest(_fk_paths.relocate((root/'manifest.json'), __file__))})
            a=make_subset(_fk_paths.relocate((root/'manifest.json'), __file__),_fk_paths.relocate((root/'one'), __file__),train=10,validation=5)
            make_subset(_fk_paths.relocate((root/'manifest.json'), __file__),_fk_paths.relocate((root/'two'), __file__),train=10,validation=5)
            self.assertEqual((_fk_paths.relocate((root/'one/manifest.json'), __file__)).read_bytes(),(_fk_paths.relocate((root/'two/manifest.json'), __file__)).read_bytes())
            self.assertEqual(a['selected_counts']['train'],{'real':4,'audio_modified':6})
            self.assertEqual(a['study_audit']['counts'],{'train':10,'validation':5})

    def test_prepared_cache_reuses_both_modalities_and_detects_corruption(self):
        from baselines.avfakenet.prepared_cache import PreparedCache
        class Population:
            def path(self,row):return _fk_paths.relocate((Path('fixture.mp4')), __file__)
        with tempfile.TemporaryDirectory() as folder:
            root=_fk_paths.relocate((Path(folder)), __file__);row={'video_id':'a','duration':2.,'source_sha256':'a'*64}
            sample={'faces':np.full((2,128,128,3),31,dtype=np.uint8),'source_times':np.array([0.,.04])}
            with patch('baselines.avfakenet.prepared_cache.faces',return_value=sample) as face, \
                 patch('baselines.avfakenet.prepared_cache.audio',return_value=np.ones(32000)) as audio, \
                 patch('baselines.avfakenet.prepared_cache.prepare_audio',return_value=torch.ones(3,175,175)):
                cache=PreparedCache(root,{'identity':'fixed'})
                first,reasons=cache.get(Population(),row,None,{})
                second,_=cache.get(Population(),row,None,{})
                np.testing.assert_array_equal(first['faces'],second['faces'])
                self.assertFalse(reasons);self.assertEqual(face.call_count,1);self.assertEqual(audio.call_count,1)
            with self.assertRaisesRegex(ValueError,'identity mismatch'):PreparedCache(root,{'identity':'changed'})
            payload=next(root.glob('*.npz'));payload.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'checksum'):cache.get(Population(),row,None,{})


if __name__=='__main__':unittest.main()
