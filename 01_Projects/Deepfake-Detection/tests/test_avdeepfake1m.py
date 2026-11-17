
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
from unittest.mock import patch

from deepfake_detection.avdeepfake1m import index_metadata, normalize_clip, prepare_avdeepfake1m


def family(split):
    original=f'{split}/real.mp4'
    rows=[]
    for kind in ('real','audio_modified','visual_modified','both_modified'):
        a=[[.2,.4]] if kind in ('audio_modified','both_modified') else []
        v=[[1.,1.3]] if kind in ('visual_modified','both_modified') else []
        rows.append({'file':original if kind=='real' else f'{split}/{kind}.mp4',
                     'original':f'voxceleb/{split}/source.mp4' if kind=='real' else original,
                     'split':split,'modify_type':kind,'video_frames':75,'audio_frames':48000,
                     'fake_segments':a+v,'audio_fake_segments':a,'visual_fake_segments':v})
    return rows


class AVDeepfake1MTests(unittest.TestCase):
    def test_modality_specific_intervals_and_val_mapping(self):
        items=next(iter(index_metadata(family('val')).values()))
        row=normalize_clip(next(r for r in items if r['modify_type']=='both_modified'),3.,'fixture')
        self.assertEqual(row['dataset'],'AV-Deepfake1M')
        self.assertEqual(row['split'],'validation')
        self.assertEqual(row['audio_manipulated_intervals'],[[.2,.4]])
        self.assertEqual(row['visual_manipulated_intervals'],[[1.,1.3]])
        self.assertEqual(len(row['manipulated_intervals']),2)
        self.assertEqual({r['source_family'] for r in items},{row['source_family']})

    def test_source_alias_and_split_relative_files(self):
        data=family('train')
        for row in data:row['file']=row['file'].removeprefix('train/')
        data[1]['original']=data[0]['original']
        groups=index_metadata(data)
        self.assertEqual(len(groups),1)
        self.assertTrue(all(r['path'].startswith('train/') for r in next(iter(groups.values()))))

    def test_unknown_modality_intervals_are_not_invented(self):
        item=next(r for r in next(iter(index_metadata(family('train')).values())) if r['modify_type']=='both_modified')
        item.pop('audio_fake_segments');item.pop('visual_fake_segments')
        row=normalize_clip(item,3.,'fixture')
        self.assertIsNone(row['audio_manipulated_intervals'])
        self.assertIsNone(row['visual_manipulated_intervals'])

    def test_official_repeated_slash_source_ids_resolve_to_one_family(self):
        data=family('train')
        data[0]['original']='VoxCeleb2//dev/mp4/person/source.mp4'
        data[1]['original']='VoxCeleb2/dev/mp4/person/source.mp4'
        data[2]['original']='train//real.mp4'
        groups=index_metadata(data)
        self.assertEqual(set(groups),{('train','voxceleb2:VoxCeleb2/dev/mp4/person/source.mp4')})
        self.assertEqual(len(next(iter(groups.values()))),4)
        self.assertEqual(data[0]['original'],'VoxCeleb2//dev/mp4/person/source.mp4')

    def test_source_normalization_preserves_path_safety_and_leak_checks(self):
        for invalid in ('VoxCeleb2//../escape.mp4','VoxCeleb2//./source.mp4',
                        '//server/share/source.mp4','/absolute.mp4',
                        'C://source.mp4','VoxCeleb2\\source.mp4'):
            data=family('train');data[0]['original']=invalid
            with self.subTest(original=invalid),self.assertRaises(ValueError):index_metadata(data)
        data=family('train');data[0]['file']='train//real.mp4'
        with self.assertRaises(ValueError):index_metadata(data)
        data=family('train')+family('val')
        data[0]['original']='VoxCeleb2//dev/source.mp4'
        data[4]['original']='VoxCeleb2/dev/source.mp4'
        with self.assertRaisesRegex(ValueError,'source-family leakage'):index_metadata(data)

    def test_fake_only_family_without_distributed_real_media(self):
        data=family('train')[1:]
        for row in data:
            row['original']='id00012/video_id/00002/real.mp4'
        groups=index_metadata(data)
        expected='voxceleb2:VoxCeleb2/dev/mp4/id00012/video_id/00002.mp4'
        self.assertEqual(set(groups),{('train',expected)})
        self.assertEqual(len(next(iter(groups.values()))),3)
        self.assertTrue(all(r['modify_type']!='real' for r in next(iter(groups.values()))))
        genuine=family('val')[0]
        genuine['original']='VoxCeleb2//dev/mp4/id00012/video_id/00002.mp4'
        with self.assertRaisesRegex(ValueError,'source-family leakage'):
            index_metadata(data+[genuine])
        data[0]['original']='unknown/missing.mp4'
        with self.assertRaisesRegex(ValueError,'missing original'):
            index_metadata(data)

    def test_missing_or_contradictory_labels_and_bounds_fail(self):
        item=next(r for r in next(iter(index_metadata(family('train')).values())) if r['modify_type']=='audio_modified')
        for change in ({'fake_segments':None},{'fake_segments':[]},
                       {'fake_segments':[[1.,.5]]},{'visual_fake_segments':[[.5,.7]]}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                normalize_clip({**item,**change},3.,'fixture')

    def test_annotations_are_intersected_with_observed_media_and_audited(self):
        item=next(r for r in next(iter(index_metadata(family('train')).values())) if r['modify_type']=='audio_modified')
        for end in (3.02,3.334):
            item.update(fake_segments=[[2.8,end]],audio_fake_segments=[[2.8,end]])
            row=normalize_clip(item,3.,'fixture')
            self.assertEqual(row['manipulated_intervals'],[[2.8,3.]])
            self.assertEqual(row['audio_manipulated_intervals'],[[2.8,3.]])
            self.assertEqual(row['annotation_bounds_adjustments']['fake_segments']['original'],[[2.8,end]])
            self.assertTrue(row['temporal_complete'])
        item.update(fake_segments=[[3.1,3.4]],audio_fake_segments=[[3.1,3.4]])
        row=normalize_clip(item,3.,'fixture')
        self.assertEqual(row['label'],1)
        self.assertFalse(row['temporal_complete'])
        self.assertEqual(row['manipulated_intervals'],[])
        self.assertIsNone(row['audio_manipulated_intervals'])

    def test_conflicting_modality_annotations_are_masked_and_preserved(self):
        item=next(r for r in next(iter(index_metadata(family('train')).values())) if r['modify_type']=='audio_modified')
        item['visual_fake_segments']=[[.2,.4]]
        row=normalize_clip(item,3.,'fixture')
        self.assertEqual(row['label'],1)
        self.assertEqual(row['audio_label'],1)
        self.assertIsNone(row['visual_label'])
        self.assertIsNone(row['visual_manipulated_intervals'])
        self.assertEqual(row['modality_annotation_conflicts'],{'visual_fake_segments':[[.2,.4]]})

    def test_family_leakage_and_invalid_paths_fail(self):
        data=family('train')+family('val')
        data[4]['original']=data[0]['original']
        with self.assertRaisesRegex(ValueError,'source-family leakage'):index_metadata(data)
        for invalid in ('../escape.mp4','/absolute.mp4','C:/absolute.mp4','val/wrong.mp4'):
            data=family('train');data[0]['file']=invalid
            with self.subTest(path=invalid),self.assertRaises(ValueError):index_metadata(data)

    def test_drop_train_preserves_validation_and_records_whole_family(self):
        rows=family('train')+family('val')
        rows[4]['original']=rows[0]['original']
        excluded=[]
        groups=index_metadata(rows,source_overlap_policy='drop-train',exclusions=excluded)
        self.assertEqual({split for split,_ in groups},{'validation'})
        self.assertEqual(len(next(iter(groups.values()))),4)
        self.assertEqual(excluded[0]['video_ids'],[r['file'] for r in rows[:4]])
        self.assertEqual(excluded[0]['retained_splits'],['validation'])
        rows=family('val')+family('test')
        rows[4]['original']=rows[0]['original']
        with self.assertRaisesRegex(ValueError,'source-family leakage'):
            index_metadata(rows,source_overlap_policy='drop-train')

    def test_hidden_test_inventory_stays_outside_training_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=_fk_paths.relocate((Path(tmp)), __file__)
            for split in ('train','val'):
                data=family(split)
                (_fk_paths.relocate((root/f'{split}_metadata.json'), __file__)).write_text(json.dumps(data),encoding='utf-8')
                for row in data:
                    path=_fk_paths.relocate((root/row['file']), __file__);path.parent.mkdir(exist_ok=True);path.write_bytes(row['file'].encode())
            (_fk_paths.relocate((root/'test_files.txt'), __file__)).write_text('000001.mp4\ntest/000002.mp4\n',encoding='utf-8')
            # No test media exists. Preparation must not inspect or label it.
            with patch('deepfake_detection.avdeepfake1m.media_duration',return_value=3.) as duration:
                result=prepare_avdeepfake1m(root,_fk_paths.relocate((root/'study'), __file__),clip_limits={'train':4,'validation':4})
            self.assertEqual(duration.call_count,8)
            self.assertEqual(result['study_audit']['counts'],{'train':4,'validation':4})
            self.assertEqual(result['test_labels'],'UNAVAILABLE_NOT_USED')
            manifest=json.loads((_fk_paths.relocate((root/'study/manifest.json'), __file__)).read_text())
            self.assertNotIn('test',{r['split'] for r in manifest})
            inventory=json.loads((_fk_paths.relocate((root/'study/test-inference-files.json'), __file__)).read_text())
            self.assertTrue(all(r['label'] is None for r in inventory))
            with patch('deepfake_detection.avdeepfake1m.media_duration',return_value=3.):
                second=prepare_avdeepfake1m(root,_fk_paths.relocate((root/'repeat'), __file__),clip_limits={'train':4,'validation':4})
            self.assertEqual(result['manifest_sha256'],second['manifest_sha256'])

    def test_labeled_test_requires_explicit_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=_fk_paths.relocate((Path(tmp)), __file__)
            for split in ('train','val','test'):
                data=family(split)
                (_fk_paths.relocate((root/f'{split}_metadata.json'), __file__)).write_text(json.dumps(data),encoding='utf-8')
                for row in data:
                    path=_fk_paths.relocate((root/row['file']), __file__);path.parent.mkdir(exist_ok=True);path.write_bytes(row['file'].encode())
            with patch('deepfake_detection.avdeepfake1m.media_duration',return_value=3.):
                result=prepare_avdeepfake1m(root,_fk_paths.relocate((root/'out'), __file__),clip_limits={'train':None,'validation':None,'test':None},
                                           test_metadata=_fk_paths.relocate((root/'test_metadata.json'), __file__))
            self.assertEqual(result['study_audit']['counts']['test'],4)
            self.assertEqual(result['test_labels'],'SUPPLIED')

    def test_keep_official_preserves_all_rows_and_reports_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=_fk_paths.relocate((Path(tmp)), __file__)
            for split in ('train','val'):
                data=family(split)
                data[0]['original']='VoxCeleb2/dev/shared.mp4'
                (_fk_paths.relocate((root/f'{split}_metadata.json'), __file__)).write_text(json.dumps(data))
                for row in data:
                    path=_fk_paths.relocate((root/row['file']), __file__);path.parent.mkdir(exist_ok=True);path.write_bytes(b'same fixture bytes')
            with patch('deepfake_detection.avdeepfake1m.media_duration',return_value=3.):
                report=prepare_avdeepfake1m(root,_fk_paths.relocate((root/'all'), __file__),clip_limits={'train':None,'validation':None},
                                           source_overlap_policy='keep-official')
            self.assertEqual(report['study_audit']['counts'],{'train':4,'validation':4})
            self.assertEqual(report['excluded_train_clips'],0)
            self.assertEqual(report['overlapping_source_families'],1)
            self.assertEqual(report['study_audit']['cross_split_content_hashes'],1)
            self.assertEqual(json.loads((_fk_paths.relocate((root/'all/source-overlap-exclusions.json'), __file__)).read_text()),[])
            self.assertEqual(len(json.loads((_fk_paths.relocate((root/'all/source-overlaps.json'), __file__)).read_text())),1)


if __name__=='__main__':unittest.main()
