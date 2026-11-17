
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

import importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(_fk_paths.relocate((ROOT/'src'), __file__)),str(_fk_paths.relocate((ROOT/'scripts'), __file__))]
import run_avtenet_frozen_test as worker
import auto_after_avhubert as controller
from baselines.common.io import write,read,digest
from baselines.avtenet.model import FeatureFusion,load_component_profile


class FrozenTest(unittest.TestCase):
    def test_frozen_inference_resume_and_unassessed_denominator(self):
        from baselines.avtenet import pipeline,assets
        from baselines.avtenet import heldout_preparation
        from baselines.common import training,population,media
        with tempfile.TemporaryDirectory() as folder:
            root=_fk_paths.relocate((Path(folder)), __file__)
            rows=[{'video_id':str(i),'source_family':'test-'+str(i),'source_sha256':str(i).zfill(64),
                   'split':'test','label':i%2,'duration':2.} for i in range(1000)]
            used=[{'video_id':'train','source_family':'train','source_sha256':'f'*64}]
            write(_fk_paths.relocate((root/'manifest.json'), __file__),rows);write(_fk_paths.relocate((root/'train.json'), __file__),used)
            profile=load_component_profile(_fk_paths.relocate((ROOT/'baselines/avtenet/configs/avtenet.hf-joint-adaptation-v1.json'), __file__))
            write(_fk_paths.relocate((root/'config.json'), __file__),profile);write(_fk_paths.relocate((root/'fusion-cache.json'), __file__),{})
            for name in ('index','train-index'):
                write(_fk_paths.relocate((root/name/'index.json'), __file__),{});(_fk_paths.relocate((root/name/'index.sqlite'), __file__)).write_bytes(b'fixture index')
            checkpoints={stage:stage+'.pt' for stage in ('audio','visual','audiovisual','fusion')}
            for path in checkpoints.values():(_fk_paths.relocate((root/path), __file__)).write_bytes(b'fixture checkpoint')
            plan={'manifest':'manifest.json','training_manifest':'train.json','config':'config.json',
                'training_fusion_cache':'fusion-cache.json','index':'index','training_index':'train-index',
                'checkpoints':checkpoints,'output':'output','files':{}}
            plan['files']={p.relative_to(root).as_posix():digest(p) for p in root.rglob('*') if p.is_file()}
            write(_fk_paths.relocate((root/'plan.json'), __file__),plan)
            class Population:
                def __init__(self,*a):self.metadata={}
            class Expert(torch.nn.Module):
                def extract_features(self,inputs):return inputs.repeat(1,768)
            fail_once=[True]
            class Prepared:
                def __init__(self,pop,stage,*a,**kw):self.stage=stage
                def records(self,split,skip_windows):
                    for row in rows[:1 if self.stage=='audiovisual' else 2]:
                        if (row['video_id'],0.,2.) not in skip_windows:
                            yield row,0.,2.,torch.ones(1,1),None
                            if self.stage=='audio' and fail_once[0]:
                                fail_once[0]=False;raise RuntimeError('simulated interruption')
            fusion=FeatureFusion(profile)
            with torch.no_grad():
                fusion.classifier.weight.zero_();fusion.classifier.bias.fill_(-1)
            def checkpoint(path):
                if _fk_paths.relocate((Path(path)), __file__).stem=='fusion':return {'identity':{'feature_cache':{},'max_steps':None},'model_state':fusion.state_dict()}
                return {'identity':{'seed':7,'max_steps':None},'model_state':{}}
            def run(stage):
                with patch.object(sys,'argv',['worker','--plan',str(_fk_paths.relocate((root/'plan.json'), __file__)),'--stage',stage,'--dataset-root',str(root),'--assets','fixture','--device','cpu']):worker.main()
            with patch.object(worker,'ROOT',root),patch.object(population,'Population',Population),\
                 patch.object(pipeline,'stage_model',return_value=(Expert(),None,{})),patch.object(heldout_preparation,'FastPrepared',Prepared),\
                 patch.object(pipeline,'stage_identity',return_value={'seed':7,'max_steps':None}),\
                 patch.object(assets,'registry',return_value={}),patch.object(media,'detector',return_value=(None,{})),\
                 patch.object(training,'load_checkpoint',side_effect=checkpoint):
                with self.assertRaisesRegex(RuntimeError,'simulated'):run('audio')
                run('audio');run('visual');run('audiovisual');run('evaluate')
                report=read(_fk_paths.relocate((root/'output/evaluate/report.json'), __file__))
                self.assertEqual(report['metrics']['videos'],1000)
                self.assertEqual(report['metrics']['assessed'],1)
                self.assertEqual(report['metrics']['unassessed'],999)
                predictions=[json.loads(line) for line in (_fk_paths.relocate((root/'output/evaluate/predictions.jsonl'), __file__)).read_text().splitlines()]
                self.assertEqual(predictions[0]['prediction'],1)
                before=digest(_fk_paths.relocate((root/'output/evaluate/report.json'), __file__));run('evaluate')
                self.assertEqual(digest(_fk_paths.relocate((root/'output/evaluate/report.json'), __file__)),before)
                (_fk_paths.relocate((root/'audio.pt'), __file__)).write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'Frozen evaluation input changed'):run('evaluate')

    def test_import_is_verified_and_never_overwrites_different_data(self):
        with tempfile.TemporaryDirectory() as folder:
            source=_fk_paths.relocate((Path(folder)/'source'), __file__);target=_fk_paths.relocate((Path(folder)/'nested/target'), __file__)
            source.write_bytes(b'good');sha=digest(source)
            controller.copy_verified(source,target,sha);controller.copy_verified(source,target,sha)
            target.write_bytes(b'different')
            with self.assertRaisesRegex(ValueError,'Destination differs'):controller.copy_verified(source,target,sha)
            self.assertEqual(target.read_bytes(),b'different')


if __name__=='__main__':unittest.main()
