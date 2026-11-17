
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

import tempfile
import itertools
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from baselines.common import training
from baselines.common.io import read


class Interrupted(Exception):pass


class ResumeTests(unittest.TestCase):
    def test_pause_saves_before_exiting(self):
        with tempfile.TemporaryDirectory() as folder:
            root=_fk_paths.relocate((Path(folder)), __file__);marker=_fk_paths.relocate((root/'pause.request'), __file__);marker.touch()
            model=torch.nn.Linear(1,2)
            schedule={'optimizer':'Adam','learning_rate':.01,'weight_decay':0.,'betas':[.9,.999],
                      'epsilon':1e-8,'maximum_epochs':1,'patience':5,'selection':'loss'}
            with self.assertRaises(training.TrainingPaused):
                training.fit(model,lambda split:[(torch.ones(1,1),torch.tensor([0]))],_fk_paths.relocate((root/'run'), __file__),
                    identity={'test':'pause'},schedule=schedule,loss_fn=torch.nn.functional.cross_entropy,
                    fake_predictions=lambda value:value.argmax(-1),pause_file=marker)
            state=training.load_checkpoint(_fk_paths.relocate((root/'run/last.pt'), __file__))
            self.assertEqual(state['partial']['steps'],1)
            self.assertEqual(state['epoch'],0)
            self.assertEqual(read(_fk_paths.relocate((root/'run/progress.json'), __file__))['status'],'PAUSED')

    def test_resume_train_validation_and_epoch_boundaries_are_exact(self):
        torch.set_num_threads(1)
        schedule={'optimizer':'Adam','learning_rate':.01,'weight_decay':0.,'betas':[.9,.999],
                  'epsilon':1e-8,'maximum_epochs':2,'patience':5,'selection':'loss'}
        def run(root,resume=None):
            training.seed_all(7)
            model=torch.nn.Sequential(torch.nn.Linear(4,5),torch.nn.Dropout(.3),torch.nn.Linear(5,2))
            def batches(split):
                for i in range(6 if split=='train' else 4):
                    yield torch.full((1,4),i/10.),torch.tensor([i%2])
            ticks=itertools.count()
            with patch.object(training.time,'monotonic',side_effect=lambda:next(ticks)/10):
                return training.fit(model,batches,root,identity={'test':'mid-epoch'},schedule=schedule,
                    loss_fn=torch.nn.functional.cross_entropy,fake_predictions=lambda value:value.argmax(-1),
                    epochs=2,resume=resume,checkpoint_seconds=.001)
        def identical(a,b):
            if isinstance(a,torch.Tensor):self.assertTrue(torch.equal(a,b))
            elif isinstance(a,dict):
                self.assertEqual(set(a),set(b))
                for k in a:identical(a[k],b[k])
            elif isinstance(a,(list,tuple)):
                self.assertEqual(len(a),len(b))
                for x,y in zip(a,b):identical(x,y)
            else:self.assertEqual(a,b)
        with tempfile.TemporaryDirectory() as folder:
            root=_fk_paths.relocate((Path(folder)), __file__)
            run(_fk_paths.relocate((root/'reference'), __file__))
            reference=training.load_checkpoint(_fk_paths.relocate((root/'reference/last.pt'), __file__))
            for number,(epoch,split,step) in enumerate([(1,'train',3),(1,'train',6),
                                                       (1,'validation',2),(1,'validation',4),
                                                       (2,'train',2),(2,'validation',2)]):
                before=_fk_paths.relocate((root/f'interrupted-{number}'), __file__);after=_fk_paths.relocate((root/f'resumed-{number}'), __file__)
                real_save=training.atomic_checkpoint
                def interrupt(path,state):
                    real_save(path,state)
                    partial=state.get('partial',{})
                    if (partial.get('epoch'),partial.get('split'),partial.get('steps'))==(epoch,split,step):
                        raise Interrupted()
                with patch.object(training,'atomic_checkpoint',side_effect=interrupt):
                    with self.assertRaises(Interrupted):run(before)
                run(after,_fk_paths.relocate((before/'last.pt'), __file__))
                actual=training.load_checkpoint(_fk_paths.relocate((after/'last.pt'), __file__))
                identical(reference,actual)
                self.assertEqual(read(_fk_paths.relocate((root/'reference/metrics.json'), __file__)),read(_fk_paths.relocate((after/'metrics.json'), __file__)))
                self.assertEqual(read(_fk_paths.relocate((after/'progress.json'), __file__))['status'],'FINISHED')

    def test_prepared_resume_skips_sources_and_failed_windows(self):
        import numpy as np
        from baselines.avtenet.pipeline import Prepared
        rows=[dict(video_id=str(i),source_sha256=str(i)*64,duration=4.,visual_label=0) for i in range(3)]
        visited=[]
        class Population:
            def rows(self,split):return iter(rows)
            def path(self,row):return row['video_id']
        class Backend:
            def prepare(self,value):return torch.from_numpy(value.copy()).float()
        class Fixture(Prepared):
            def raw_inputs(self,path,start):
                visited.append((path,start))
                if path=='1' and start==0:raise ValueError('fixture failure')
                return {'faces':np.full((1,4),int(path)*10+start,dtype=np.float32)}
        batches=Fixture(Population(),'visual',Backend(),{},None)
        iterator=iter(batches('train'))
        for _ in range(4):next(iterator)
        cursor=batches.cursor();iterator.close()
        self.assertEqual(cursor,{'video_id':'1','start':1.,'end':3.,'windows_processed':5})
        visited.clear()
        resumed=list(batches.iter_from('train',cursor))
        self.assertEqual(len(resumed),4)
        self.assertEqual(visited,[('1',2.),('2',0.),('2',1.),('2',2.)])
        self.assertEqual(batches.progress('train')['windows_processed'],9)
        with self.assertRaisesRegex(ValueError,'order differs'):
            list(batches.iter_from('train',{**cursor,'windows_processed':4}))
        with self.assertRaisesRegex(ValueError,'source not'):
            list(batches.iter_from('train',{**cursor,'video_id':'missing'}))


if __name__=='__main__':unittest.main()
