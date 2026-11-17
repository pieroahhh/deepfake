"""AVFakeNet full-video audio and temporally supervised face-frame training."""

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

import torch

from baselines.common.io import fingerprint, write
from baselines.common.media import faces, audio, event
from baselines.common.population import target
from baselines.common.training import fit, load_checkpoint, seed_all
from baselines.common.metrics import metrics
from .model import DSTNet, load_profile, classification_loss
from .preprocessing import prepare_face, prepare_audio


PROFILE = _fk_paths.relocate((Path(__file__).parent/'configs/avfakenet.reconstruction-v1.json'), __file__)
PREPARATION = {'version': 1, 'frames_per_second': 25, 'audio': 'whole_source_interval_16khz',
               'visual_targets': 'source_timestamp_half_open_intervals',
               'missing_frames': 'whole_video_unassessed', 'augmentation': 'none_reconstruction_default'}


class Loader:
    def __init__(self, population, profile, detect, modality, failure_log=None, prepared_cache=None):
        self.population, self.profile, self.detect, self.modality = population, profile, detect, modality
        self.failure_log = failure_log
        self.prepared_cache = prepared_cache

    def __call__(self, split):
        size = self.profile['training']['batch_size']
        images, labels = [], []
        for row in self.population.rows(split):
            path = self.population.path(row)  # Integrity failures are fatal, never data exclusions.
            cached=None
            if self.prepared_cache:
                cached,reasons=self.prepared_cache.get(self.population,row,self.detect,self.profile)
                if self.modality in reasons:
                    event(self.failure_log,row,reasons[self.modality]);continue
            try:
                if self.modality == 'audio':
                    label = target(row, 'audio')
                    if label is None:
                        event(self.failure_log, row, 'unknown_audio_target')
                        continue
                    mel=torch.from_numpy(cached['mel']) if cached is not None else prepare_audio(audio(path,0,row['duration']),16000,self.profile)
                    values = [(mel, label)]
                else:
                    if row['visual_label'] is None:
                        event(self.failure_log, row, 'unknown_visual_target')
                        continue
                    sample = cached if cached is not None else faces(path,0,row['duration'],self.detect)
                    values = []
                    for rgb, timestamp in zip(sample['faces'], sample['source_times']):
                        label = target(row,'visual',float(timestamp))
                        if label is not None:
                            values.append((prepare_face(rgb,self.profile),label))
                    if not values:
                        event(self.failure_log,row,'unknown_visual_temporal_targets')
            except (ValueError, OSError) as error:
                event(self.failure_log,row,error)
                continue
            for image, label in values:
                images.append(image); labels.append(label)
                if len(images) == size:
                    yield torch.stack(images), torch.tensor(labels,dtype=torch.long)
                    images, labels = [], []
        if images:
            yield torch.stack(images), torch.tensor(labels,dtype=torch.long)


def identity(population, profile, detector_provenance, stage, seed):
    return {'model':'avfakenet','stage':stage,'profile':profile,'preparation':PREPARATION,
            'index_sha256':population.metadata['index_sha256'], 'population_sha256':population.metadata['population_sha256'],
            'source_overlap_policy':population.metadata['source_overlap_policy'],
            'population_audit':population.metadata['population_audit'], 'detector':detector_provenance,'seed':seed}


def train(population, output, detect, detector_provenance, *, stage, profile_path=PROFILE,
          device='cpu', seed=7, epochs=None, resume=None, max_steps=None, prepared_cache=None):
    profile = load_profile(profile_path)
    if stage not in ('visual','audio'):
        raise ValueError('AVFakeNet stage must be visual or audio')
    seed_all(seed)
    model = DSTNet(profile,stage)
    settings = profile['training']
    schedule = {**settings,'selection':'accuracy','patience':settings['early_stopping_patience']}
    cache=None
    if prepared_cache:
        from .prepared_cache import PreparedCache
        cache=PreparedCache(prepared_cache,identity(population,profile,detector_provenance,'prepared-inputs',seed))
    loader = Loader(population,profile,detect,stage,_fk_paths.relocate((Path(output)/'preparation-events.jsonl'), __file__),cache)
    return fit(model,loader,output,identity=identity(population,profile,detector_provenance,stage,seed),
               schedule=schedule,loss_fn=lambda logits,labels:classification_loss(logits,labels,settings['label_smoothing']),
               fake_predictions=lambda logits:logits.argmax(-1),device=device,epochs=epochs,resume=resume,max_steps=max_steps)


def load_branch(path, stage, population, profile, detector_provenance, device):
    checkpoint = load_checkpoint(path)
    recorded = checkpoint['identity']
    expected = identity(population,profile,detector_provenance,stage,recorded['seed'])
    if any(recorded.get(key) != value for key,value in expected.items()):
        raise ValueError('branch checkpoint population/config/detector/stage mismatch')
    model = DSTNet(profile,stage).to(device)
    model.load_state_dict(checkpoint['model_state'],strict=True)
    return model.eval(), checkpoint


@torch.inference_mode()
def evaluate(population, output, detect, detector_provenance, *, visual_checkpoint,audio_checkpoint,
             profile_path=PROFILE,split='validation',device='cpu',compare_with=None):
    from deepfake_detection.baselines import avfakenet_decision
    from baselines.common.metrics import compare,read_predictions
    from baselines.common.io import digest
    profile = load_profile(profile_path)
    visual, vc = load_branch(visual_checkpoint,'visual',population,profile,detector_provenance,device)
    acoustic, ac = load_branch(audio_checkpoint,'audio',population,profile,detector_provenance,device)
    output=_fk_paths.relocate((Path(output)), __file__);output.mkdir(parents=True,exist_ok=False)
    results=[]
    with (_fk_paths.relocate((output/'predictions.jsonl'), __file__)).open('x',encoding='utf-8') as stream:
        for row in population.rows(split):
            path=population.path(row)
            prediction={'video_id':row['video_id'],'split':split,'label':row['label'],
                        'status':'UNASSESSED','prediction':None,'score':None}
            try:
                sample=faces(path,0,row['duration'],detect)
                mel=prepare_audio(audio(path,0,row['duration']),16000,profile)
            except (ValueError,OSError) as error:
                prediction['reason']=str(error)
            else:
                votes=[]
                for start in range(0,len(sample['faces']),profile['training']['batch_size']):
                    pixels=torch.stack([prepare_face(rgb,profile) for rgb in sample['faces'][start:start+profile['training']['batch_size']]])
                    votes.extend(visual(pixels.to(device)).argmax(-1).cpu().tolist())
                audio_vote=int(acoustic(mel.unsqueeze(0).to(device)).argmax(-1).item())
                prediction.update(avfakenet_decision(votes,audio_vote,tie_break=profile['decision']['tie_break']))
                if prediction['prediction'] is None:
                    prediction['reason']='visual_majority_tie'
            stream.write(json.dumps(prediction,allow_nan=False)+'\n');stream.flush();results.append(prediction)
            if len(results)%100==0:print('evaluated',len(results),flush=True)
    report={'model':'AVFakeNet-paper-guided-reconstruction','split':split,'metrics':metrics(results),
            'population':population.metadata,'preparation':PREPARATION,
            'checkpoints':{'visual':digest(visual_checkpoint),'audio':digest(audio_checkpoint)},
            'integration_only':vc['identity']['max_steps'] is not None or ac['identity']['max_steps'] is not None}
    if compare_with:
        report['comparison']=compare(results,read_predictions(compare_with))
    write(_fk_paths.relocate((output/'report.json'), __file__),report)
    return report
