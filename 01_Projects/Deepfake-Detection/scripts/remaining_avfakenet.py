"""Frozen AVFakeNet raw-media adapter; no fitting or eligibility changes."""

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

import time,json
import numpy as np
import torch
from baselines.common.io import read,digest,fingerprint
from baselines.common.stream_identity import stream_identity
from baselines.common.training import load_checkpoint
from run_avfakenet_one_epoch import make_model
from baselines.common.media import detector,faces,audio
from baselines.common.video_reader import VideoWindowReader
from baselines.avfakenet.preprocessing import prepare_audio
from deepfake_detection.baselines import avfakenet_decision


class FrozenAVFakeNet:
    def __init__(self,root):
        training=_fk_paths.relocate((root/'runs/avfakenet-one-epoch-001'), __file__);identity=read(_fk_paths.relocate((training/'identity.json'), __file__));done=read(_fk_paths.relocate((training/'complete.json'), __file__))
        self.sha=done['identity']
        # Reuse only identical clean robustness references, never timed profiles.
        heldout=_fk_paths.relocate((root/'runs/avfakenet-heldout-002'), __file__)
        retained=read(_fk_paths.relocate((heldout/'complete.json'), __file__))
        if any(digest(heldout/name)!=sha for name,sha in retained['files'].items()):raise ValueError('Retained clean AVFakeNet result changed')
        if read(_fk_paths.relocate((heldout/'identity.json'), __file__))['training_identity_sha256']!=self.sha:raise ValueError('Clean model identity differs')
        self.clean={r['video_id']:r for r in (json.loads(line) for line in (_fk_paths.relocate((heldout/'predictions.jsonl'), __file__)).read_text().splitlines() if line.strip())}
        self.original={r['video_id']:r for r in read(_fk_paths.relocate((root.parent/'Heldout-1000/population/manifest.json'), __file__))}
        self.dataset=_fk_paths.relocate((root.parent/'Heldout-1000/dataset'), __file__);self.prepared=_fk_paths.relocate((heldout/'prepared'), __file__);self.stream_cache={}
        self.profile=identity['profile'];self.models={}
        self.detect,provenance=detector('cuda')
        if provenance!=identity['detector']:raise ValueError('AVFakeNet detector changed')
        for branch in ('visual','audio'):
            path=_fk_paths.relocate((training/branch/'best.pt'), __file__)
            if digest(path)!=done['files'][branch+'/best.pt']:raise ValueError('AVFakeNet checkpoint changed')
            state=load_checkpoint(path)
            self.models[branch]=make_model(self.profile,branch,state['model_state'],'cuda').eval().requires_grad_(False)

    def retained_input(self,path,row,branch):
        """Return a verified unchanged-modality input/error, or None to recompute."""
        condition=row.get('condition','')
        eligible=(branch=='visual' and condition.startswith(('audio_compression-','av_offset-'))
            or branch=='audio' and condition.startswith('video_compression-'))
        if not eligible:return None
        original=self.original[row['video_id']]
        if (row.get('original_source_sha256')!=original['source_sha256']
                or row['duration']!=original['duration'] or row['label']!=original['label']):raise ValueError('Derivative source binding differs')
        kind='video' if branch=='visual' else 'audio';key=(row['video_id'],kind)
        if key not in self.stream_cache:
            source=self.dataset/original['path']
            if digest(source)!=original['source_sha256']:raise ValueError('Original AVFakeNet source changed')
            self.stream_cache[key]=stream_identity(source,kind)
        identity=stream_identity(path,kind)
        if identity is None or identity!=self.stream_cache[key]:return None
        folder=self.prepared/fingerprint(original);header=_fk_paths.relocate((folder/'record.json'), __file__)
        if not header.exists():return None
        record=read(header)
        if record['video_id']!=original['video_id'] or record['source_sha256']!=original['source_sha256']:raise ValueError('Retained preparation source differs')
        if branch in record['reasons']:return {'reason':record['reasons'][branch],'stream_identity':identity}
        name=branch+'.npy'
        if name not in record['files'] or digest(folder/name)!=record['files'][name]:raise ValueError('Retained preparation array changed')
        array=np.load(_fk_paths.relocate_value((folder/name), __file__),allow_pickle=False)
        return {'array':array,'stream_identity':identity,'array_sha256':record['files'][name]}

    @torch.inference_mode()
    def predict(self,path,row):
        if row.get('condition')=='clean':
            original=self.original[row['video_id']]
            if row['source_sha256']!=original['source_sha256'] or row['label']!=original['label'] or digest(path)!=original['source_sha256']:raise ValueError('Clean AVFakeNet source differs')
            return {**self.clean[row['video_id']],'execution':'reused_verified_clean_prediction','timings':None}
        result={'video_id':row['video_id'],'label':row['label'],'status':'UNASSESSED','prediction':None,'score':None}
        reasons={};reused={};start=time.perf_counter()
        kept=self.retained_input(path,row,'visual')
        if kept is not None:
            reused['visual']={k:v for k,v in kept.items() if k!='array'}
            if 'reason' in kept:reasons['visual']=kept['reason']
            else:visual=kept['array']
        else:
            reader=VideoWindowReader()
            try:
                try:visual=faces(path,0,row['duration'],self.detect,sampler=reader.sample,prefetch_decode=True)['faces']
                except (ValueError,OSError) as error:reasons['visual']=str(error)
            finally:reader.close()
        kept=self.retained_input(path,row,'audio')
        if kept is not None:
            reused['audio']={k:v for k,v in kept.items() if k!='array'}
            if 'reason' in kept:reasons['audio']=kept['reason']
            else:mel=torch.from_numpy(kept['array'])
        else:
            try:mel=prepare_audio(audio(path,0,row['duration']),16000,self.profile)
            except (ValueError,OSError) as error:reasons['audio']=str(error)
        torch.cuda.synchronize();prepare_seconds=time.perf_counter()-start;start=time.perf_counter()
        if reasons:result['reasons']=reasons
        else:
            votes=[]
            for i in range(0,len(visual),8):
                x=torch.from_numpy(np.array(visual[i:i+8],copy=True)).permute(0,3,1,2).contiguous().float().cuda()/255
                logits=self.models['visual'](x)
                if not torch.isfinite(logits).all():raise ValueError('Invalid visual logits')
                votes.extend(logits.argmax(-1).cpu().tolist())
            logits=self.models['audio'](mel.unsqueeze(0).cuda())
            if not torch.isfinite(logits).all():raise ValueError('Invalid audio logits')
            result.update(avfakenet_decision(votes,int(logits.argmax(-1)),tie_break=self.profile['decision']['tie_break']))
            result.update(audio_probability=float(logits.softmax(-1)[0,1]),visual_fake_vote_fraction=sum(votes)/len(votes))
        torch.cuda.synchronize()
        result['timings']={'preprocessing_seconds':prepare_seconds,'model_seconds':time.perf_counter()-start}
        if reused:result['reused_identical_modality_inputs']=reused
        return result
