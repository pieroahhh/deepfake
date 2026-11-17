"""One-command dataset-independent check bundle; every sample is labeled synthetic."""

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


import argparse
from pathlib import Path
import subprocess
import sys

import torch
from torch import nn

from .cli import load_json,save
from .commands import profile_synthetic_fk
from .experiments import prepare_experiments
from .infrastructure import readiness
from .reporting import write_detection_report


def synthetic_video_check(output):
    from .inference import run_fk_video
    from .model import EvidenceModel
    from .forensics import sha256_file
    from .training import model_spec
    root=_fk_paths.relocate((Path(output)), __file__);root.mkdir(parents=True,exist_ok=False)
    source=_fk_paths.relocate((root/'generated-colorbars-and-tone.mkv'), __file__)
    ffmpeg=_fk_paths.relocate((Path(sys.executable).parent/'ffmpeg'), __file__)
    subprocess.run([str(ffmpeg),'-v','error','-f','lavfi','-i','testsrc=size=96x96:rate=25:duration=3',
                    '-f','lavfi','-i','sine=frequency=440:sample_rate=16000:duration=3',
                    '-c:v','ffv1','-pix_fmt','bgr0','-c:a','pcm_s16le','-threads','1',str(source)],
                   check=True,capture_output=True,timeout=30)
    class Detector:
        method='SYNTHETIC_FIXED_GEOMETRY_NOT_REAL_FACE_DETECTION'
        def __call__(self,rgb):
            return [{'box':[10,10,86,86],'score':1.,'landmarks':[[30,30],[66,30],[48,48],[36,65],[60,65]]}]
    class Encoders(nn.Module):
        def __init__(self):
            super().__init__();self.visual=nn.Linear(3,1280);self.audio=nn.Linear(1,768)
        def encode_window(self,faces,waveform,times):
            return {'visual_features':self.visual(faces.mean((-2,-1))).unsqueeze(0),
                    'audio_features':self.audio(waveform.unfold(0,400,320).mean(-1,keepdim=True)).unsqueeze(0),
                    'visual_times':times,'audio_times':(torch.arange(99)*320+199.5)/16000}
    torch.manual_seed(7)
    model=EvidenceModel()
    encoders=Encoders()
    torch.save({'status':'SYNTHETIC_UNTRAINED_NOT_DETECTION_WEIGHTS','model_state':model.state_dict(),
                'encoder_fixture_state':encoders.state_dict(),'model_spec':model_spec(model),'seed':7},_fk_paths.relocate((root/'fixture.pt'), __file__))
    return run_fk_video(source,3.,model,encoders,Detector(),lambda audio:[(0.,3.)],
                        lambda q:torch.tensor([[q['face_coverage'],q['speech_coverage']]]),_fk_paths.relocate((root/'inference'), __file__),
                        model_version='SYNTHETIC-'+sha256_file(_fk_paths.relocate((root/'fixture.pt'), __file__)),
                        quality_profile='SYNTHETIC_AVAILABILITY_AND_SPEECH',provider_status='synthetic')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,help='new bundle directory')
    parser.add_argument('--assets',help='actual assets.local.json; defaults to the empty example registry')
    args=parser.parse_args(argv)
    root=_fk_paths.relocate((Path(args.output)), __file__);root.mkdir(parents=True,exist_ok=False)
    project=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[2]
    result=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(_fk_paths.relocate((project/'tests'), __file__)),'-v'],
                          capture_output=True,text=True,cwd=project,timeout=300)
    (_fk_paths.relocate((root/'tests.log'), __file__)).write_text(result.stdout+result.stderr)
    if result.returncode:
        raise RuntimeError('test suite failed; see '+str(_fk_paths.relocate((root/'tests.log'), __file__)))
    summary={'status':'LOCAL_SOFTWARE_CHECKS_ONLY','data_kind':'synthetic',
             'tests':'PASSED','test_log':str(_fk_paths.relocate((root/'tests.log'), __file__))}
    asset_registry=_fk_paths.relocate((Path(args.assets)), __file__).resolve() if args.assets else _fk_paths.relocate((project/'configs/assets.example.json'), __file__)
    summary['asset_registry']=str(asset_registry)
    summary['asset_registry_kind']='explicit_local_registry' if args.assets else 'empty_example_template'
    summary['readiness']=readiness(load_json(asset_registry))
    save(_fk_paths.relocate((root/'readiness.json'), __file__),summary['readiness'])
    plan=prepare_experiments(load_json(_fk_paths.relocate((project/'configs/ours.provisional.json'), __file__)),_fk_paths.relocate((root/'experiments'), __file__))
    summary['planned_unexecuted_runs']=len(plan['runs'])
    summary['synthetic_video']=synthetic_video_check(_fk_paths.relocate((root/'synthetic-video'), __file__))
    profile_synthetic_fk(_fk_paths.relocate((root/'head-only-profile.json'), __file__),iterations=10,threads=2)
    rows=[dict(video_id=str(i),source_family='fixture-family-'+str(i//2),label=label,
               status='ASSESSED',prediction=label,score=.9 if label else .1)
          for i,label in enumerate((0,0,1,1))]
    write_detection_report({'synthetic_continuous':rows,'synthetic_hard':[{**row,'score':None} for row in rows]},
                           _fk_paths.relocate((root/'example-report'), __file__),provenance={'protocol_id':'SYNTHETIC_SOFTWARE_FIXTURE',
                           'data_kind':'synthetic'},reference='synthetic_continuous',replicates=100)
    summary['external_validation']='REAL_ASSET_AND_REAL_FACE_EXECUTION_NOT_PERFORMED_BY_SYNTHETIC_BUNDLE'
    save(_fk_paths.relocate((root/'summary.json'), __file__),summary)
    print('Dataset-independent check bundle:',root)
    print('Generated-video record:',summary['synthetic_video']['record'])
    print('Asset registry checked:',asset_registry)
    print('This bundle checks synthetic mechanics and asset checksums; real runtime/face validation is separate.')


if __name__=='__main__':main()
