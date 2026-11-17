"""Resume immutable, time-audited robustness derivatives; no GPU or training."""

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

import os,sys,time,json,subprocess,shutil,argparse
from pathlib import Path
ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]
os.environ['PATH']=str(_fk_paths.relocate((ROOT/'.cache/ffmpeg-tools/static_ffmpeg/bin/win32'), __file__))+os.pathsep+os.environ.get('PATH','')
sys.path[:0]=[str(ROOT),str(_fk_paths.relocate((ROOT/'src'), __file__)),str(_fk_paths.relocate((ROOT/'scripts'), __file__))]
from baselines.common.io import read,digest,fingerprint
from avfakenet_progress_io import write_json
RUN=_fk_paths.relocate((ROOT/'runs/remaining-derivatives-001'), __file__)
DATA=_fk_paths.relocate((ROOT.parent/'Heldout-1000/dataset'), __file__)


def timeline(path):
    import av
    with av.open(str(path)) as container:
        origin=0 if container.start_time is None else container.start_time/av.time_base
        pts=[float(frame.pts*frame.time_base)-origin for frame in container.decode(video=0)]
    if not pts:raise ValueError('Empty video timeline')
    return pts


def validate_timeline(source,derived,tolerance=.002):
    if len(source)!=len(derived):raise ValueError('Frame count changed')
    error=max(abs(a-b) for a,b in zip(source,derived))
    if error>tolerance:raise ValueError(f'Source timeline shifted: {error} seconds')
    return {'frames':len(source),'max_timestamp_error_seconds':error,'tolerance_seconds':tolerance}


def build(row,kind,strength):
    from deepfake_detection.media import media_tool,probe
    name=f'{kind}-{strength}';key=fingerprint(row);folder=_fk_paths.relocate((RUN/name/key), __file__)
    folder.mkdir(parents=True,exist_ok=True);header=_fk_paths.relocate((folder/'transformation.json'), __file__)
    source=_fk_paths.relocate((DATA/row['path']), __file__);destination=_fk_paths.relocate((folder/'derived.mp4'), __file__)
    if digest(source)!=row['source_sha256']:raise ValueError('Source changed')
    if header.exists():
        record=read(header)
        if digest(destination)!=record['derived_sha256'] or record['source_sha256']!=row['source_sha256']:raise ValueError('Derivative changed')
        return record['row']
    if shutil.disk_usage(RUN).free<20*1024**3:raise RuntimeError('Less than 20 GiB free; completed files retained')
    # MP4 edit lists retain AAC priming instead of moving the video timeline as
    # some Matroska muxes do. The decoded video PTS check below remains mandatory.
    partial=_fk_paths.relocate((folder/'derived.partial.mp4'), __file__)
    args=[str(media_tool('ffmpeg')),'-v','error','-y','-copyts','-start_at_zero','-i',str(source),'-map','0:v:0','-map','0:a:0']
    if kind=='video_compression':args+=['-c:v','libx264','-crf',str(strength),'-c:a','copy']
    elif kind=='audio_compression':args+=['-c:v','copy','-c:a','aac','-b:a',f'{strength}k']
    elif kind=='av_offset':args+=['-c:v','copy','-af',f'adelay={round(strength*1000)}:all=1','-c:a','aac','-b:a','128k','-t',str(row['duration'])]
    else:raise ValueError('Unknown derivative')
    args+=['-fps_mode','passthrough','-threads','1','-avoid_negative_ts','disabled',str(partial)]
    subprocess.run(args,check=True,capture_output=True,timeout=300,
                   creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    audit=validate_timeline(timeline(source),timeline(partial))
    info=probe(partial)
    duration=float(info['format']['duration'])
    if abs(duration-row['duration'])>.1:raise ValueError('Unexpected duration change')
    if digest(source)!=row['source_sha256']:raise ValueError('Source changed during transformation')
    partial.replace(destination)
    result={**row,'path':destination.relative_to(RUN).as_posix(),'source_sha256':digest(destination),
        'original_source_sha256':row['source_sha256'],'condition':name,'original_video_id':row['video_id']}
    # Keep source-aligned annotations/duration; never extend the assessment unit
    # to encoder padding or label a genuine delayed clip as a manipulation.
    write_json(header,{'kind':kind,'strength':strength,'source_sha256':row['source_sha256'],
        'derived_sha256':result['source_sha256'],'arguments':args,'timeline_audit':audit,
        'actual_container_duration':duration,'label_policy':'original source timeline; delayed real clips are false-alarm stress only',
        'row':result})
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');args=p.parse_args()
    RUN.mkdir(exist_ok=True)
    from deepfake_detection.quality_fitting import exclusive_run
    with exclusive_run(_fk_paths.relocate((RUN/'worker.lock'), __file__)):
        package=_fk_paths.relocate((ROOT/'experiments/remaining-001'), __file__)
        for name,sha in read(_fk_paths.relocate((package/'SHA256SUMS.json'), __file__)).items():
            if digest(_fk_paths.relocate((package/name), __file__))!=sha:raise ValueError('Sealed study plan changed')
        rows=read(_fk_paths.relocate((package/'robustness-manifest.json'), __file__));plan=read(_fk_paths.relocate((package/'robustness-plan.json'), __file__))
        jobs=[(r,c['kind'],c['strength']) for c in plan['conditions'] if c['kind']!='clean' for r in rows]
        jobs += [(r,'av_offset',strength) for strength in plan['separate_sync_stress']['audio_delay_seconds'] for r in rows if r['label']==0]
        if args.smoke:jobs=[jobs[0],jobs[200],jobs[400]]
        write_json(_fk_paths.relocate((RUN/'identity.json'), __file__),{'manifest_sha256':digest(_fk_paths.relocate((package/'robustness-manifest.json'), __file__)),
            'plan_sha256':digest(_fk_paths.relocate((package/'robustness-plan.json'), __file__)),'script_sha256':digest(_fk_paths.relocate((Path(__file__)), __file__)),
            'protocol':'time-audited MP4 compression and real-only audio-delay stress; exploratory'})
        finished=[];start=time.monotonic()
        try:
            for i,(row,kind,strength) in enumerate(jobs):
                if (_fk_paths.relocate((RUN/'pause.request'), __file__)).exists():raise RuntimeError('PAUSED between derivatives')
                write_json(_fk_paths.relocate((RUN/'progress.json'), __file__),{'status':'RUNNING','stage':'transcode_and_timeline_audit','processed':i,'total':len(jobs),'condition':f'{kind}-{strength}','updated_unix':time.time()})
                finished.append(build(row,kind,strength))
            if not args.smoke:
                clean=[]
                for row in rows:
                    # A relative path into the separately retained original dataset.
                    clean.append({**row,'condition':'clean','original_video_id':row['video_id'],'absolute_source':str(_fk_paths.relocate((DATA/row['path']), __file__))})
                write_json(_fk_paths.relocate((RUN/'manifest.json'), __file__),clean+finished)
                write_json(_fk_paths.relocate((RUN/'complete.json'), __file__),{'derivatives':len(finished),'clean':len(clean),'manifest_sha256':digest(_fk_paths.relocate((RUN/'manifest.json'), __file__)),'source_manifest_sha256':digest(_fk_paths.relocate((package/'robustness-manifest.json'), __file__))})
            write_json(_fk_paths.relocate((RUN/'progress.json'), __file__),{'status':'FINISHED','stage':'smoke_complete' if args.smoke else 'complete','processed':len(finished),'total':len(jobs),'seconds':time.monotonic()-start,'updated_unix':time.time()})
        except Exception as error:
            write_json(_fk_paths.relocate((RUN/'progress.json'), __file__),{'status':'FAILED','stage':'transform_error','processed':len(finished),'total':len(jobs),'error':str(error),'updated_unix':time.time()});raise


if __name__=='__main__':main()
