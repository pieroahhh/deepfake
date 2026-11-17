"""Training-only annotation/Mel inspection. No fitting or checkpoint changes."""

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

import os,sys,json,argparse,math
from pathlib import Path
from fractions import Fraction
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='2'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(_fk_paths.relocate((ROOT/'src'), __file__))]
from baselines.common.io import read,write,digest,checked_path


def merged_intervals(row):
    if row['split']!='train' or row['audio_label']!=1:raise ValueError('Known fake-audio training clips only')
    intervals=row.get('audio_manipulated_intervals')
    if not intervals:raise ValueError('Known positive audio annotations required')
    merged=[]
    for a,b in sorted(intervals):
        if not math.isfinite(a+b) or not 0<=a<b<=row['duration']+1e-7:raise ValueError('Invalid source audio annotation')
        if merged and a<=merged[-1][1]:merged[-1][1]=max(b,merged[-1][1])
        else:merged.append([a,b])
    return merged


def column_geometry(duration,intervals,frames,pixels,hop_seconds,fft_seconds):
    centers=[];supports=[]
    for column in range(pixels):
        position=(column+.5)*frames/pixels-.5
        lo=max(0,min(frames-1,math.floor(position)));hi=max(0,min(frames-1,math.floor(position)+1))
        center=max(0,min(frames-1,position))*hop_seconds
        if any(a<=center<b for a,b in intervals):centers.append(column)
        left=max(0,lo*hop_seconds-fft_seconds/2)
        right=min(duration,hi*hop_seconds+fft_seconds/2)
        if any(max(left,a)<min(right,b) for a,b in intervals):supports.append(column)
    return {'stft_center_inside_edit_columns':centers,'stft_support_overlapping_edit_columns':supports,
            'note':'Geometry of align_corners=False bilinear resize with centered STFT; not evidence of discriminative fake artifacts'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--profile-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();output=args.output.resolve()
    if output.exists():raise ValueError('Use a new output folder')
    manifest=read(args.manifest);profile=read(args.profile_run)['identity']['profile']
    candidates=[]
    for row in manifest:
        if row['split']!='train' or row['audio_label']!=1 or not row.get('audio_manipulated_intervals'):continue
        intervals=merged_intervals(row);fraction=sum(b-a for a,b in intervals)/row['duration']
        candidates.append((fraction,row['video_id'],row,intervals))
    candidates.sort(key=lambda item:(item[0],item[1]))
    # Predetermined metadata quantiles; never choose by a model score or outcome.
    targets=[.1,.25,.45,.55,.75,.9];selected=[];families=set()
    for quantile in targets:
        index=round(quantile*(len(candidates)-1))
        nearest=sorted(range(len(candidates)),key=lambda i:(abs(i-index),i))
        for i in nearest:
            entry=candidates[i]
            if entry[2]['source_family'] not in families:
                families.add(entry[2]['source_family']);selected.append((quantile,entry));break
    output.mkdir(parents=True)
    os.environ['MPLCONFIGDIR']=str(_fk_paths.relocate((output/'.matplotlib-cache'), __file__))
    import numpy as np
    import torch
    import librosa
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from baselines.common.media import audio
    from baselines.avfakenet.preprocessing import prepare_audio
    torch.set_num_threads(2)
    settings=profile['preprocessing']
    if not settings['stft_center']:raise ValueError('Review expects the recorded centered STFT profile')
    sealed={args.manifest.resolve():digest(args.manifest),args.profile_run.resolve():digest(args.profile_run)}
    results=[]
    for number,(quantile,(_,vid,row,intervals)) in enumerate(selected,1):
        source=checked_path(args.dataset_root,row['path'])
        if digest(source)!=row['source_sha256']:raise ValueError('Source checksum differs')
        sealed[source]=row['source_sha256']
        entry={'video_id':vid,'split':'train','source_family':row['source_family'],'source_sha256':row['source_sha256'],
            'duration_seconds':row['duration'],'audio_edit_intervals':intervals,'selection_quantile':quantile,
            'edited_seconds':sum(b-a for a,b in intervals)}
        entry['edited_fraction']=entry['edited_seconds']/row['duration']
        entry['approximate_edited_width_in_175_column_input']=entry['edited_fraction']*settings['audio_pixels']
        try:
            waveform=audio(source,0,row['duration'])
            model_input=prepare_audio(waveform,16000,profile)
        except (ValueError,OSError) as error:
            entry.update(status='PREPARATION_FAILED',reason=str(error));results.append(entry);continue
        spectrum=librosa.feature.melspectrogram(y=waveform,sr=settings['audio_sample_rate'],
            n_fft=settings['n_fft'],hop_length=settings['hop_length'],n_mels=settings['n_mels'],
            win_length=settings['n_fft'],window=settings['stft_window'],center=settings['stft_center'],
            pad_mode=settings['stft_pad_mode'],power=settings['mel_power'],htk=settings['mel_htk'],
            norm=settings['mel_norm'],fmin=settings['mel_fmin'],fmax=settings['mel_fmax'])
        db=librosa.power_to_db(spectrum,ref=np.max,amin=settings['power_to_db_amin'],top_db=-settings['db_floor'])
        grayscale=np.clip((db-settings['db_floor'])/-settings['db_floor'],0,1)
        reproduced=torch.nn.functional.interpolate(torch.from_numpy(grayscale)[None,None],
            size=(settings['audio_pixels'],settings['audio_pixels']),mode='bilinear',align_corners=False)[0].expand(3,-1,-1)
        if not torch.equal(reproduced,model_input):raise ValueError('Whole-clip Mel input parity failed')
        hop=settings['hop_length']/settings['audio_sample_rate'];fft=settings['n_fft']/settings['audio_sample_rate']
        entry.update(status='REVIEWED',whole_clip_mel_input_parity=True,raw_mel_time_columns=int(spectrum.shape[1]),
            fft_seconds=fft,hop_seconds=hop,columns=column_geometry(row['duration'],intervals,spectrum.shape[1],settings['audio_pixels'],hop,fft))
        np.savez_compressed(output/f'clip-{number:02d}-model-input.npz',mel=model_input.numpy())
        fig,axes=plt.subplots(3,1,figsize=(10,7),layout='constrained')
        image=model_input[0].numpy()
        axes[0].imshow(image,aspect='auto',origin='upper',extent=(0,row['duration'],settings['n_mels'],0),cmap='gray',vmin=0,vmax=1)
        axes[0].set(title=f'Exact whole-clip model input: {100*entry["edited_fraction"]:.2f}% annotated edited audio',ylabel='Mel row (model orientation)')
        # Show annotation markers separately from the saved exact model pixels.
        for a,b in intervals:axes[0].axvspan(a,b,color='orange',alpha=.25)
        step=max(1,len(waveform)//20000);times=np.arange(0,len(waveform),step)/16000
        axes[1].plot(times,waveform[::step],lw=.5,color='#276b91')
        axes[1].set(ylabel='PCM amplitude',xlim=(0,row['duration']))
        for a,b in intervals:axes[1].axvspan(a,b,color='orange',alpha=.25)
        a,b=max(intervals,key=lambda pair:pair[1]-pair[0]);left=max(0,a-.5);right=min(row['duration'],b+.5)
        frame_times=np.arange(spectrum.shape[1])*hop
        mask=(frame_times>=left)&(frame_times<=right)
        local=grayscale[:,mask]
        axes[2].imshow(local,aspect='auto',origin='upper',extent=(left,right,settings['n_mels'],0),cmap='gray',vmin=0,vmax=1)
        axes[2].axvspan(a,b,color='orange',alpha=.25)
        axes[2].set(title='Diagnostic zoom from raw Mel; not a replacement model input',xlabel='Source time (seconds)',ylabel='Mel row')
        fig.suptitle(vid,fontsize=8)
        fig.savefig(output/f'clip-{number:02d}.png',dpi=150);plt.close(fig)
        entry['figure']=f'clip-{number:02d}.png';results.append(entry)
        print('Reviewed training audio',number,'/',len(selected),flush=True)
    write(_fk_paths.relocate((output/'report.json'), __file__),{'scope':'Annotation and model-input geometry only; no training and no authenticity judgment from image appearance',
        'selection':'Six predetermined edited-duration quantiles, distinct source families, training only',
        'full_validation_pilot':'ON_HOLD','clips':results})
    rows=['# Annotated fake-audio training input review','',
        'No audio model was trained. No validation or held-out clip was used for this review. The full validation pilot remains on hold.','',
        '| Clip | Seconds | Edited seconds | Edited % | Approx. width / 175 columns | Status |',
        '|---|---:|---:|---:|---:|---|']
    for i,r in enumerate(results,1):
        rows.append(f"| {i} | {r['duration_seconds']:.3f} | {r['edited_seconds']:.3f} | {100*r['edited_fraction']:.2f}% | {r['approximate_edited_width_in_175_column_input']:.2f} | {r['status']} |")
    rows+=['','Orange spans identify metadata annotations. They do not prove that a forgery artifact is visually distinguishable in the Mel image.',
        'The whole-clip model inputs were independently reproduced with exact tensor parity. Resizing can reduce short edits to very few columns; STFT windows and bilinear interpolation also mix surrounding context.',
        'A clean paired reference or controlled ablation is needed to establish whether any visible spectral difference is caused by manipulation.','']
    for i,r in enumerate(results,1):
        if r['status']=='REVIEWED':rows+=[f"## Clip {i}: {r['video_id']}",'',f"![Audio input and annotated interval]({r['figure']})",'']
    (_fk_paths.relocate((output/'REVIEW.md'), __file__)).write_text('\n'.join(rows),encoding='utf-8')
    for source,sha in sealed.items():
        if digest(source)!=sha:raise ValueError('Original input changed')
    write(_fk_paths.relocate((output/'source-preservation.json'), __file__),{'all_unchanged':True,'files':{str(p):sha for p,sha in sealed.items()}})
    print(json.dumps({'output':str(output),'reviewed':sum(r['status']=='REVIEWED' for r in results)}),flush=True)


if __name__=='__main__':main()
