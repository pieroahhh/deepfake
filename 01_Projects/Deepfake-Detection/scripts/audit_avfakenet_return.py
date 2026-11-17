"""Read-only AVFakeNet return audit; no weights, thresholds or results are changed."""

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

import argparse,json,math,sys
from collections import Counter
from pathlib import Path
ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(_fk_paths.relocate((ROOT/'src'), __file__))]
from baselines.common.io import read,digest,write,checked_path
from baselines.common.metrics import metrics
from deepfake_detection.baselines import avfakenet_decision


def inspect_predictions(rows,manifest):
    known={r['video_id']:r for r in manifest}
    if len(known)!=len(manifest):raise ValueError('Duplicate manifest ID')
    if len({r['video_id'] for r in rows})!=len(rows) or {r['video_id'] for r in rows}!=set(known):
        raise ValueError('Prediction IDs differ from manifest')
    assessed=[];reasons=Counter();branches={'visual':Counter(),'audio':Counter()};inconsistent=[]
    modality={name:Counter() for name in branches}
    for row in rows:
        source=known[row['video_id']]
        if row['split']!='test' or row['label']!=source['label']:raise ValueError('Label/split mismatch')
        if row['status']!='ASSESSED':
            if row['prediction'] is not None:raise ValueError('Unassessed row has a decision')
            reasons[row.get('reason','unknown')]+=1;continue
        assessed.append(row)
        visual,audio=row.get('visual_prediction'),row.get('audio_prediction')
        if type(visual) is not int or visual not in (0,1) or type(audio) is not int or audio not in (0,1):
            raise ValueError('Missing binary branch decisions')
        expected=avfakenet_decision([visual],audio,tie_break='unassessed')['prediction']
        if row['prediction']!=expected:inconsistent.append(row['video_id'])
        for name,prediction in [('visual',visual),('audio',audio)]:
            branches[name][prediction]+=1
            label=source[name+'_label']
            if label is not None:modality[name]['tp' if label and prediction else 'fn' if label else 'fp' if prediction else 'tn']+=1
    result=metrics(rows)
    result['accuracy']=(result['confusion']['tp']+result['confusion']['tn'])/len(assessed) if assessed else None
    fake_fraction=sum(r['label']==1 for r in assessed)/len(assessed) if assessed else None
    return {'metrics':result,'assessment_failure_reasons':dict(reasons.most_common()),
        'branch_predictions':{k:dict(v) for k,v in branches.items()},
        'branch_confusion_against_modality_labels':{k:dict(v) for k,v in modality.items()},
        'fusion_inconsistent_ids':inconsistent,'continuous_scores_present':sum(r.get('score') is not None for r in assessed),
        'always_fake_accuracy_on_same_assessed_clips':fake_fraction,
        'constant_fake_baseline_matches_return':bool(assessed) and all(r['prediction']==1 for r in assessed)}


def training_metadata(rows):
    output={}
    for split in ('train','validation'):
        part=[r for r in rows if r['split']==split]
        stats={name:dict(Counter(str(r[name]) for r in part)) for name in ('label','visual_label','audio_label')}
        duration_total=0.;fake_duration=0.
        for row in part:
            if row['visual_label'] is None:continue
            if row['visual_label']==1 and row.get('visual_manipulated_intervals') is None:continue
            duration_total+=row['duration']
            for a,b in row.get('visual_manipulated_intervals') or []:
                fake_duration+=max(0,min(b,row['duration'])-max(a,0))
        output[split]={'clips':len(part),'clip_label_counts':stats,
            'metadata_visual_manipulated_time_fraction':fake_duration/duration_total if duration_total else None,
            'limitation':'Source metadata only, before media failures; this is not the actual sampled training class count'}
    return output


def visual_duration_diagnostic(manifest,assessed_ids):
    """Annotation-time diagnostic, not actual decoded-frame or model accuracy."""
    output={}
    for title,part in [('all_heldout',manifest),('assessed_heldout',[r for r in manifest if r['video_id'] in assessed_ids])]:
        fractions=[]
        for row in part:
            if row['visual_label']!=1 or row.get('visual_manipulated_intervals') is None:continue
            modified=sum(max(0,min(b,row['duration'])-max(0,a)) for a,b in row['visual_manipulated_intervals'])
            fractions.append(modified/row['duration'])
        fractions.sort();n=len(fractions)
        median=(fractions[(n-1)//2]+fractions[n//2])/2 if n else None
        output[title]={'known_visually_manipulated_clips':n,
            'below_half_manipulated_duration':sum(f<.5 for f in fractions),
            'median_manipulated_duration_fraction':median,'maximum_manipulated_duration_fraction':max(fractions) if n else None}
    output['interpretation']='Under uniform dense frame sampling and perfect local frame labels, a majority-real clip decision is expected when manipulated time is below half. Actual source PTS, rejected frames and learned logits must still be measured.'
    return output


def media_probes(rows,manifest,dataset_root):
    """Bounded CPU reproductions without a face detector or any model."""
    import numpy as np
    from deepfake_detection.media import sample_video,aligned_audio
    known={r['video_id']:r for r in manifest};result=[]
    categories=('source-frame gap exceeds declared sampling tolerance',
                'audio does not fully cover the requested source-time interval',
                'PCM must be normalized to [-1, 1]')
    for reason in categories:
        example=next((r for r in rows if r.get('reason')==reason),None)
        if example is None:continue
        definition=known[example['video_id']];path=checked_path(dataset_root,definition['path'])
        if digest(path)!=definition['source_sha256']:raise ValueError('Probe source checksum mismatch')
        entry={'video_id':definition['video_id'],'reported_reason':reason,'source_sha256_verified':True}
        try:
            if reason.startswith('source-frame'):
                count=max(1,math.ceil(definition['duration']*25-1e-9))
                for offset in range(0,count,25):
                    sample_video(path,offset/25,min(25,count-offset)/25,fps=25)
            else:
                waveform=aligned_audio(path,0,definition['duration'])['samples']
                entry['decoded_peak_absolute_pcm']=float(np.abs(waveform).max())
                if entry['decoded_peak_absolute_pcm']>1:raise ValueError('PCM must be normalized to [-1, 1]')
        except (ValueError,OSError) as error:
            entry['reproduced_error']=str(error)
        entry['exact_reported_reason_reproduced']=entry.get('reproduced_error')==reason
        result.append(entry)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--return-root',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--training-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--dataset-root',type=Path)
    args=p.parse_args();source=args.return_root.resolve();output=args.output.resolve()
    if output.exists():raise ValueError('Use a new output folder')
    if output==source or source in output.parents:raise ValueError('Output must be outside the original return')
    returned=read(_fk_paths.relocate((source/'return.json'), __file__));report=read(_fk_paths.relocate((source/'report.json'), __file__))
    bound={_fk_paths.relocate((source/'return.json'), __file__):digest(_fk_paths.relocate((source/'return.json'), __file__)),args.manifest:digest(args.manifest),
           args.training_manifest:digest(args.training_manifest)}
    for name,sha in returned['files'].items():
        path=checked_path(source,name)
        if digest(path)!=sha:raise ValueError('Returned artifact checksum mismatch')
        bound[path]=sha
    if bound[args.manifest]!=returned['population_sha256'] or bound[args.training_manifest]!=returned['training_population_sha256']:
        raise ValueError('Training/test population differs from return')
    if report['checkpoints']!=returned['checkpoints']:raise ValueError('Returned checkpoint declarations differ')
    rows=[json.loads(line) for line in (_fk_paths.relocate((source/'predictions.jsonl'), __file__)).read_text().splitlines() if line.strip()]
    manifest=read(args.manifest);result=inspect_predictions(rows,manifest)
    for field,value in report['metrics'].items():
        if result['metrics'][field]!=value:raise ValueError('Recomputed metric differs: '+field)
    result['training_metadata']=training_metadata(read(args.training_manifest))
    result['visual_duration_diagnostic']=visual_duration_diagnostic(manifest,{r['video_id'] for r in rows if r['status']=='ASSESSED'})
    result['checkpoint_hash_declarations']=returned['checkpoints']
    result['checkpoint_payloads_present']=any(source.rglob('*.pt'))
    result['media_probes']=media_probes(rows,manifest,args.dataset_root) if args.dataset_root else []
    result['confirmed_findings']=[
        'Returned decisions follow hard OR; the fusion code is not forcing a fake result independently of branch votes.',
        'The returned audio branch decides fake on every assessed clip; the visual branch decides real on every assessed clip.',
        'The returned accuracy equals an always-fake predictor on the same eligible subset.',
        'All 500 visually manipulated held-out clips have less than half their annotated duration manipulated. Whole-video frame majority voting is poorly matched to this localized-edit population.',
        'Whole-video preparation rejects a clip if any required frame/crop/audio interval fails.',
        'Hard-decision fusion deliberately stores no continuous fused score; null AUC is not proof of a broken classifier.']
    result['not_yet_established']=[
        'Why audio hard decisions are always fake: undertraining, optimization collapse, checkpoint selection or preprocessing mismatch require checkpoint payloads and training logs.',
        'Visual clip votes alone do not prove a collapsed frame classifier. Per-frame probabilities are required to separate model collapse from majority-vote dilution.',
        'The actual training class distribution after source-time labels and preparation exclusions is not available in this return.',
        'Current source code can be inspected, but the held-out return does not bind the exact Machine B source revision.']
    output.mkdir(parents=True);write(_fk_paths.relocate((output/'report.json'), __file__),result)
    failures=result['assessment_failure_reasons'];metadata=result['training_metadata']
    lines=['# AVFakeNet returned-result diagnosis','',
        'Original predictions, reports and checkpoints were not modified. No model was retrained.','',
        '## Confirmed','',
        f"- Assessed {result['metrics']['assessed']}/1,000. All 417 assessed rows have visual=0 (real), audio=1 (fake), final=1.",
        f"- Recomputed confusion: {result['metrics']['confusion']}. Accuracy {result['metrics']['accuracy']:.4%}; balanced accuracy 50%.",
        '- This equals the always-fake baseline on the assessed subset. It is not evidence that the baseline learned useful discrimination.',
        '- No hard-OR inconsistencies were found. Both modalities have known class semantics in the current reconstruction.',
        '- Missing AUC is by design: the native adapter exports hard votes and null fused scores. Do not fabricate a continuous score to fill this field.','',
        '## Localized-edit versus majority-vote mismatch','',
        '- All 500 visually manipulated held-out clips have manipulated duration below 50%; the maximum is approximately 25.9%, and the median approximately 4.5%.',
        '- Within the assessed subset, all 185 visually manipulated clips are below 50%, with median approximately 4.3% and maximum approximately 13.3%.',
        '- A frame classifier can correctly detect a short manipulation yet produce a real video vote because genuine frames dominate. Therefore the returned visual clip votes do not establish visual-model collapse.',
        '- This is an annotation-duration diagnostic under dense uniform sampling, not a measured perfect-frame oracle or an actual decoded frame count.',
        '- Keep the paper-style majority-vote result unchanged. A localized variant would need a separately declared aggregation rule chosen on validation; changing it silently would invalidate the reproduction claim.','',
        '## Coverage failures','',*['- '+reason+': '+str(count) for reason,count in failures.items()],'',
        'The current evaluator calls face preparation for the whole video before audio preparation. Any preparation exception makes the whole clip unassessed, so these are first-failure counts, not independent modality failure totals.',
        'The shared face helper also computes lip crops although the AVFakeNet image branch uses only faces. Whether this caused any particular rejection needs a per-crop diagnostic; counts cannot be attributed to it from the returned reason alone.','',
        '## Training risks requiring evidence','',
        '- The limited-budget runner requests three epochs. The current loader streams videos ordered by ID, with consecutive frames per video and no training shuffle.',
        '- Temporal frame targets can be strongly dominated by real frames on AV-Deepfake1M; choosing checkpoints by ordinary validation accuracy can favor a majority-class visual model.',
        '- In the shared training metadata, manipulated visual time is only approximately 2.39% of known visual duration (validation: 2.49%), despite near-balanced visual clip labels. These are pre-exclusion duration fractions, not actual model training sample counts.',
        '- The audio branch labels an entire Mel image using the clip-level audio label, even for short localized edits. This differs from the source dataset assumptions and may be difficult to learn.',
        '- These are hypotheses, not confirmed explanations of the returned weights. The actual checkpoint payloads and per-epoch metrics are missing.','',
        '## Required Machine B evidence','',
        'Export the existing completed training return, which includes both selected best.pt files, run.json, metrics.json and validation predictions. No retraining is required:',
        '```powershell',
        '& .\\.venv-gpu\\Scripts\\python.exe scripts/transfer_worker_results.py export --worker-root .\\runs\\worker-b --role B --output .\\return-b-diagnostics-001',
        '```','',
        'Bring that folder to Machine A. Verify the checkpoint hashes match this held-out report before probing logits. Then inspect validation confusion, class balance, class probabilities, entropy and training history per branch. Use validation data for any revised training or calibration decisions; retain this held-out result unchanged.','',
        'Do not invert audio labels, change OR to AND, relax crop rules or replace missing scores merely to improve test numbers. Any changed protocol needs a separately named run and validation justification.','']
    (_fk_paths.relocate((output/'DIAGNOSIS.md'), __file__)).write_text('\n'.join(lines),encoding='utf-8')
    for path,sha in bound.items():
        if digest(path)!=sha:raise ValueError('Original evidence changed')
    write(_fk_paths.relocate((output/'source-preservation.json'), __file__),{'all_unchanged':True,'files':{str(p):sha for p,sha in bound.items()}})
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
