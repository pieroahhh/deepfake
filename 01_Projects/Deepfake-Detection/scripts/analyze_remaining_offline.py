"""CPU-only summaries from existing results; no fitting or raw-media inference."""

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

import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
import sys,json,argparse
from pathlib import Path
ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(_fk_paths.relocate((ROOT/'src'), __file__)),str(_fk_paths.relocate((ROOT/'scripts'), __file__))]
from baselines.common.io import read,digest
from avfakenet_progress_io import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task',choices=('localization','bootstrap'),required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    output=args.output.resolve()
    if output.exists():raise ValueError('Use a new output folder; preserve previous analyses')
    package=_fk_paths.relocate((ROOT/'experiments/remaining-001'), __file__);plan=read(_fk_paths.relocate((package/'plan.json'), __file__))
    for relative,sha in read(_fk_paths.relocate((package/'SHA256SUMS.json'), __file__)).items():
        if digest(_fk_paths.relocate((package/relative), __file__))!=sha:raise ValueError('Prepared protocol changed')
    manifest={r['video_id']:r for r in read(_fk_paths.relocate((ROOT.parent/'Heldout-1000/population/manifest.json'), __file__))}
    fkpath=_fk_paths.relocate((ROOT.parent/'fk-heldout-copy.json'), __file__);fk=read(fkpath)
    if args.task=='bootstrap':
        heldout=_fk_paths.relocate((ROOT/'runs/avfakenet-heldout-002'), __file__)
        if not (_fk_paths.relocate((heldout/'complete.json'), __file__)).exists():raise ValueError('Wait for the current held-out evaluation to complete')
        from compare_three_heldout import checked_rows
        from deepfake_detection.reporting import write_detection_report
        paths={'FK':fkpath,'AVTENet adaptation':_fk_paths.relocate((ROOT.parent/'avh-a/code/runs/auto-next-001/heldout/evaluate/predictions.jsonl'), __file__),
               'AVFakeNet adaptation':_fk_paths.relocate((heldout/'predictions.jsonl'), __file__)}
        models={}
        for name,path in paths.items():
            values=fk['videos'] if name=='FK' else [json.loads(line) for line in path.read_text().splitlines()]
            checked_rows(values,manifest)
            models[name]=[{**row,'source_family':manifest[row['video_id']]['source_family'],
                'status':row.get('status','ASSESSED' if row['prediction'] is not None else 'UNASSESSED')} for row in values]
        write_detection_report(models,output,provenance={'protocol_id':'remaining-001-exploratory-paired-bootstrap',
            'data_kind':'research','sources':{name:digest(path) for name,path in paths.items()},
            'note':'Default AVTENet threshold only. Calibrated post-inspection analysis stays separate; intervals are not training-seed variability.'},
            reference='FK',replicates=plan['bootstrap']['replicates'])
        return
    output.mkdir(parents=True)
    summary=read(_fk_paths.relocate((package/'localization-summary.json'), __file__))
    if summary['source_sha256']!=digest(fkpath):raise ValueError('Localization source changed')
    write_json(_fk_paths.relocate((output/'summary.json'), __file__),summary)
    coverage=summary['time_assessment_coverage']
    metrics=summary['duration_weighted_assessed_region']
    (_fk_paths.relocate((output/'README.md'), __file__)).write_text('\n'.join(['# FK existing localization analysis','',
        'This summarizes the stored endpoint-partition/mean-of-covering-bins adapter at its existing validation-selected threshold.',
        'It does not refit any threshold or claim native temporal localization for the two external baselines.','',
        f'- Videos: {summary["videos"]}',f'- Duration-weighted assessment coverage: {coverage:.4%}',
        f'- Assessed-region precision: {metrics["precision"]:.4f}',f'- Assessed-region recall: {metrics["recall"]:.4f}',
        f'- Assessed-region F1: {metrics["f1"]:.4f}',f'- Assessed-region IoU: {metrics["iou"]:.4f}',
        f'- Detected share of all manipulated duration, including unassessed positive time in the denominator: {summary["detected_fraction_of_all_manipulated_time"]:.4%}',
        '', 'Example selection is based on source metadata and a fixed hash, not successful predictions.']),encoding='utf-8')
    os.environ['MPLCONFIGDIR']=str(_fk_paths.relocate((ROOT/'.cache/matplotlib'), __file__))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    examples=read(_fk_paths.relocate((package/'forensic-plan.json'), __file__))['selected_source_examples'];lookup={t['video_id']:t for t in fk['timeline_metrics_supplement']}
    fig,axes=plt.subplots(4,1,figsize=(12,9))
    for axis,row in zip(axes,examples):
        timeline=lookup[row['video_id']];segments=timeline['segments'];xs=[];ys=[]
        for segment in segments:
            a,b=segment['interval'];score=segment['score']
            xs.extend([a,b]);ys.extend([np.nan if score is None else score]*2)
            if score is None:axis.axvspan(a,b,color='gray',alpha=.25)
        for a,b in row['manipulated_intervals']:axis.axvspan(a,b,color='red',alpha=.18)
        axis.plot(xs,ys,color='#2563eb',linewidth=1);axis.axhline(timeline['threshold'],linestyle='--',color='black',linewidth=.8)
        axis.set(xlim=(0,row['duration']),ylim=(0,1),ylabel='FK score',title=row['source_modify_type']+' | '+row['video_id'])
        axis.title.set_fontsize(8)
    axes[-1].set_xlabel('Source time (seconds)')
    fig.suptitle('FK localization examples — red: annotated edit; gray: unassessed; dashed: fixed threshold',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(_fk_paths.relocate((output/'localization-examples.png'), __file__),dpi=160);plt.close(fig)
    write_json(_fk_paths.relocate((output/'complete.json'), __file__),{'source_sha256':digest(fkpath),'files':{name:digest(output/name) for name in ('summary.json','README.md','localization-examples.png')}})


if __name__=='__main__':main()
