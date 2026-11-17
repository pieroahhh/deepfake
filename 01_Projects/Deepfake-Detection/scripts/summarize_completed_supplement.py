"""Compile completed robustness, forensic and timing results without GPU work."""

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

import json,hashlib,statistics,os,shutil
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1];RUN=_fk_paths.relocate((ROOT/'runs/remaining-supplement-002'), __file__)
OUT=_fk_paths.relocate((ROOT/'runs/completed-results-update-001'), __file__)
os.environ['MPLCONFIGDIR']=str(_fk_paths.relocate((ROOT/'.cache/matplotlib'), __file__))
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def pct(v):return 'N/A' if v is None else f'{v*100:.2f}%'


def main():
    OUT.mkdir(exist_ok=True);sources={};data={}
    for stage in ('forensic','profiling-fk','robustness-fk','profiling-avfakenet','robustness-avfakenet'):
        path=_fk_paths.relocate((RUN/stage/'summary.json'), __file__)
        if sha(path)!=read(_fk_paths.relocate((path.parent/'complete.json'), __file__))['summary_sha256']:raise ValueError('Completed summary changed: '+stage)
        data[stage]=read(path);sources[str(path.relative_to(ROOT))]=sha(path)
    base=_fk_paths.relocate((ROOT/'runs/report-0900-001/REPORT.md'), __file__)
    text=base.read_text(encoding='utf-8');sources[str(base.relative_to(ROOT))]=sha(base)
    text+='\n# Completed supplementary results update\n\nSnapshot '+datetime.now(ZoneInfo('Asia/Bangkok')).isoformat(timespec='seconds')+'\n'
    text+='\n## Robustness: 100 fixed clips per compression condition\n\n| Condition | FK BA | FK coverage | AVFakeNet BA | AVFakeNet coverage |\n|---|---:|---:|---:|---:|\n'
    conditions=['clean','video_compression-28','video_compression-38','audio_compression-64','audio_compression-32']
    for c in conditions:
        a=data['robustness-fk']['metrics'][c];b=data['robustness-avfakenet']['metrics'][c]
        text+=f'| {c} | {pct(a["balanced_accuracy"])} | {pct(a["coverage"])} | {pct(b["balanced_accuracy"])} | {pct(b["coverage"])} |\n'
    text+='\nThese are different assessed denominators, not paired condition/method comparisons. Coverage changes mean nonmonotonic BA across severity is not proof that stronger compression improves robustness. AVFakeNet remains a weak, low-coverage local adaptation.\n'
    text+='\n## Genuine audio-delay stress: false alarms\n\n| Delay | FK false alarms / assessed | AVFakeNet false alarms / assessed |\n|---|---:|---:|\n'
    for delay in (.08,.16,.32):
        key='av_offset-'+str(delay);a=data['robustness-fk']['metrics'][key];b=data['robustness-avfakenet']['metrics'][key]
        text+=f'| {delay:.2f} s | {a["confusion"]["fp"]}/{a["assessed"]} | {b["confusion"]["fp"]}/{b["assessed"]} |\n'
    text+='\nEach delay attempts 25 genuine clips. FK assesses 25, AVFakeNet 10. Undelayed genuine false alarms are 4/25 for FK and 9/10 for AVFakeNet. A delayed genuine clip keeps label 0; do not present fake recall/F1/BA for this single-class stress study.\n'
    a=data['profiling-fk']['clips'];b=data['profiling-avfakenet']['clips'];paired={r['row']['video_id'] for r in b if all(s['status']=='ASSESSED' for s in r['samples'][1:])}
    text+='\n## Raw profiling: loaded models, 12 shared clips\n\n| Model | Assessed clips | Median of 12 clip-median times | Median on 5 common assessed clips |\n|---|---:|---:|---:|\n'
    for name,clips in [('FK',a),('AVFakeNet adaptation',b)]:
        assessed=sum(all(s['status']=='ASSESSED' for s in r['samples'][1:]) for r in clips)
        text+=f'| {name} | {assessed}/12 | {statistics.median(r["median_seconds"] for r in clips):.2f} s | {statistics.median(r["median_seconds"] for r in clips if r["row"]["video_id"] in paired):.2f} s |\n'
    text+='\nOne warmup plus three measured passes, raw source preparation/inference, models already loaded, OS disk cache uncontrolled. FK includes canonical record construction but not artifact persistence. Full 12-clip timing includes AVFakeNet exclusions and cannot imply equal completed work. The 5-clip paired values are small-sample descriptive timings, not general throughput guarantees. Per-clip CUDA allocation/reservation and RSS are retained in source JSON; RSS is an after-call snapshot, not peak RAM. AVTENet staged timings are pending.\n'
    rows=data['forensic']['real_source_bound_records'];rejected=sum(sum(v.get('status')=='REJECTED_AS_EXPECTED' for v in r['checks'].values()) for r in rows)
    text+=f'\n## Forensic checks\n\nAll {len(rows)} original records verify; all {rejected} applicable tampering checks reject. Four cases are assessed real dataset sources; the no-audio and short derivatives correctly remain UNASSESSED, with four nonapplicable evidence/artifact checks across them. Stored-contribution reconstruction error is zero. This validates integrity and reconstruction, not authenticity or classification correctness.\n'
    text+='\n## Pending\n\nAVTENet audiovisual expert and fusion remain pending; no full three-model robustness/runtime ranking is claimed. Cross-dataset transfer and independent training seeds remain outside the current completed results.\n'
    (_fk_paths.relocate((OUT/'REPORT.md'), __file__)).write_text(text,encoding='utf-8')
    for name in ('controls-and-ablations.png','three-model-comparison.png','localization-examples.png'):
        source=_fk_paths.relocate((base.parent/name), __file__)
        if source.exists():shutil.copyfile(source,_fk_paths.relocate((OUT/name), __file__))
    (_fk_paths.relocate((OUT/'summary.json'), __file__)).write_text(json.dumps({'completed':data,'sources':sources,'paired_profiling_ids':sorted(paired)},indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained');x=np.arange(5)
    for model,label in [('fk','FK'),('avfakenet','AVFakeNet adaptation')]:
        metrics=data['robustness-'+model]['metrics']
        axes[0].plot(x,[metrics[c]['balanced_accuracy'] for c in conditions],marker='o',label=label)
        axes[1].plot(x,[metrics[c]['coverage'] for c in conditions],marker='o',label=label)
    for ax,title in zip(axes,['BA on each method/condition assessed subset','Assessment coverage (100 attempted per condition)']):
        ax.set_xticks(x,['Clean','Video CRF28','Video CRF38','Audio 64k','Audio 32k'],rotation=20);ax.set_ylim(0,1);ax.set_title(title,fontsize=10);ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Exploratory robustness; assessed populations differ; AVTENet pending',fontsize=11)
    fig.savefig(_fk_paths.relocate((OUT/'robustness.png'), __file__),dpi=180);plt.close(fig)
    (_fk_paths.relocate((OUT/'snapshot.json'), __file__)).write_text(json.dumps({'files':{p.name:sha(p) for p in OUT.iterdir() if p.name!='snapshot.json'}},indent=2))
    print(OUT)


if __name__=='__main__':main()
