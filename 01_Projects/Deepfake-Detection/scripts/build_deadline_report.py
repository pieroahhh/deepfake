"""Read-only results compilation; completed evidence only, no fitting or GPU work."""

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

import os,sys,json,time,hashlib,argparse
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(_fk_paths.relocate((ROOT/'.cache/matplotlib'), __file__))
OUT=_fk_paths.relocate((ROOT/'runs/report-0900-001'), __file__)
POP='c60bc02bf696360c3dfce8eca2a9a895b84db864e51255639e1529e23893c5b5'
def read(path):return json.loads(_fk_paths.relocate((Path(path)), __file__).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(_fk_paths.relocate((Path(path)), __file__).read_bytes()).hexdigest()
def write(path,text):
    temporary=path.with_suffix(path.suffix+'.writing');temporary.write_text(text,encoding='utf-8');temporary.replace(path)
def percent(value):return 'N/A' if value is None else f'{100*value:.2f}%'
def table(models):
    lines=['| Method | Assessed | Coverage | Accuracy | BA | F1 | ROC AUC |','|---|---:|---:|---:|---:|---:|---:|']
    for name,m in models.items():
        count=m.get('assessed');c=m.get('confusion',m)
        accuracy=m.get('accuracy',(c.get('tp',0)+c.get('tn',0))/count if count else None)
        auc=m.get('roc_auc')
        lines.append(f'| {name} | {count} | {percent(m.get("coverage"))} | {percent(accuracy)} | {percent(m.get("balanced_accuracy"))} | {percent(m.get("f1"))} | {"N/A" if auc is None else f"{auc:.4f}"} |')
    return '\n'.join(lines)


def build():
    OUT.mkdir(exist_ok=True);sources={}
    def source(relative):
        path=_fk_paths.relocate((ROOT/relative), __file__);value=read(path);sources[relative]=sha(path);return value
    main=source('runs/three-model-heldout-002/comparison.json')
    controls=source('runs/shared-controls-001/summary.json')
    if main['population_sha256']!=POP or controls['population_sha256']!=POP:raise ValueError('Evaluation populations differ')
    source('runs/three-model-heldout-002/complete.json')
    source('runs/remaining-ablation-evaluation-001/complete.json')
    core=source('runs/remaining-ablation-evaluation-001/comparison/metrics.json')
    localization=source('runs/remaining-offline-001/localization/summary.json')
    bootstrap=source('runs/remaining-offline-001/bootstrap/metrics.json')['bootstrap']
    shared={k:v['video_metrics'] for k,v in controls['models'].items()}
    aggregation={};pending=[]
    for name in ('window_max','window_mean','video_max','video_mean'):
        path=_fk_paths.relocate((ROOT/'runs/remaining-supplement-001/aggregation'/name/'test.json'), __file__)
        locked=_fk_paths.relocate((path.parent/'locked.json'), __file__)
        if path.exists() and locked.exists():
            value=source(str(path.relative_to(ROOT)));lock=source(str(locked.relative_to(ROOT)))
            if value['checkpoint_sha256']!=lock['checkpoint_sha256'] or len(value['videos'])!=1000:raise ValueError('Incomplete aggregation result')
            aggregation[name]=value['video_metrics']
        else:pending.append(name)
    statuses={}
    for folder in ('remaining-supplement-001','remaining-derivatives-001'):
        if folder=='remaining-supplement-001' and (_fk_paths.relocate((ROOT/'runs/remaining-supplement-active.json'), __file__)).exists():
            active=read(_fk_paths.relocate((ROOT/'runs/remaining-supplement-active.json'), __file__))['folder']
            if active not in ('remaining-supplement-001','remaining-supplement-002'):raise ValueError('Unknown continuation')
            folder=active
        path=_fk_paths.relocate((ROOT/'runs'/folder/'progress.json'), __file__)
        if path.exists():statuses[folder]=read(path)
    now=datetime.now(ZoneInfo('Asia/Bangkok')).isoformat(timespec='seconds')
    result={'generated_bangkok':now,'population_sha256':POP,'attempted':1000,
        'external_comparison':main,'shared_controls':shared,'core_ablations':core,
        'localization':localization,'primary_bootstrap':bootstrap,'completed_aggregation':aggregation,
        'pending_aggregation':pending,'live_status':statuses,'source_sha256':sources}
    write(_fk_paths.relocate((OUT/'results.json'), __file__),json.dumps(result,indent=2,allow_nan=False))
    text=[f'# Results available for the 09:00 report\n\nUpdated {now}. This is a verified snapshot of completed experiments; the full queue continues.\n',
        '## Population and interpretation\n\nAll primary methods attempted the same locked 1,000 local held-out clips from unused official validation (250 per manipulation type). This is not the official test set. Training attempted 3,000 clips and validation 500; the FK frozen caches retained 2,912/485 usable entries. The official train/validation source-family overlap was retained and documented.\n',
        '## Primary external comparison: each method\'s assessed clips\n\n'+table(main['own_assessed'])+'\n',
        'Coverage differs substantially. Scores in this table have different assessed denominators; inspect the paired subset below as well. AVFakeNet provides hard fused decisions, so fused ROC AUC is unavailable.\n',
        f'## Paired common assessed subset ({main["common_assessed_count"]} clips)\n\n'+table(main['common_assessed'])+'\n',
        'All methods have coverage 100% within this selected common subset. This does not remove selection bias caused by excluding clips that any method could not assess. Calibrated AVTENet, if listed, is supplementary post-inspection analysis; retain default 0.5 as the primary row.\n',
        '## Shared encoder controls\n\n'+table(shared)+'\n',
        'These are five-epoch, seed-7 controls using the same FK encoder caches. Late fusion exceeds FK on some retained metrics; report it alongside the other controls rather than claiming FK wins every comparison.\n',
        '## Matched retrained core ablations\n\n'+table(core['per_method_assessed'])+'\n',
        'All four variants completed five epochs with seed 7 and validation-only calibration. The new Full reference is a matched rerun. Differences below use 2,000 paired source-family bootstrap resamples on common assessed clips. Intervals measure test-population sampling, not training-seed variability.\n',
        '| Variant minus Full | BA difference [95% CI], percentage points | F1 difference [95% CI], percentage points |\n|---|---:|---:|']
    for name,values in core['bootstrap']['paired_differences'].items():
        if name=='full':continue
        formatted=[]
        for metric in ('balanced_accuracy','f1'):
            m=values[metric];formatted.append(f'{100*m["observed_difference"]:+.2f} [{100*m["ci95"][0]:+.2f}, {100*m["ci95"][1]:+.2f}]')
        text.append('| '+name+' | '+' | '.join(formatted)+' |')
    text+=['\nFor BA, the no-AV and no-temporal intervals include zero; uniform weighting has a negative interval in this run. Do not generalize a single training seed into a universal ranking.\n',
        '## Aggregation variants completed so far\n\n'+(table(aggregation) if aggregation else 'No aggregation held-out results completed at this snapshot.')+'\n\nPending: '+', '.join(pending)+'.\n',
        '## Temporal localization\n\nThis summarizes the retained frozen-threshold FK timeline adapter. Time coverage: '+percent(localization['time_assessment_coverage'])+'; assessed-region precision '+percent(localization['duration_weighted_assessed_region']['precision'])+', recall '+percent(localization['duration_weighted_assessed_region']['recall'])+', F1 '+percent(localization['duration_weighted_assessed_region']['f1'])+', IoU '+percent(localization['duration_weighted_assessed_region']['iou'])+'. Detected share of all manipulated time, including unassessed positive time in the denominator: '+percent(localization['detected_fraction_of_all_manipulated_time'])+'. Native temporal localization is not provided by these AVFakeNet/AVTENet adaptations.\n',
        '## Primary external-comparison uncertainty\n\n2,000 paired source-family bootstrap resamples, seed 7, on 415 common assessed clips. Differences and intervals below are percentage points relative to FK. This supplements point estimates; it cannot fix comparator fidelity, unequal training budgets, or low coverage.\n'+
        '\n| Method minus FK | BA difference [95% CI] | F1 difference [95% CI] |\n|---|---:|---:|\n'+
        '\n'.join('| '+name+' | '+' | '.join(f'{100*values[key]["observed_difference"]:+.2f} [{100*values[key]["ci95"][0]:+.2f}, {100*values[key]["ci95"][1]:+.2f}]' for key in ('balanced_accuracy','f1'))+' |' for name,values in bootstrap['paired_differences'].items() if name!='FK')+'\n',
        '## Limits that must accompany the results\n\n- AVFakeNet is a local paper-guided reconstruction with normalization/training adaptations and a one-epoch recovery pilot; its low coverage and weak performance do not establish that the authors\' method is weak.\n- AVTENet uses the approved joint Google/Hugging Face ViViT adaptation with AST and native AV-HuBERT; it is not the original factorized-ViViT configuration. Its validation-calibrated threshold was explored after earlier held-out inspection.\n- Follow-up ablations and robustness are exploratory because this held-out population has already been inspected. No fresh confirmatory test or independent-seed result is claimed.\n- Robustness derivatives are prepared, but derivative preparation is not robustness evaluation. Full raw-media profiling, forensic cases and robustness remain pending until their outputs pass checks.\n- FakeAVCeleb transfer is deferred by user choice while the dataset is unavailable.\n',
        '## Live queue at this snapshot\n\n'+ '\n'.join(f'- {k}: {v.get("status")} / {v.get("stage")}' for k,v in statuses.items())+'\n',
        '## Figures\n\n![Controls and core ablations](controls-and-ablations.png)\n\n![Three-model held-out comparison](three-model-comparison.png)\n\n![Localization examples](localization-examples.png)\n',
        '## Reproducibility\n\nPopulation SHA256: `'+POP+'`. All consumed source file digests are in `results.json`. Original predictions and checkpoints are read only. Rebuild this report to include newly completed aggregation results; pending experiments never populate completed result tables.\n']
    write(_fk_paths.relocate((OUT/'REPORT.md'), __file__),'\n'.join(text))
    thai=f'''# รายงานผลที่มีสำหรับส่งก่อน 09:00

อัปเดต {now}

ผลที่เสร็จครบ: เปรียบเทียบ FK/AVTENet/AVFakeNet, shared controls 4 แบบ, core ablation 4 แบบ, localization และ paired bootstrap

## Core ablation

{table(core['per_method_assessed'])}

ทั้ง 4 ตัวเทรนครบ 5 epochs, seed 7 ใช้ validation เลือก threshold แล้วทดสอบบนชุดเดิม 1,000 คลิป มี feature ใช้ได้ 980 คลิป ผล Full สูงกว่า ablation ใน BA/F1 รอบนี้ แต่ช่วงความเชื่อมั่นของ BA เมื่อปิด AV หรือ temporal ยังคร่อมศูนย์ จึงยังสรุปผลทั่วไปจาก seed เดียวไม่ได้

## ข้อจำกัด

- AVFakeNet และ AVTENet เป็น implementation/adaptation ของโครงการนี้ ไม่ใช่ผลจากโค้ดต้นฉบับครบเงื่อนไข จึงไม่ควรอ้างว่าชนะวิธีต้นฉบับ
- coverage แต่ละโมเดลต่างกัน ต้องดูทั้งผลบนคลิปที่แต่ละตัวประเมินได้ และผลบน 415 คลิปที่ประเมินได้ร่วมกัน
- Late fusion ทำได้ดีกว่า FK ในบาง metric ต้องรายงานด้วย
- ชุดนี้เป็น local held-out จาก official validation ไม่ใช่ official test; การทดลองต่อยอดเป็น exploratory
- aggregation ที่ยังไม่จบ, forensic, profiling และ robustness ยังไม่ถือว่ามีผลสมบูรณ์ ส่วน FakeAVCeleb พักไว้ตามที่เลือก

อ่านตารางทั้งหมด กราฟ และรายละเอียดข้อจำกัดใน [REPORT.md](REPORT.md) และข้อมูลพร้อม source hashes ใน [results.json](results.json)

สถานะคิว ณ เวลาสร้างรายงาน: {statuses.get('remaining-supplement-001',{}).get('stage','unknown')}
'''
    write(_fk_paths.relocate((OUT/'REPORT_TH.md'), __file__),thai)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    panels=[('Shared encoder controls',{'FK':main['own_assessed']['FK'],**shared}),('Matched core ablations',core['per_method_assessed'])]
    for ax,(title,models) in zip(axes,panels):
        x=np.arange(len(models));width=.25
        for i,(metric,label) in enumerate([('balanced_accuracy','BA'),('f1','F1'),('roc_auc','AUC')]):
            ax.bar(x+(i-1)*width,[m[metric] for m in models.values()],width,label=label)
        ax.set_xticks(x,list(models),rotation=24,ha='right');ax.set_ylim(0,1);ax.set_title(title);ax.set_ylabel('Score');ax.legend(ncol=3);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Local held-out results — 980 assessed / 1,000 attempted; exploratory, seed 7')
    fig.savefig(_fk_paths.relocate((OUT/'controls-and-ablations.png'), __file__),dpi=170);fig.savefig(_fk_paths.relocate((OUT/'controls-and-ablations.pdf'), __file__));plt.close(fig)
    # Existing checked comparison and localization figures remain authoritative.
    import shutil
    for original,name in [(_fk_paths.relocate((ROOT/'runs/three-model-heldout-002/comparison.png'), __file__),'three-model-comparison.png'),(_fk_paths.relocate((ROOT/'runs/remaining-offline-001/localization/localization-examples.png'), __file__),'localization-examples.png')]:
        if original.exists():shutil.copyfile(original,_fk_paths.relocate((OUT/name), __file__))
    files={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='snapshot.json' and not p.name.endswith('.writing')}
    write(_fk_paths.relocate((OUT/'snapshot.json'), __file__),json.dumps({'generated_bangkok':now,'files':files,'sources':sources,'completed_scope':'Only completed results are included; full queue continues'},indent=2))
    print(f'Report ready: {OUT}; completed aggregation: {len(aggregation)}/4',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--until-nine',action='store_true');args=parser.parse_args()
    while True:
        build()
        now=datetime.now(ZoneInfo('Asia/Bangkok'))
        if not args.until_nine or now.hour>=9:break
        time.sleep(60)
