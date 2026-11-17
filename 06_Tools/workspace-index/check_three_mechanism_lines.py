"""Recompute plotted points from raw replays and scores, and verify sources."""
import csv,hashlib,json,re,sys,zipfile
from pathlib import Path
from urllib.parse import unquote,urlsplit
import numpy as np
from sklearn.metrics import roc_curve,roc_auc_score
W=next(p for p in Path(__file__).resolve().parents if (p/'workspace-layout.json').is_file())
O=W/'04_Results/publication/three-mechanisms-lines-001';R=W/'04_Results/experiments/fk';B=R/'training-scaling-001'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def csv_read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
items=read(O/'catalog.json')['items'];assert len(items)==3
assert [i['label'] for i in items]==['Fig. M1','Fig. M2','Fig. M3']
assert read(B/'progress.json')['status']==read(B/'extraction-progress.json')['status']=='FINISHED'
protocol=read(B/'protocol.json');assert protocol['window_counts']==[1,2,4,8] and protocol['row']['split']=='train'
assert protocol['row']['label']==0 and len(protocol['windows'])==2
for window in protocol['windows']:
    assert window['end']-window['start']==2 and sha(B/'prepared'/f"window-{window['index']:02d}.npz")==window['prepared_sha256']
stages={'Ours':['ours'],'AVFakeNet':['avf-visual','avf-audio'],'AVTENet':['avt-audio','avt-visual','avt-audiovisual','avt-fusion']}
updates={'Ours':1,'AVFakeNet':8,'AVTENet':4};protected={};point_count=0;raw_replays=0
for extract,item in [(False,items[0]),(True,items[2])]:
    plotted=csv_read(O/item['data'])
    assert {p['method'] for p in plotted if p['panel']=='a'}==set(stages)
    for method,names in stages.items():
        names=names[:-1] if extract and method=='AVTENet' else names
        parts=[]
        for name in names:
            d=read(B/(('extract-' if extract else '')+name+'.json'));assert d['checkpoint_unchanged']
            for p,h in d['source_files'].items():
                assert p not in protected or protected[p]==h;protected[p]=h
            points=d['curves'];parts.append(points['online'] if isinstance(points,dict) else points)
            raw_replays+=sum(len(p['samples']) for p in parts[-1])
        for index,n in enumerate([1,2,4,8]):
            point=next(p for p in plotted if p['method']==method and int(p['window_presentations'])==n)
            sums=[sum(part[index]['samples'][r]['seconds'] for part in parts) for r in range(3)]
            assert abs(float(point['seconds'])-np.median(sums))<1e-12
            for r in range(3):assert float(point[f'repeat_{r}_seconds'])==sums[r]
            assert int(point['optimizer_updates'])==(0 if extract else n*updates[method])
            if not extract:assert all(s['optimizer_updates']==p['updates_per_presentation']*n for part in parts for p in [part[index]] for s in p['samples'])
            point_count+=1
    if not extract:
        cached=read(B/'ours.json')['curves']['cached'];raw_replays+=sum(len(p['samples']) for p in cached)
        for point in cached:
            row=next(r for r in plotted if r['method']=='Ours: cached' and int(r['window_presentations'])==point['window_presentations'])
            assert float(row['seconds'])==np.median([s['seconds'] for s in point['samples']]);point_count+=1
external=csv_read(R/'paper-style-visuals-001/data/fig05-roc-pr.csv');score=csv_read(O/items[1]['data'])
mapping={'FK':'Ours','AVFakeNet adaptation':'AVFakeNet','AVTENet adaptation (0.5)':'AVTENet'}
expected=[(mapping[r['method']],r['curve'],float(r['x']),float(r['y'])) for m in ['Ours','AVFakeNet','AVTENet'] for r in external if mapping[r['method']]==m and r['curve'] in ('ROC','hard ROC point')]
observed=[(r['method'],r['curve'],float(r['x']),float(r['y'])) for r in score if r['panel']=='a'];assert observed==expected
assert sum(r['method']=='AVFakeNet' for r in score)==1
ab={v:{r['video_id']:r for r in read(R/'remaining-ablation-evaluation-001'/v/'test.json')['videos'] if r['prediction'] is not None and r['score'] is not None} for v in ['full','uniform_weights']}
ids=sorted(set(ab['full'])&set(ab['uniform_weights']));assert len(ids)==980
for v,label in [('full','Ours: learned weights'),('uniform_weights','Ours: uniform weights')]:
    labels=[ab[v][i]['label'] for i in ids];scores=[ab[v][i]['score'] for i in ids];x,y,_=roc_curve(labels,scores)
    rr=[r for r in score if r['method']==label]
    assert [(float(r['x']),float(r['y'])) for r in rr]==list(zip(x,y))
    assert abs(roc_auc_score(labels,scores)-read(O/'measurements.json')['scoring']['internal_metrics'][v]['auc'])<1e-12
sources=read(O/'source-manifest.json')['files']
assert all(sha(W/p)==entry['sha256'] for p,entry in sources.items())
for p,h in protected.items():
    path=W/p[len('/inputs/'):] if p.startswith('/inputs/') else Path(p)
    assert path.is_file() and sha(path)==h,(str(path),'original checkpoint changed')
for item in items:
    svg=(O/item['svg']).read_text(encoding='utf-8');assert not re.search(r'\bFK\b|adaptation|\badapt\.',svg,re.I)
    assert all(m in svg for m in stages)
    assert item['source_files'] and all((O/item[key]).is_file() for key in ['png','svg','data'])
with zipfile.ZipFile(O/'THREE_MECHANISMS_PNG.zip') as z:assert len(z.namelist())==3 and z.testzip() is None
sys.path.insert(0,str(W/'09_Runtime/review-cache/archive-v0/1PiHw63lG2QUPDDZ'));import pymupdf as fitz
with fitz.open(O/'THREE_MECHANISMS.pdf') as doc:
    assert len(doc)==3
    for page in doc:
        for block in page.get_text('blocks'):assert block[0]>=0 and block[1]>=0 and block[2]<=page.rect.width+.5 and block[3]<=page.rect.height+.5
links=0
for p in [O/'index.html',W/'START_HERE.md',W/'04_Results/README.md',W/'05_Documents/README.md']:
    for match in re.finditer(r'''(?:href|src)=["']([^"']+)|\]\(([^)]+)\)''',p.read_text(encoding='utf-8')):
        target=match.group(1) or match.group(2);u=urlsplit(target)
        if u.scheme or u.netloc or not u.path:continue
        assert (p.parent/unquote(u.path)).exists(),(str(p),target);links+=1
reviewed='--visual-reviewed' in sys.argv
v=read(O/'validation.json');v.update(status='PASSED',numeric_points_recomputed=True,workload_points_checked=point_count,raw_stage_replays=raw_replays,source_files=len(sources),protected_checkpoint_files=len(set(protected)),native_stage_coverage_checked=True,links_checked=links,manual_visual_review='PASSED' if reviewed else 'pending',pages_visually_reviewed=3 if reviewed else 0)
(O/'validation.json').write_text(json.dumps(v,indent=2),encoding='utf-8')
if reviewed:
    (O/'complete.json').write_text(json.dumps({'status':'PASSED','files':{p.relative_to(O).as_posix():sha(p) for p in O.rglob('*') if p.is_file() and p.name!='complete.json' and 'qa' not in p.relative_to(O).parts}},indent=2),encoding='utf-8')
print(json.dumps(v))
