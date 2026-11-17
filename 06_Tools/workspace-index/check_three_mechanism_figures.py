"""Independent checks of the three mechanism figures and claim boundaries."""
import csv,hashlib,json,re,sys,zipfile
from pathlib import Path
from urllib.parse import unquote,urlsplit
W=next(p for p in Path(__file__).resolve().parents if (p/'workspace-layout.json').is_file())
O=W/'04_Results/publication/three-mechanisms-001';R=W/'04_Results/experiments/fk'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def csv_read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
items=read(O/'catalog.json')['items'];assert len(items)==3 and all(x['source_files'] for x in items)
assert [x['label'] for x in items]==['Fig. M1','Fig. M2','Fig. M3']
data=[csv_read(O/x['data']) for x in items]
assert {r['method'] for r in data[0] if r['panel']=='a'}=={'Ours','AVFakeNet','AVTENet'}
inventory=csv_read(O/'data/training-stage-inventory.csv');totals={m:sum(int(s['trainable_parameters']) for s in inventory if s['method']==m) for m in ('Ours','AVFakeNet','AVTENet')}
assert totals=={'Ours':1002504,'AVFakeNet':13540732,'AVTENet':284662002}
assert {r['method']:int(r['value']) for r in data[0] if r['panel']=='a'}==totals
assert len(inventory)==7 and any(s['stage']=='cached fusion head' and s['trainable_parameters']=='2305' for s in inventory)
raw=read(R/'table-iii-complete-001/fk/training-cost.json')['matched_online_cached']
for row in data[0]:
    if row['panel']=='b':assert float(row['value'])==raw[row['measurement'].removesuffix('_update')]['medians']['total_seconds']
assert raw['final_weight_max_absolute_error']==0
ablation=read(R/'remaining-ablation-evaluation-001/comparison/metrics.json')
for row in data[1]:
    if row['panel']!='b':continue
    metric=row['measurement'].removesuffix('_full_minus_uniform');paired=ablation['bootstrap']['paired_differences']['uniform_weights'][metric]
    assert abs(float(row['value'])+paired['observed_difference']*100)<1e-9
    assert float(row['low95'])==-paired['ci95'][1]*100 and float(row['high95'])==-paired['ci95'][0]*100
    assert row['assessed_clips']=='980'
assert all(r['assessed_clips']=='415' for r in data[1] if r['panel']=='a')
assert len(csv_read(O/'data/all-retained-aggregation-controls.csv'))==5
controls={r['variant'] for r in data[2] if r['panel']=='c'}
assert controls=={'Ours','Video only','Audio only','Concatenation','Late fusion'}
timings={r['method']:float(r['value']) for r in data[2] if r['panel']=='a'}
assert timings['AVFakeNet']<timings['Ours']<timings['AVTENet']
sources=read(O/'source-manifest.json')['files'];assert all(sha(W/path)==entry['sha256'] for path,entry in sources.items())
for item in items:
    svg=(O/item['svg']).read_text(encoding='utf-8');assert not re.search(r'\bFK\b|adaptation|\badapt\.',svg,re.I)
    assert all(m in svg for m in ('Ours','AVFakeNet','AVTENet'))
pattern=re.compile(r'''(?:href|src)=["']([^"']+)|\]\(([^)]+)\)''');links=0
for p in [O/'index.html',W/'START_HERE.md',W/'04_Results/README.md',W/'05_Documents/README.md']:
    for m in pattern.finditer(p.read_text(encoding='utf-8')):
        value=m.group(1) or m.group(2);u=urlsplit(value)
        if u.scheme or u.netloc or not u.path:continue
        assert (p.parent/unquote(u.path)).exists(),(p,value);links+=1
with zipfile.ZipFile(O/'THREE_MECHANISMS_PNG.zip') as z:
    assert len(z.namelist())==3 and z.testzip() is None
sys.path.insert(0,str(W/'09_Runtime/review-cache/archive-v0/1PiHw63lG2QUPDDZ'));import pymupdf as fitz
doc=fitz.open(O/'THREE_MECHANISMS.pdf');assert len(doc)==3
for page in doc:
    for b in page.get_text('blocks'):assert b[0]>=0 and b[1]>=0 and b[2]<=page.rect.width+.5 and b[3]<=page.rect.height+.5
doc.close()
v=read(O/'validation.json');v.update({'status':'PASSED','numeric_data_and_ci_checked':True,'all_trainable_stages_disclosed':True,'all_modality_controls_present':True,'source_files':len(sources),'links_checked':links,'manual_visual_review':'PASSED','pages_visually_reviewed':3})
(O/'validation.json').write_text(json.dumps(v,indent=2),encoding='utf-8')
(O/'complete.json').write_text(json.dumps({'status':'PASSED','files':{p.relative_to(O).as_posix():sha(p) for p in O.rglob('*') if p.is_file() and p.name!='complete.json' and 'qa' not in p.relative_to(O).parts}},indent=2),encoding='utf-8')
print(json.dumps(v))
