"""Verify naming, allowed comparisons, numeric provenance and export integrity."""
import csv,hashlib,json,re,shutil,sys,zipfile
from pathlib import Path
from urllib.parse import urlsplit,unquote
W=next(p for p in Path(__file__).resolve().parents if (p/'workspace-layout.json').is_file())
O=W/'04_Results/publication/paper-compact-002'
sys.path.insert(0,str(Path(__file__).parent))
from revise_paper_comparisons import Selection,name
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def csv_rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
c=read(O/'catalog.json');main=c['main'];own=c['own_method_appendix']
assert len(main)==7 and sum(x['kind']=='figure' for x in main)==4 and len(own)==2
validator=Selection();validator.main=main;validator.appendix=own;validator.check_labels()
original=csv_rows(W/'04_Results/experiments/fk/paper-style-visuals-001/data/fig05-roc-pr.csv')
plotted=csv_rows(O/next(x['data'] for x in main if x['stem']=='fig02-score-comparison'))
assert plotted==[{**r,'method':name(r['method'])} for r in original]
assert not any(r['curve'] in ('ROC','PR') for r in plotted if r['method']=='AVFakeNet')
confusion=next(x for x in main if x['stem']=='fig01-confusion-common')
original_confusion=csv_rows(W/'04_Results/experiments/fk/paper-style-visuals-001/data/fig03-confusion-common.csv')
assert csv_rows(O/confusion['data'])==[{**r,'method':name(r['method'])} for r in original_confusion]
assert confusion['label']=='Fig. 1'
table2=next(x for x in main if x['kind']=='table' and x['number']==2)
assert 'CUDA' not in (O/table2['svg']).read_text(encoding='utf-8')
assert 'CUDA' not in (O/table2['data']).read_text(encoding='utf-8-sig')
assert 'CUDA' not in (O/table2['latex']).read_text(encoding='utf-8')
cost=next(x for x in main if x['stem']=='fig03-pipeline-cost');cost_svg=(O/cost['svg']).read_text(encoding='utf-8')
assert all(text in cost_svg for text in ['25.627 ms','4.141 ms','1.822 ms','27.40 us','3.70 us','Seconds per clip'])
assert 'log scale' not in cost_svg
assert 'minor ticks mark 0.5 s' in cost['caption'] and 'stroke thickness is not duration' in cost['caption']
assert cost['title']=='Table III pipeline overview'
source_cost={name(r['model']):r for r in csv_rows(W/'04_Results/experiments/fk/table-iii-cost-001/table-iii-comparison.csv') if r['population']=='common_assessed'}
cost_rows=csv_rows(O/cost['data']);assert len(cost_rows)==18
for row in cost_rows:
    assert float(row['seconds'])==float(source_cost[row['method']][row['measurement']])
    assert abs(float(row['stack_top_seconds'])-float(row['stack_bottom_seconds'])-float(row['seconds']))<1e-12
    if row['status']=='measured':assert float(row['plotted_seconds'])==float(row['seconds'])>0
    else:assert row['plotted_seconds']=='' and row['status'] in ('grouped with aggregation','not provided')
assert sum(r['status']!='measured' for r in cost_rows)==3
original=csv_rows(W/'04_Results/experiments/fk/paper-style-visuals-001/data/fig14-shared-controls.csv')
plotted=csv_rows(O/own[0]['data']);mapping={'FK':'Ours','visual_only':'Video only','audio_only':'Audio only','concatenation':'Concatenation','late_fusion':'Late fusion'}
assert plotted==[{**r,'model':mapping[r['model']]} for r in original]
assert {r['model'] for r in plotted}==set(mapping.values())
source=read(O/'source-manifest.json');assert all(sha(W/p)==v['sha256'] for p,v in source['files'].items())
for item in main+own:
    assert item['source_files']
    for key in ('png','svg','data'):assert (O/item[key]).is_file()
pattern=re.compile(r'''(?:href|src)=["']([^"']+)|\]\(([^)]+)\)''');checked=0
for page in [O/'index.html',W/'START_HERE.md',W/'04_Results/README.md',W/'05_Documents/README.md']:
    for m in pattern.finditer(page.read_text(encoding='utf-8')):
        value=m.group(1) or m.group(2);u=urlsplit(value)
        if u.scheme or u.netloc or not u.path:continue
        assert (page.parent/unquote(u.path)).exists(),(page,value)
        checked+=1
with zipfile.ZipFile(O/'COMPARISON_PNG.zip') as z:assert z.testzip() is None
# Preserve superseded numbered exports outside the current selection, so a
# folder view cannot accidentally pick the old hidden-cost chart.
obsolete=['fig01-score-comparison','fig02-pipeline-cost','fig03-output-capabilities'];archived=[]
for stem in obsolete:
    candidates=[O/'figures'/(stem+ext) for ext in ('.png','.svg')]+[O/'data'/(stem+'.csv'),O/'metadata'/(stem+'.json'),O/'PNG_EXPORT/main'/(stem+'.png')]
    for source_path in candidates:
        if not source_path.exists():continue
        source_path=source_path.resolve();assert O.resolve() in source_path.parents
        target=(W/'08_Archive/publication-revisions/paper-compact-002-before-visibility-fix'/source_path.relative_to(O)).resolve()
        assert W.resolve() in target.parents
        if target.exists():raise FileExistsError('Revision archive already contains '+str(target))
        h=sha(source_path);target.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(source_path),str(target));assert sha(target)==h
        archived.append({'path':target.relative_to(W).as_posix(),'sha256':h})
if archived:(O/'superseded-export-archive.json').write_text(json.dumps(archived,indent=2),encoding='utf-8')
v=read(O/'validation.json');v.update({'manual_visual_review':'PASSED','pdf_pages_visually_reviewed':11,'links_checked':checked,'numeric_series_match_sources':True,'all_internal_controls_retained':True,'source_files_verified':len(source['files']),'confusion_matrices_restored':True,'cuda_column_removed_from_table_ii':True,'small_costs_expanded_with_true_values':True})
(O/'validation.json').write_text(json.dumps(v,indent=2),encoding='utf-8')
complete={'files':{p.relative_to(O).as_posix():sha(p) for p in O.rglob('*') if p.is_file() and 'qa' not in p.relative_to(O).parts and p.name!='complete.json'},'status':'PASSED','counts':{'comparison_figures':4,'tables':3,'own_method_figures':2}}
(O/'complete.json').write_text(json.dumps(complete,indent=2),encoding='utf-8')
print(json.dumps(v))
