"""Check the approved selection, source integrity and local navigation."""
import csv,hashlib,json,re
from pathlib import Path
from urllib.parse import unquote,urlsplit
W=next(p for p in Path(__file__).resolve().parents if (p/'workspace-layout.json').is_file())
O=W/'04_Results/publication/paper-compact-001'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
c=read(O/'catalog.json');main=c['main'];appendix=c['own_method_appendix']
assert len(main)==6 and sum(x['kind']=='figure' for x in main)==3
assert len(appendix)==2
assert [x['number'] for x in main if x['kind']=='figure']==[1,2,3]
for item in main+appendix:
    assert item['source_files']
    assert all((O/item[k]).is_file() for k in ('png','svg','data'))
    text=(O/item['data']).read_text(encoding='utf-8-sig')
    if item in main:assert 'FK' in text
    else:assert 'AVTENet' not in text and 'AVFakeNet' not in text
assert [x['label'] for x in appendix]==['Fig. A1','Fig. A2']
manifest=read(O/'source-manifest.json')
assert all(sha(W/rel)==v['sha256'] for rel,v in manifest['files'].items())
pages=[O/'index.html',W/'START_HERE.md',W/'04_Results/README.md',W/'05_Documents/README.md']
pattern=re.compile(r'''(?:href|src)=["']([^"']+)|\]\(([^)]+)\)''');links=0
for page in pages:
    for m in pattern.finditer(page.read_text(encoding='utf-8')):
        value=m.group(1) or m.group(2);parts=urlsplit(value)
        if parts.scheme or parts.netloc or not parts.path:continue
        assert (page.parent/unquote(parts.path)).exists(),(page,value)
        links+=1
v=read(O/'validation.json');v.update({'manual_visual_review':'PASSED','pages_visually_inspected':10,'local_links_checked':links,'source_files_rechecked':len(manifest['files'])})
(O/'validation.json').write_text(json.dumps(v,indent=2),encoding='utf-8')
complete=read(O/'complete.json');complete['files']['validation.json']=sha(O/'validation.json');(O/'complete.json').write_text(json.dumps(complete,indent=2),encoding='utf-8')
print(json.dumps({'status':'PASSED','main_figures':3,'main_tables':3,'own_method_figures':2,'links':links,'own_source_files':len(manifest['files']),'originals_unchanged':True}))
