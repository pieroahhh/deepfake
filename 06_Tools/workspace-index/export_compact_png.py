"""Export main comparison pages and original high-resolution PNG assets."""
import hashlib,json,shutil,sys,zipfile
from pathlib import Path
W=next(p for p in Path(__file__).resolve().parents if (p/'workspace-layout.json').is_file())
ROOT=W/'04_Results/publication/paper-compact-001'
OUT=ROOT/'PNG_EXPORT'
sys.path.insert(0,str(W/'09_Runtime/review-cache/archive-v0/1PiHw63lG2QUPDDZ'))
import pymupdf as fitz
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    catalog=json.loads((ROOT/'catalog.json').read_text(encoding='utf-8'))
    pdf=ROOT/'MAIN_COMPARISONS.pdf';original=sha(pdf)
    for name in ('figures-and-tables','pdf-pages'):(OUT/name).mkdir(parents=True,exist_ok=True)
    files=[]
    for item in catalog['main']:
        source=ROOT/item['png'];target=OUT/'figures-and-tables'/source.name
        shutil.copy2(source,target)
        assert sha(source)==sha(target)
        pix=fitz.Pixmap(str(target));assert pix.width>1000 and pix.height>500
        files.append({'file':target.relative_to(OUT).as_posix(),'title':item['title'],'width':pix.width,'height':pix.height,'scope':'original figure/table PNG; retained 600-dpi export','sha256':sha(target)})
    doc=fitz.open(pdf)
    for index,page in enumerate(doc):
        target=OUT/'pdf-pages'/f'MAIN_COMPARISONS-page-{index+1:02d}.png'
        pix=page.get_pixmap(dpi=300,alpha=False);pix.save(target)
        files.append({'file':target.relative_to(OUT).as_posix(),'page':index+1,'width':pix.width,'height':pix.height,'dpi':300,'sha256':sha(target)})
    doc.close();assert sha(pdf)==original
    (OUT/'README.txt').write_text('MAIN COMPARISONS — PNG\n\nfigures-and-tables/: 3 comparison figures + 3 tables, original high-resolution PNG exports.\npdf-pages/: complete PDF pages including cover and captions, 300 dpi.\n\nNumerical results are unchanged. No new training or inference.\n',encoding='utf-8')
    (OUT/'manifest.json').write_text(json.dumps({'source_pdf_sha256':original,'files':files,'source_pdf_unchanged':True},indent=2),encoding='utf-8')
    archive=ROOT/'MAIN_COMPARISONS_PNG.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(OUT).as_posix())
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(json.dumps({'figures_and_tables':6,'pdf_pages':7,'archive':str(archive),'bytes':archive.stat().st_size}))
if __name__=='__main__':main()
