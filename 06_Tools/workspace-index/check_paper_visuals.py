"""Verify artifact provenance, links and PDF layouts, then package the result."""

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

import csv
import hashlib
import html.parser
import json
import re
import sys
import zipfile
from pathlib import Path
from urllib.parse import unquote

ROOT = _fk_paths.workspace_root(__file__)
OUT = _fk_paths.relocate((ROOT / 'Deepfake-Detection/runs/paper-style-visuals-001'), __file__)
TMP = _fk_paths.relocate((ROOT / 'tmp/pdfs/paper-style-qa'), __file__)
TMP.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(_fk_paths.relocate((ROOT / '.review-cache/archive-v0/1PiHw63lG2QUPDDZ'), __file__)))
import pymupdf as fitz
from PIL import Image, ImageDraw


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    catalog = read(_fk_paths.relocate((OUT / 'catalog.json'), __file__))
    sources = read(_fk_paths.relocate((OUT / 'source-manifest.json'), __file__))
    # Manifest shape is deliberately inspected before taking any shortcut.
    files = sources.get('files', sources.get('sources', sources))
    if not isinstance(files, dict):
        raise ValueError('Unknown source manifest')
    bad_sources = []
    for name, entry in files.items():
        if not isinstance(entry, dict) or 'sha256' not in entry:
            continue
        if digest(_fk_paths.relocate((ROOT / name), __file__)) != entry['sha256']:
            bad_sources.append(name)
        if not (name.startswith(('Deepfake-Detection/runs/', 'avh-a/code/runs/', 'Heldout-1000/population/')) or name == 'fk-heldout-copy.json'):
            raise ValueError('Foreign data source: ' + name)
    assert not bad_sources
    items = catalog['items']
    assert len(items) == 60 and catalog['counts'] == {'table': 25, 'figure': 35}
    for item in items:
        assert item['data_origin'] == 'own_experiment_only'
        for field in ('png', 'svg', 'data'):
            assert (_fk_paths.relocate((OUT / item[field]), __file__)).is_file(), item
        assert item['source_files'] and not any('Downloads' in p or 'Paper22' in p or 'Paper16' in p for p in item['source_files'])
        with (_fk_paths.relocate((OUT / item['data']), __file__)).open(encoding='utf-8-sig', newline='') as stream:
            assert sum(1 for _ in csv.reader(stream)) > 1
    class Links(html.parser.HTMLParser):
        def __init__(self):
            super().__init__(); self.urls = []
        def handle_starttag(self, tag, attrs):
            for key, value in attrs:
                if key in ('href', 'src'):
                    self.urls.append(value)
    links = Links(); links.feed((_fk_paths.relocate((OUT / 'index.html'), __file__)).read_text(encoding='utf-8'))
    missing = [u for u in links.urls if not u.startswith(('#', 'http')) and not (_fk_paths.relocate((OUT / unquote(u)), __file__)).exists()]
    assert not missing, missing
    doc = fitz.open(_fk_paths.relocate((OUT / 'FIGURES_AND_TABLES.pdf'), __file__))
    assert len(doc) == 61
    outside = []
    for i, page in enumerate(doc):
        for block in page.get_text('dict')['blocks']:
            box = fitz.Rect(block['bbox'])
            if box.x0 < -1 or box.y0 < -1 or box.x1 > page.rect.width + 1 or box.y1 > page.rect.height + 1:
                outside.append(i + 1)
        if i > 0:
            assert items[i - 1]['label'] in page.get_text(), (i, page.get_text())
    assert not outside, outside
    # Rasterize every PDF page for a visual pass, plus larger selected pages.
    for start in range(0, len(doc), 12):
        sheet = Image.new('RGB', (1600, 1920), 'white')
        draw = ImageDraw.Draw(sheet)
        for offset in range(12):
            i = start + offset
            if i >= len(doc):
                break
            page = doc[i]
            pix = page.get_pixmap(matrix=fitz.Matrix(.5, .5))
            image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
            image.thumbnail((375, 445))
            x, y = (offset % 4) * 400 + 12, (offset // 4) * 640 + 35
            sheet.paste(image, (x, y))
            draw.text((x, y - 23), 'Page %d' % (i + 1), fill='black')
        sheet.save(_fk_paths.relocate((TMP / ('pdf-sheet-%02d.png' % (start // 12 + 1))), __file__))
    chosen = [0, 2, 6, 25, 40, 49, 58, 60]
    for i in chosen:
        doc[i].get_pixmap(matrix=fitz.Matrix(1.3, 1.3)).save(_fk_paths.relocate((TMP / ('page-%02d.png' % (i + 1))), __file__))
    # Figure contact sheets let us inspect every plot at its intended aspect ratio.
    for kind in ('figure', 'table'):
        selected = [x for x in items if x['kind'] == kind]
        for start in range(0, len(selected), 9):
            sheet = Image.new('RGB', (1800, 1500), 'white'); draw = ImageDraw.Draw(sheet)
            for offset, item in enumerate(selected[start:start + 9]):
                image = Image.open(_fk_paths.relocate((OUT / item['png']), __file__)).convert('RGB')
                image.thumbnail((570, 440))
                x, y = (offset % 3) * 600 + 15, (offset // 3) * 500 + 38
                sheet.paste(image, (x, y)); draw.text((x, y - 24), item['label'] + ' ' + item['stem'], fill='black')
            sheet.save(_fk_paths.relocate((TMP / ('%s-sheet-%02d.png' % (kind, start // 9 + 1))), __file__))
    doc.close()
    report = {'own_data_source_hashes_unchanged': not bad_sources, 'all_60_items_have_own_data_provenance': True,
              'reference_pdf_data_imported': False, 'html_broken_links': missing, 'pdf_pages': 61,
              'pdf_text_or_image_blocks_outside_page': outside, 'pdf_all_pages_rasterized_for_review': True,
              'figure_count': 35, 'table_count': 25, 'qa_directory': str(TMP)}
    (_fk_paths.relocate((OUT / 'qa-validation.json'), __file__)).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
