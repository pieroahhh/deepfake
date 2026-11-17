"""Finalize the visually reviewed own-data pack and create its ZIP archive."""

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

import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = _fk_paths.workspace_root(__file__)
OUT = _fk_paths.relocate((ROOT / 'Deepfake-Detection/runs/paper-style-visuals-001'), __file__)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    catalog = json.loads((_fk_paths.relocate((OUT / 'catalog.json'), __file__)).read_text(encoding='utf-8'))
    qa = json.loads((_fk_paths.relocate((OUT / 'qa-validation.json'), __file__)).read_text(encoding='utf-8'))
    assert qa['own_data_source_hashes_unchanged'] and not qa['reference_pdf_data_imported']
    assert not qa['html_broken_links'] and not qa['pdf_text_or_image_blocks_outside_page']
    assert catalog['counts'] == {'table': 25, 'figure': 35}
    for name in ('check_paper_visuals.py', 'package_paper_visuals.py'):
        shutil.copy2(_fk_paths.relocate((Path(__file__)), __file__).with_name(name), _fk_paths.relocate((OUT / 'reproduce' / name), __file__))
    stamp = datetime.now(timezone(timedelta(hours=7))).isoformat(timespec='seconds')
    review = {'reviewed_at_bangkok': stamp, 'all_35_figure_thumbnails_inspected': True,
              'all_25_table_thumbnails_inspected': True, 'all_61_pdf_page_layouts_reviewed': True,
              'detailed_pages_checked': [1, 3, 7, 26, 41, 50, 59, 61],
              'corrected': ['Table number/title spacing', 'Manipulation-type legend placed outside data', 'PDF cover heading fit'],
              'reference_use': 'Style only; reference PDFs contain no numerical sources for this pack',
              'data_origin': 'Own retained experiments only; no new training, model inference or threshold fitting'}
    (_fk_paths.relocate((OUT / 'visual-review.json'), __file__)).write_text(json.dumps(review, indent=2) + '\n', encoding='utf-8')
    complete = {'counts': catalog['counts'], 'source_integrity': True, 'visually_reviewed': True,
                'finished_bangkok': stamp,
                'files': {p.relative_to(OUT).as_posix(): digest(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'complete.json'}}
    (_fk_paths.relocate((OUT / 'complete.json'), __file__)).write_text(json.dumps(complete, indent=2) + '\n', encoding='utf-8')
    archive = OUT.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in sorted(OUT.rglob('*')):
            if path.is_file():
                bundle.write(path, path.relative_to(OUT.parent).as_posix())
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        assert not any(name.endswith(('Paper22.pdf', 'Paper16.pdf')) for name in bundle.namelist())
    print(json.dumps({'archive': str(archive), 'archive_mb': round(_fk_paths.relocate((archive.stat().st_size / 1e6), __file__), 2),
                      'archive_sha256': digest(archive), 'counts': catalog['counts'], 'files': len(complete['files']),
                      'own_sources': len(json.loads((_fk_paths.relocate((OUT / 'source-manifest.json'), __file__)).read_text())['files'])}, indent=2))


if __name__ == '__main__':
    main()
