"""Portable workspace paths and read-time relocation of historical path strings.

No filesystem aliases are created. Stored experiment metadata stays unchanged.
"""
import json
import os
from functools import lru_cache
from pathlib import Path, PurePath


def workspace_root(start=None):
    override = os.environ.get('FK_WORKSPACE_ROOT')
    if override:
        root = Path(override).resolve()
        if not (root / 'workspace-layout.json').is_file():
            raise FileNotFoundError('FK_WORKSPACE_ROOT has no workspace-layout.json')
        return root
    start = Path(start or __file__).resolve()
    for parent in (start, *start.parents):
        if (parent / 'workspace-layout.json').is_file():
            return parent
    if os.name!='nt' and Path('/inputs/workspace-layout.json').is_file():return Path('/inputs')
    raise FileNotFoundError('Cannot find workspace-layout.json above ' + str(start))


def map_relative(value, mappings):
    value = value.replace('\\', '/').strip('/')
    folded = value.casefold()
    for old, new in sorted(mappings.items(), key=lambda item: len(item[0]), reverse=True):
        prefix = old.replace('\\', '/').strip('/')
        if folded == prefix.casefold() or folded.startswith(prefix.casefold() + '/'):
            return new.rstrip('/') + value[len(prefix):]
    return value


@lru_cache(maxsize=8)
def _layout(root):
    return json.loads((Path(root) / 'workspace-layout.json').read_text(encoding='utf-8'))


def relocate(value, caller=None):
    """Resolve known old paths; leave numbers and unrelated relative paths alone."""
    if not isinstance(value, PurePath):
        return value
    root = workspace_root(caller)
    layout = _layout(str(root))
    if layout.get('state') != 'migrated':
        return value
    text = os.path.normpath(str(value)).replace('\\', '/')
    root_text = root.as_posix().rstrip('/')
    is_absolute = value.is_absolute()
    legacy_root=str(layout.get('original_root','')).replace('\\','/').rstrip('/')
    if legacy_root and text.casefold().startswith(legacy_root.casefold()+'/'):
        relative=text[len(legacy_root)+1:]
    elif os.name!='nt' and (text=='/workspace' or text.startswith('/workspace/')):
        relative='avh-a/code'+text[len('/workspace'):]
    elif text.casefold().startswith(root_text.casefold() + '/'):
        relative = text[len(root_text) + 1:]
    elif is_absolute:
        return value
    else:
        relative = text
    mapped = map_relative(relative, layout['path_mappings'])
    if mapped == relative:
        for context in ('01_Projects/', '06_Tools/'):
            if relative.startswith(context):
                candidate=relative[len(context):]
                result=map_relative(candidate,layout['path_mappings'])
                if result!=candidate:mapped=result;break
    if mapped != relative:
        return root / mapped
    if not is_absolute and caller:
        caller_text = str(Path(caller).resolve()).replace('\\', '/').casefold()
        for project, code_root in layout['project_roots'].items():
            if caller_text.startswith((root / code_root).as_posix().casefold() + '/'):
                for name in ('runs', 'checkpoints', '.cache', '.venv', '.venv-gpu', '.venv-baselines'):
                    if relative == name or relative.startswith(name + '/'):
                        key = layout['original_project_roots'][project] + '/' + relative
                        return root / map_relative(key, layout['path_mappings'])
    return value


def relocate_value(value,caller=None):
    """Normalize file arguments passed directly to external loaders."""
    if isinstance(value,PurePath):return relocate(value,caller)
    if isinstance(value,str):
        original=Path(value);result=relocate(original,caller)
        if result!=original:return str(result)
    return value


def workspace_path(name, *parts, start=None):
    root = workspace_root(start)
    layout = _layout(str(root))
    return root / layout['roots'][name] / Path(*parts)
