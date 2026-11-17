"""Python 3.8-compatible dependency, native-extension and AV-HuBERT registration check."""

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


import argparse
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def check(repository):
    result={'python':sys.version.split()[0],'checks':{},'status':'NOT_READY'}
    if sys.version_info[:2]!=(3,8):
        result['checks']['python']={'status':'FAILED','reason':'Python 3.8 is required'}
        return result
    modules=('torch','torchaudio','numpy','scipy','cv2','python_speech_features',
             'sentencepiece','editdistance','hydra','omegaconf','fairseq')
    for name in modules:
        try:
            module=importlib.import_module(name)
            result['checks'][name]={'status':'PASSED','version':str(getattr(module,'__version__','imported'))}
        except Exception as error:
            result['checks'][name]={'status':'FAILED','reason':str(error),'error_type':type(error).__name__}
    if any(value['status']!='PASSED' for value in result['checks'].values()):
        return result
    for name in ('fairseq.data.data_utils_fast','fairseq.data.token_block_utils_fast'):
        try:
            importlib.import_module(name)
            result['checks'][name]={'status':'PASSED'}
        except Exception as error:
            result['checks'][name]={'status':'FAILED','reason':str(error)}
    try:
        from fairseq import utils
        # Upstream selects standalone debug imports when len(sys.argv) == 1.
        # Registration needs its package-relative imports, including when this
        # checker is invoked without CLI options.
        original_argv = sys.argv
        try:
            sys.argv = list(original_argv) + ['--avhubert-package-registration']
            utils.import_user_module(SimpleNamespace(user_dir=str(_fk_paths.relocate((Path(repository).resolve()/'avhubert'), __file__))))
        finally:
            sys.argv = original_argv
        result['checks']['avhubert_registration']={'status':'PASSED'}
    except Exception as error:
        result['checks']['avhubert_registration']={'status':'FAILED','reason':str(error),'error_type':type(error).__name__}
    if all(value['status']=='PASSED' for value in result['checks'].values()):
        result['status']='ENVIRONMENT_IMPORTS_VERIFIED_CHECKPOINT_NOT_TESTED'
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository',default=str(_fk_paths.relocate((Path(__file__).resolve().parents[1]/'external/av_hubert'), __file__)))
    parser.add_argument('--output')
    args=parser.parse_args()
    result=check(args.repository)
    text=json.dumps(result,indent=2)+'\n'
    if args.output:
        path=_fk_paths.relocate((Path(args.output)), __file__);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
    print(text,end='')
    raise SystemExit(0 if result['status']!='NOT_READY' else 2)


if __name__=='__main__':main()
