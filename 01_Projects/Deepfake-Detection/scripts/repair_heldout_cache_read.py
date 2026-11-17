"""One-time admin ACL repair, limited to the user-approved legacy heldout cache."""

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

import ctypes,subprocess,json,hashlib,traceback,time
from pathlib import Path
ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]
TARGET=_fk_paths.relocate((ROOT/'runs/heldout-001/fk/cache'), __file__)
OUT=_fk_paths.relocate((ROOT/'runs/heldout-cache-read-repair-001'), __file__)
OUT.mkdir(exist_ok=True)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def command(args,name):
    value=subprocess.run(args,capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW)
    (_fk_paths.relocate((OUT/name), __file__)).write_bytes(value.stdout+value.stderr)
    if value.returncode:raise RuntimeError(f'ACL operation failed: {name}; exit {value.returncode}')
try:
    if TARGET.resolve()!=_fk_paths.relocate((Path('C:/Users/teeku/Desktop/fk/Deepfake-Detection/runs/heldout-001/fk/cache')), __file__):raise ValueError('Unexpected target')
    if not ctypes.windll.shell32.IsUserAnAdmin():raise RuntimeError('Administrator token required')
    manifest=_fk_paths.relocate((TARGET/'training-manifest.json'), __file__);before=sha(manifest)
    if not (_fk_paths.relocate((OUT/'acl-before.txt'), __file__)).exists():
        command(['icacls.exe',str(TARGET),'/save',str(_fk_paths.relocate((OUT/'acl-before.txt'), __file__)),'/T','/C'],'acl-backup.log')
    # Explicit grants are needed on legacy child objects with protected DACLs.
    command(['icacls.exe',str(TARGET),'/grant','*S-1-5-21-1502947017-976547090-1717846262-1001:(RX)','/T','/C'],'acl-repair-explicit.log')
    command(['icacls.exe',str(manifest)],'manifest-acl-after.log')
    after=sha(manifest)
    if before!=after:raise ValueError('Manifest content changed')
    result={'status':'FINISHED','target':str(TARGET),'manifest_sha256':after,'manifest_unchanged':True,'ownership_changed':False,'updated_unix':time.time()}
except Exception:
    result={'status':'FAILED','error':traceback.format_exc(),'updated_unix':time.time()}
(_fk_paths.relocate((OUT/'result.json'), __file__)).write_text(json.dumps(result,indent=2),encoding='utf-8')
if result['status']=='FINISHED':
    # Scheduler retains each task's Limited user token even when requested here.
    started=time.time()
    command(['schtasks.exe','/Run','/TN','FK-Remaining-Ablation-Evaluation-001'],'restart-evaluation.log')
    for _ in range(60):
        time.sleep(2)
        try:state=json.loads((_fk_paths.relocate((ROOT/'runs/remaining-ablation-evaluation-001/progress.json'), __file__)).read_text())
        except (OSError,ValueError):continue
        if state.get('updated_unix',0)>started:
            if state.get('status') in ('RUNNING','FINISHED'):
                command(['schtasks.exe','/Run','/TN','FK-Remaining-Supplement-001'],'restart-supplement.log')
            break
