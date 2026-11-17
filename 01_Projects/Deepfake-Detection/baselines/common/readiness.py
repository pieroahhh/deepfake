"""Readiness distinguishes missing dependencies, verified weights and actual execution."""
import importlib
import sys


def check_runtime(model, assets=None, device='cpu'):
    checks={}
    for name in ('torch','torchvision','numpy','av','cv2','librosa','facenet_pytorch','sklearn'):
        try:
            imported=importlib.import_module(name)
            checks[name]={'ok':True,'version':str(getattr(imported,'__version__','installed'))}
        except Exception as error:
            checks[name]={'ok':False,'reason':str(error)}
    if all(v['ok'] for v in checks.values()):
        try:
            from .media import detector
            _,info=detector(device)
            checks['detector']={'ok':True,'provenance':info}
        except Exception as error:
            checks['detector']={'ok':False,'reason':str(error)}
    if model=='avtenet':
        from baselines.avtenet.assets import check_assets
        checks.update(check_assets(assets))
    return {'model':model,'python':sys.version,'ready':all(v['ok'] for v in checks.values()),
            'checks':checks,'meaning':'dependencies and asset validation; real-media training is a separate test'}
