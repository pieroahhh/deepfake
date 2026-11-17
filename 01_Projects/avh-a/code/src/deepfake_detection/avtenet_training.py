"""Compatibility exports; implementation lives in baselines.avtenet.legacy_training."""
from baselines.avtenet.legacy_training import *  # noqa: F401,F403
from baselines.avtenet.legacy_training import __dict__ as _implementation

def __getattr__(name):
    try:
        return _implementation[name]
    except KeyError:
        raise AttributeError(name) from None
