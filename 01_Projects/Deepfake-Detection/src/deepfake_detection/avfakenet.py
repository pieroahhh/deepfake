"""Compatibility exports; implementation lives in baselines.avfakenet.model."""
from baselines.avfakenet.model import *  # noqa: F401,F403
from baselines.avfakenet.model import __dict__ as _implementation

def __getattr__(name):
    try:
        return _implementation[name]
    except KeyError:
        raise AttributeError(name) from None
