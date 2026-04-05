# Root conftest.py
# Patches pluggy to skip broken third-party pytest plugins (web3/eth_typing
# version mismatch) before they crash test collection.

import pluggy._manager as _pm

_orig_load = _pm.PluginManager.load_setuptools_entrypoints


def _safe_load(self, group, name=None):
    try:
        return _orig_load(self, group, name)
    except ImportError:
        pass


_pm.PluginManager.load_setuptools_entrypoints = _safe_load
