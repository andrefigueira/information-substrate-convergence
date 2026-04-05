#!/usr/bin/env python
"""
Test runner for the ISC project.

Works around a broken web3/eth_typing entrypoint that prevents pytest from
starting in environments where web3 is installed with a mismatched eth_typing.
Usage: python run_tests.py [pytest args...]
"""
import sys

import pluggy._manager as _pm

_orig_load = _pm.PluginManager.load_setuptools_entrypoints


def _safe_load(self, group, name=None):
    try:
        return _orig_load(self, group, name)
    except ImportError:
        pass


_pm.PluginManager.load_setuptools_entrypoints = _safe_load

import pytest  # noqa: E402

sys.exit(pytest.main(sys.argv[1:] or ["tests/", "-v"]))
