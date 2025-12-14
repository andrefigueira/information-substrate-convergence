#!/usr/bin/env python
"""Backup script for ISC storage"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
from isc.storage import StorageManager

storage = StorageManager('isc_storage')
path = storage.backup()
print(f'Backup created: {path}')