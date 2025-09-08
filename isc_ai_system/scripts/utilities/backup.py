#!/usr/bin/env python
"""Backup script for ISC storage"""

import sys
sys.path.insert(0, 'src')
from isc_ai.storage import StorageManager

storage = StorageManager('isc_storage')
path = storage.backup()
print(f'Backup created: {path}')