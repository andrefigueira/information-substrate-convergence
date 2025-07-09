#!/usr/bin/env python
"""Clean storage script"""

import sys
sys.path.insert(0, 'src')
from isc_ai.storage import StorageManager

storage = StorageManager('isc_storage')
storage._cleanup_old_versions()
storage._cleanup_old_backups()
print('Storage cleaned.')