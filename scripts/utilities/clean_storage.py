#!/usr/bin/env python
"""Clean storage script"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
from isc.storage import StorageManager

storage = StorageManager('isc_storage')
storage._cleanup_old_versions()
storage._cleanup_old_backups()
print('Storage cleaned.')