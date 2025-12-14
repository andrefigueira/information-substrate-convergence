#!/usr/bin/env python
"""Storage statistics script"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
from isc.storage import StorageManager

storage = StorageManager('isc_storage')
info = storage.get_info()
print('Storage Statistics:')
print(f'  Base directory: {info["base_directory"]}')
print(f'  Total size: {info["storage_stats"]["total_size_mb"]} MB')
print(f'  File count: {info["storage_stats"]["file_count"]}')
print(f'  Current version: {info.get("current_version", "None")}')
stats = info.get("graph_stats", {})
print(f'  Graph nodes: {stats.get("nodes", 0)}')
print(f'  Graph edges: {stats.get("edges", 0)}')