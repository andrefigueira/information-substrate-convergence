#!/usr/bin/env python
"""Export graph script"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
from isc.storage import StorageManager

format_type = sys.argv[1] if len(sys.argv) > 1 else 'json'

storage = StorageManager('isc_storage')
try:
    storage.load()
    
    if format_type == 'all':
        formats = ['json', 'graphml', 'text', 'adjacency', 'edgelist']
        for fmt in formats:
            path = storage.export(fmt)
            print(f'{fmt}: {path}')
    else:
        path = storage.export(format_type)
        print(f'Exported to: {path}')
except Exception as e:
    print(f'Export failed: {e}')