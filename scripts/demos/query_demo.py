#!/usr/bin/env python
"""Query demonstration script"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
from isc.storage import StorageManager

print('=== Graph Query Demo ===\n')
storage = StorageManager('demo_storage')

# Load existing or create new
try:
    storage.load()
    print('Loaded existing graph')
except:
    print('Creating new graph...')
    # Create a more complex graph
    nodes = ['mind', 'brain', 'consciousness', 'awareness', 
             'thought', 'memory', 'learning', 'pattern']
    
    for node in nodes:
        storage.graph_db.add_node(node)
    
    # Add connections
    connections = [
        ('mind', 'consciousness'), ('brain', 'mind'),
        ('consciousness', 'awareness'), ('thought', 'mind'),
        ('memory', 'brain'), ('learning', 'memory'),
        ('pattern', 'learning'), ('pattern', 'thought')
    ]
    
    for src, tgt in connections:
        storage.graph_db.add_edge(src, tgt)
    
    storage.save('query_demo', 'Complex graph for queries')

print('\nExample queries:\n')

queries = [
    'find node mind',
    'path from brain to awareness',
    'neighbors of consciousness',
    'contains "learn"',
    'central nodes',
    'clusters',
    'statistics'
]

for query in queries:
    print(f'Query: {query}')
    results = storage.query(query)
    
    if results:
        result = results[0]
        print(f'  Type: {result.get("type", "unknown")}')
        
        if result['type'] == 'path' and 'nodes' in result:
            print(f'  Path: {" -> ".join(result["nodes"])}')
        elif result['type'] == 'neighbors' and 'neighbors' in result:
            print(f'  Found {len(result["neighbors"])} neighbors')
        elif result['type'] == 'statistics':
            stats = result.get('basic_stats', {})
            print(f'  Nodes: {stats.get("nodes", 0)}, Edges: {stats.get("edges", 0)}')
    else:
        print('  No results')
    print()

print('\nQuery syntax help:')
print(storage.query_engine.explain_query_syntax()[:500] + '...')