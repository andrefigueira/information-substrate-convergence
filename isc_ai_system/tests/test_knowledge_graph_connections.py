#!/usr/bin/env python3
"""
Test that knowledge graph connections are being formed properly
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.isc_ai.core import ISCCore


def test_connections():
    """Test that connections are formed between concepts"""
    print("Testing Knowledge Graph Connections...\n")
    
    # Initialize core
    core = ISCCore()
    core.session_active = True
    
    # Test 1: Simple co-occurrence
    print("Test 1: Co-occurrence connections")
    response1 = core.process_input("The cat and dog are playing together")
    
    # Check concepts were added
    concepts = list(core.knowledge_graph.graph.nodes())
    print(f"Concepts found: {concepts}")
    
    # Check connections
    edges = list(core.knowledge_graph.graph.edges(data=True))
    print(f"Connections formed: {len(edges)}")
    for edge in edges[:5]:  # Show first 5
        print(f"  {edge[0]} <-> {edge[1]} (weight: {edge[2].get('weight', 1.0):.2f})")
    
    # Test 2: Check specific connection
    print("\nTest 2: Checking specific connections")
    if 'cat' in core.knowledge_graph.graph and 'dog' in core.knowledge_graph.graph:
        if core.knowledge_graph.graph.has_edge('cat', 'dog'):
            weight = core.knowledge_graph.graph['cat']['dog']['weight']
            print(f"✓ 'cat' and 'dog' are connected (weight: {weight:.2f})")
        else:
            print("✗ 'cat' and 'dog' are NOT connected")
    
    # Test 3: Related concepts
    print("\nTest 3: Getting related concepts")
    related_to_cat = core.knowledge_graph.get_related_concepts('cat', k=5)
    print(f"Concepts related to 'cat': {related_to_cat}")
    
    # Test 4: Multiple inputs to strengthen connections
    print("\nTest 4: Strengthening connections")
    response2 = core.process_input("The cat is a pet animal")
    response3 = core.process_input("Dogs and cats are common pets")
    
    # Check if connections strengthened
    if core.knowledge_graph.graph.has_edge('cat', 'pet'):
        weight = core.knowledge_graph.graph['cat']['pet']['weight']
        print(f"'cat' <-> 'pet' connection weight: {weight:.2f}")
    
    # Test 5: Response connections
    print("\nTest 5: Input-Response connections")
    response4 = core.process_input("What is consciousness?")
    
    # Show some connections
    all_edges = list(core.knowledge_graph.graph.edges(data=True))
    print(f"\nTotal concepts: {len(core.knowledge_graph.graph.nodes())}")
    print(f"Total connections: {len(all_edges)}")
    
    # Show concepts with most connections
    print("\nMost connected concepts:")
    node_degrees = [(node, core.knowledge_graph.graph.degree(node)) 
                    for node in core.knowledge_graph.graph.nodes()]
    node_degrees.sort(key=lambda x: x[1], reverse=True)
    
    for concept, degree in node_degrees[:10]:
        print(f"  {concept}: {degree} connections")
    
    # Test the get_central_concepts method
    print("\nCentral concepts:")
    central = core.knowledge_graph.get_central_concepts(k=5)
    for concept in central:
        print(f"  - {concept}")
    
    print("\n✅ Connection tests complete!")


if __name__ == "__main__":
    test_connections()