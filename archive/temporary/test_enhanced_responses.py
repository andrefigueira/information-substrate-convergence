#!/usr/bin/env python3
"""
Test the enhanced response generation
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.isc_ai.core import ISCCore


def test_responses():
    print("Testing Enhanced Response Generation\n")
    print("=" * 60)
    
    # Initialize core
    core = ISCCore()
    core.session_active = True
    
    # Test inputs
    test_inputs = [
        "Hello, how are you?",
        "What is consciousness?",
        "Can you explain how your understanding develops?",
        "Tell me about the connections you see between concepts.",
        "How does information integration relate to awareness?"
    ]
    
    print("Starting with fresh system...\n")
    
    for i, user_input in enumerate(test_inputs):
        print(f"User: {user_input}")
        
        response = core.process_input(user_input)
        
        # Get metrics
        phi = core.metrics.get("phi_value", 0.0)
        
        print(f"ISC: {response}")
        print(f"[Φ = {phi:.4f}]")
        print("-" * 60)
        print()
    
    # Show knowledge graph stats
    print("\nKnowledge Graph Statistics:")
    print(f"Total concepts: {len(core.knowledge_graph.graph.nodes())}")
    print(f"Total connections: {len(core.knowledge_graph.graph.edges())}")
    
    # Show some connections
    print("\nSample connections:")
    edges = list(core.knowledge_graph.graph.edges(data=True))[:5]
    for edge in edges:
        print(f"  {edge[0]} <-> {edge[1]} (weight: {edge[2].get('weight', 1.0):.2f})")


if __name__ == "__main__":
    test_responses()