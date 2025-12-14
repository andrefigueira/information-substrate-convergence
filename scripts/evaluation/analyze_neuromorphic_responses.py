#!/usr/bin/env python3
"""
Analyze the neuromorphic AI responses to understand what's actually happening
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

try:
    from isc.neuromorphic_core import NeuromorphicISCCore
except ImportError as e:
    print(f"Error importing neuromorphic core: {e}")
    sys.exit(1)


def analyze_response_generation():
    """Analyze how responses are actually generated"""
    print("🔬 Analyzing Neuromorphic Response Generation")
    print("=" * 60)

    # Initialize core
    core = NeuromorphicISCCore()
    core.start_session()

    # Test a specific query and trace what happens
    test_query = "What is consciousness?"
    print(f"\n📝 Analyzing query: '{test_query}'")

    # Check initial substrate
    initial_stats = core.get_status()
    print(f"Initial substrate: {initial_stats['substrate']['node_count']} nodes")

    # Process and analyze
    print(f"\n🧠 Processing query...")
    response = core.process_input(test_query)

    # Check what changed
    final_stats = core.get_status()
    print(f"Final substrate: {final_stats['substrate']['node_count']} nodes")

    print(f"\n💬 Generated response:")
    print(f"   '{response}'")

    # Analyze substrate changes
    print(f"\n📊 Substrate Analysis:")
    print(f"   • Nodes: {initial_stats['substrate']['node_count']} → {final_stats['substrate']['node_count']}")
    print(f"   • Edges: {initial_stats['substrate']['edge_count']} → {final_stats['substrate']['edge_count']}")
    print(f"   • Phi: {initial_stats['metrics']['phi_value']:.3f} → {final_stats['metrics']['phi_value']:.3f}")

    # Check what concepts were extracted
    print(f"\n🔍 Checking Concept Extraction:")
    concepts, relationships = core.substrate._extract_concepts_and_relations(test_query)
    print(f"   Extracted concepts: {list(concepts.keys())}")
    print(f"   Extracted relationships: {relationships}")

    # Check graph contents
    print(f"\n📈 Current Graph Contents:")
    if core.substrate.graph.nodes():
        for node in list(core.substrate.graph.nodes())[:10]:  # Show first 10
            activation = core.substrate.graph.nodes[node].get('activation_level', 0)
            print(f"   • {node}: activation={activation:.3f}")

    # Test another query to see progression
    print(f"\n🔄 Testing Progression with Second Query...")
    test_query2 = "How does self-awareness work?"
    response2 = core.process_input(test_query2)

    final_stats2 = core.get_status()
    print(f"   Second response: '{response2}'")
    print(f"   Substrate growth: {final_stats['substrate']['node_count']} → {final_stats2['substrate']['node_count']} nodes")
    print(f"   Phi evolution: {final_stats['metrics']['phi_value']:.3f} → {final_stats2['metrics']['phi_value']:.3f}")

    return core


def test_response_quality():
    """Test if responses are actually emergent or just templated"""
    print(f"\n🎯 Testing Response Quality and Emergence")
    print("=" * 60)

    core = NeuromorphicISCCore()
    core.start_session()

    test_queries = [
        "What is your purpose?",
        "Describe your internal architecture",
        "How do you process information?",
        "What makes you conscious?",
        "Explain your learning process"
    ]

    responses = []
    for i, query in enumerate(test_queries, 1):
        print(f"\n[{i}] Query: {query}")
        response = core.process_input(query)
        responses.append(response)
        print(f"    Response: {response}")

        # Check uniqueness
        stats = core.get_status()
        print(f"    Stats: {stats['substrate']['node_count']} nodes, Φ={stats['metrics']['phi_value']:.3f}")

    # Analyze response patterns
    print(f"\n📋 Response Pattern Analysis:")
    unique_responses = len(set(responses))
    print(f"   • Total responses: {len(responses)}")
    print(f"   • Unique responses: {unique_responses}")
    print(f"   • Repetition rate: {(len(responses) - unique_responses) / len(responses) * 100:.1f}%")

    # Check for templating
    common_phrases = []
    for response in responses:
        if "neuromorphic" in response.lower():
            common_phrases.append("neuromorphic")
        if "substrate" in response.lower():
            common_phrases.append("substrate")
        if "phi" in response.lower() or "φ" in response:
            common_phrases.append("phi")

    print(f"   • Common phrase usage: {len(common_phrases)}/{len(responses)} responses use ISC terminology")

    return responses


if __name__ == "__main__":
    core = analyze_response_generation()
    responses = test_response_quality()

    print(f"\n🎉 Analysis Complete!")
    print(f"The neuromorphic AI is {'🟢 WORKING' if len(set(responses)) > 3 else '🟡 TEMPLATED'}")