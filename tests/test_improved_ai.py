#!/usr/bin/env python3
"""
Test script for improved AI responses after fixing concept extraction
"""

from isc.neuromorphic_core import NeuromorphicISCCore

def test_improved_responses():
    """Test the AI responses with the improved concept extraction"""

    print("🧠 Testing improved neuromorphic AI responses...")

    # Initialize the core
    core = NeuromorphicISCCore()
    core.verbose = True

    # Test queries that previously gave templated responses
    test_queries = [
        "hi",
        "i wanted to chat and meet you",
        "well just saying hi",
        "ok",
        "well im trying to understand you",
        "stuff",
        "what do you think about consciousness?",
        "how are you feeling today?"
    ]

    print(f"📊 Current substrate: {core.substrate.graph.number_of_nodes()} concepts")
    print(f"🎓 Training sessions: {core.metrics.get('training_sessions', 0)}")
    print()

    for i, query in enumerate(test_queries, 1):
        print(f"[{i}] Testing: '{query}'")
        response = core.process_input(query)
        print(f"    Response: {response}")
        print()

    print("✅ Testing complete!")

if __name__ == "__main__":
    test_improved_responses()