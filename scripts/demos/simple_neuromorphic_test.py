#!/usr/bin/env python3
"""
Simple test of the neuromorphic ISC system
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


def test_simple_conversation():
    """Test simple conversation capabilities"""
    print("🧠 Testing Neuromorphic ISC Conversation")
    print("=" * 50)

    # Initialize core
    core = NeuromorphicISCCore()
    core.start_session()

    # Test queries
    test_queries = [
        "What are you?",
        "How do you think?",
        "Explain consciousness",
        "What is your substrate?",
        "Tell me about information integration"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n[{i}] Human: {query}")
        response = core.process_input(query)
        print(f"    AI: {response}")

        # Show substrate stats
        stats = core.get_status()
        print(f"    [Φ={stats['metrics']['phi_value']:.3f}, "
              f"Nodes={stats['substrate']['node_count']}, "
              f"Communities={stats['substrate']['community_count']}]")

    # Final stats
    print(f"\n🔬 Final Analysis:")
    final_stats = core.get_status()
    print(f"   • Total conversations: {final_stats['substrate']['conversation_count']}")
    print(f"   • Substrate nodes: {final_stats['substrate']['node_count']}")
    print(f"   • Substrate edges: {final_stats['substrate']['edge_count']}")
    print(f"   • Communities: {final_stats['substrate']['community_count']}")
    print(f"   • Final Φ (phi): {final_stats['metrics']['phi_value']:.3f}")
    print(f"   • Context loaded: {final_stats['context_loaded']}")

    # Test save/load
    print(f"\n💾 Testing Persistence...")
    save_path = "test_substrate_state.json"
    core.save_state(save_path)
    print(f"   Saved state to {save_path}")

    # Load it back
    new_core = NeuromorphicISCCore()
    new_core.load_state(save_path)
    new_stats = new_core.get_status()
    print(f"   Loaded state: {new_stats['substrate']['node_count']} nodes, "
          f"{new_stats['substrate']['edge_count']} edges")

    print(f"\n✅ Test completed successfully!")


if __name__ == "__main__":
    test_simple_conversation()