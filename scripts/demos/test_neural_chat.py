#!/usr/bin/env python3
"""
Test the ISC system with the neural language model.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from isc.neuromorphic_core import NeuromorphicISCCore


def main():
    print("=" * 60)
    print("ISC Neural Chat Test")
    print("=" * 60)

    print("\nInitializing ISC system...")
    core = NeuromorphicISCCore()
    core.start_session()

    status = core.get_status()
    print(f"Concepts: {status['substrate']['node_count']}")
    print(f"Neural LM loaded: {core.substrate.neural_lm is not None}")

    print("\n" + "=" * 60)
    print("Conversation Test")
    print("=" * 60)

    test_queries = [
        "Hello!",
        "How are you?",
        "What are you?",
        "Are you conscious?",
        "What is the meaning of life?",
        "I'm feeling a bit sad today",
        "Can you help me think through a problem?",
        "What do you think about artificial intelligence?",
        "Tell me about yourself",
        "Goodbye"
    ]

    for query in test_queries:
        response = core.process_input(query)
        print(f"\nYou: {query}")
        print(f"ISC: {response}")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
