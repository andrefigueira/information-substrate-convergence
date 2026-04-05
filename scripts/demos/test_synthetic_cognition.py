#!/usr/bin/env python3
"""
Test the Synthetic Cognition Platform

This demonstrates the revolutionary features:
1. Cognitive Architecture Modeling - learning HOW users think
2. Cognitive Composability - combining cognitive profiles
3. Continuous Cross-Domain Learning - never-forgetting accumulation
4. Novel Reasoning Evolution - discovering new thinking patterns
5. Insight Translation - personalized explanations
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from isc.neuromorphic_core import NeuromorphicISCCore


def main():
    print("=" * 70)
    print("Synthetic Cognition Platform Test")
    print("=" * 70)

    # Initialize system
    print("\nInitializing ISC with Cognitive Architecture...")
    core = NeuromorphicISCCore()
    core.start_session()

    # Get status
    status = core.get_status()
    print(f"\nSystem Status:")
    print(f"  Cognitive Architecture: {status.get('cognitive_architecture_available', False)}")
    print(f"  Cognitive Profiles: {status.get('cognitive_profiles_count', 0)}")
    print(f"  Domain Count: {status.get('domain_count', 0)}")
    print(f"  Evolution Generation: {status.get('evolution_generation', 0)}")

    # Test 1: Cognitive Architecture Modeling
    print("\n" + "=" * 70)
    print("Test 1: Cognitive Architecture Modeling")
    print("=" * 70)

    # Simulate interactions from different users
    user_alice = "alice"
    user_bob = "bob"

    alice_queries = [
        "Why does this system work?",
        "What is the logical basis for consciousness?",
        "How can we analyze this systematically?",
        "What evidence supports this theory?"
    ]

    bob_queries = [
        "This feels like something new",
        "I have a sense that there's more here",
        "What if we tried something different?",
        "Imagine the possibilities!"
    ]

    print("\nAlice's interactions (analytical style):")
    for query in alice_queries:
        response = core.process_input(query, user_id=user_alice)
        print(f"  Q: {query[:50]}...")

    print("\nBob's interactions (creative/intuitive style):")
    for query in bob_queries:
        response = core.process_input(query, user_id=user_bob)
        print(f"  Q: {query[:50]}...")

    # Get cognitive summaries
    print("\nCognitive Summaries:")
    print(f"\n{core.get_cognitive_summary(user_alice)}")
    print(f"\n{core.get_cognitive_summary(user_bob)}")

    # Test 2: Cognitive Composability
    print("\n" + "=" * 70)
    print("Test 2: Cognitive Composability")
    print("=" * 70)

    compatibility = core.compute_cognitive_compatibility(user_alice, user_bob)
    print(f"\nCognitive compatibility between Alice and Bob: {compatibility:.2%}")

    composed = core.compose_cognitive_profiles([user_alice, user_bob])
    if composed:
        print(f"\nComposed profile cognitive style:")
        print(f"  Analytical: {composed.get('analytical_tendency', 0):.2f}")
        print(f"  Intuitive: {composed.get('intuitive_tendency', 0):.2f}")
        print(f"  Systematic: {composed.get('systematic_tendency', 0):.2f}")
        print(f"  Creative: {composed.get('creative_tendency', 0):.2f}")

    # Test 3: Continuous Cross-Domain Learning
    print("\n" + "=" * 70)
    print("Test 3: Continuous Cross-Domain Learning")
    print("=" * 70)

    domain_stats = core.get_domain_knowledge()
    if 'error' not in domain_stats:
        print(f"\nLearned domains:")
        for domain, stats in domain_stats.items():
            print(f"  {domain}: {stats.get('concept_count', 0)} concepts, "
                  f"consolidation: {stats.get('consolidation_level', 0):.2%}")

    # Test knowledge recall
    recall = core.recall_knowledge("consciousness thinking")
    if 'error' not in recall:
        print(f"\nKnowledge recall for 'consciousness thinking':")
        print(f"  Primary domain: {recall.get('primary_domain', 'unknown')}")
        print(f"  Related concepts: {len(recall.get('related_concepts', []))}")
        print(f"  Cross-domain connections: {len(recall.get('cross_domain_connections', []))}")

    # Test 4: Novel Reasoning Evolution
    print("\n" + "=" * 70)
    print("Test 4: Novel Reasoning Evolution")
    print("=" * 70)

    problem = "How can we create artificial consciousness?"

    # Get initial strategy
    strategy = core.generate_novel_reasoning(problem)
    if 'error' not in strategy:
        print(f"\nBest reasoning strategy for: '{problem}'")
        print(f"  Strategy ID: {strategy.get('strategy_id', 'unknown')}")
        print(f"  Reasoning steps: {' -> '.join(strategy.get('reasoning_steps', []))}")
        print(f"  Description: {strategy.get('description', '')[:100]}...")

    # Evolve strategies
    print("\nEvolving reasoning strategies...")
    evolution_result = core.evolve_reasoning([problem], generations=3)
    if 'error' not in evolution_result:
        print(f"  Evolved {evolution_result.get('generations_evolved', 0)} generations")
        print(f"  Best fitness: {evolution_result.get('best_fitness', 0):.3f}")

    # Get improved strategy
    strategy = core.generate_novel_reasoning(problem)
    if 'error' not in strategy:
        print(f"\nImproved reasoning strategy:")
        print(f"  Generation: {strategy.get('generation', 0)}")
        print(f"  Fitness: {strategy.get('fitness', 0):.3f}")
        print(f"  Cognitive style: ", end="")
        style = strategy.get('cognitive_style', {})
        print(f"analytical={style.get('analytical', 0):.2f}, "
              f"creative={style.get('creative', 0):.2f}")

    # Test 5: Full conversation with cognitive architecture
    print("\n" + "=" * 70)
    print("Test 5: Full Conversation")
    print("=" * 70)

    test_queries = [
        "Hello, what can you do?",
        "Are you conscious?",
        "What have you learned from our conversation?"
    ]

    for query in test_queries:
        response = core.process_input(query, user_id="test_user")
        print(f"\nYou: {query}")
        print(f"ISC: {response[:200]}...")

    # Final status
    print("\n" + "=" * 70)
    print("Final Status")
    print("=" * 70)

    final_status = core.get_status()
    print(f"\nTotal interactions: {final_status['metrics']['total_interactions']}")
    print(f"Cognitive profiles: {final_status.get('cognitive_profiles_count', 0)}")
    print(f"Domain knowledge areas: {final_status.get('domain_count', 0)}")
    print(f"Evolution generation: {final_status.get('evolution_generation', 0)}")

    print("\n" + "=" * 70)
    print("Synthetic Cognition Platform Test Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
