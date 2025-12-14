#!/usr/bin/env python3
"""
Neuromorphic ISC Demo Script
Tests the enhanced substrate-driven AI with dual context integration
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

try:
    from isc.neuromorphic_core import NeuromorphicISCCore
except ImportError as e:
    print(f"Error importing neuromorphic core: {e}")
    print("Please ensure all dependencies are installed")
    sys.exit(1)


def print_banner():
    """Print startup banner"""
    print("=" * 80)
    print("🧠 NEUROMORPHIC ISC AI SYSTEM")
    print("Information Substrate Convergence - Dual Context Implementation")
    print("=" * 80)
    print()


def print_status(core):
    """Print current system status"""
    status = core.get_status()
    substrate = status['substrate']

    print(f"📊 SYSTEM STATUS")
    print(f"   Session: {'Active' if status['session_active'] else 'Inactive'}")
    print(f"   Φ (phi): {status['metrics']['phi_value']:.3f}")
    print(f"   Substrate: {substrate['node_count']} concepts, {substrate['edge_count']} edges")
    print(f"   Communities: {substrate['community_count']}")
    print(f"   Conversations: {substrate['conversation_count']}")
    print(f"   Context loaded: {status['context_loaded']}")
    print(f"   Embeddings: {'Available' if status['embeddings_available'] else 'Random fallback'}")

    if substrate.get('top_concepts'):
        print(f"   Top concepts: {', '.join(substrate['top_concepts'][:3])}")

    print()


def run_test_queries(core):
    """Run a series of test queries to demonstrate neuromorphic behavior"""

    test_queries = [
        "What is self-awareness?",
        "How does consciousness emerge?",
        "Explain your AI architecture",
        "What is information integration?",
        "Describe your substrate",
        "How do you process queries?",
        "What is the phi metric?",
        "Explain neuromorphic computation",
        "How do you learn and adapt?",
        "What makes you conscious?"
    ]

    print("🧪 RUNNING TEST QUERIES")
    print("Each query will create neuromorphic spikes that modify the substrate")
    print("-" * 80)

    results = []

    for i, query in enumerate(test_queries, 1):
        print(f"\n[Query {i}/10] {query}")
        print("-" * 50)

        start_time = time.time()
        response = core.process_input(query)
        processing_time = time.time() - start_time

        print(f"Response: {response}")
        print(f"Processing time: {processing_time:.3f}s")

        # Get updated status
        status = core.get_status()
        substrate = status['substrate']

        print(f"After spike: Φ={status['metrics']['phi_value']:.3f}, "
              f"Nodes={substrate['node_count']}, "
              f"Communities={substrate['community_count']}")

        results.append({
            'query': query,
            'response': response,
            'processing_time': processing_time,
            'phi': status['metrics']['phi_value'],
            'nodes': substrate['node_count'],
            'edges': substrate['edge_count'],
            'communities': substrate['community_count']
        })

        # Brief pause between queries
        time.sleep(0.5)

    return results


def analyze_results(results):
    """Analyze test results to show neuromorphic evolution"""
    print("\n🔍 NEUROMORPHIC EVOLUTION ANALYSIS")
    print("-" * 80)

    # Track phi evolution
    phi_values = [r['phi'] for r in results]
    node_counts = [r['nodes'] for r in results]
    community_counts = [r['communities'] for r in results]

    print(f"Φ (phi) evolution: {phi_values[0]:.3f} → {phi_values[-1]:.3f}")
    print(f"Substrate growth: {node_counts[0]} → {node_counts[-1]} concepts")
    print(f"Community formation: {community_counts[0]} → {community_counts[-1]} clusters")

    # Check for reorganization events
    reorganizations = 0
    for i in range(1, len(community_counts)):
        if community_counts[i] != community_counts[i-1]:
            reorganizations += 1

    print(f"Substrate reorganizations: {reorganizations}")

    # Response complexity evolution
    response_lengths = [len(r['response']) for r in results]
    avg_early = sum(response_lengths[:3]) / 3
    avg_late = sum(response_lengths[-3:]) / 3

    print(f"Response complexity: {avg_early:.0f} → {avg_late:.0f} chars (average)")

    # Performance metrics
    processing_times = [r['processing_time'] for r in results]
    avg_time = sum(processing_times) / len(processing_times)
    max_time = max(processing_times)

    print(f"Processing performance: {avg_time:.3f}s average, {max_time:.3f}s max")


def save_results(core, results):
    """Save experiment results and substrate state"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Save test results
    results_file = results_dir / f"neuromorphic_test_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'test_results': results,
            'final_status': core.get_status()
        }, f, indent=2)

    # Save substrate state
    substrate_file = results_dir / f"substrate_state_{timestamp}.json"
    core.save_state(str(substrate_file))

    # Create substrate visualization
    viz_file = results_dir / f"substrate_viz_{timestamp}.png"
    core.visualize_substrate(str(viz_file))

    print(f"\n💾 RESULTS SAVED")
    print(f"   Test results: {results_file}")
    print(f"   Substrate state: {substrate_file}")
    print(f"   Visualization: {viz_file}")


def interactive_mode(core):
    """Run interactive chat mode"""
    print("\n💬 INTERACTIVE MODE")
    print("Type 'quit' to exit, 'status' for system info, 'save' to save state")
    print("-" * 80)

    while True:
        try:
            user_input = input("\n[ISC] > ").strip()

            if not user_input:
                continue

            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'status':
                print_status(core)
                continue
            elif user_input.lower().startswith('save'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"results/interactive_session_{timestamp}.json"
                core.save_state(save_path)
                print(f"State saved to {save_path}")
                continue

            # Process user input
            start_time = time.time()
            response = core.process_input(user_input)
            processing_time = time.time() - start_time

            print(f"\n{response}")

            if core.verbose:
                status = core.get_status()
                print(f"\n[Φ={status['metrics']['phi_value']:.3f}, "
                      f"t={processing_time:.3f}s]")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main demo function"""
    print_banner()

    # Initialize neuromorphic core
    print("🚀 Initializing Neuromorphic ISC Core...")
    try:
        core = NeuromorphicISCCore()
        core.verbose = True
    except Exception as e:
        print(f"Failed to initialize core: {e}")
        return 1

    # Start session
    session_msg = core.start_session()
    print(f"✓ {session_msg}")

    # Show initial status
    print_status(core)

    # Ask user what to do
    print("Choose an option:")
    print("1. Run automated test queries")
    print("2. Interactive chat mode")
    print("3. Both (tests then interactive)")

    # For demo purposes, automatically run test queries
    choice = '1'
    print(f"Auto-selected choice: {choice}")

    results = None

    if choice in ['1', '3']:
        # Run test queries
        results = run_test_queries(core)
        analyze_results(results)
        save_results(core, results)

    if choice in ['2', '3']:
        # Interactive mode
        interactive_mode(core)

    print("\n🎯 NEUROMORPHIC ISC DEMO COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())