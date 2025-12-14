"""
Basic conversation example with the ISC AI System
"""

from isc_ai import ISCCore
import time


def main():
    # Initialize the system
    print("Initializing ISC AI System...")
    core = ISCCore()
    
    # Start a session
    core.session_active = True
    core.current_session_id = "example_session"
    
    print("\n=== Basic Conversation Example ===\n")
    
    # Example conversation
    conversations = [
        "Hello! I'd like to explore how you learn from our conversation.",
        "When I mention concepts like learning and adaptation, how do you process them?",
        "Learning seems connected to change and growth over time.",
        "Can you predict what I might ask about next?",
        "Tell me about the patterns you've noticed in our conversation.",
    ]
    
    for i, user_input in enumerate(conversations, 1):
        print(f"Human: {user_input}")
        
        # Process input
        response = core.process_input(user_input)
        print(f"ISC: {response}")
        
        # Show metrics
        print(f"\nMetrics after interaction {i}:")
        print(f"  Φ (Phi): {core.metrics['phi_value']:.4f}")
        print(f"  Coherence: {core.metrics['coherence_score']:.4f}")
        print(f"  Total concepts: {len(core.knowledge_graph.graph.nodes())}")
        print(f"  Connections: {core.knowledge_graph.graph.number_of_edges()}")
        
        print("-" * 50 + "\n")
        time.sleep(1)  # Brief pause between interactions
    
    # Show final introspection
    print("\n=== System Introspection ===")
    print(core.introspect())
    
    # Show concept connections
    print("\n=== Concept Network ===")
    print(core.knowledge_graph.visualize_ascii(max_nodes=10))
    
    # Demonstrate learning
    print("\n=== Testing Learning ===")
    print("Human: What have you learned about learning?")
    response = core.process_input("What have you learned about learning?")
    print(f"ISC: {response}")
    
    # Test prediction
    print("\n=== Prediction Test ===")
    prediction = core.predict_next_input()
    print(f"ISC prediction: {prediction}")
    
    # Save state
    print("\n=== Saving State ===")
    result = core.save_state("example_conversation.pt")
    print(result)


if __name__ == "__main__":
    main()