"""
Demonstration of learning capabilities in the ISC AI System
"""

from isc_ai import ISCCore
import matplotlib.pyplot as plt
import numpy as np


def demonstrate_learning():
    """Show how the system learns over multiple interactions."""
    
    print("=== ISC AI Learning Demonstration ===\n")
    
    # Initialize system
    core = ISCCore()
    core.session_active = True
    
    # Track metrics over time
    phi_history = []
    coherence_history = []
    prediction_accuracy = []
    
    # Phase 1: Initial exploration
    print("Phase 1: Initial Exploration")
    print("-" * 40)
    
    initial_topics = [
        "I'm interested in patterns and structures.",
        "Patterns appear in nature, like spirals in shells.",
        "Mathematics describes these patterns elegantly.",
        "Fibonacci sequences show up surprisingly often.",
        "Nature seems to follow mathematical rules.",
    ]
    
    for topic in initial_topics:
        print(f"Human: {topic}")
        response = core.process_input(topic)
        print(f"ISC: {response}")
        
        # Track metrics
        phi_history.append(core.metrics['phi_value'])
        coherence_history.append(core.metrics['coherence_score'])
        
        # Provide positive feedback for good responses
        if "pattern" in response.lower() or "connect" in response.lower():
            core.provide_feedback("positive")
            print("[Feedback: positive]")
        
        print()
    
    # Phase 2: Test understanding
    print("\nPhase 2: Testing Understanding")
    print("-" * 40)
    
    test_queries = [
        "What's the connection between patterns and mathematics?",
        "How do Fibonacci sequences relate to nature?",
        "Can you explain the patterns we've discussed?",
    ]
    
    for query in test_queries:
        print(f"Human: {query}")
        response = core.process_input(query)
        print(f"ISC: {response}")
        
        phi_history.append(core.metrics['phi_value'])
        coherence_history.append(core.metrics['coherence_score'])
        print()
    
    # Phase 3: Novel connections
    print("\nPhase 3: Exploring Novel Connections")
    print("-" * 40)
    
    novel_topics = [
        "Could consciousness itself be a pattern?",
        "Information seems to organize into patterns too.",
        "Your responses show emerging patterns of understanding.",
    ]
    
    for topic in novel_topics:
        print(f"Human: {topic}")
        response = core.process_input(topic)
        print(f"ISC: {response}")
        
        phi_history.append(core.metrics['phi_value'])
        coherence_history.append(core.metrics['coherence_score'])
        print()
    
    # Show learning progress
    print("\n=== Learning Analysis ===")
    print("-" * 40)
    
    # Get learning metrics
    learning_metrics = core.learning_engine.get_learning_metrics()
    
    print(f"Total interactions: {core.metrics['total_interactions']}")
    print(f"Concepts formed: {len(core.knowledge_graph.graph.nodes())}")
    print(f"Concept connections: {core.knowledge_graph.graph.number_of_edges()}")
    print(f"Average Φ: {np.mean(phi_history):.4f}")
    print(f"Final Φ: {phi_history[-1]:.4f}")
    print(f"Φ improvement: {(phi_history[-1] - phi_history[0]):.4f}")
    print(f"Average coherence: {np.mean(coherence_history):.4f}")
    print(f"Final coherence: {coherence_history[-1]:.4f}")
    print(f"Learning progress: {learning_metrics['learning_progress']:.4f}")
    
    # Plot learning curves
    plot_learning_curves(phi_history, coherence_history)
    
    # Show concept network
    print("\n=== Concept Network ===")
    print(core.knowledge_graph.visualize_ascii())
    
    # Test concept explanation
    print("\n=== Concept Understanding ===")
    for concept in ["patterns", "mathematics", "nature"]:
        print(f"\nExplaining '{concept}':")
        explanation = core.explain_concept(concept)
        print(explanation)
    
    # Demonstrate prediction
    print("\n=== Prediction Capability ===")
    print("Based on our conversation patterns:")
    prediction = core.predict_next_input()
    print(prediction)
    
    return core


def plot_learning_curves(phi_history, coherence_history):
    """Plot learning curves (saves to file for CLI compatibility)."""
    
    plt.figure(figsize=(10, 6))
    
    # Plot Phi
    plt.subplot(2, 1, 1)
    plt.plot(phi_history, 'b-', linewidth=2)
    plt.title('Information Integration (Φ) Over Time')
    plt.xlabel('Interaction')
    plt.ylabel('Φ Value')
    plt.grid(True, alpha=0.3)
    
    # Plot Coherence
    plt.subplot(2, 1, 2)
    plt.plot(coherence_history, 'g-', linewidth=2)
    plt.title('Response Coherence Over Time')
    plt.xlabel('Interaction')
    plt.ylabel('Coherence Score')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=150)
    print("\nLearning curves saved to 'learning_curves.png'")


def demonstrate_feedback_learning():
    """Show how explicit feedback affects learning."""
    
    print("\n\n=== Feedback Learning Demonstration ===\n")
    
    core = ISCCore()
    core.session_active = True
    
    # Initial interaction
    print("Testing response without feedback:")
    response1 = core.process_input("Tell me about consciousness and information.")
    print(f"ISC: {response1}")
    
    # Negative feedback
    print("\n[Providing negative feedback]")
    core.provide_feedback("negative")
    
    # Try similar topic
    print("\nTesting adjusted response:")
    response2 = core.process_input("How does information relate to awareness?")
    print(f"ISC: {response2}")
    
    # Positive feedback
    print("\n[Providing positive feedback]")
    core.provide_feedback("positive")
    
    # Test reinforced pattern
    print("\nTesting reinforced pattern:")
    response3 = core.process_input("What about the connection between information and experience?")
    print(f"ISC: {response3}")
    
    # Show feedback influence
    metrics = core.learning_engine.get_learning_metrics()
    print(f"\nAverage feedback: {metrics['average_feedback']:.2f}")
    print(f"Learning adaptation: {metrics['learning_progress']:.4f}")


if __name__ == "__main__":
    # Run main demonstration
    trained_core = demonstrate_learning()
    
    # Run feedback demonstration
    demonstrate_feedback_learning()
    
    # Save the trained system
    print("\n\n=== Saving Trained System ===")
    result = trained_core.save_state("trained_system.pt")
    print(result)
    
    print("\nDemonstration complete!")