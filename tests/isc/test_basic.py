"""
Basic tests for ISC AI System
"""

import pytest
import torch
from isc import ISCCore, InformationIntegrator, KnowledgeGraph, LearningEngine


def test_core_initialization():
    """Test that core system initializes properly."""
    core = ISCCore()
    
    assert core is not None
    assert core.network is not None
    assert core.integrator is not None
    assert core.knowledge_graph is not None
    assert core.learning_engine is not None
    assert core.memory is not None
    assert not core.session_active


def test_process_input():
    """Test basic input processing."""
    core = ISCCore()
    core.session_active = True
    
    response = core.process_input("Hello, this is a test.")
    
    assert isinstance(response, str)
    assert len(response) > 0
    assert core.metrics["total_interactions"] == 1


def test_phi_calculation():
    """Test information integration calculation."""
    integrator = InformationIntegrator()
    
    # Create dummy states
    states = [
        torch.randn(1, 100),
        torch.randn(1, 100),
        torch.randn(1, 100),
    ]
    
    phi = integrator.calculate_phi(states)
    
    assert isinstance(phi, float)
    assert phi >= 0.0


def test_knowledge_graph():
    """Test knowledge graph operations."""
    kg = KnowledgeGraph()
    
    # Add concepts
    kg.add_concept("test")
    kg.add_concept("example")
    kg.add_connection("test", "example", strength=1.0)
    
    assert "test" in kg.graph
    assert "example" in kg.graph
    assert kg.graph.has_edge("test", "example")
    
    # Test retrieval
    related = kg.get_related_concepts("test")
    assert "example" in related


def test_learning_engine():
    """Test learning engine basics."""
    from isc.core import SelfModifyingNetwork
    
    network = SelfModifyingNetwork()
    engine = LearningEngine(network)
    
    # Create dummy experience
    states = [torch.randn(1, 512) for _ in range(4)]
    engine.learn_from_interaction("test input", "test response", states)
    
    metrics = engine.get_learning_metrics()
    assert "average_loss" in metrics
    assert "prediction_accuracy" in metrics
    assert metrics["experience_count"] == 1


def test_feedback_mechanism():
    """Test feedback processing."""
    core = ISCCore()
    core.session_active = True
    
    # Process an input
    core.process_input("Test input for feedback.")
    
    # Apply feedback
    response = core.provide_feedback("positive")
    assert "positive" in response.lower()
    
    # Check that feedback was recorded
    assert len(core.learning_engine.feedback_history) > 0


def test_save_load_state():
    """Test state persistence."""
    import tempfile
    import os
    
    core = ISCCore()
    core.session_active = True
    
    # Process some inputs
    core.process_input("First test input.")
    core.process_input("Second test input.")
    
    # Save state
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        temp_path = tmp.name
    
    core.save_state(temp_path)
    assert os.path.exists(temp_path)
    
    # Create new core and load state
    core2 = ISCCore()
    core2.load_state(temp_path)
    
    # Verify state was loaded
    assert core2.metrics["total_interactions"] == 2
    
    # Cleanup
    os.unlink(temp_path)


def test_concept_explanation():
    """Test concept explanation functionality."""
    core = ISCCore()
    core.session_active = True
    
    # Build some concepts
    core.process_input("Let's talk about patterns in nature.")
    core.process_input("Patterns appear in mathematics too.")
    
    # Explain a concept
    explanation = core.explain_concept("patterns")
    
    assert isinstance(explanation, str)
    assert "patterns" in explanation.lower()
    assert "encountered" in explanation.lower()


def test_introspection():
    """Test system introspection."""
    core = ISCCore()
    core.session_active = True
    
    core.process_input("Test input for introspection.")
    
    introspection = core.introspect()
    
    assert isinstance(introspection, str)
    assert "Information Integration" in introspection
    assert "Coherence Score" in introspection
    assert "Knowledge Structure" in introspection


if __name__ == "__main__":
    pytest.main([__file__, "-v"])