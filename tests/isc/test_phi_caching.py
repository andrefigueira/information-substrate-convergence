#!/usr/bin/env python3
"""
Test script for phi tracking and caching functionality
"""

import sys
from pathlib import Path
import time
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.isc_ai.core import ISCCore
from src.isc_ai.cache_manager import CacheManager
from src.isc_ai.enhanced_learning import EnhancedLearningEngine
from src.isc_ai.enhanced_information_integration import EnhancedInformationIntegrator


def test_cache_manager():
    """Test cache manager functionality"""
    print("Testing Cache Manager...")
    
    cache = CacheManager(cache_dir="test_cache", max_memory_items=10)
    
    # Test ChatGPT response caching
    prompt = "Test prompt for caching"
    response = "Test response"
    
    # Save response
    cache.save_chatgpt_response(prompt, response, model="test-model")
    
    # Retrieve response
    cached = cache.get_chatgpt_response(prompt, model="test-model")
    assert cached == response, "Cache retrieval failed"
    
    # Test phi caching
    state_hash = "test_hash_123"
    phi_value = 2.345
    
    cache.save_phi_value(state_hash, phi_value, computation_time=0.1)
    cached_phi = cache.get_phi_value(state_hash)
    assert cached_phi == phi_value, "Phi cache retrieval failed"
    
    # Check statistics
    stats = cache.get_cache_stats()
    print(f"  Cache stats: {stats}")
    
    print("✓ Cache Manager tests passed!\n")


def test_enhanced_information_integrator():
    """Test enhanced information integrator with caching"""
    print("Testing Enhanced Information Integrator...")
    
    cache = CacheManager(cache_dir="test_cache")
    integrator = EnhancedInformationIntegrator(cache_manager=cache)
    
    # Create test states
    states = [
        np.random.randn(10, 20),
        np.random.randn(15, 20),
        np.random.randn(20, 20)
    ]
    
    # First calculation (should miss cache)
    start_time = time.time()
    phi1 = integrator.calculate_phi(states, use_cache=True)
    time1 = time.time() - start_time
    
    print(f"  First phi calculation: {phi1:.3f} (took {time1:.3f}s)")
    
    # Second calculation (should hit cache)
    start_time = time.time()
    phi2 = integrator.calculate_phi(states, use_cache=True)
    time2 = time.time() - start_time
    
    print(f"  Second phi calculation: {phi2:.3f} (took {time2:.3f}s)")
    
    assert phi1 == phi2, "Cached phi value mismatch"
    assert time2 < time1 * 0.1, "Cache doesn't seem to be working (second call not faster)"
    
    # Test metrics
    metrics = integrator.get_integration_metrics()
    print(f"  Integration metrics: {metrics}")
    
    # Test trend analysis
    for i in range(10):
        new_states = [s + np.random.randn(*s.shape) * 0.1 for s in states]
        integrator.calculate_phi(new_states)
    
    trend = integrator.get_phi_trend()
    print(f"  Phi trend: {trend}")
    
    print("✓ Enhanced Information Integrator tests passed!\n")


def test_enhanced_learning_engine():
    """Test enhanced learning engine with phi optimization"""
    print("Testing Enhanced Learning Engine...")
    
    # Create a simple network
    import torch
    import torch.nn as nn
    
    class SimpleNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 10)
            
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    network = SimpleNetwork()
    engine = EnhancedLearningEngine(network)
    
    # Simulate training with phi values
    for i in range(10):
        # Simulate interaction
        input_tensor = torch.randn(1, 10)
        output = network(input_tensor)
        
        states = [
            torch.randn(10, 10),
            torch.randn(20, 20),
            torch.randn(10, 10)
        ]
        
        # Simulate increasing phi
        phi_value = 1.0 + i * 0.1 + np.random.randn() * 0.05
        
        engine.learn_from_interaction(
            user_input=f"Test input {i}",
            response=f"Test response {i}",
            internal_states=states,
            phi_value=phi_value
        )
    
    # Check metrics
    metrics = engine.get_learning_metrics()
    print(f"  Learning metrics: {metrics}")
    
    assert len(engine.phi_history) == 10, "Phi history not tracked correctly"
    assert metrics["average_phi"] > 0, "Average phi should be positive"
    
    print("✓ Enhanced Learning Engine tests passed!\n")


def test_integration_with_core():
    """Test integration with ISC Core"""
    print("Testing Integration with ISC Core...")
    
    # Initialize core
    core = ISCCore()
    
    # Replace components with enhanced versions
    cache = CacheManager(cache_dir="test_cache")
    
    old_integrator = core.integrator
    core.integrator = EnhancedInformationIntegrator(cache_manager=cache)
    if hasattr(old_integrator, 'phi_history'):
        core.integrator.phi_history = old_integrator.phi_history
    
    core.learning_engine = EnhancedLearningEngine(
        network=core.network,
        config={
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "buffer_size": 1000,
            "prediction_window": 5,
            "batch_size": 16,
            "phi_weight": 0.4,
            "phi_target": 2.0,
            "phi_growth_rate": 0.001,
            "adaptive_phi_target": True
        }
    )
    
    # Start session
    core.session_active = True
    core.current_session_id = "test_session"
    
    # Process some inputs
    test_inputs = [
        "What is consciousness?",
        "How does information integrate in complex systems?",
        "Explain emergent properties of neural networks.",
        "What is the relationship between complexity and consciousness?"
    ]
    
    phi_values = []
    for i, input_text in enumerate(test_inputs):
        response = core.process_input(input_text)
        
        # Check phi
        current_phi = core.integrator.phi_history[-1] if core.integrator.phi_history else 0
        phi_values.append(current_phi)
        
        print(f"  Input {i+1}: Phi = {current_phi:.3f}")
    
    # Check that phi is being tracked
    assert len(phi_values) == len(test_inputs), "Phi not tracked for all inputs"
    assert any(p > 0 for p in phi_values), "Some phi values should be positive"
    
    # Check cache performance
    cache_stats = cache.get_cache_stats()
    print(f"  Final cache stats: {cache_stats}")
    
    # End session
    core.session_active = False
    
    print("✓ Integration tests passed!\n")


def main():
    """Run all tests"""
    print("=== Testing Phi Tracking and Caching ===\n")
    
    try:
        test_cache_manager()
        test_enhanced_information_integrator()
        test_enhanced_learning_engine()
        test_integration_with_core()
        
        print("\n✅ All tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())