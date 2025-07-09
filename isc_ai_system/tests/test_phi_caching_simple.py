#!/usr/bin/env python3
"""
Simple test for phi tracking and caching functionality
"""

import sys
from pathlib import Path
import time
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.isc_ai.cache_manager import CacheManager
from src.isc_ai.enhanced_information_integration import EnhancedInformationIntegrator


def test_basic_caching():
    """Test basic caching functionality"""
    print("Testing basic caching...")
    
    # Test cache manager
    cache = CacheManager(cache_dir="test_cache", max_memory_items=100)
    
    # Test saving and retrieving ChatGPT response
    prompt = "What is consciousness?"
    response = "Consciousness is the state of being aware..."
    
    cache.save_chatgpt_response(prompt, response)
    cached = cache.get_chatgpt_response(prompt)
    
    assert cached == response, "Basic cache test failed"
    print("✓ Basic caching works!")
    
    # Test cache statistics
    stats = cache.get_cache_stats()
    print(f"  Cache hit rate: {stats['hit_rate']:.1%}")
    print(f"  Cached responses: {stats['chatgpt_cache_count']}")
    

def test_phi_caching():
    """Test phi calculation with caching"""
    print("\nTesting phi calculation with caching...")
    
    cache = CacheManager(cache_dir="test_cache")
    integrator = EnhancedInformationIntegrator(cache_manager=cache)
    
    # Create test states
    states = [
        np.random.randn(10, 20),
        np.random.randn(15, 20),
        np.random.randn(20, 20)
    ]
    
    # Time first calculation
    start = time.time()
    phi1 = integrator.calculate_phi(states)
    time1 = time.time() - start
    print(f"  First calculation: phi={phi1:.3f}, time={time1:.3f}s")
    
    # Time second calculation (should be cached)
    start = time.time()
    phi2 = integrator.calculate_phi(states)
    time2 = time.time() - start
    print(f"  Second calculation: phi={phi2:.3f}, time={time2:.3f}s")
    
    # Verify caching worked
    assert phi1 == phi2, "Phi values don't match"
    assert time2 < time1 * 0.5, f"Caching doesn't seem to work (time2={time2:.3f} not much faster than time1={time1:.3f})"
    
    print("✓ Phi caching works!")
    
    # Show metrics
    metrics = integrator.get_integration_metrics()
    print(f"  Cache hit rate: {metrics['cache_hit_rate']:.1%}")
    print(f"  Average computation time: {metrics['average_computation_time']:.3f}s")


def test_phi_trends():
    """Test phi trend analysis"""
    print("\nTesting phi trend analysis...")
    
    integrator = EnhancedInformationIntegrator()
    
    # Generate increasing phi values
    base_states = [np.random.randn(10, 20) for _ in range(3)]
    
    for i in range(20):
        # Slightly modify states to get different phi
        noise_level = 0.1 + i * 0.01  # Increasing noise
        states = [s + np.random.randn(*s.shape) * noise_level for s in base_states]
        
        phi = integrator.calculate_phi(states, use_cache=False)
    
    # Analyze trend
    trend = integrator.get_phi_trend(window=10)
    
    print(f"  Phi trend: {trend['trend']:.4f}")
    print(f"  Stability: {trend['stability']:.2f}")
    print(f"  Recent mean: {trend['recent_mean']:.3f}")
    print(f"  Range: [{trend['recent_min']:.3f}, {trend['recent_max']:.3f}]")
    
    print("✓ Phi trend analysis works!")


def test_cache_performance():
    """Test cache performance with many queries"""
    print("\nTesting cache performance...")
    
    cache = CacheManager(cache_dir="test_cache")
    integrator = EnhancedInformationIntegrator(cache_manager=cache)
    
    # Generate multiple similar queries
    n_unique = 10
    n_repeats = 5
    
    unique_states = []
    for i in range(n_unique):
        states = [np.random.randn(10, 20) for _ in range(3)]
        unique_states.append(states)
    
    # Calculate phi for each unique state multiple times
    total_time = 0
    cache_times = []
    
    for repeat in range(n_repeats):
        for states in unique_states:
            start = time.time()
            phi = integrator.calculate_phi(states)
            elapsed = time.time() - start
            total_time += elapsed
            
            if repeat > 0:  # Should be cached after first repeat
                cache_times.append(elapsed)
    
    avg_cache_time = np.mean(cache_times) if cache_times else 0
    
    print(f"  Total queries: {n_unique * n_repeats}")
    print(f"  Unique states: {n_unique}")
    print(f"  Average cached query time: {avg_cache_time:.4f}s")
    
    # Final statistics
    final_stats = cache.get_cache_stats()
    print(f"  Final cache hit rate: {final_stats['hit_rate']:.1%}")
    print(f"  Total cached phi values: {final_stats['phi_cache_count']}")
    
    print("✓ Cache performance test complete!")


def main():
    """Run all tests"""
    print("=== Testing Phi Tracking and Caching (Simplified) ===\n")
    
    try:
        test_basic_caching()
        test_phi_caching()
        test_phi_trends()
        test_cache_performance()
        
        print("\n✅ All tests passed!")
        
    except AssertionError as e:
        print(f"\n❌ Test assertion failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())