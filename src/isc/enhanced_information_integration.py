"""
Enhanced Information Integration module with caching and optimization
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from itertools import combinations
import torch.nn.functional as F
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
import pickle


class EnhancedInformationIntegrator:
    """
    Enhanced calculator for Integrated Information (Φ) with caching
    and performance optimizations.
    """
    
    def __init__(self, cache_manager=None, max_workers: int = 4):
        self.phi_history = []
        self.partition_cache = {}
        self.mutual_info_cache = {}
        self.cache_manager = cache_manager
        self.computation_times = []
        self.max_workers = max_workers
        
        # Performance metrics
        self.cache_hits = 0
        self.cache_misses = 0
        
    def calculate_phi(self, states: List[torch.Tensor], use_cache: bool = True) -> float:
        """
        Calculate Φ with caching and optimizations.
        """
        if not states or len(states) < 2:
            return 0.0
        
        start_time = time.time()
        
        # Generate cache key if caching is enabled
        cache_key = None
        if use_cache and self.cache_manager:
            cache_key = self._generate_cache_key(states)
            
            # Check cache
            cached_phi = self.cache_manager.get_phi_value(cache_key)
            if cached_phi is not None:
                self.cache_hits += 1
                self.phi_history.append(cached_phi)
                return cached_phi
            
            self.cache_misses += 1
        
        # Convert states to numpy for easier manipulation
        state_arrays = [self._to_numpy(s) for s in states]
        
        # Use approximation for large systems
        if len(states) > 8:
            phi = self._calculate_phi_approximation(state_arrays)
        else:
            # Full calculation for smaller systems
            phi = self._calculate_phi_exact(state_arrays)
        
        self.phi_history.append(phi)
        
        # Cache the result
        computation_time = time.time() - start_time
        self.computation_times.append(computation_time)
        
        if use_cache and self.cache_manager and cache_key:
            self.cache_manager.save_phi_value(cache_key, phi, computation_time)
        
        return phi
    
    def _calculate_phi_exact(self, states: List[np.ndarray]) -> float:
        """
        Exact phi calculation for small systems.
        """
        # Calculate total system information
        system_info = self._calculate_system_information(states)
        
        # Find the partition that minimizes information loss
        min_partition_info = self._find_minimum_partition(states)
        
        # Φ is the difference
        phi = system_info - min_partition_info
        return max(0.0, phi)
    
    def _calculate_phi_approximation(self, states: List[np.ndarray]) -> float:
        """
        Fast approximation for large systems using sampling.
        """
        n_layers = len(states)
        
        # Sample random partitions instead of trying all
        n_samples = min(100, 2 ** (n_layers - 1))
        
        system_info = self._calculate_system_information(states)
        min_partition_info = float('inf')
        
        # Random sampling of partitions
        for _ in range(n_samples):
            # Random partition size
            partition_size = np.random.randint(1, n_layers // 2 + 1)
            # Random partition
            partition_indices = np.random.choice(n_layers, partition_size, replace=False)
            
            partition_info = self._calculate_partition_information_fast(
                states, tuple(sorted(partition_indices))
            )
            min_partition_info = min(min_partition_info, partition_info)
        
        phi = system_info - min_partition_info
        return max(0.0, phi)
    
    def _generate_cache_key(self, states: List[torch.Tensor]) -> str:
        """
        Generate a cache key from states.
        """
        key_data = []
        for s in states:
            if isinstance(s, torch.Tensor):
                s_numpy = s.detach().cpu().numpy()
            else:
                s_numpy = s
            
            # Use shape and statistics for key
            key_data.append({
                'shape': s_numpy.shape,
                'mean': float(np.mean(s_numpy)),
                'std': float(np.std(s_numpy)),
                'sample': s_numpy.flatten()[:20].tobytes()
            })
        
        return hashlib.sha256(pickle.dumps(key_data)).hexdigest()
    
    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert tensor to numpy array.
        """
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return tensor
    
    def _calculate_system_information(self, states: List[np.ndarray]) -> float:
        """
        Calculate total information with caching.
        """
        total_info = 0.0
        
        # Parallel computation of mutual information
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for i in range(len(states) - 1):
                future = executor.submit(
                    self._mutual_information_cached,
                    states[i], states[i + 1], i
                )
                futures.append(future)
            
            for future in futures:
                total_info += future.result()
        
        return total_info
    
    def _mutual_information_cached(self, x: np.ndarray, y: np.ndarray, pair_idx: int) -> float:
        """
        Calculate mutual information with caching.
        """
        # Create cache key
        cache_key = f"{pair_idx}_{x.shape}_{y.shape}_{np.mean(x):.4f}_{np.mean(y):.4f}"
        
        if cache_key in self.mutual_info_cache:
            return self.mutual_info_cache[cache_key]
        
        mi = self._mutual_information(x, y)
        
        # Cache with size limit
        if len(self.mutual_info_cache) < 1000:
            self.mutual_info_cache[cache_key] = mi
        
        return mi
    
    def _mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Enhanced mutual information estimation.
        """
        # Flatten if needed
        x_flat = x.flatten()
        y_flat = y.flatten()
        
        # Handle size mismatch by truncating or padding
        min_size = min(len(x_flat), len(y_flat))
        x_flat = x_flat[:min_size]
        y_flat = y_flat[:min_size]
        
        # Normalize
        x_norm = (x_flat - x_flat.mean()) / (x_flat.std() + 1e-8)
        y_norm = (y_flat - y_flat.mean()) / (y_flat.std() + 1e-8)
        
        # Multiple measures for robustness
        # 1. Correlation-based
        correlation = np.abs(np.corrcoef(x_norm, y_norm)[0, 1])
        
        # 2. Entropy-based approximation
        # Discretize for entropy calculation
        n_bins = 10
        x_discrete = np.digitize(x_norm, np.linspace(-3, 3, n_bins))
        y_discrete = np.digitize(y_norm, np.linspace(-3, 3, n_bins))
        
        # Joint histogram
        hist_2d = np.histogram2d(x_discrete, y_discrete, bins=n_bins)[0]
        hist_2d = hist_2d / hist_2d.sum()
        
        # Marginal distributions
        px = hist_2d.sum(axis=1)
        py = hist_2d.sum(axis=0)
        
        # Mutual information
        mi_entropy = 0.0
        for i in range(n_bins):
            for j in range(n_bins):
                if hist_2d[i, j] > 0 and px[i] > 0 and py[j] > 0:
                    mi_entropy += hist_2d[i, j] * np.log(hist_2d[i, j] / (px[i] * py[j]))
        
        # Combine measures
        if correlation < 0.999:
            mi_corr = -np.log(1 - correlation**2)
        else:
            mi_corr = 10.0
        
        # Weighted combination
        mi = 0.7 * mi_corr + 0.3 * max(0, mi_entropy)
        
        return mi
    
    def _find_minimum_partition(self, states: List[np.ndarray]) -> float:
        """
        Find minimum partition with caching.
        """
        n_layers = len(states)
        if n_layers <= 2:
            return 0.0
        
        # Create partition cache key
        partition_key = hashlib.md5(
            ''.join([f"{s.shape}{np.mean(s):.4f}" for s in states]).encode()
        ).hexdigest()
        
        if partition_key in self.partition_cache:
            return self.partition_cache[partition_key]
        
        min_info = float('inf')
        
        # Limit partition search for large systems
        max_partitions = 1000 if n_layers <= 10 else 100
        partition_count = 0
        
        # Try different bipartitions
        for partition_size in range(1, n_layers // 2 + 1):
            for partition_indices in combinations(range(n_layers), partition_size):
                if partition_count >= max_partitions:
                    break
                
                partition_info = self._calculate_partition_information_fast(
                    states, partition_indices
                )
                min_info = min(min_info, partition_info)
                partition_count += 1
            
            if partition_count >= max_partitions:
                break
        
        # Cache the result
        self.partition_cache[partition_key] = min_info
        
        # Limit cache size
        if len(self.partition_cache) > 500:
            # Remove oldest entries
            keys_to_remove = list(self.partition_cache.keys())[:100]
            for key in keys_to_remove:
                del self.partition_cache[key]
        
        return min_info
    
    def _calculate_partition_information_fast(
        self, states: List[np.ndarray], partition_indices: Tuple[int, ...]
    ) -> float:
        """
        Fast partition information calculation.
        """
        partition_a = list(partition_indices)
        partition_b = [i for i in range(len(states)) if i not in partition_indices]
        
        partition_info = 0.0
        
        # Information within partition A
        if len(partition_a) > 1:
            states_a = [states[i] for i in partition_a]
            for i in range(len(states_a) - 1):
                partition_info += self._mutual_information_cached(
                    states_a[i], states_a[i + 1], 
                    f"a_{partition_a[i]}_{partition_a[i+1]}"
                )
        
        # Information within partition B
        if len(partition_b) > 1:
            states_b = [states[i] for i in partition_b]
            for i in range(len(states_b) - 1):
                partition_info += self._mutual_information_cached(
                    states_b[i], states_b[i + 1],
                    f"b_{partition_b[i]}_{partition_b[i+1]}"
                )
        
        return partition_info
    
    def get_integration_metrics(self) -> Dict[str, float]:
        """
        Get metrics about phi calculation performance.
        """
        avg_time = np.mean(self.computation_times) if self.computation_times else 0.0
        avg_phi = np.mean(self.phi_history) if self.phi_history else 0.0
        
        cache_hit_rate = 0.0
        if self.cache_hits + self.cache_misses > 0:
            cache_hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses)
        
        return {
            "average_computation_time": avg_time,
            "average_phi": avg_phi,
            "current_phi": self.phi_history[-1] if self.phi_history else 0.0,
            "cache_hit_rate": cache_hit_rate,
            "partition_cache_size": len(self.partition_cache),
            "mutual_info_cache_size": len(self.mutual_info_cache),
        }
    
    def get_phi_trend(self, window: int = 10) -> Dict[str, float]:
        """
        Analyze recent phi trends.
        """
        if len(self.phi_history) < window:
            return {"trend": 0.0, "stability": 0.0}
        
        recent = self.phi_history[-window:]
        
        # Linear trend
        x = np.arange(window)
        y = np.array(recent)
        trend = np.polyfit(x, y, 1)[0]
        
        # Stability (inverse of variance)
        stability = 1.0 / (np.std(recent) + 0.1)
        
        return {
            "trend": trend,
            "stability": stability,
            "recent_mean": np.mean(recent),
            "recent_max": np.max(recent),
            "recent_min": np.min(recent)
        }