"""
Efficient caching system for ChatGPT responses and phi calculations.
Uses SQLite for persistence and LRU caching for memory efficiency.
"""

import sqlite3
import json
import hashlib
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from collections import OrderedDict
from threading import Lock
import pickle
import zlib


class CacheManager:
    """
    Manages caching for ChatGPT responses and phi calculations.
    Uses SQLite for persistence and in-memory LRU cache for speed.
    """
    
    def __init__(self, cache_dir: str = "cache", max_memory_items: int = 1000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.db_path = self.cache_dir / "response_cache.db"
        self.max_memory_items = max_memory_items
        
        # Thread-safe in-memory LRU cache
        self.memory_cache = OrderedDict()
        self.cache_lock = Lock()
        
        # Initialize database
        self._init_database()
        
        # Cache statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "saves": 0
        }
    
    def _init_database(self):
        """Initialize SQLite database for persistent caching."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chatgpt_cache (
                    prompt_hash TEXT PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    response TEXT NOT NULL,
                    model TEXT,
                    timestamp REAL,
                    usage_count INTEGER DEFAULT 1,
                    last_accessed REAL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS phi_cache (
                    state_hash TEXT PRIMARY KEY,
                    phi_value REAL NOT NULL,
                    computation_time REAL,
                    timestamp REAL,
                    usage_count INTEGER DEFAULT 1
                )
            """)
            
            # Create indices for faster lookups
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chatgpt_timestamp ON chatgpt_cache(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chatgpt_usage ON chatgpt_cache(usage_count)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_phi_timestamp ON phi_cache(timestamp)")
    
    def _compute_hash(self, data: Any) -> str:
        """Compute a hash for caching keys."""
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        else:
            # For complex objects, use pickle
            data_bytes = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        
        return hashlib.sha256(data_bytes).hexdigest()
    
    def get_chatgpt_response(self, prompt: str, model: str = None) -> Optional[str]:
        """
        Get cached ChatGPT response if available.
        
        Returns:
            Cached response string or None if not found
        """
        prompt_hash = self._compute_hash(prompt + (model or ""))
        
        # Check memory cache first
        with self.cache_lock:
            if prompt_hash in self.memory_cache:
                # Move to end (LRU)
                self.memory_cache.move_to_end(prompt_hash)
                self.stats["hits"] += 1
                return self.memory_cache[prompt_hash]["response"]
        
        # Check database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT response FROM chatgpt_cache WHERE prompt_hash = ?",
                (prompt_hash,)
            )
            row = cursor.fetchone()
            
            if row:
                response = row[0]
                
                # Update usage statistics
                conn.execute(
                    "UPDATE chatgpt_cache SET usage_count = usage_count + 1, last_accessed = ? WHERE prompt_hash = ?",
                    (time.time(), prompt_hash)
                )
                
                # Add to memory cache
                self._add_to_memory_cache(prompt_hash, {"response": response})
                
                self.stats["hits"] += 1
                return response
        
        self.stats["misses"] += 1
        return None
    
    def save_chatgpt_response(self, prompt: str, response: str, model: str = None):
        """Save ChatGPT response to cache."""
        prompt_hash = self._compute_hash(prompt + (model or ""))
        
        # Add to memory cache
        self._add_to_memory_cache(prompt_hash, {"response": response})
        
        # Save to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO chatgpt_cache 
                (prompt_hash, prompt, response, model, timestamp, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (prompt_hash, prompt, response, model, time.time(), time.time()))
        
        self.stats["saves"] += 1
    
    def get_phi_value(self, states_hash: str) -> Optional[float]:
        """Get cached phi value if available."""
        with self.cache_lock:
            if states_hash in self.memory_cache:
                self.memory_cache.move_to_end(states_hash)
                return self.memory_cache[states_hash].get("phi_value")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT phi_value FROM phi_cache WHERE state_hash = ?",
                (states_hash,)
            )
            row = cursor.fetchone()
            
            if row:
                phi_value = row[0]
                conn.execute(
                    "UPDATE phi_cache SET usage_count = usage_count + 1 WHERE state_hash = ?",
                    (states_hash,)
                )
                
                self._add_to_memory_cache(states_hash, {"phi_value": phi_value})
                return phi_value
        
        return None
    
    def save_phi_value(self, states_hash: str, phi_value: float, computation_time: float = 0.0):
        """Save phi calculation to cache."""
        self._add_to_memory_cache(states_hash, {"phi_value": phi_value})
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO phi_cache 
                (state_hash, phi_value, computation_time, timestamp)
                VALUES (?, ?, ?, ?)
            """, (states_hash, phi_value, computation_time, time.time()))
    
    def _add_to_memory_cache(self, key: str, value: Dict[str, Any]):
        """Add item to memory cache with LRU eviction."""
        with self.cache_lock:
            # Remove oldest if at capacity
            if len(self.memory_cache) >= self.max_memory_items:
                self.memory_cache.popitem(last=False)
            
            self.memory_cache[key] = value
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with sqlite3.connect(self.db_path) as conn:
            chatgpt_count = conn.execute("SELECT COUNT(*) FROM chatgpt_cache").fetchone()[0]
            phi_count = conn.execute("SELECT COUNT(*) FROM phi_cache").fetchone()[0]
            
            # Get database size
            db_size = self.db_path.stat().st_size / (1024 * 1024)  # MB
        
        return {
            "memory_cache_size": len(self.memory_cache),
            "chatgpt_cache_count": chatgpt_count,
            "phi_cache_count": phi_count,
            "database_size_mb": round(db_size, 2),
            "hit_rate": self.stats["hits"] / (self.stats["hits"] + self.stats["misses"]) if (self.stats["hits"] + self.stats["misses"]) > 0 else 0,
            **self.stats
        }
    
    def cleanup_old_entries(self, days: int = 30):
        """Remove cache entries older than specified days."""
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM chatgpt_cache WHERE timestamp < ?", (cutoff_time,))
            conn.execute("DELETE FROM phi_cache WHERE timestamp < ?", (cutoff_time,))
            conn.execute("VACUUM")  # Reclaim space
    
    def export_frequent_prompts(self, min_usage: int = 5) -> List[Tuple[str, str, int]]:
        """Export frequently used prompts for analysis."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT prompt, response, usage_count 
                FROM chatgpt_cache 
                WHERE usage_count >= ?
                ORDER BY usage_count DESC
            """, (min_usage,))
            
            return cursor.fetchall()