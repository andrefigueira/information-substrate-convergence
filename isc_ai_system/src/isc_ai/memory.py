"""
Memory module for storing and retrieving conversation history
"""

import sqlite3
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import torch
import torch.nn.functional as F


class ConversationMemory:
    """
    Manages conversation history and enables retrieval of relevant past interactions.
    """
    
    def __init__(self, db_path: str = "conversation_memory.db"):
        self.db_path = db_path
        self.interactions = []
        self._init_database()
        self._load_interactions()
    
    def _init_database(self):
        """Initialize SQLite database for persistent storage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_input TEXT,
                system_response TEXT,
                embedding BLOB,
                metadata TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                start_time DATETIME,
                end_time DATETIME,
                interaction_count INTEGER,
                metadata TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _load_interactions(self, limit: int = 1000):
        """Load recent interactions from database into memory."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, session_id, timestamp, user_input, system_response, embedding, metadata
            FROM interactions
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        self.interactions = []
        for row in reversed(rows):  # Reverse to get chronological order
            interaction = {
                "id": row[0],
                "session_id": row[1],
                "timestamp": row[2],
                "input": row[3],
                "response": row[4],
                "embedding": np.frombuffer(row[5], dtype=np.float32) if row[5] else None,
                "metadata": json.loads(row[6]) if row[6] else {},
            }
            self.interactions.append(interaction)
    
    def add_interaction(
        self, 
        user_input: str, 
        system_response: str, 
        session_id: Optional[str] = None,
        embedding: Optional[np.ndarray] = None,
        metadata: Optional[Dict] = None
    ):
        """Add a new interaction to memory."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Convert embedding to bytes if provided
        embedding_bytes = embedding.tobytes() if embedding is not None else None
        metadata_json = json.dumps(metadata) if metadata else None
        
        cursor.execute("""
            INSERT INTO interactions (session_id, user_input, system_response, embedding, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (session_id, user_input, system_response, embedding_bytes, metadata_json))
        
        interaction_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Add to in-memory list
        interaction = {
            "id": interaction_id,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "input": user_input,
            "response": system_response,
            "embedding": embedding,
            "metadata": metadata or {},
        }
        self.interactions.append(interaction)
        
        # Keep memory size manageable
        if len(self.interactions) > 1000:
            self.interactions.pop(0)
    
    def get_recent_interactions(self, n: int = 10) -> List[Dict]:
        """Get the n most recent interactions."""
        return self.interactions[-n:] if self.interactions else []
    
    def get_similar_interactions(
        self, 
        query: str, 
        k: int = 5,
        encoder=None
    ) -> List[Dict]:
        """Find interactions similar to the query."""
        if not self.interactions or not encoder:
            return []
        
        # Encode query
        query_embedding = encoder(query)
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.detach().cpu().numpy()
        
        # Calculate similarities
        similarities = []
        for i, interaction in enumerate(self.interactions):
            if interaction.get("embedding") is not None:
                # Cosine similarity
                interaction_emb = interaction["embedding"]
                if len(interaction_emb.shape) > 1:
                    interaction_emb = interaction_emb.flatten()
                if len(query_embedding.shape) > 1:
                    query_emb_flat = query_embedding.flatten()
                else:
                    query_emb_flat = query_embedding
                
                similarity = np.dot(query_emb_flat, interaction_emb) / (
                    np.linalg.norm(query_emb_flat) * np.linalg.norm(interaction_emb) + 1e-8
                )
                similarities.append((i, similarity))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return [self.interactions[i] for i, _ in similarities[:k]]
    
    def search_interactions(self, keyword: str) -> List[Dict]:
        """Search interactions containing a keyword."""
        keyword_lower = keyword.lower()
        results = []
        
        for interaction in self.interactions:
            if (keyword_lower in interaction["input"].lower() or 
                keyword_lower in interaction["response"].lower()):
                results.append(interaction)
        
        return results
    
    def get_session_history(self, session_id: str) -> List[Dict]:
        """Get all interactions from a specific session."""
        return [
            interaction for interaction in self.interactions
            if interaction.get("session_id") == session_id
        ]
    
    def get_interaction_patterns(self) -> Dict[str, any]:
        """Analyze patterns in conversation history."""
        if not self.interactions:
            return {
                "total_interactions": 0,
                "avg_input_length": 0,
                "avg_response_length": 0,
                "common_topics": [],
            }
        
        # Basic statistics
        input_lengths = [len(i["input"].split()) for i in self.interactions]
        response_lengths = [len(i["response"].split()) for i in self.interactions]
        
        # Extract common words (simple topic modeling)
        from collections import Counter
        import re
        
        all_words = []
        for interaction in self.interactions:
            words = re.findall(r'\b\w{4,}\b', interaction["input"].lower())
            all_words.extend(words)
        
        word_counts = Counter(all_words)
        common_topics = [word for word, _ in word_counts.most_common(10)]
        
        return {
            "total_interactions": len(self.interactions),
            "avg_input_length": np.mean(input_lengths),
            "avg_response_length": np.mean(response_lengths),
            "common_topics": common_topics,
            "interaction_frequency": self._calculate_interaction_frequency(),
        }
    
    def _calculate_interaction_frequency(self) -> Dict[str, int]:
        """Calculate interaction frequency by time period."""
        from collections import defaultdict
        
        frequency = defaultdict(int)
        
        for interaction in self.interactions:
            if interaction.get("timestamp"):
                try:
                    # Parse timestamp and get hour
                    dt = datetime.fromisoformat(interaction["timestamp"])
                    hour = dt.hour
                    frequency[f"hour_{hour}"] += 1
                except:
                    pass
        
        return dict(frequency)
    
    def export_data(self) -> Dict:
        """Export memory data for saving."""
        return {
            "interactions": [
                {
                    "id": i["id"],
                    "session_id": i["session_id"],
                    "timestamp": i["timestamp"],
                    "input": i["input"],
                    "response": i["response"],
                    "metadata": i["metadata"],
                }
                for i in self.interactions
            ],
            "total_count": len(self.interactions),
        }
    
    def import_data(self, data: Dict):
        """Import memory data."""
        self.interactions = []
        
        for interaction_data in data.get("interactions", []):
            self.interactions.append({
                "id": interaction_data["id"],
                "session_id": interaction_data["session_id"],
                "timestamp": interaction_data["timestamp"],
                "input": interaction_data["input"],
                "response": interaction_data["response"],
                "embedding": None,  # Embeddings not serialized
                "metadata": interaction_data.get("metadata", {}),
            })
    
    def clear_old_interactions(self, days: int = 30):
        """Clear interactions older than specified days."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            DELETE FROM interactions
            WHERE timestamp < datetime('now', '-{} days')
        """.format(days))
        
        conn.commit()
        conn.close()
        
        # Reload interactions
        self._load_interactions()