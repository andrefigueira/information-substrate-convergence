"""
Local Graph Database implementation using SQLite and NetworkX
"""

import sqlite3
import json
import pickle
import gzip
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
import networkx as nx
import numpy as np
from collections import defaultdict
import hashlib


class LocalGraphDB:
    """
    A fully local graph database that combines SQLite for persistence
    and NetworkX for graph operations. No external services required.
    """
    
    def __init__(self, storage_dir: str = "isc_graph_storage"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.versions_dir = self.storage_dir / "versions"
        self.snapshots_dir = self.storage_dir / "snapshots"
        self.exports_dir = self.storage_dir / "exports"
        
        for dir_path in [self.versions_dir, self.snapshots_dir, self.exports_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Initialize SQLite database
        self.db_path = self.storage_dir / "graph.db"
        self.conn = None
        self._init_database()
        
        # In-memory graph
        self.graph = nx.Graph()
        
        # Version tracking
        self.current_version = None
        self.version_history = []
        
        # Change tracking for incremental updates
        self.pending_changes = {
            "added_nodes": set(),
            "removed_nodes": set(),
            "added_edges": set(),
            "removed_edges": set(),
            "updated_attributes": {}
        }
    
    def _init_database(self):
        """Initialize SQLite database schema."""
        self.conn = sqlite3.connect(str(self.db_path))
        cursor = self.conn.cursor()
        
        # Nodes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                data TEXT,
                attributes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Edges table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                source TEXT,
                target TEXT,
                weight REAL DEFAULT 1.0,
                attributes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (source, target),
                FOREIGN KEY (source) REFERENCES nodes(id),
                FOREIGN KEY (target) REFERENCES nodes(id)
            )
        """)
        
        # Versions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS versions (
                version_id TEXT PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                description TEXT,
                node_count INTEGER,
                edge_count INTEGER,
                metadata TEXT
            )
        """)
        
        # Node embeddings table (for semantic search)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS node_embeddings (
                node_id TEXT PRIMARY KEY,
                embedding BLOB,
                FOREIGN KEY (node_id) REFERENCES nodes(id)
            )
        """)
        
        # Create indexes for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_created ON nodes(created_at)")
        
        self.conn.commit()
    
    def add_node(self, node_id: str, **attributes):
        """Add a node to the graph."""
        self.graph.add_node(node_id, **attributes)
        self.pending_changes["added_nodes"].add(node_id)
        
        # Remove from removed_nodes if it was pending removal
        self.pending_changes["removed_nodes"].discard(node_id)
    
    def add_edge(self, source: str, target: str, weight: float = 1.0, **attributes):
        """Add an edge to the graph."""
        self.graph.add_edge(source, target, weight=weight, **attributes)
        edge_tuple = (source, target) if source < target else (target, source)
        self.pending_changes["added_edges"].add(edge_tuple)
        
        # Remove from removed_edges if it was pending removal
        self.pending_changes["removed_edges"].discard(edge_tuple)
    
    def remove_node(self, node_id: str):
        """Remove a node from the graph."""
        if node_id in self.graph:
            # Track edges that will be removed
            for neighbor in self.graph.neighbors(node_id):
                edge_tuple = (node_id, neighbor) if node_id < neighbor else (neighbor, node_id)
                self.pending_changes["removed_edges"].add(edge_tuple)
            
            self.graph.remove_node(node_id)
            self.pending_changes["removed_nodes"].add(node_id)
            self.pending_changes["added_nodes"].discard(node_id)
    
    def remove_edge(self, source: str, target: str):
        """Remove an edge from the graph."""
        if self.graph.has_edge(source, target):
            self.graph.remove_edge(source, target)
            edge_tuple = (source, target) if source < target else (target, source)
            self.pending_changes["removed_edges"].add(edge_tuple)
            self.pending_changes["added_edges"].discard(edge_tuple)
    
    def update_node_attributes(self, node_id: str, **attributes):
        """Update node attributes."""
        if node_id in self.graph:
            self.graph.nodes[node_id].update(attributes)
            if node_id not in self.pending_changes["updated_attributes"]:
                self.pending_changes["updated_attributes"][node_id] = {}
            self.pending_changes["updated_attributes"][node_id].update(attributes)
    
    def save_graph(self, version_tag: Optional[str] = None, description: str = "") -> str:
        """
        Save the current graph state with optional version tag.
        Returns the version ID.
        """
        # Generate version ID
        timestamp = datetime.now()
        version_id = version_tag or timestamp.strftime("%Y%m%d_%H%M%S")
        
        # Apply pending changes to database
        self._apply_pending_changes()
        
        # Save full snapshot
        snapshot_path = self.snapshots_dir / f"{version_id}.gpz"
        self._save_snapshot(snapshot_path)
        
        # Record version
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO versions (version_id, description, node_count, edge_count, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (
            version_id,
            description,
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
            json.dumps({
                "snapshot_path": str(snapshot_path),
                "timestamp": timestamp.isoformat()
            })
        ))
        self.conn.commit()
        
        self.current_version = version_id
        self.version_history.append(version_id)
        
        # Clear pending changes
        self._clear_pending_changes()
        
        return version_id
    
    def _apply_pending_changes(self):
        """Apply pending changes to the SQLite database."""
        cursor = self.conn.cursor()
        
        # Remove nodes
        for node_id in self.pending_changes["removed_nodes"]:
            cursor.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
            cursor.execute("DELETE FROM node_embeddings WHERE node_id = ?", (node_id,))
        
        # Remove edges
        for source, target in self.pending_changes["removed_edges"]:
            cursor.execute(
                "DELETE FROM edges WHERE (source = ? AND target = ?) OR (source = ? AND target = ?)",
                (source, target, target, source)
            )
        
        # Add nodes
        for node_id in self.pending_changes["added_nodes"]:
            node_data = self.graph.nodes[node_id]
            cursor.execute("""
                INSERT OR REPLACE INTO nodes (id, data, attributes)
                VALUES (?, ?, ?)
            """, (
                node_id,
                json.dumps(node_data.get("data", {})),
                json.dumps({k: v for k, v in node_data.items() if k != "data"})
            ))
        
        # Add edges
        for source, target in self.pending_changes["added_edges"]:
            edge_data = self.graph[source][target]
            cursor.execute("""
                INSERT OR REPLACE INTO edges (source, target, weight, attributes)
                VALUES (?, ?, ?, ?)
            """, (
                source,
                target,
                edge_data.get("weight", 1.0),
                json.dumps({k: v for k, v in edge_data.items() if k != "weight"})
            ))
        
        # Update node attributes
        for node_id, attributes in self.pending_changes["updated_attributes"].items():
            if node_id in self.graph:
                node_data = self.graph.nodes[node_id]
                cursor.execute("""
                    UPDATE nodes 
                    SET attributes = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (
                    json.dumps({k: v for k, v in node_data.items() if k != "data"}),
                    node_id
                ))
        
        self.conn.commit()
    
    def _save_snapshot(self, path: Path):
        """Save a compressed snapshot of the graph."""
        data = {
            "graph": nx.node_link_data(self.graph),
            "metadata": {
                "node_count": self.graph.number_of_nodes(),
                "edge_count": self.graph.number_of_edges(),
                "timestamp": datetime.now().isoformat()
            }
        }
        
        with gzip.open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def load_graph(self, version: Optional[str] = None) -> nx.Graph:
        """
        Load a specific version of the graph or the latest if version is None.
        """
        cursor = self.conn.cursor()
        
        if version is None:
            # Get latest version
            cursor.execute("""
                SELECT version_id, metadata FROM versions
                ORDER BY timestamp DESC LIMIT 1
            """)
        else:
            cursor.execute("""
                SELECT version_id, metadata FROM versions
                WHERE version_id = ?
            """, (version,))
        
        result = cursor.fetchone()
        if not result:
            raise ValueError(f"Version {version} not found")
        
        version_id, metadata_str = result
        metadata = json.loads(metadata_str)
        
        # Load snapshot
        snapshot_path = Path(metadata["snapshot_path"])
        if snapshot_path.exists():
            with gzip.open(snapshot_path, 'rb') as f:
                data = pickle.load(f)
                self.graph = nx.node_link_graph(data["graph"])
        else:
            # Rebuild from database
            self._rebuild_graph_from_db()
        
        self.current_version = version_id
        self._clear_pending_changes()
        
        return self.graph
    
    def _rebuild_graph_from_db(self):
        """Rebuild the graph from SQLite database."""
        self.graph.clear()
        cursor = self.conn.cursor()
        
        # Load nodes
        cursor.execute("SELECT id, data, attributes FROM nodes")
        for node_id, data_str, attr_str in cursor.fetchall():
            data = json.loads(data_str) if data_str else {}
            attributes = json.loads(attr_str) if attr_str else {}
            attributes["data"] = data
            self.graph.add_node(node_id, **attributes)
        
        # Load edges
        cursor.execute("SELECT source, target, weight, attributes FROM edges")
        for source, target, weight, attr_str in cursor.fetchall():
            attributes = json.loads(attr_str) if attr_str else {}
            self.graph.add_edge(source, target, weight=weight, **attributes)
    
    def query_graph(self, concept_query: str, query_type: str = "node") -> List[Dict]:
        """
        Query the graph for concepts/patterns.
        
        Args:
            concept_query: The search query
            query_type: "node", "edge", "path", "neighbors", "pattern"
        
        Returns:
            List of matching results
        """
        results = []
        
        if query_type == "node":
            # Search nodes by ID and attributes
            for node_id, data in self.graph.nodes(data=True):
                if self._matches_query(concept_query, node_id, data):
                    results.append({
                        "type": "node",
                        "id": node_id,
                        "data": data,
                        "degree": self.graph.degree(node_id)
                    })
        
        elif query_type == "edge":
            # Search edges
            for source, target, data in self.graph.edges(data=True):
                edge_str = f"{source}-{target}"
                if self._matches_query(concept_query, edge_str, data):
                    results.append({
                        "type": "edge",
                        "source": source,
                        "target": target,
                        "data": data
                    })
        
        elif query_type == "neighbors":
            # Find neighbors of a node
            if concept_query in self.graph:
                neighbors = list(self.graph.neighbors(concept_query))
                results.append({
                    "type": "neighbors",
                    "node": concept_query,
                    "neighbors": neighbors,
                    "count": len(neighbors)
                })
        
        elif query_type == "path":
            # Find paths between nodes (format: "node1->node2")
            if "->" in concept_query:
                source, target = concept_query.split("->")
                source, target = source.strip(), target.strip()
                if source in self.graph and target in self.graph:
                    try:
                        paths = list(nx.all_simple_paths(
                            self.graph, source, target, cutoff=5
                        ))
                        results.append({
                            "type": "path",
                            "source": source,
                            "target": target,
                            "paths": paths[:10]  # Limit to 10 paths
                        })
                    except nx.NetworkXNoPath:
                        results.append({
                            "type": "path",
                            "source": source,
                            "target": target,
                            "paths": []
                        })
        
        elif query_type == "pattern":
            # Search for graph patterns (e.g., triangles, stars)
            results.extend(self._find_patterns(concept_query))
        
        return results
    
    def _matches_query(self, query: str, text: str, data: Dict) -> bool:
        """Check if query matches text or data attributes."""
        query_lower = query.lower()
        
        # Check ID/text
        if query_lower in text.lower():
            return True
        
        # Check attributes
        for key, value in data.items():
            if isinstance(value, str) and query_lower in value.lower():
                return True
            elif query_lower in str(key).lower():
                return True
        
        return False
    
    def _find_patterns(self, pattern_type: str) -> List[Dict]:
        """Find specific graph patterns."""
        results = []
        
        if pattern_type == "triangles":
            # Find all triangles
            triangles = [list(clique) for clique in nx.enumerate_all_cliques(self.graph) 
                        if len(clique) == 3]
            results.append({
                "type": "pattern",
                "pattern": "triangles",
                "instances": triangles[:20],  # Limit results
                "count": len(triangles)
            })
        
        elif pattern_type == "hubs":
            # Find hub nodes (high degree)
            degrees = dict(self.graph.degree())
            avg_degree = np.mean(list(degrees.values()))
            hubs = [(node, deg) for node, deg in degrees.items() 
                   if deg > avg_degree * 2]
            hubs.sort(key=lambda x: x[1], reverse=True)
            results.append({
                "type": "pattern",
                "pattern": "hubs",
                "instances": hubs[:10],
                "avg_degree": avg_degree
            })
        
        elif pattern_type == "communities":
            # Find communities
            communities = list(nx.community.greedy_modularity_communities(self.graph))
            results.append({
                "type": "pattern",
                "pattern": "communities",
                "instances": [list(c)[:10] for c in communities[:5]],  # Sample
                "count": len(communities)
            })
        
        return results
    
    def update_graph(self, updates: Dict[str, Any]):
        """
        Apply incremental updates to the graph.
        
        Args:
            updates: Dictionary containing:
                - nodes_to_add: List of (node_id, attributes) tuples
                - nodes_to_remove: List of node_ids
                - edges_to_add: List of (source, target, attributes) tuples
                - edges_to_remove: List of (source, target) tuples
                - node_updates: Dict of {node_id: {attr: value}}
        """
        # Add nodes
        for node_info in updates.get("nodes_to_add", []):
            if isinstance(node_info, tuple):
                node_id, attrs = node_info
                self.add_node(node_id, **attrs)
            else:
                self.add_node(node_info)
        
        # Remove nodes
        for node_id in updates.get("nodes_to_remove", []):
            self.remove_node(node_id)
        
        # Add edges
        for edge_info in updates.get("edges_to_add", []):
            if len(edge_info) >= 3:
                source, target, attrs = edge_info[0], edge_info[1], edge_info[2]
                self.add_edge(source, target, **attrs)
            else:
                source, target = edge_info
                self.add_edge(source, target)
        
        # Remove edges
        for edge in updates.get("edges_to_remove", []):
            self.remove_edge(edge[0], edge[1])
        
        # Update node attributes
        for node_id, attrs in updates.get("node_updates", {}).items():
            self.update_node_attributes(node_id, **attrs)
    
    def export_to_text(self, output_path: Optional[str] = None) -> str:
        """Export graph to human-readable text format."""
        lines = []
        lines.append("ISC Graph Export")
        lines.append("=" * 50)
        lines.append(f"Nodes: {self.graph.number_of_nodes()}")
        lines.append(f"Edges: {self.graph.number_of_edges()}")
        lines.append(f"Version: {self.current_version}")
        lines.append("")
        
        # Node list
        lines.append("NODES:")
        lines.append("-" * 30)
        for node_id, data in sorted(self.graph.nodes(data=True)):
            degree = self.graph.degree(node_id)
            lines.append(f"[{node_id}] (degree: {degree})")
            for key, value in data.items():
                if key != "data":
                    lines.append(f"  {key}: {value}")
        
        lines.append("")
        lines.append("EDGES:")
        lines.append("-" * 30)
        for source, target, data in sorted(self.graph.edges(data=True)):
            weight = data.get("weight", 1.0)
            lines.append(f"{source} -> {target} (weight: {weight:.3f})")
            for key, value in data.items():
                if key != "weight":
                    lines.append(f"  {key}: {value}")
        
        text = "\n".join(lines)
        
        if output_path:
            path = self.exports_dir / output_path
            path.write_text(text)
            return f"Exported to {path}"
        
        return text
    
    def visualize_ascii(self, node_subset: Optional[Set[str]] = None, max_nodes: int = 20) -> str:
        """Create ASCII visualization of graph section."""
        if node_subset:
            subgraph = self.graph.subgraph(node_subset)
        else:
            # Get most connected nodes
            degrees = dict(self.graph.degree())
            top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
            subgraph = self.graph.subgraph([n for n, _ in top_nodes])
        
        lines = []
        lines.append("Graph Visualization")
        lines.append("=" * 50)
        
        # Simple adjacency representation
        for node in sorted(subgraph.nodes()):
            neighbors = sorted(subgraph.neighbors(node))
            if neighbors:
                lines.append(f"[{node}] --> {', '.join(neighbors)}")
            else:
                lines.append(f"[{node}] (isolated)")
        
        lines.append("")
        lines.append(f"Shown: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")
        lines.append(f"Total: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        
        return "\n".join(lines)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        if self.graph.number_of_nodes() == 0:
            return {"nodes": 0, "edges": 0}
        
        degrees = list(dict(self.graph.degree()).values())
        
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "avg_degree": np.mean(degrees),
            "max_degree": max(degrees),
            "min_degree": min(degrees),
            "connected_components": nx.number_connected_components(self.graph),
            "clustering_coefficient": nx.average_clustering(self.graph),
            "versions_saved": len(self.version_history)
        }
    
    def list_versions(self) -> List[Dict[str, Any]]:
        """List all saved versions."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT version_id, timestamp, description, node_count, edge_count
            FROM versions
            ORDER BY timestamp DESC
        """)
        
        versions = []
        for row in cursor.fetchall():
            versions.append({
                "version_id": row[0],
                "timestamp": row[1],
                "description": row[2],
                "node_count": row[3],
                "edge_count": row[4]
            })
        
        return versions
    
    def _clear_pending_changes(self):
        """Clear all pending changes."""
        self.pending_changes = {
            "added_nodes": set(),
            "removed_nodes": set(),
            "added_edges": set(),
            "removed_edges": set(),
            "updated_attributes": {}
        }
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()