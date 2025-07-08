"""
Knowledge Graph module for concept representation and connections
"""

import networkx as nx
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import json


class KnowledgeGraph:
    """
    Represents the system's knowledge as a graph where nodes are concepts
    and edges represent learned relationships.
    """
    
    def __init__(self):
        self.graph = nx.Graph()
        self.concept_embeddings = {}
        self.concept_frequencies = defaultdict(int)
        self.edge_strengths = {}
        self.concept_contexts = defaultdict(list)
        
    def add_concept(self, concept: str, embedding: Optional[torch.Tensor] = None):
        """Add a concept to the knowledge graph."""
        concept = concept.lower()
        self.concept_frequencies[concept] += 1
        
        if concept not in self.graph:
            self.graph.add_node(concept)
            
        if embedding is not None:
            if concept not in self.concept_embeddings:
                self.concept_embeddings[concept] = []
            self.concept_embeddings[concept].append(
                embedding.detach().cpu().numpy()
            )
    
    def add_connection(self, concept1: str, concept2: str, strength: float = 1.0):
        """Add or strengthen a connection between two concepts."""
        concept1, concept2 = concept1.lower(), concept2.lower()
        
        # Ensure both concepts exist
        for concept in [concept1, concept2]:
            if concept not in self.graph:
                self.add_concept(concept)
        
        # Add edge with weight
        if self.graph.has_edge(concept1, concept2):
            # Strengthen existing connection
            current_weight = self.graph[concept1][concept2].get('weight', 1.0)
            new_weight = current_weight + strength * 0.1
            self.graph[concept1][concept2]['weight'] = min(new_weight, 10.0)
        else:
            self.graph.add_edge(concept1, concept2, weight=strength)
    
    def get_related_concepts(self, concept: str, k: int = 5) -> List[str]:
        """Get the k most related concepts to a given concept."""
        concept = concept.lower()
        
        if concept not in self.graph:
            return []
        
        # Get neighbors with weights
        neighbors = []
        for neighbor in self.graph.neighbors(concept):
            weight = self.graph[concept][neighbor].get('weight', 1.0)
            neighbors.append((neighbor, weight))
        
        # Sort by weight and return top k
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return [n[0] for n in neighbors[:k]]
    
    def get_concept_path(self, concept1: str, concept2: str) -> Optional[List[str]]:
        """Find the shortest path between two concepts."""
        concept1, concept2 = concept1.lower(), concept2.lower()
        
        if concept1 not in self.graph or concept2 not in self.graph:
            return None
        
        try:
            path = nx.shortest_path(self.graph, concept1, concept2)
            return path
        except nx.NetworkXNoPath:
            return None
    
    def get_central_concepts(self, k: int = 10) -> List[str]:
        """Get the k most central concepts in the knowledge graph."""
        if not self.graph.nodes():
            return []
        
        # Calculate centrality
        centrality = nx.eigenvector_centrality_numpy(self.graph, max_iter=100)
        
        # Sort by centrality
        sorted_concepts = sorted(
            centrality.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return [concept for concept, _ in sorted_concepts[:k]]
    
    def get_concept_clusters(self) -> List[Set[str]]:
        """Identify clusters of related concepts."""
        if len(self.graph) < 3:
            return [set(self.graph.nodes())]
        
        # Find connected components
        clusters = []
        for component in nx.connected_components(self.graph):
            if len(component) > 1:
                clusters.append(component)
        
        return clusters
    
    def merge_concepts(self, concept1: str, concept2: str, new_name: str):
        """Merge two concepts into one."""
        concept1, concept2 = concept1.lower(), concept2.lower()
        new_name = new_name.lower()
        
        if concept1 not in self.graph or concept2 not in self.graph:
            return
        
        # Combine connections
        all_neighbors = set()
        for concept in [concept1, concept2]:
            all_neighbors.update(self.graph.neighbors(concept))
        all_neighbors.discard(concept1)
        all_neighbors.discard(concept2)
        
        # Remove old concepts
        self.graph.remove_node(concept1)
        self.graph.remove_node(concept2)
        
        # Add new concept with combined connections
        self.add_concept(new_name)
        for neighbor in all_neighbors:
            self.add_connection(new_name, neighbor)
        
        # Combine embeddings
        if concept1 in self.concept_embeddings and concept2 in self.concept_embeddings:
            combined_embeddings = (
                self.concept_embeddings[concept1] + 
                self.concept_embeddings[concept2]
            )
            self.concept_embeddings[new_name] = combined_embeddings
            del self.concept_embeddings[concept1]
            del self.concept_embeddings[concept2]
    
    def prune_weak_connections(self, threshold: float = 0.5):
        """Remove connections below a certain strength threshold."""
        edges_to_remove = []
        
        for u, v, data in self.graph.edges(data=True):
            if data.get('weight', 1.0) < threshold:
                edges_to_remove.append((u, v))
        
        for edge in edges_to_remove:
            self.graph.remove_edge(*edge)
    
    def get_graph_metrics(self) -> Dict[str, float]:
        """Calculate various metrics about the knowledge graph."""
        if not self.graph.nodes():
            return {
                "num_concepts": 0,
                "num_connections": 0,
                "avg_connections": 0.0,
                "density": 0.0,
                "clustering_coefficient": 0.0,
            }
        
        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()
        
        return {
            "num_concepts": num_nodes,
            "num_connections": num_edges,
            "avg_connections": 2 * num_edges / num_nodes if num_nodes > 0 else 0,
            "density": nx.density(self.graph),
            "clustering_coefficient": nx.average_clustering(self.graph),
        }
    
    def visualize_ascii(self, max_nodes: int = 20) -> str:
        """Create an ASCII visualization of the knowledge graph."""
        if not self.graph.nodes():
            return "Knowledge graph is empty."
        
        # Get most central nodes for visualization
        central_concepts = self.get_central_concepts(min(max_nodes, len(self.graph)))
        
        # Create subgraph with central concepts
        subgraph = self.graph.subgraph(central_concepts)
        
        # Simple ASCII representation
        ascii_viz = "Knowledge Graph Structure:\n"
        ascii_viz += "=" * 50 + "\n\n"
        
        # Show nodes and their connections
        for node in subgraph.nodes():
            neighbors = list(subgraph.neighbors(node))
            if neighbors:
                ascii_viz += f"[{node}] --> {', '.join(neighbors)}\n"
            else:
                ascii_viz += f"[{node}] (isolated)\n"
        
        ascii_viz += "\n" + "=" * 50 + "\n"
        ascii_viz += f"Total concepts: {len(self.graph)}\n"
        ascii_viz += f"Total connections: {self.graph.number_of_edges()}\n"
        
        return ascii_viz
    
    def export_to_dict(self) -> Dict:
        """Export the knowledge graph to a dictionary."""
        return {
            "nodes": list(self.graph.nodes()),
            "edges": [
                {
                    "source": u,
                    "target": v,
                    "weight": data.get("weight", 1.0)
                }
                for u, v, data in self.graph.edges(data=True)
            ],
            "concept_frequencies": dict(self.concept_frequencies),
        }
    
    def import_from_dict(self, data: Dict):
        """Import a knowledge graph from a dictionary."""
        self.graph.clear()
        self.concept_frequencies.clear()
        
        # Add nodes
        for node in data.get("nodes", []):
            self.graph.add_node(node)
        
        # Add edges
        for edge in data.get("edges", []):
            self.graph.add_edge(
                edge["source"],
                edge["target"],
                weight=edge.get("weight", 1.0)
            )
        
        # Restore frequencies
        self.concept_frequencies.update(data.get("concept_frequencies", {}))