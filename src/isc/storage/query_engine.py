"""
Query engine for searching and analyzing the graph
"""

import re
from typing import List, Dict, Any, Optional, Tuple, Set
import networkx as nx
import numpy as np
from collections import defaultdict, Counter
from datetime import datetime
import json


class QueryEngine:
    """
    Advanced query engine for graph search and analysis.
    Supports natural language-like queries and complex graph operations.
    """
    
    def __init__(self, graph_db):
        self.graph_db = graph_db
        self.query_cache = {}
        
        # Query patterns
        self.patterns = {
            'find_node': re.compile(r'find\s+node\s+(\w+)', re.I),
            'find_path': re.compile(r'path\s+from\s+(\w+)\s+to\s+(\w+)', re.I),
            'neighbors_of': re.compile(r'neighbors\s+of\s+(\w+)', re.I),
            'connected_to': re.compile(r'connected\s+to\s+(\w+)', re.I),
            'degree_of': re.compile(r'degree\s+of\s+(\w+)', re.I),
            'shortest_path': re.compile(r'shortest\s+path\s+(\w+)\s+to\s+(\w+)', re.I),
            'contains': re.compile(r'contains\s+"([^"]+)"', re.I),
            'weight_above': re.compile(r'weight\s*>\s*([\d.]+)', re.I),
            'weight_below': re.compile(r'weight\s*<\s*([\d.]+)', re.I),
        }
    
    def query(self, query_string: str) -> List[Dict[str, Any]]:
        """
        Execute a natural language-like query.
        
        Examples:
            - "find node consciousness"
            - "path from input to output"
            - "neighbors of learning"
            - "nodes connected to memory"
            - "edges with weight > 0.5"
            - "nodes contains 'pattern'"
        """
        # Check cache
        cache_key = query_string.lower().strip()
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]
        
        results = []
        graph = self.graph_db.graph
        
        # Parse query type
        query_lower = query_string.lower()
        
        # Node search
        if 'find node' in query_lower or 'find nodes' in query_lower:
            match = self.patterns['find_node'].search(query_string)
            if match:
                pattern = match.group(1)
                results = self._find_nodes_by_pattern(pattern)
            else:
                # Find all nodes if no pattern specified
                results = self._all_nodes_info()
        
        # Path queries
        elif 'path from' in query_lower:
            match = self.patterns['find_path'].search(query_string)
            if match:
                source, target = match.groups()
                results = self._find_paths(source, target)
        
        elif 'shortest path' in query_lower:
            match = self.patterns['shortest_path'].search(query_string)
            if match:
                source, target = match.groups()
                results = self._shortest_path(source, target)
        
        # Neighbor queries
        elif 'neighbors of' in query_lower:
            match = self.patterns['neighbors_of'].search(query_string)
            if match:
                node = match.group(1)
                results = self._get_neighbors(node)
        
        # Connection queries
        elif 'connected to' in query_lower:
            match = self.patterns['connected_to'].search(query_string)
            if match:
                node = match.group(1)
                results = self._find_connected_nodes(node)
        
        # Degree queries
        elif 'degree' in query_lower:
            match = self.patterns['degree_of'].search(query_string)
            if match:
                node = match.group(1)
                results = self._get_degree_info(node)
            else:
                results = self._degree_distribution()
        
        # Content search
        elif 'contains' in query_lower:
            match = self.patterns['contains'].search(query_string)
            if match:
                search_term = match.group(1)
                results = self._search_content(search_term)
        
        # Weight-based queries
        elif 'weight' in query_lower:
            if match := self.patterns['weight_above'].search(query_string):
                threshold = float(match.group(1))
                results = self._edges_by_weight(min_weight=threshold)
            elif match := self.patterns['weight_below'].search(query_string):
                threshold = float(match.group(1))
                results = self._edges_by_weight(max_weight=threshold)
        
        # Complex queries
        elif 'clusters' in query_lower or 'communities' in query_lower:
            results = self._find_clusters()
        
        elif 'central' in query_lower or 'important' in query_lower:
            results = self._find_central_nodes()
        
        elif 'bridges' in query_lower:
            results = self._find_bridges()
        
        elif 'cycles' in query_lower:
            results = self._find_cycles()
        
        elif 'statistics' in query_lower or 'stats' in query_lower:
            results = [self._graph_statistics()]
        
        else:
            # Default: search everything
            results = self._general_search(query_string)
        
        # Cache results
        self.query_cache[cache_key] = results
        
        return results
    
    def _find_nodes_by_pattern(self, pattern: str) -> List[Dict[str, Any]]:
        """Find nodes matching a pattern."""
        results = []
        pattern_lower = pattern.lower()
        
        for node_id, data in self.graph_db.graph.nodes(data=True):
            if pattern_lower in str(node_id).lower():
                results.append({
                    'type': 'node',
                    'id': node_id,
                    'attributes': data,
                    'degree': self.graph_db.graph.degree(node_id),
                    'neighbors': list(self.graph_db.graph.neighbors(node_id))[:5]
                })
        
        return sorted(results, key=lambda x: x['degree'], reverse=True)
    
    def _find_paths(self, source: str, target: str, max_paths: int = 5) -> List[Dict[str, Any]]:
        """Find paths between two nodes."""
        graph = self.graph_db.graph
        
        if source not in graph or target not in graph:
            return [{
                'type': 'error',
                'message': f'Node not found: {source if source not in graph else target}'
            }]
        
        try:
            paths = list(nx.all_simple_paths(graph, source, target, cutoff=6))[:max_paths]
            
            results = []
            for i, path in enumerate(paths):
                path_weight = sum(
                    graph[path[j]][path[j+1]].get('weight', 1.0)
                    for j in range(len(path)-1)
                )
                
                results.append({
                    'type': 'path',
                    'path_id': i + 1,
                    'nodes': path,
                    'length': len(path) - 1,
                    'total_weight': path_weight,
                    'edges': [
                        {
                            'from': path[j],
                            'to': path[j+1],
                            'weight': graph[path[j]][path[j+1]].get('weight', 1.0)
                        }
                        for j in range(len(path)-1)
                    ]
                })
            
            return results
        
        except nx.NetworkXNoPath:
            return [{
                'type': 'no_path',
                'source': source,
                'target': target,
                'message': 'No path exists between these nodes'
            }]
    
    def _shortest_path(self, source: str, target: str) -> List[Dict[str, Any]]:
        """Find shortest path between nodes."""
        graph = self.graph_db.graph
        
        if source not in graph or target not in graph:
            return [{
                'type': 'error',
                'message': f'Node not found'
            }]
        
        try:
            path = nx.shortest_path(graph, source, target)
            path_length = nx.shortest_path_length(graph, source, target)
            
            return [{
                'type': 'shortest_path',
                'path': path,
                'length': path_length,
                'nodes_in_path': len(path)
            }]
        
        except nx.NetworkXNoPath:
            return [{
                'type': 'no_path',
                'message': 'No path exists'
            }]
    
    def _get_neighbors(self, node: str) -> List[Dict[str, Any]]:
        """Get neighbors of a node."""
        graph = self.graph_db.graph
        
        if node not in graph:
            return [{'type': 'error', 'message': f'Node {node} not found'}]
        
        neighbors = []
        for neighbor in graph.neighbors(node):
            edge_data = graph[node][neighbor]
            neighbors.append({
                'node': neighbor,
                'weight': edge_data.get('weight', 1.0),
                'attributes': edge_data
            })
        
        # Sort by weight
        neighbors.sort(key=lambda x: x['weight'], reverse=True)
        
        return [{
            'type': 'neighbors',
            'node': node,
            'neighbor_count': len(neighbors),
            'neighbors': neighbors
        }]
    
    def _find_connected_nodes(self, node: str, max_distance: int = 2) -> List[Dict[str, Any]]:
        """Find all nodes within a certain distance."""
        graph = self.graph_db.graph
        
        if node not in graph:
            return [{'type': 'error', 'message': f'Node {node} not found'}]
        
        connected = defaultdict(list)
        
        # BFS to find connected nodes
        for distance in range(1, max_distance + 1):
            if distance == 1:
                nodes_at_distance = set(graph.neighbors(node))
            else:
                nodes_at_distance = set()
                for n in connected[distance - 1]:
                    nodes_at_distance.update(graph.neighbors(n))
                nodes_at_distance.discard(node)
                for d in range(1, distance):
                    nodes_at_distance -= set(connected[d])
            
            connected[distance] = list(nodes_at_distance)
        
        return [{
            'type': 'connected_nodes',
            'center': node,
            'connections_by_distance': dict(connected),
            'total_connected': sum(len(nodes) for nodes in connected.values())
        }]
    
    def _search_content(self, search_term: str) -> List[Dict[str, Any]]:
        """Search node and edge attributes for content."""
        results = []
        search_lower = search_term.lower()
        
        # Search nodes
        for node_id, data in self.graph_db.graph.nodes(data=True):
            if self._contains_term(data, search_lower) or search_lower in str(node_id).lower():
                results.append({
                    'type': 'node_match',
                    'node': node_id,
                    'attributes': data,
                    'match_context': self._get_match_context(data, search_lower)
                })
        
        # Search edges
        for source, target, data in self.graph_db.graph.edges(data=True):
            if self._contains_term(data, search_lower):
                results.append({
                    'type': 'edge_match',
                    'source': source,
                    'target': target,
                    'attributes': data,
                    'match_context': self._get_match_context(data, search_lower)
                })
        
        return results
    
    def _contains_term(self, data: Dict, term: str) -> bool:
        """Check if dictionary contains search term."""
        for key, value in data.items():
            if term in str(key).lower() or term in str(value).lower():
                return True
        return False
    
    def _get_match_context(self, data: Dict, term: str) -> str:
        """Get context where term was found."""
        contexts = []
        for key, value in data.items():
            if term in str(key).lower():
                contexts.append(f"{key}: ...")
            if term in str(value).lower():
                contexts.append(f"{key}: {str(value)[:50]}...")
        return " | ".join(contexts[:3])
    
    def _edges_by_weight(self, min_weight: Optional[float] = None, 
                        max_weight: Optional[float] = None) -> List[Dict[str, Any]]:
        """Find edges within weight range."""
        edges = []
        
        for source, target, data in self.graph_db.graph.edges(data=True):
            weight = data.get('weight', 1.0)
            
            if min_weight is not None and weight < min_weight:
                continue
            if max_weight is not None and weight > max_weight:
                continue
            
            edges.append({
                'source': source,
                'target': target,
                'weight': weight,
                'attributes': {k: v for k, v in data.items() if k != 'weight'}
            })
        
        # Sort by weight
        edges.sort(key=lambda x: x['weight'], reverse=True)
        
        return [{
            'type': 'edges_by_weight',
            'count': len(edges),
            'weight_range': f"{min_weight or '-∞'} to {max_weight or '+∞'}",
            'edges': edges[:50]  # Limit to 50 results
        }]
    
    def _find_clusters(self) -> List[Dict[str, Any]]:
        """Find communities/clusters in the graph."""
        graph = self.graph_db.graph
        
        if graph.number_of_nodes() == 0:
            return [{'type': 'error', 'message': 'Graph is empty'}]
        
        # Find communities
        communities = list(nx.community.greedy_modularity_communities(graph))
        
        results = []
        for i, community in enumerate(communities):
            community_list = list(community)
            subgraph = graph.subgraph(community_list)
            
            results.append({
                'cluster_id': i + 1,
                'size': len(community_list),
                'nodes': community_list[:20],  # Sample
                'density': nx.density(subgraph),
                'internal_edges': subgraph.number_of_edges()
            })
        
        return [{
            'type': 'clusters',
            'total_clusters': len(communities),
            'modularity': nx.community.modularity(graph, communities),
            'clusters': sorted(results, key=lambda x: x['size'], reverse=True)
        }]
    
    def _find_central_nodes(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """Find most central/important nodes."""
        graph = self.graph_db.graph
        
        if graph.number_of_nodes() == 0:
            return [{'type': 'error', 'message': 'Graph is empty'}]
        
        # Calculate different centrality measures
        degree_cent = nx.degree_centrality(graph)
        between_cent = nx.betweenness_centrality(graph) if graph.number_of_nodes() < 1000 else {}
        close_cent = nx.closeness_centrality(graph) if nx.is_connected(graph) else {}
        
        # Combine scores
        nodes_scores = []
        for node in graph.nodes():
            score = {
                'node': node,
                'degree_centrality': degree_cent.get(node, 0),
                'betweenness_centrality': between_cent.get(node, 0),
                'closeness_centrality': close_cent.get(node, 0),
                'degree': graph.degree(node)
            }
            # Combined score
            score['combined_score'] = (
                score['degree_centrality'] + 
                score['betweenness_centrality'] + 
                score['closeness_centrality']
            ) / 3
            nodes_scores.append(score)
        
        # Sort by combined score
        nodes_scores.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return [{
            'type': 'central_nodes',
            'top_nodes': nodes_scores[:top_k]
        }]
    
    def _find_bridges(self) -> List[Dict[str, Any]]:
        """Find bridge edges (whose removal disconnects the graph)."""
        graph = self.graph_db.graph
        
        if graph.number_of_edges() == 0:
            return [{'type': 'error', 'message': 'No edges in graph'}]
        
        bridges = list(nx.bridges(graph))
        
        bridge_info = []
        for source, target in bridges[:20]:  # Limit to 20
            bridge_info.append({
                'source': source,
                'target': target,
                'weight': graph[source][target].get('weight', 1.0)
            })
        
        return [{
            'type': 'bridges',
            'total_bridges': len(bridges),
            'bridges': bridge_info,
            'message': f'Found {len(bridges)} bridge edges'
        }]
    
    def _find_cycles(self, max_length: int = 5) -> List[Dict[str, Any]]:
        """Find cycles in the graph."""
        graph = self.graph_db.graph
        
        cycles = []
        for cycle in nx.simple_cycles(graph.to_directed()):
            if len(cycle) <= max_length:
                cycles.append(cycle)
            if len(cycles) >= 20:  # Limit results
                break
        
        return [{
            'type': 'cycles',
            'max_length': max_length,
            'cycles_found': len(cycles),
            'cycles': cycles
        }]
    
    def _graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics."""
        stats = self.graph_db.get_statistics()
        
        return {
            'type': 'statistics',
            'basic_stats': stats,
            'advanced_stats': self._advanced_statistics()
        }
    
    def _advanced_statistics(self) -> Dict[str, Any]:
        """Calculate advanced graph metrics."""
        graph = self.graph_db.graph
        
        if graph.number_of_nodes() == 0:
            return {}
        
        stats = {}
        
        # Diameter and radius (for connected graphs)
        if nx.is_connected(graph):
            stats['diameter'] = nx.diameter(graph)
            stats['radius'] = nx.radius(graph)
        
        # Assortativity
        stats['degree_assortativity'] = nx.degree_assortativity_coefficient(graph)
        
        # Transitivity
        stats['transitivity'] = nx.transitivity(graph)
        
        # Node connectivity
        stats['node_connectivity'] = nx.node_connectivity(graph)
        
        # Edge connectivity  
        stats['edge_connectivity'] = nx.edge_connectivity(graph)
        
        return stats
    
    def _general_search(self, query: str) -> List[Dict[str, Any]]:
        """General search when no specific pattern matches."""
        results = []
        
        # Search nodes
        node_results = self._find_nodes_by_pattern(query)
        if node_results:
            results.extend(node_results[:10])
        
        # Search content
        content_results = self._search_content(query)
        if content_results:
            results.extend(content_results[:10])
        
        if not results:
            results.append({
                'type': 'no_results',
                'query': query,
                'suggestions': [
                    'Try: find node <name>',
                    'Try: path from <node1> to <node2>',
                    'Try: neighbors of <node>',
                    'Try: contains "<text>"'
                ]
            })
        
        return results
    
    def clear_cache(self):
        """Clear the query cache."""
        self.query_cache.clear()
    
    def explain_query_syntax(self) -> str:
        """Return explanation of query syntax."""
        return """
Query Syntax Guide:

Basic Queries:
- find node <pattern>         : Find nodes matching pattern
- path from <A> to <B>       : Find paths between nodes
- shortest path <A> to <B>   : Find shortest path
- neighbors of <node>        : Get immediate neighbors
- connected to <node>        : Find nodes within 2 hops
- degree of <node>           : Get degree information

Content Search:
- contains "<text>"          : Search in node/edge attributes

Weight Queries:
- weight > 0.5              : Edges with weight above threshold
- weight < 0.5              : Edges with weight below threshold

Analysis Queries:
- clusters                   : Find communities
- central nodes             : Find important nodes
- bridges                   : Find critical edges
- cycles                    : Find cycles in graph
- statistics                : Get graph statistics

Examples:
- find node learning
- path from input to output
- neighbors of consciousness
- contains "pattern"
- weight > 0.8
"""