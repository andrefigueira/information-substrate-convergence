"""
Graph serialization utilities for efficient storage
"""

import json
import pickle
import gzip
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
import networkx as nx
import msgpack
import base64


class GraphSerializer:
    """
    Handles serialization and deserialization of graph data
    with multiple format options for different use cases.
    """
    
    @staticmethod
    def to_json(graph: nx.Graph, compress: bool = False) -> str:
        """
        Serialize graph to JSON format.
        Good for human readability and web APIs.
        """
        data = nx.node_link_data(graph)
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            return obj
        
        data = convert_numpy(data)
        json_str = json.dumps(data, indent=2 if not compress else None)
        
        if compress:
            return base64.b64encode(
                gzip.compress(json_str.encode())
            ).decode()
        
        return json_str
    
    @staticmethod
    def from_json(json_str: str, compressed: bool = False) -> nx.Graph:
        """Deserialize graph from JSON format."""
        if compressed:
            json_str = gzip.decompress(
                base64.b64decode(json_str.encode())
            ).decode()
        
        data = json.loads(json_str)
        return nx.node_link_graph(data)
    
    @staticmethod
    def to_pickle(graph: nx.Graph, path: Path, compress: bool = True):
        """
        Serialize graph to pickle format.
        Most efficient for large graphs with complex data.
        """
        if compress:
            with gzip.open(path, 'wb') as f:
                pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(path, 'wb') as f:
                pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    @staticmethod
    def from_pickle(path: Path, compressed: bool = True) -> nx.Graph:
        """Deserialize graph from pickle format."""
        if compressed:
            with gzip.open(path, 'rb') as f:
                return pickle.load(f)
        else:
            with open(path, 'rb') as f:
                return pickle.load(f)
    
    @staticmethod
    def to_msgpack(graph: nx.Graph, path: Path):
        """
        Serialize graph to MessagePack format.
        Good balance between efficiency and compatibility.
        """
        data = nx.node_link_data(graph)
        
        # Convert numpy arrays for msgpack
        def convert_for_msgpack(obj):
            if isinstance(obj, np.ndarray):
                return {'__numpy__': True, 'data': obj.tolist(), 
                       'dtype': str(obj.dtype), 'shape': obj.shape}
            elif isinstance(obj, dict):
                return {k: convert_for_msgpack(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_msgpack(i) for i in obj]
            return obj
        
        data = convert_for_msgpack(data)
        
        with open(path, 'wb') as f:
            msgpack.pack(data, f)
    
    @staticmethod
    def from_msgpack(path: Path) -> nx.Graph:
        """Deserialize graph from MessagePack format."""
        def restore_numpy(obj):
            if isinstance(obj, dict) and obj.get('__numpy__'):
                return np.array(obj['data'], dtype=obj['dtype']).reshape(obj['shape'])
            elif isinstance(obj, dict):
                return {k: restore_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [restore_numpy(i) for i in obj]
            return obj
        
        with open(path, 'rb') as f:
            data = msgpack.unpack(f, strict_map_key=False)
        
        data = restore_numpy(data)
        return nx.node_link_graph(data)
    
    @staticmethod
    def to_graphml(graph: nx.Graph, path: Path):
        """
        Export to GraphML format.
        Good for interoperability with other graph tools.
        """
        # Convert complex attributes to strings
        for node, attrs in graph.nodes(data=True):
            for key, value in list(attrs.items()):
                if isinstance(value, (dict, list, np.ndarray)):
                    attrs[key] = json.dumps(value)
        
        for u, v, attrs in graph.edges(data=True):
            for key, value in list(attrs.items()):
                if isinstance(value, (dict, list, np.ndarray)):
                    attrs[key] = json.dumps(value)
        
        nx.write_graphml(graph, path)
    
    @staticmethod
    def to_adjacency_list(graph: nx.Graph, path: Path):
        """
        Export to simple adjacency list format.
        Human-readable and easy to process.
        """
        with open(path, 'w') as f:
            for node in sorted(graph.nodes()):
                neighbors = sorted(graph.neighbors(node))
                if neighbors:
                    f.write(f"{node}: {' '.join(neighbors)}\n")
                else:
                    f.write(f"{node}:\n")
    
    @staticmethod
    def to_edge_list(graph: nx.Graph, path: Path, include_weights: bool = True):
        """
        Export to edge list format.
        Simple and widely supported.
        """
        with open(path, 'w') as f:
            for source, target, data in sorted(graph.edges(data=True)):
                if include_weights and 'weight' in data:
                    f.write(f"{source} {target} {data['weight']}\n")
                else:
                    f.write(f"{source} {target}\n")
    
    @staticmethod
    def estimate_size(graph: nx.Graph) -> Dict[str, int]:
        """
        Estimate storage size for different formats.
        Returns size in bytes.
        """
        import tempfile
        import os
        
        sizes = {}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # JSON size
            json_str = GraphSerializer.to_json(graph)
            sizes['json'] = len(json_str.encode())
            sizes['json_compressed'] = len(
                gzip.compress(json_str.encode())
            )
            
            # Pickle size
            pickle_path = tmpdir / "test.pkl"
            GraphSerializer.to_pickle(graph, pickle_path, compress=False)
            sizes['pickle'] = os.path.getsize(pickle_path)
            
            pickle_gz_path = tmpdir / "test.pkl.gz"
            GraphSerializer.to_pickle(graph, pickle_gz_path, compress=True)
            sizes['pickle_compressed'] = os.path.getsize(pickle_gz_path)
            
            # MessagePack size
            msgpack_path = tmpdir / "test.msgpack"
            GraphSerializer.to_msgpack(graph, msgpack_path)
            sizes['msgpack'] = os.path.getsize(msgpack_path)
        
        return sizes