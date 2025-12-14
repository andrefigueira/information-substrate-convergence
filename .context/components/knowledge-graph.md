# Knowledge Graph Component

> Location: `isc_ai_system/src/isc_ai/knowledge_graph.py`

## Overview

The Knowledge Graph stores and manages the system's conceptual understanding. It represents concepts as nodes and relationships as weighted edges, enabling semantic reasoning and context retrieval.

## Architecture

### Graph Structure

```
           ┌─────────────┐
           │ consciousness│
           │   [0.89]     │
           └──────┬───────┘
                  │ 0.7
         ┌────────┴────────┐
         │                 │
    ┌────▼────┐       ┌────▼────┐
    │ emergence│       │information│
    │  [0.72]  │       │  [0.85]   │
    └────┬────┘       └────┬─────┘
         │ 0.5              │ 0.6
         │            ┌─────┴─────┐
    ┌────▼────┐      │           │
    │ patterns │◄────▼─────►┌────▼────┐
    │  [0.61]  │    0.4     │ quantum │
    └─────────┘             │  [0.78] │
                            └─────────┘

[Number] = centrality score
Edge weight = connection strength
```

### Data Model

```python
class KnowledgeGraph:
    """
    Dynamic semantic graph for concept representation.

    Node attributes:
    - embedding: np.ndarray (384-dim semantic vector)
    - frequency: int (occurrence count)
    - first_seen: datetime
    - last_updated: datetime

    Edge attributes:
    - weight: float (connection strength, 0-1)
    - co_occurrences: int (times seen together)
    """

    def __init__(self):
        self.graph: nx.Graph = nx.Graph()
        self.embeddings: Dict[str, np.ndarray] = {}
```

## Key Methods

### add_concept

```python
def add_concept(self, concept: str, embedding: np.ndarray) -> None:
    """
    Add concept node to graph.

    If concept exists:
    - Increment frequency
    - Update embedding (rolling average)
    - Update last_updated timestamp

    If new:
    - Create node with initial attributes
    - Store embedding
    """
    if concept in self.graph:
        # Update existing
        self.graph.nodes[concept]['frequency'] += 1
        self.graph.nodes[concept]['last_updated'] = datetime.now()
        # Rolling average embedding
        old_emb = self.embeddings[concept]
        self.embeddings[concept] = 0.9 * old_emb + 0.1 * embedding
    else:
        # Create new
        self.graph.add_node(
            concept,
            frequency=1,
            first_seen=datetime.now(),
            last_updated=datetime.now()
        )
        self.embeddings[concept] = embedding
```

### add_connection

```python
def add_connection(self, source: str, target: str, weight: float = 0.1) -> None:
    """
    Create or strengthen connection between concepts.

    If edge exists:
    - Increment weight by given amount
    - Cap at 1.0
    - Increment co_occurrence count

    If new:
    - Create edge with initial weight
    """
    if self.graph.has_edge(source, target):
        self.graph[source][target]['weight'] += weight
        self.graph[source][target]['weight'] = min(1.0, self.graph[source][target]['weight'])
        self.graph[source][target]['co_occurrences'] += 1
    else:
        self.graph.add_edge(
            source, target,
            weight=weight,
            co_occurrences=1
        )
```

### get_related_concepts

```python
def get_related_concepts(
    self,
    concept: str,
    limit: int = 10,
    method: str = 'hybrid'
) -> List[Tuple[str, float]]:
    """
    Find concepts related to given concept.

    Methods:
    - 'graph': Use graph proximity (neighbors, path length)
    - 'embedding': Use embedding similarity
    - 'hybrid': Combine both (default)

    Returns:
        List of (concept, relevance_score) tuples
    """
    if method == 'graph':
        return self._get_graph_neighbors(concept, limit)
    elif method == 'embedding':
        return self._get_embedding_similar(concept, limit)
    else:
        # Hybrid: weighted combination
        graph_related = self._get_graph_neighbors(concept, limit * 2)
        embed_related = self._get_embedding_similar(concept, limit * 2)
        return self._merge_results(graph_related, embed_related, limit)
```

### get_concept_path

```python
def get_concept_path(self, source: str, target: str) -> Optional[List[str]]:
    """
    Find shortest path between concepts.

    Uses Dijkstra with edge weights inverted (higher weight = shorter distance).

    Returns:
        List of concepts forming path, or None if no path exists
    """
    if source not in self.graph or target not in self.graph:
        return None

    try:
        path = nx.shortest_path(
            self.graph, source, target,
            weight=lambda u, v, d: 1.0 / (d['weight'] + 0.01)
        )
        return path
    except nx.NetworkXNoPath:
        return None
```

### get_central_concepts

```python
def get_central_concepts(self, k: int = 10) -> List[str]:
    """
    Get the k most central concepts in the knowledge graph.

    Uses eigenvector centrality for connected graphs,
    falls back to degree centrality for disconnected graphs.

    Returns:
        List of concept names (most central first)
    """
    if not self.graph.nodes():
        return []

    # Check if graph is connected
    if not nx.is_connected(self.graph):
        # For disconnected graphs, use degree centrality as fallback
        centrality = nx.degree_centrality(self.graph)
    else:
        # Calculate eigenvector centrality for connected graphs
        try:
            centrality = nx.eigenvector_centrality_numpy(self.graph, max_iter=100)
        except nx.NetworkXError:
            # Fallback to degree centrality if eigenvector fails
            centrality = nx.degree_centrality(self.graph)

    # Sort by centrality
    sorted_concepts = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    return [concept for concept, _ in sorted_concepts[:k]]
```

## Graph Operations

### Pruning

```python
def prune_infrequent(self, min_frequency: int = 3) -> int:
    """
    Remove concepts below frequency threshold.

    Returns:
        Number of concepts removed
    """
    to_remove = [
        n for n, d in self.graph.nodes(data=True)
        if d['frequency'] < min_frequency
    ]
    self.graph.remove_nodes_from(to_remove)
    for concept in to_remove:
        del self.embeddings[concept]
    return len(to_remove)
```

### Decay

```python
def apply_decay(self, decay_factor: float = 0.99) -> None:
    """
    Apply temporal decay to edge weights.

    Older, unused connections weaken over time.
    """
    for u, v, d in self.graph.edges(data=True):
        d['weight'] *= decay_factor
        if d['weight'] < 0.01:
            self.graph.remove_edge(u, v)
```

### Clustering

```python
def get_concept_clusters(self, num_clusters: int = 5) -> Dict[int, List[str]]:
    """
    Cluster concepts by semantic similarity.

    Uses embedding-based clustering (K-means on embeddings).

    Returns:
        Dict mapping cluster_id to list of concepts
    """
    from sklearn.cluster import KMeans

    concepts = list(self.embeddings.keys())
    embeddings = np.array([self.embeddings[c] for c in concepts])

    kmeans = KMeans(n_clusters=min(num_clusters, len(concepts)))
    labels = kmeans.fit_predict(embeddings)

    clusters = {}
    for concept, label in zip(concepts, labels):
        clusters.setdefault(label, []).append(concept)
    return clusters
```

## Additional Methods

### Graph Metrics

```python
def get_graph_metrics(self) -> Dict[str, float]:
    """
    Calculate various metrics about the knowledge graph.

    Returns:
        dict: {
            'num_concepts': int,
            'num_connections': int,
            'avg_connections': float,
            'density': float,
            'clustering_coefficient': float
        }
    """
```

### Visualization

```python
def visualize_ascii(self, max_nodes: int = 20) -> str:
    """
    Create an ASCII visualization of the knowledge graph.

    Shows most central nodes and their connections.
    Returns formatted string for terminal display.
    """
```

### Pruning

```python
def prune_weak_connections(self, threshold: float = 0.5) -> None:
    """
    Remove connections below a certain strength threshold.
    Helps keep graph focused on strong relationships.
    """
```

### Concept Clusters

```python
def get_concept_clusters(self) -> List[Set[str]]:
    """
    Identify clusters of related concepts.
    Uses connected components of the graph.

    Returns:
        List of sets, each containing concepts in a cluster
    """
```

## Serialization

### export_to_dict / import_from_dict

```python
def export_to_dict(self) -> Dict:
    """
    Export the knowledge graph to a dictionary.

    Returns:
        dict: {
            'nodes': List[str],
            'edges': List[{'source': str, 'target': str, 'weight': float}],
            'concept_frequencies': Dict[str, int]
        }
    """

def import_from_dict(self, data: Dict) -> None:
    """
    Import a knowledge graph from a dictionary.
    Clears existing graph and replaces with imported data.
    """
```

## Usage Examples

### Building Understanding

```python
from isc_ai.knowledge_graph import KnowledgeGraph

kg = KnowledgeGraph()

# Add concepts from conversation
kg.add_concept("consciousness", embedding_model.encode("consciousness"))
kg.add_concept("information", embedding_model.encode("information"))
kg.add_concept("emergence", embedding_model.encode("emergence"))

# Connect related concepts
kg.add_connection("consciousness", "information", weight=0.3)
kg.add_connection("consciousness", "emergence", weight=0.5)
kg.add_connection("information", "emergence", weight=0.2)
```

### Querying

```python
# Find related concepts
related = kg.get_related_concepts("consciousness", limit=5)
for concept, score in related:
    print(f"  {concept}: {score:.3f}")

# Find path between concepts
path = kg.get_concept_path("quantum", "consciousness")
if path:
    print(" -> ".join(path))

# Get central concepts
central = kg.get_central_concepts(limit=5)
print("Most important concepts:")
for concept, centrality in central:
    print(f"  {concept}: {centrality:.4f}")
```

### Analysis

```python
# Graph statistics
print(f"Concepts: {kg.graph.number_of_nodes()}")
print(f"Connections: {kg.graph.number_of_edges()}")
print(f"Density: {nx.density(kg.graph):.4f}")

# Cluster analysis
clusters = kg.get_concept_clusters(num_clusters=4)
for cluster_id, concepts in clusters.items():
    print(f"Cluster {cluster_id}: {concepts}")
```

## Integration with ISCCore

```python
# In core.py
def _update_knowledge(self, user_input: str, response: str):
    """Update knowledge graph from interaction."""
    # Extract concepts
    concepts = self._extract_concepts(user_input + " " + response)

    # Add to graph
    for concept in concepts:
        embedding = self.encode_text(concept)
        self.knowledge_graph.add_concept(concept, embedding.numpy())

    # Connect co-occurring concepts
    for c1, c2 in itertools.combinations(concepts, 2):
        self.knowledge_graph.add_connection(c1, c2)
```

## Related Components

- [isc-core.md](isc-core.md) - System orchestrator
- [memory.md](memory.md) - Conversation persistence
- [../architecture/overview.md](../architecture/overview.md) - System architecture
