"""
Example demonstrating the local storage system
"""

from isc_ai import ISCCore
from isc_ai.storage import StorageManager
import time


def main():
    print("=== ISC AI Storage System Example ===\n")
    
    # Initialize components
    core = ISCCore()
    storage = StorageManager("example_storage")
    
    # Start a session
    core.session_active = True
    core.current_session_id = "storage_example"
    
    print("1. Building knowledge through conversation...\n")
    
    # Have a conversation to build knowledge
    conversations = [
        "Let's explore the concept of emergence.",
        "Emergence occurs when simple rules create complex patterns.",
        "Conway's Game of Life is a perfect example of emergence.",
        "Simple cellular automaton rules lead to gliders, oscillators, and complex structures.",
        "This connects to how consciousness might emerge from simple information processing rules.",
        "The whole becomes greater than the sum of its parts in emergent systems.",
    ]
    
    for i, user_input in enumerate(conversations):
        print(f"Human: {user_input}")
        response = core.process_input(user_input)
        print(f"ISC: {response}\n")
        time.sleep(0.5)
    
    # Transfer knowledge graph to storage
    storage.graph_db.graph = core.knowledge_graph.graph
    
    print("\n2. Saving knowledge graph...")
    save_result = storage.save("emergence_v1", "Knowledge about emergence and complexity")
    print(f"   Saved as version: {save_result['version_id']}")
    print(f"   Nodes: {save_result['stats']['graph_stats']['nodes']}")
    print(f"   Edges: {save_result['stats']['graph_stats']['edges']}")
    
    print("\n3. Querying the saved graph...\n")
    
    # Example queries
    queries = [
        "find node emergence",
        "neighbors of patterns",
        "path from rules to consciousness",
        "contains 'complex'",
        "central nodes",
        "statistics"
    ]
    
    for query in queries:
        print(f"Query: '{query}'")
        results = storage.query(query)
        
        if results:
            result = results[0]  # Show first result
            result_type = result.get('type', 'unknown')
            
            if result_type == 'node':
                print(f"  Found node: {result['id']} (degree: {result['degree']})")
            elif result_type == 'neighbors':
                print(f"  Node '{result['node']}' has {result['neighbor_count']} neighbors")
            elif result_type == 'path' and 'nodes' in result:
                path_str = " → ".join(result['nodes'])
                print(f"  Path found: {path_str}")
            elif result_type == 'statistics':
                stats = result.get('basic_stats', {})
                print(f"  Graph has {stats.get('nodes', 0)} nodes and {stats.get('edges', 0)} edges")
            elif result_type == 'central_nodes':
                top_nodes = result.get('top_nodes', [])[:3]
                for node in top_nodes:
                    print(f"  - {node['node']} (centrality: {node['combined_score']:.3f})")
            else:
                print(f"  Result type: {result_type}")
        else:
            print("  No results found")
        print()
    
    print("\n4. Exporting graph...")
    
    # Export in different formats
    formats = ["text", "json", "graphml"]
    for fmt in formats:
        path = storage.export(fmt)
        print(f"   Exported as {fmt}: {path}")
    
    print("\n5. Graph visualization (ASCII):")
    print(storage.visualize())
    
    print("\n6. Adding new knowledge...")
    
    # Add more nodes and connections
    storage.graph_db.add_node("self-organization", type="concept")
    storage.graph_db.add_edge("emergence", "self-organization", weight=0.9)
    storage.graph_db.add_edge("self-organization", "patterns", weight=0.8)
    
    # Save updated version
    save_result = storage.save("emergence_v2", "Added self-organization concept")
    print(f"   Saved as version: {save_result['version_id']}")
    
    print("\n7. Version history:")
    versions = storage.graph_db.list_versions()
    for v in versions:
        print(f"   {v['version_id']}: {v['node_count']} nodes, {v['edge_count']} edges - {v['description']}")
    
    print("\n8. Storage statistics:")
    info = storage.get_info()
    print(f"   Total size: {info['storage_stats']['total_size_mb']} MB")
    print(f"   File count: {info['storage_stats']['file_count']}")
    print(f"   Versions saved: {len(versions)}")
    
    print("\nExample complete! Check 'example_storage/' directory for saved files.")


if __name__ == "__main__":
    main()