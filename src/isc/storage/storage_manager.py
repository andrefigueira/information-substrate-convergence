"""
High-level storage manager that coordinates all storage operations
"""

import os
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json
import numpy as np

from .local_graph_db import LocalGraphDB
from .graph_serializer import GraphSerializer
from .query_engine import QueryEngine


class StorageManager:
    """
    Manages the complete storage system for ISC AI,
    coordinating between graph database, serialization, and queries.
    """
    
    def __init__(self, base_dir: str = "isc_storage"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.graph_db = LocalGraphDB(str(self.base_dir / "graph"))
        self.query_engine = QueryEngine(self.graph_db)
        self.serializer = GraphSerializer()
        
        # Storage configuration
        self.config = self._load_config()
        
        # Metadata tracking
        self.metadata_file = self.base_dir / "metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load or create storage configuration."""
        config_file = self.base_dir / "storage_config.json"
        
        default_config = {
            "auto_save_interval": 300,  # seconds
            "max_versions": 50,
            "compression": True,
            "export_formats": ["json", "graphml", "text"],
            "backup_days": 7,
            "query_cache_size": 100
        }
        
        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
        else:
            config = default_config
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
        
        return config
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load or create metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                return json.load(f)
        
        metadata = {
            "created_at": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat(),
            "total_saves": 0,
            "total_queries": 0,
            "storage_stats": {}
        }
        
        self._save_metadata(metadata)
        return metadata
    
    def _save_metadata(self, metadata: Optional[Dict] = None):
        """Save metadata to file."""
        if metadata:
            self.metadata = metadata
        
        self.metadata["last_modified"] = datetime.now().isoformat()
        
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def save(self, version_tag: Optional[str] = None, 
             description: str = "", 
             auto_export: bool = True) -> Dict[str, Any]:
        """
        Save current graph state with all configured options.
        
        Returns:
            Dictionary with save results and statistics
        """
        # Save to graph database
        version_id = self.graph_db.save_graph(version_tag, description)
        
        # Update metadata
        self.metadata["total_saves"] += 1
        self.metadata["last_save"] = datetime.now().isoformat()
        self.metadata["last_version"] = version_id
        
        # Calculate storage statistics
        stats = self._calculate_storage_stats()
        self.metadata["storage_stats"] = stats
        
        # Auto-export if enabled
        export_results = {}
        if auto_export:
            for format_type in self.config["export_formats"]:
                try:
                    export_path = self.export(format_type, version_id)
                    export_results[format_type] = str(export_path)
                except Exception as e:
                    export_results[format_type] = f"Error: {str(e)}"
        
        # Clean old versions if needed
        self._cleanup_old_versions()
        
        # Save metadata
        self._save_metadata()
        
        return {
            "version_id": version_id,
            "timestamp": datetime.now().isoformat(),
            "stats": stats,
            "exports": export_results,
            "description": description
        }
    
    def load(self, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Load a specific version or latest.
        
        Returns:
            Dictionary with load results
        """
        graph = self.graph_db.load_graph(version)
        
        # Clear query cache when loading new graph
        self.query_engine.clear_cache()
        
        return {
            "version": self.graph_db.current_version,
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "loaded_at": datetime.now().isoformat()
        }
    
    def query(self, query_string: str) -> List[Dict[str, Any]]:
        """
        Execute a query on the graph.
        
        Supports natural language-like queries.
        """
        self.metadata["total_queries"] += 1
        
        results = self.query_engine.query(query_string)
        
        # Log popular queries
        if "popular_queries" not in self.metadata:
            self.metadata["popular_queries"] = {}
        
        query_key = query_string.lower().strip()
        self.metadata["popular_queries"][query_key] = \
            self.metadata["popular_queries"].get(query_key, 0) + 1
        
        return results
    
    def update(self, updates: Dict[str, Any], auto_save: bool = False) -> Dict[str, Any]:
        """
        Apply updates to the graph.
        
        Args:
            updates: Update dictionary (see LocalGraphDB.update_graph)
            auto_save: Whether to save after updates
        
        Returns:
            Update statistics
        """
        # Track stats before update
        before_nodes = self.graph_db.graph.number_of_nodes()
        before_edges = self.graph_db.graph.number_of_edges()
        
        # Apply updates
        self.graph_db.update_graph(updates)
        
        # Calculate changes
        after_nodes = self.graph_db.graph.number_of_nodes()
        after_edges = self.graph_db.graph.number_of_edges()
        
        stats = {
            "nodes_added": after_nodes - before_nodes,
            "edges_added": after_edges - before_edges,
            "timestamp": datetime.now().isoformat()
        }
        
        # Auto-save if requested
        if auto_save:
            save_result = self.save(
                description=f"Auto-save after update: +{stats['nodes_added']} nodes, +{stats['edges_added']} edges"
            )
            stats["saved_version"] = save_result["version_id"]
        
        return stats
    
    def export(self, format_type: str, version: Optional[str] = None) -> Path:
        """
        Export graph to various formats.
        
        Supported formats: json, graphml, text, adjacency, edgelist, pickle
        """
        # Ensure we have the right version loaded
        if version and version != self.graph_db.current_version:
            self.load(version)
        
        export_dir = self.base_dir / "exports" / (version or self.graph_db.current_version or "latest")
        export_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format_type == "json":
            path = export_dir / f"graph_{timestamp}.json"
            json_str = self.serializer.to_json(self.graph_db.graph, compress=False)
            path.write_text(json_str)
        
        elif format_type == "json_compressed":
            path = export_dir / f"graph_{timestamp}.json.gz"
            json_str = self.serializer.to_json(self.graph_db.graph, compress=True)
            path.write_text(json_str)
        
        elif format_type == "graphml":
            path = export_dir / f"graph_{timestamp}.graphml"
            self.serializer.to_graphml(self.graph_db.graph, path)
        
        elif format_type == "text":
            path = export_dir / f"graph_{timestamp}.txt"
            text = self.graph_db.export_to_text()
            path.write_text(text)
        
        elif format_type == "adjacency":
            path = export_dir / f"graph_{timestamp}_adj.txt"
            self.serializer.to_adjacency_list(self.graph_db.graph, path)
        
        elif format_type == "edgelist":
            path = export_dir / f"graph_{timestamp}_edges.txt"
            self.serializer.to_edge_list(self.graph_db.graph, path)
        
        elif format_type == "pickle":
            path = export_dir / f"graph_{timestamp}.pkl.gz"
            self.serializer.to_pickle(self.graph_db.graph, path)
        
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        return path
    
    def import_graph(self, file_path: Path, format_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Import a graph from file.
        
        Auto-detects format if not specified.
        """
        file_path = Path(file_path)
        
        # Auto-detect format
        if format_type is None:
            if file_path.suffix == ".json":
                format_type = "json"
            elif file_path.suffix == ".graphml":
                format_type = "graphml"
            elif file_path.suffix in [".pkl", ".pickle"]:
                format_type = "pickle"
            else:
                raise ValueError("Cannot auto-detect format. Please specify format_type.")
        
        # Import based on format
        if format_type == "json":
            graph = self.serializer.from_json(file_path.read_text())
        elif format_type == "pickle":
            graph = self.serializer.from_pickle(file_path)
        else:
            raise ValueError(f"Unsupported import format: {format_type}")
        
        # Replace current graph
        self.graph_db.graph = graph
        
        # Save as new version
        save_result = self.save(
            description=f"Imported from {file_path.name}"
        )
        
        return {
            "imported_from": str(file_path),
            "format": format_type,
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "saved_as": save_result["version_id"]
        }
    
    def backup(self) -> Path:
        """
        Create a full backup of the storage system.
        """
        backup_dir = self.base_dir / "backups"
        backup_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"backup_{timestamp}"
        
        # Copy entire storage directory
        shutil.copytree(
            self.base_dir,
            backup_path,
            ignore=shutil.ignore_patterns("backups", "*.tmp")
        )
        
        # Create backup metadata
        backup_meta = {
            "created_at": datetime.now().isoformat(),
            "storage_stats": self._calculate_storage_stats(),
            "versions": self.graph_db.list_versions()
        }
        
        with open(backup_path / "backup_metadata.json", 'w') as f:
            json.dump(backup_meta, f, indent=2)
        
        # Clean old backups
        self._cleanup_old_backups()
        
        return backup_path
    
    def restore(self, backup_path: Path) -> Dict[str, Any]:
        """
        Restore from a backup.
        """
        backup_path = Path(backup_path)
        
        if not backup_path.exists():
            raise ValueError(f"Backup not found: {backup_path}")
        
        # Create restore point
        restore_backup = self.backup()
        
        try:
            # Close current connections
            self.graph_db.close()
            
            # Copy backup files
            for item in backup_path.iterdir():
                if item.name not in ["backups", "backup_metadata.json"]:
                    dest = self.base_dir / item.name
                    if item.is_dir():
                        shutil.rmtree(dest, ignore_errors=True)
                        shutil.copytree(item, dest)
                    else:
                        shutil.copy2(item, dest)
            
            # Reinitialize components
            self.graph_db = LocalGraphDB(str(self.base_dir / "graph"))
            self.query_engine = QueryEngine(self.graph_db)
            self.metadata = self._load_metadata()
            
            # Load latest version
            self.load()
            
            return {
                "restored_from": str(backup_path),
                "restore_backup": str(restore_backup),
                "status": "success"
            }
        
        except Exception as e:
            # Restore from restore point
            self.restore(restore_backup)
            raise Exception(f"Restore failed: {e}")
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the storage system.
        """
        stats = self._calculate_storage_stats()
        
        return {
            "base_directory": str(self.base_dir),
            "current_version": self.graph_db.current_version,
            "graph_stats": self.graph_db.get_statistics(),
            "storage_stats": stats,
            "metadata": self.metadata,
            "config": self.config,
            "versions": self.graph_db.list_versions()[-10:],  # Last 10 versions
            "query_syntax": self.query_engine.explain_query_syntax()
        }
    
    def visualize(self, node_subset: Optional[List[str]] = None) -> str:
        """
        Get ASCII visualization of the graph.
        """
        return self.graph_db.visualize_ascii(
            set(node_subset) if node_subset else None
        )
    
    def _calculate_storage_stats(self) -> Dict[str, Any]:
        """Calculate storage usage statistics."""
        total_size = 0
        file_count = 0
        
        for path in self.base_dir.rglob("*"):
            if path.is_file():
                total_size += path.stat().st_size
                file_count += 1
        
        # Size breakdown by directory
        size_by_dir = {}
        for subdir in ["graph", "exports", "backups"]:
            dir_path = self.base_dir / subdir
            if dir_path.exists():
                dir_size = sum(
                    f.stat().st_size for f in dir_path.rglob("*") if f.is_file()
                )
                size_by_dir[subdir] = dir_size
        
        return {
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "file_count": file_count,
            "size_by_directory": size_by_dir,
            "largest_files": self._find_largest_files(5)
        }
    
    def _find_largest_files(self, n: int = 5) -> List[Dict[str, Any]]:
        """Find the n largest files in storage."""
        files = []
        
        for path in self.base_dir.rglob("*"):
            if path.is_file():
                files.append({
                    "path": str(path.relative_to(self.base_dir)),
                    "size_bytes": path.stat().st_size,
                    "size_mb": round(path.stat().st_size / (1024 * 1024), 2)
                })
        
        files.sort(key=lambda x: x["size_bytes"], reverse=True)
        return files[:n]
    
    def _cleanup_old_versions(self):
        """Remove old versions beyond the configured maximum."""
        versions = self.graph_db.list_versions()
        
        if len(versions) > self.config["max_versions"]:
            versions_to_remove = versions[self.config["max_versions"]:]
            
            for version_info in versions_to_remove:
                version_id = version_info["version_id"]
                
                # Remove snapshot
                snapshot_dir = self.base_dir / "graph" / "snapshots"
                for snapshot in snapshot_dir.glob(f"{version_id}*"):
                    snapshot.unlink()
                
                # Remove exports
                export_dir = self.base_dir / "exports" / version_id
                if export_dir.exists():
                    shutil.rmtree(export_dir)
    
    def _cleanup_old_backups(self):
        """Remove backups older than configured days."""
        backup_dir = self.base_dir / "backups"
        cutoff_date = datetime.now() - timedelta(days=self.config["backup_days"])
        
        for backup_path in backup_dir.iterdir():
            if backup_path.is_dir():
                # Parse timestamp from directory name
                try:
                    timestamp_str = backup_path.name.split("_", 1)[1]
                    backup_date = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    
                    if backup_date < cutoff_date:
                        shutil.rmtree(backup_path)
                except (ValueError, IndexError):
                    # Skip if can't parse timestamp
                    pass