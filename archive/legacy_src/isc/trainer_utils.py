"""
Utilities for trainer scripts to maintain consistent checkpoint behavior
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


class CheckpointManager:
    """Manages checkpoint files with fixed names to avoid file proliferation"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Fixed filenames
        self.state_file = self.checkpoint_dir / "current_state.pt"
        self.metrics_file = self.checkpoint_dir / "current_metrics.json"
        self.visualization_file = self.checkpoint_dir / "current_progress.png"
        
        # Backup files (keep one previous version)
        self.state_backup = self.checkpoint_dir / "previous_state.pt"
        self.metrics_backup = self.checkpoint_dir / "previous_metrics.json"
        self.visualization_backup = self.checkpoint_dir / "previous_progress.png"
    
    def save_checkpoint(self, core, metrics: Dict[str, Any], exchange_num: int) -> str:
        """Save checkpoint with fixed filenames"""
        # Backup existing files if they exist
        if self.state_file.exists():
            if self.state_backup.exists():
                self.state_backup.unlink()
            self.state_file.rename(self.state_backup)
        
        if self.metrics_file.exists():
            if self.metrics_backup.exists():
                self.metrics_backup.unlink()
            self.metrics_file.rename(self.metrics_backup)
        
        # Save new checkpoint
        core.save_state(str(self.state_file))
        
        # Add metadata to metrics
        metrics_data = {
            "timestamp": datetime.now().isoformat(),
            "exchange_num": exchange_num,
            **metrics
        }
        
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)
        
        return f"Checkpoint saved to {self.checkpoint_dir}"
    
    def save_visualization(self, fig):
        """Save visualization with fixed filename"""
        # Backup existing visualization
        if self.visualization_file.exists():
            if self.visualization_backup.exists():
                self.visualization_backup.unlink()
            self.visualization_file.rename(self.visualization_backup)
        
        # Save new visualization
        fig.savefig(self.visualization_file, dpi=150, bbox_inches='tight')
        
    def load_checkpoint(self, core) -> Optional[Dict[str, Any]]:
        """Load the current checkpoint if it exists"""
        if not self.state_file.exists() or not self.metrics_file.exists():
            return None
        
        # Load state
        core.load_state(str(self.state_file))
        
        # Load metrics
        with open(self.metrics_file, 'r') as f:
            metrics = json.load(f)
        
        return metrics
    
    def get_checkpoint_info(self) -> Dict[str, Any]:
        """Get information about existing checkpoints"""
        info = {
            "checkpoint_dir": str(self.checkpoint_dir),
            "has_current": self.state_file.exists(),
            "has_backup": self.state_backup.exists(),
        }
        
        if self.state_file.exists():
            info["current_size_mb"] = self.state_file.stat().st_size / (1024 * 1024)
            info["current_modified"] = datetime.fromtimestamp(
                self.state_file.stat().st_mtime
            ).isoformat()
        
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                metrics = json.load(f)
                info["current_exchange"] = metrics.get("exchange_num", 0)
                info["current_timestamp"] = metrics.get("timestamp", "unknown")
        
        return info
    
    def cleanup_old_files(self, pattern: str = None):
        """Clean up old timestamped files if they exist"""
        if pattern is None:
            patterns = ["isc_state_*.pt", "training_*.json", "progress_*.png"]
        else:
            patterns = [pattern]
        
        removed_count = 0
        for pattern in patterns:
            for file in self.checkpoint_dir.glob(pattern):
                if file.name not in ["current_state.pt", "previous_state.pt", 
                                   "current_metrics.json", "previous_metrics.json",
                                   "current_progress.png", "previous_progress.png"]:
                    file.unlink()
                    removed_count += 1
        
        return removed_count