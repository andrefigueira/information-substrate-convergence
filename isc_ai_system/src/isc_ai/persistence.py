"""
Persistence module for ISC AI System
Handles automatic loading and saving of state with single file approach
"""

import os
import glob
import torch
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import shutil


class PersistenceManager:
    """Manages persistent state for ISC AI with automatic loading"""
    
    def __init__(self, state_dir: str = "isc_state", state_filename: str = "isc_ai_state.pt"):
        self.state_dir = Path(state_dir)
        self.state_filename = state_filename
        self.state_path = self.state_dir / self.state_filename
        self.backup_dir = self.state_dir / "backups"
        
        # Create directories if they don't exist
        self.state_dir.mkdir(exist_ok=True)
        self.backup_dir.mkdir(exist_ok=True)
        
    def get_latest_state_path(self) -> Optional[Path]:
        """Get the path to the latest state file"""
        if self.state_path.exists():
            return self.state_path
        
        # Check for legacy timestamped files
        # Exclude LM head files and other non-state files
        legacy_files = [
            f for f in self.state_dir.glob("*.pt")
            if not f.name.endswith('_lm_head.pt') 
            and not f.name.endswith('_head.pt')
            and 'enhanced_model' not in f.name
        ]
        
        if legacy_files:
            # Sort by modification time and return the newest
            latest = max(legacy_files, key=lambda p: p.stat().st_mtime)
            return latest
        
        return None
    
    def load_latest_state(self) -> Optional[Dict[str, Any]]:
        """Load the latest available state"""
        state_path = self.get_latest_state_path()
        
        if not state_path:
            return None
        
        try:
            # Try safe loading first
            state = torch.load(state_path, weights_only=True, map_location='cpu')
            
            # Validate it's a proper ISC state file
            if not isinstance(state, dict):
                print(f"✗ Invalid state file format in {state_path.name}: not a dictionary")
                return None
            
            required_keys = ["network_state", "knowledge_graph", "memory", "metrics"]
            if not all(key in state for key in required_keys):
                print(f"✗ Invalid ISC state file {state_path.name}: missing required keys")
                return None
            
            print(f"✓ Loaded state from {state_path.name}")
            return state
        except Exception:
            try:
                # Fall back to unsafe loading with warning
                import warnings
                warnings.warn(
                    f"Loading {state_path} with weights_only=False. "
                    "This file may contain arbitrary code.",
                    RuntimeWarning
                )
                state = torch.load(state_path, weights_only=False, map_location='cpu')
                
                # Validate even in unsafe mode
                if not isinstance(state, dict):
                    print(f"✗ Invalid state file format in {state_path.name}: not a dictionary")
                    return None
                
                required_keys = ["network_state", "knowledge_graph", "memory", "metrics"]
                if not all(key in state for key in required_keys):
                    print(f"✗ Invalid ISC state file {state_path.name}: missing required keys")
                    return None
                
                print(f"✓ Loaded state from {state_path.name} (unsafe mode)")
                return state
            except Exception as e:
                print(f"✗ Failed to load state from {state_path.name}: {e}")
                return None
    
    def save_state(self, state: Dict[str, Any], create_backup: bool = True) -> str:
        """Save state to the standard location with optional backup"""
        # Create backup of existing state if requested
        if create_backup and self.state_path.exists():
            self._create_backup()
        
        # Save state to standard location
        torch.save(state, self.state_path)
        
        return str(self.state_path)
    
    def _create_backup(self):
        """Create a backup of the current state file"""
        if not self.state_path.exists():
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"isc_ai_state_backup_{timestamp}.pt"
        
        shutil.copy2(self.state_path, backup_path)
        
        # Keep only the last 10 backups
        self._cleanup_old_backups()
    
    def _cleanup_old_backups(self, keep_count: int = 10):
        """Remove old backup files, keeping only the most recent ones"""
        backup_files = sorted(
            self.backup_dir.glob("isc_ai_state_backup_*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        # Remove older backups
        for backup_file in backup_files[keep_count:]:
            backup_file.unlink()
    
    def migrate_legacy_files(self):
        """Migrate from timestamped files to single state file"""
        legacy_files = [f for f in self.state_dir.glob("*.pt") if f != self.state_path]
        
        if not legacy_files:
            return
        
        print(f"Found {len(legacy_files)} legacy state files")
        
        # Find the most recent legacy file
        latest_legacy = max(legacy_files, key=lambda p: p.stat().st_mtime)
        
        # Copy it to the standard location
        shutil.copy2(latest_legacy, self.state_path)
        print(f"✓ Migrated latest state from {latest_legacy.name}")
        
        # Move all legacy files to backup directory
        for legacy_file in legacy_files:
            backup_path = self.backup_dir / legacy_file.name
            shutil.move(str(legacy_file), str(backup_path))
        
        print(f"✓ Moved {len(legacy_files)} legacy files to backup directory")