#!/usr/bin/env python3
"""
Migrate old timestamped state files to the new single-file system
"""

import sys
import glob
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from isc.persistence import PersistenceManager

def main():
    """Run state migration"""
    print("ISC AI State Migration Tool")
    print("=" * 40)
    
    # Find all .pt files in current directory
    pt_files = glob.glob("*.pt")
    
    if not pt_files:
        print("No .pt files found in current directory")
        return
    
    print(f"\nFound {len(pt_files)} state files:")
    for i, f in enumerate(pt_files[:10]):  # Show first 10
        print(f"  {i+1}. {f}")
    
    if len(pt_files) > 10:
        print(f"  ... and {len(pt_files) - 10} more")
    
    # Initialize persistence manager
    pm = PersistenceManager()
    
    # Run migration
    print("\nRunning migration...")
    pm.migrate_legacy_files()
    
    print("\n✓ Migration complete!")
    print(f"  - State saved to: {pm.state_path}")
    print(f"  - Backups in: {pm.backup_dir}")
    
    # Cleanup recommendation
    remaining_files = glob.glob("*.pt")
    if remaining_files:
        print(f"\n{len(remaining_files)} .pt files remain in current directory")
        print("These have been backed up and can be safely removed if desired")

if __name__ == "__main__":
    main()