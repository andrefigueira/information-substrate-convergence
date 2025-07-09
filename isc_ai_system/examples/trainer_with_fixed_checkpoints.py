#!/usr/bin/env python3
"""
Example of using the CheckpointManager for fixed checkpoint files
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.isc_ai.core import ISCCore
from src.isc_ai.trainer_utils import CheckpointManager
import matplotlib.pyplot as plt
import numpy as np


def main():
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(checkpoint_dir="checkpoints")
    
    # Check existing checkpoints
    info = checkpoint_manager.get_checkpoint_info()
    print(f"Checkpoint directory: {info['checkpoint_dir']}")
    print(f"Has current checkpoint: {info['has_current']}")
    print(f"Has backup: {info['has_backup']}")
    
    # Clean up old timestamped files if they exist
    removed = checkpoint_manager.cleanup_old_files()
    if removed > 0:
        print(f"Cleaned up {removed} old checkpoint files")
    
    # Initialize ISC Core
    core = ISCCore()
    core.session_active = True
    
    # Load existing checkpoint if available
    if info['has_current']:
        print(f"\nLoading checkpoint from exchange {info.get('current_exchange', 0)}...")
        metrics = checkpoint_manager.load_checkpoint(core)
        starting_exchange = metrics.get('exchange_num', 0) + 1
    else:
        starting_exchange = 0
    
    # Simulate training
    print(f"\nStarting training from exchange {starting_exchange}...")
    
    phi_values = []
    for i in range(starting_exchange, starting_exchange + 10):
        # Process some input
        response = core.process_input(f"Training input {i}")
        
        # Get current phi
        current_phi = core.integrator.phi_history[-1] if core.integrator.phi_history else 0.0
        phi_values.append(current_phi)
        
        print(f"Exchange {i}: Phi = {current_phi:.3f}")
        
        # Save checkpoint every 5 exchanges
        if (i + 1) % 5 == 0:
            metrics = {
                "session_metrics": {
                    "total_exchanges": i + 1,
                    "phi_progression": phi_values,
                    "average_phi": np.mean(phi_values)
                }
            }
            
            message = checkpoint_manager.save_checkpoint(core, metrics, i + 1)
            print(f"  {message}")
            
            # Create and save visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(range(len(phi_values)), phi_values, 'b-', linewidth=2)
            ax.set_xlabel("Exchange")
            ax.set_ylabel("Phi (Φ)")
            ax.set_title("Training Progress")
            ax.grid(True, alpha=0.3)
            
            checkpoint_manager.save_visualization(fig)
            plt.close()
            print(f"  Visualization saved")
    
    print("\nTraining complete!")
    
    # Show final checkpoint info
    final_info = checkpoint_manager.get_checkpoint_info()
    print(f"\nFinal checkpoint info:")
    print(f"  Size: {final_info.get('current_size_mb', 0):.2f} MB")
    print(f"  Modified: {final_info.get('current_modified', 'unknown')}")
    print(f"  Exchange: {final_info.get('current_exchange', 0)}")


if __name__ == "__main__":
    main()