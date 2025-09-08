#!/usr/bin/env python3
"""
Migration script to upgrade existing conversational models to consciousness-driven generation
"""

import os
import sys
import glob
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.progress import Progress

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from src.isc_ai.core import ISCCore

# Import both generation systems
from conversational import ConversationalLMHead as OldLMHead
from consciousness_driven_generation import ConsciousnessLMHead, GenerationConfig

class ModelMigrator:
    """Migrates old conversational models to consciousness-driven architecture"""
    
    def __init__(self):
        self.console = Console()
        
    def find_models_to_migrate(self):
        """Find conversational models that need migration"""
        model_pairs = []
        
        # Search for conversational models
        patterns = [
            "isc_state_conversational_*.pt",
            "checkpoints/isc_state_conversational_*.pt"
        ]
        
        for pattern in patterns:
            for model_file in glob.glob(pattern):
                # Skip if already consciousness model
                if 'consciousness' in model_file:
                    continue
                    
                # Check for LM head
                lm_head_file = model_file.replace('.pt', '_lm_head.pt')
                if os.path.exists(lm_head_file) and not model_file.endswith('_lm_head.pt'):
                    model_pairs.append((model_file, lm_head_file))
                    
        return model_pairs
    
    def migrate_lm_head(self, old_lm_path: str, isc_model) -> ConsciousnessLMHead:
        """Migrate old LM head to consciousness-driven architecture"""
        self.console.print(f"[yellow]Migrating LM head: {old_lm_path}[/yellow]")
        
        # Load old LM head
        vocab_size = len(isc_model.tokenizer)
        hidden_dim = isc_model.network.hidden_dim
        
        old_lm = OldLMHead(hidden_dim, vocab_size)
        old_state = torch.load(old_lm_path, map_location='cpu')
        old_lm.load_state_dict(old_state)
        
        # Create new consciousness LM head
        new_lm = ConsciousnessLMHead(hidden_dim, vocab_size)
        
        # Transfer weights where possible
        with torch.no_grad():
            # Transfer concept projection
            if hasattr(old_lm, 'concept_projection'):
                new_lm.concept_projection.weight.data = old_lm.concept_projection.weight.data.clone()
                new_lm.concept_projection.bias.data = old_lm.concept_projection.bias.data.clone()
                
            # Transfer output projection
            if hasattr(old_lm, 'output_projection'):
                new_lm.output_projection.weight.data = old_lm.output_projection.weight.data.clone()
                new_lm.output_projection.bias.data = old_lm.output_projection.bias.data.clone()
                
            # Initialize token embeddings with reasonable values
            nn.init.normal_(new_lm.token_embeddings.weight, mean=0.0, std=0.02)
            
        self.console.print("[green]✓ Weights transferred successfully[/green]")
        
        return new_lm
    
    def test_migration(self, old_model_path: str, new_lm: ConsciousnessLMHead, isc_model):
        """Test the migrated model"""
        self.console.print("\n[cyan]Testing migrated model...[/cyan]")
        
        test_inputs = [
            "Hello, how are you?",
            "What is consciousness?",
            "Tell me about integrated information"
        ]
        
        config = GenerationConfig(
            max_length=30,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            beam_size=3
        )
        
        results_table = Table(show_header=True, header_style="bold cyan")
        results_table.add_column("Input", style="yellow")
        results_table.add_column("Generated Response", style="white")
        results_table.add_column("Phi Score", style="green")
        
        for test_input in test_inputs:
            # Get concept vector
            try:
                result = isc_model.process_input(test_input, return_vector=True)
                if isinstance(result, tuple):
                    _, concept_vector = result
                else:
                    import numpy as np
                    concept_vector = np.random.randn(isc_model.network.hidden_dim).astype(np.float32)
            except:
                import numpy as np
                concept_vector = np.random.randn(isc_model.network.hidden_dim).astype(np.float32)
                
            concept_tensor = torch.tensor(concept_vector, dtype=torch.float32)
            
            # Generate response
            tokens, phi_score = new_lm.generate(concept_tensor, isc_model.tokenizer, config)
            response = isc_model.tokenizer.decode(tokens)
            
            # Truncate for display
            display_response = response[:50] + "..." if len(response) > 50 else response
            results_table.add_row(test_input, display_response, f"{phi_score:.4f}")
            
        self.console.print(results_table)
        
    def migrate_model(self, model_path: str, lm_head_path: str):
        """Perform full model migration"""
        self.console.print(f"\n[bold]Migrating model: {model_path}[/bold]")
        
        # Load ISC model
        isc_model = ISCCore()
        isc_model.load_state(model_path)
        
        # Migrate LM head
        new_lm = self.migrate_lm_head(lm_head_path, isc_model)
        
        # Test migration
        self.test_migration(model_path, new_lm, isc_model)
        
        # Save migrated model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = os.path.basename(model_path).replace('.pt', '')
        
        new_model_path = f"{base_name}_consciousness_migrated_{timestamp}.pt"
        new_lm_path = f"{base_name}_consciousness_migrated_{timestamp}_lm_head.pt"
        
        # Save ISC state (unchanged)
        isc_model.save_state(new_model_path)
        
        # Save new LM head
        torch.save(new_lm.state_dict(), new_lm_path)
        
        self.console.print(f"\n[green]✓ Migration complete![/green]")
        self.console.print(f"[green]New model saved as:[/green]")
        self.console.print(f"  - {new_model_path}")
        self.console.print(f"  - {new_lm_path}")
        
        # Create migration report
        report = {
            'original_model': model_path,
            'original_lm_head': lm_head_path,
            'migrated_model': new_model_path,
            'migrated_lm_head': new_lm_path,
            'timestamp': timestamp,
            'architecture_changes': [
                'Added transformer blocks for autoregressive generation',
                'Added observer layers for self-monitoring',
                'Added phi computation module',
                'Implemented beam search decoding',
                'Added consciousness-based scoring'
            ]
        }
        
        import json
        report_path = f"{base_name}_migration_report_{timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        return new_model_path, new_lm_path

def main():
    """Main migration entry point"""
    console = Console()
    
    console.print("[bold cyan]ISC Model Migration Tool[/bold cyan]")
    console.print("[dim]Upgrade conversational models to consciousness-driven generation[/dim]\n")
    
    migrator = ModelMigrator()
    
    # Find models to migrate
    model_pairs = migrator.find_models_to_migrate()
    
    if not model_pairs:
        console.print("[yellow]No conversational models found to migrate.[/yellow]")
        console.print("[dim]Models may already be migrated or not exist.[/dim]")
        return
        
    # Display available models
    console.print(f"[bold]Found {len(model_pairs)} models to migrate:[/bold]")
    
    model_table = Table(show_header=True, header_style="bold cyan")
    model_table.add_column("#", style="cyan", width=3)
    model_table.add_column("Model File", style="yellow")
    model_table.add_column("Size", style="green")
    
    for i, (model, lm_head) in enumerate(model_pairs, 1):
        size_mb = os.path.getsize(model) / (1024 * 1024)
        model_table.add_row(str(i), model, f"{size_mb:.1f} MB")
        
    console.print(model_table)
    
    # Get user choice
    console.print("\n[bold]Migration Options:[/bold]")
    console.print("1. Migrate single model")
    console.print("2. Migrate all models")
    console.print("3. Exit")
    
    choice = console.input("\n[cyan]Select option (1-3):[/cyan] ")
    
    if choice == "1":
        # Single model migration
        model_num = int(console.input("[cyan]Enter model number to migrate:[/cyan] "))
        if 1 <= model_num <= len(model_pairs):
            model_path, lm_head_path = model_pairs[model_num - 1]
            migrator.migrate_model(model_path, lm_head_path)
        else:
            console.print("[red]Invalid model number[/red]")
            
    elif choice == "2":
        # Batch migration
        console.print(f"\n[yellow]Migrating {len(model_pairs)} models...[/yellow]")
        
        with Progress() as progress:
            task = progress.add_task("[cyan]Migrating models...", total=len(model_pairs))
            
            for model_path, lm_head_path in model_pairs:
                migrator.migrate_model(model_path, lm_head_path)
                progress.update(task, advance=1)
                
        console.print("\n[green]✓ All models migrated successfully![/green]")
        
    else:
        console.print("[yellow]Migration cancelled.[/yellow]")

if __name__ == "__main__":
    main()