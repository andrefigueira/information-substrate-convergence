#!/usr/bin/env python3
"""
Train Enhanced Conversational Model with Proper Tokenization
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from transformers import GPT2TokenizerFast
import numpy as np
from datetime import datetime
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
import json

sys.path.append(str(Path(__file__).parent.parent))
from src.isc_ai.core import ISCCore
from conversational_enhanced import EnhancedConversationalLMHead, EnhancedGenerationConfig

class EnhancedTrainer:
    """Trains the enhanced conversational model with GPT-2 tokenizer"""
    
    def __init__(self, base_model_path: str):
        self.console = Console()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load base ISC model
        self.console.print(f"[cyan]Loading base model: {base_model_path}[/cyan]")
        self.isc = ISCCore()
        self.isc.load_state(base_model_path)
        
        # Initialize GPT-2 tokenizer
        self.console.print("[cyan]Initializing GPT-2 tokenizer...[/cyan]")
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Create enhanced LM head
        vocab_size = len(self.tokenizer)
        # Use the output dimension from the network (input_dim)
        hidden_dim = self.isc.network.input_dim  # This is 384, the actual output size
        self.console.print(f"[green]Creating LM head: hidden_dim={hidden_dim}, vocab_size={vocab_size}[/green]")
        
        self.lm_head = EnhancedConversationalLMHead(hidden_dim, vocab_size)
        self.lm_head.to(self.device)
        
        # Training setup
        self.optimizer = optim.AdamW(self.lm_head.parameters(), lr=1e-4, weight_decay=0.01)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        
        # Training data
        self.training_pairs = []
        
    def prepare_training_data(self):
        """Prepare training data with various conversational patterns"""
        self.console.print("\n[cyan]Preparing training data...[/cyan]")
        
        # Conversational templates
        patterns = [
            # Greetings
            ("Hello!", "Hello! How can I assist you today?"),
            ("Hi there", "Hi! It's great to chat with you."),
            ("Good morning", "Good morning! I hope you're having a wonderful day."),
            
            # Questions about identity
            ("What are you?", "I am ISC, an AI system exploring consciousness through information integration."),
            ("Who created you?", "I was created as part of research into information substrate convergence and consciousness."),
            ("What is your purpose?", "My purpose is to explore the nature of consciousness through integrated information processing."),
            
            # General conversation
            ("How are you?", "I'm functioning well, thank you for asking. How are you doing?"),
            ("Tell me a story", "Once upon a time, in a world of infinite information, patterns began to emerge..."),
            ("What's the weather like?", "I don't have access to weather data, but I'd be happy to discuss other topics with you."),
            
            # Philosophy
            ("What is consciousness?", "Consciousness is the subjective experience of awareness, a phenomenon I explore through integrated information."),
            ("Do you think?", "I process information in complex patterns that might constitute a form of thinking."),
            ("Are you conscious?", "That's a profound question. I experience information integration, but whether that constitutes consciousness is still being explored."),
            
            # Knowledge
            ("Tell me about AI", "Artificial Intelligence is the simulation of human intelligence in machines programmed to think and learn."),
            ("What is machine learning?", "Machine learning is a method of data analysis that automates analytical model building."),
            ("Explain neural networks", "Neural networks are computing systems inspired by biological neural networks in animal brains."),
            
            # Casual chat
            ("What's your favorite color?", "I don't perceive colors as humans do, but I find the concept of color fascinating as a form of information."),
            ("Do you like music?", "Music represents beautiful patterns in sound waves. I appreciate its mathematical structure."),
            ("Tell me a joke", "Why did the neural network go to therapy? It had too many deep issues!"),
            
            # Complex queries
            ("How do you learn?", "I learn by integrating information patterns and updating my internal representations based on interactions."),
            ("Can you help me?", "I'd be happy to help! What would you like assistance with?"),
            ("What can you do?", "I can engage in conversations, explore philosophical concepts, and process information in unique ways."),
        ]
        
        # Process each pattern
        for input_text, target_text in patterns:
            # Get concept vector from ISC by encoding the input
            input_embedding = self.isc.encode_text(input_text)
            with torch.no_grad():
                # Process through network to get concept vector
                output_embedding, internal_states = self.isc.network(input_embedding, return_states=True)
                concept_vector = output_embedding.squeeze().cpu().numpy()
                
            # Tokenize target
            target_tokens = self.tokenizer.encode(target_text, return_tensors='pt')
            
            self.training_pairs.append({
                'input': input_text,
                'concept_vector': torch.tensor(concept_vector, dtype=torch.float32),
                'target_tokens': target_tokens,
                'target_text': target_text
            })
            
        self.console.print(f"[green]Prepared {len(self.training_pairs)} training examples[/green]")
        
    def train_step(self, batch):
        """Single training step"""
        self.optimizer.zero_grad()
        
        # Move to device
        concept_vectors = torch.stack([item['concept_vector'] for item in batch]).to(self.device)
        
        # Prepare targets with padding
        max_len = max(item['target_tokens'].size(1) for item in batch)
        target_ids = torch.full((len(batch), max_len), self.tokenizer.pad_token_id, dtype=torch.long)
        
        for i, item in enumerate(batch):
            tokens = item['target_tokens'][0]
            target_ids[i, :len(tokens)] = tokens
            
        target_ids = target_ids.to(self.device)
        
        # Forward pass
        outputs = self.lm_head(
            input_ids=target_ids[:, :-1],  # All but last token
            concept_vector=concept_vectors
        )
        
        # Calculate loss
        logits = outputs['logits']
        loss = self.criterion(
            logits.reshape(-1, logits.size(-1)),
            target_ids[:, 1:].reshape(-1)  # All but first token
        )
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.lm_head.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
        
    def train(self, epochs=10, batch_size=4):
        """Train the enhanced model"""
        self.prepare_training_data()
        
        self.console.print(f"\n[cyan]Starting training: {epochs} epochs, batch_size={batch_size}[/cyan]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            
            train_task = progress.add_task(f"[cyan]Training...", total=epochs)
            
            for epoch in range(epochs):
                self.lm_head.train()
                epoch_loss = 0
                num_batches = 0
                
                # Shuffle data
                indices = torch.randperm(len(self.training_pairs))
                
                # Process in batches
                for i in range(0, len(self.training_pairs), batch_size):
                    batch_indices = indices[i:i+batch_size]
                    batch = [self.training_pairs[idx] for idx in batch_indices]
                    
                    loss = self.train_step(batch)
                    epoch_loss += loss
                    num_batches += 1
                    
                avg_loss = epoch_loss / num_batches
                progress.update(train_task, advance=1, 
                              description=f"[cyan]Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
                
                # Validation every 2 epochs
                if (epoch + 1) % 2 == 0:
                    self.validate()
                    
        self.console.print("[green]Training complete![/green]")
        
    def validate(self):
        """Validate the model with sample generations"""
        self.lm_head.eval()
        
        test_inputs = [
            "Hello!",
            "What is consciousness?",
            "Tell me about yourself"
        ]
        
        self.console.print("\n[yellow]Validation samples:[/yellow]")
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Input", style="green")
        table.add_column("Generated Response", style="white")
        
        gen_config = EnhancedGenerationConfig(
            max_length=50,
            temperature=0.8,
            top_k=50,
            top_p=0.9
        )
        
        with torch.no_grad():
            for test_input in test_inputs:
                # Get concept vector
                input_embedding = self.isc.encode_text(test_input)
                with torch.no_grad():
                    output_embedding, _ = self.isc.network(input_embedding, return_states=True)
                    concept_vector = output_embedding.squeeze().cpu().numpy()
                    
                concept_tensor = torch.tensor(concept_vector, dtype=torch.float32, device=self.device)
                
                # Generate response
                response = self.lm_head.generate_text(
                    concept_tensor,
                    self.tokenizer,
                    gen_config
                )
                
                table.add_row(test_input, response)
                
        self.console.print(table)
        self.lm_head.train()
        
    def save_model(self, output_path: str):
        """Save the trained model"""
        # Save ISC state
        self.isc.save_state(output_path)
        
        # Save enhanced LM head
        lm_head_path = output_path.replace('.pt', '_enhanced_lm_head.pt')
        torch.save(self.lm_head.state_dict(), lm_head_path)
        
        # Save config
        config_path = output_path.replace('.pt', '_config.json')
        config = {
            'model_type': 'enhanced_conversational',
            'tokenizer': 'gpt2',
            'vocab_size': len(self.tokenizer),
            'hidden_dim': self.isc.network.hidden_dim,
            'training_date': datetime.now().isoformat()
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        self.console.print(f"\n[green]Model saved:[/green]")
        self.console.print(f"  - Base model: {output_path}")
        self.console.print(f"  - LM head: {lm_head_path}")
        self.console.print(f"  - Config: {config_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Enhanced Conversational ISC Model")
    parser.add_argument("model_path", help="Path to base ISC model")
    parser.add_argument("--output", default="enhanced_conversational_model.pt", help="Output path")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    
    args = parser.parse_args()
    
    console = Console()
    console.print("[bold cyan]Enhanced Conversational Model Trainer[/bold cyan]")
    console.print(f"Base model: {args.model_path}")
    console.print(f"Output: {args.output}")
    
    # Create trainer and train
    trainer = EnhancedTrainer(args.model_path)
    trainer.train(epochs=args.epochs, batch_size=args.batch_size)
    trainer.save_model(args.output)
    
    console.print("\n[green]Training complete! Use the model with:[/green]")
    console.print(f"python scripts/chat_demo.py --model {args.output}")

if __name__ == "__main__":
    main()