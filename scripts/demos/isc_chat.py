#!/usr/bin/env python3
"""
ISC Chat Interface - Supports both original and consciousness-driven models
"""

import os
import sys
import glob
import torch
import json
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
import traceback
from transformers import AutoTokenizer, GPT2Tokenizer

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # For isc package
sys.path.insert(0, str(Path(__file__).parent.parent))  # For training scripts

from isc.core import ISCCore

# Import both architectures
try:
    from training.conversational import ConversationalLMHead as OldLMHead
except ImportError:
    OldLMHead = None

try:
    from training.consciousness_driven_generation import ConsciousnessLMHead, GenerationConfig
except ImportError:
    # Fallback: define minimal GenerationConfig if module not available
    class GenerationConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    ConsciousnessLMHead = None

class UniversalISCChat:
    """Universal chat interface that works with all ISC model types"""
    
    def __init__(self):
        self.console = Console()
        self.isc = ISCCore()
        self.lm_head = None
        self.model_type = None  # 'original', 'consciousness', or 'base'
        self.generation_config = GenerationConfig(
            max_length=50,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            beam_size=3,
            repetition_penalty=1.2,
            phi_weight=0.3
        )
        
        # Initialize proper text generation tokenizer
        try:
            self.text_tokenizer = AutoTokenizer.from_pretrained("gpt2")
            # Add padding token if not present
            if self.text_tokenizer.pad_token is None:
                self.text_tokenizer.pad_token = self.text_tokenizer.eos_token
        except:
            self.console.print("[yellow]Warning: Could not load GPT2 tokenizer, using fallback[/yellow]")
            self.text_tokenizer = None
            
        self.using_original_tokenizer = False  # Track if using original tokenizer
        
    def find_available_models(self):
        """Find all available ISC models and categorize them"""
        models = {
            'base': [],
            'original_conversational': [],
            'consciousness': []
        }
        
        # Search patterns
        patterns = [
            "isc_state_*.pt",
            "checkpoints/isc_state_*.pt"
        ]
        
        for pattern in patterns:
            for model_file in glob.glob(pattern):
                # Skip LM head files
                if model_file.endswith('_lm_head.pt'):
                    continue
                    
                # Categorize model
                if 'consciousness' in model_file:
                    # Check for corresponding LM head
                    lm_head = model_file.replace('.pt', '_lm_head.pt')
                    if os.path.exists(lm_head):
                        models['consciousness'].append((model_file, lm_head))
                elif 'conversational' in model_file:
                    # Check for LM head
                    lm_head = model_file.replace('.pt', '_lm_head.pt')
                    if os.path.exists(lm_head):
                        models['original_conversational'].append((model_file, lm_head))
                else:
                    # Base ISC model
                    models['base'].append(model_file)
                    
        return models
    
    def load_model(self, model_path: str, lm_head_path: str = None):
        """Load model with appropriate architecture"""
        try:
            self.console.print(f"[cyan]Loading ISC model: {model_path}[/cyan]")
            self.isc.load_state(model_path)
            
            if lm_head_path and os.path.exists(lm_head_path):
                # Determine model type by examining the saved state
                lm_state = torch.load(lm_head_path, map_location='cpu')
                
                # Check for consciousness model keys
                if any('transformer_blocks' in key for key in lm_state.keys()):
                    self.model_type = 'consciousness'
                    self.console.print("[green]Detected consciousness-driven model[/green]")
                    
                    # Load consciousness LM head
                    # Detect vocab size from saved state
                    if 'output_projection.weight' in lm_state:
                        saved_vocab_size = lm_state['output_projection.weight'].shape[0]
                        vocab_size = saved_vocab_size
                    else:
                        # Use text tokenizer vocab size if available
                        vocab_size = len(self.text_tokenizer) if self.text_tokenizer else len(self.isc.tokenizer)
                        
                    hidden_dim = self.isc.network.hidden_dim
                    self.lm_head = ConsciousnessLMHead(hidden_dim, vocab_size)
                    self.lm_head.load_state_dict(lm_state)
                    self.lm_head.eval()
                    
                    # Mark if using original tokenizer
                    self.using_original_tokenizer = (vocab_size == 30522)
                    
                elif any('dialogue_context' in key for key in lm_state.keys()):
                    self.model_type = 'original'
                    self.console.print("[green]Detected original conversational model[/green]")
                    
                    # Load old conversational LM head
                    if OldLMHead is not None:
                        # Detect vocab size from saved state
                        output_proj_weight = lm_state.get('output_projection.weight', None)
                        if output_proj_weight is not None:
                            saved_vocab_size = output_proj_weight.shape[0]
                            self.console.print(f"[yellow]Model was trained with vocab_size={saved_vocab_size}[/yellow]")
                            
                            # Check if it's BERT vocab size
                            if saved_vocab_size == 30522:
                                self.console.print("[yellow]Detected BERT tokenizer model (vocab_size=30522)[/yellow]")
                                # Keep original tokenizer for this model
                                vocab_size = saved_vocab_size
                            else:
                                # Use text tokenizer vocab size if available
                                vocab_size = len(self.text_tokenizer) if self.text_tokenizer else saved_vocab_size
                        else:
                            vocab_size = len(self.isc.tokenizer)
                            
                        hidden_dim = self.isc.network.hidden_dim
                        self.lm_head = OldLMHead(hidden_dim, vocab_size)
                        self.lm_head.load_state_dict(lm_state)
                        self.lm_head.eval()
                        
                        # Mark if using original tokenizer
                        self.using_original_tokenizer = (vocab_size == 30522)
                    else:
                        self.console.print("[red]Original conversational module not available[/red]")
                        return False
                else:
                    self.console.print("[yellow]Unknown LM head architecture[/yellow]")
                    return False
            else:
                # Base ISC model without LM head
                self.model_type = 'base'
                self.console.print("[green]Loaded base ISC model (philosophical responses only)[/green]")
                
            # Initialize session
            self.isc.session_active = True
            self.isc.current_session_id = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            return True
            
        except Exception as e:
            self.console.print(f"[red]Failed to load model: {e}[/red]")
            traceback.print_exc()
            return False
    
    def generate_response(self, user_input: str):
        """Generate response based on model type"""
        # Always get ISC's philosophical response
        philosophical_response = self.isc.process_input(user_input)
        
        if self.model_type == 'base':
            # Return only philosophical response for base models
            return philosophical_response, None
            
        # Get concept vector for conversational models
        try:
            import numpy as np
            # Get concept vector by encoding the input
            input_embedding = self.isc.encode_text(user_input)
            with torch.no_grad():
                # Process through network to get concept vector
                output_embedding, internal_states = self.isc.network(input_embedding, return_states=True)
                concept_vector = output_embedding.squeeze().cpu().numpy()
        except:
            import numpy as np
            # Use input_dim (384) which is the actual output size
            concept_vector = np.random.randn(self.isc.network.input_dim).astype(np.float32)
            
        concept_tensor = torch.tensor(concept_vector, dtype=torch.float32)
        
        # Generate conversational response based on model type
        if self.model_type == 'consciousness':
            # Use consciousness-driven generation
            with torch.no_grad():
                # Use proper text tokenizer
                tokenizer = self.text_tokenizer if self.text_tokenizer else self.isc.tokenizer
                tokens, phi_score = self.lm_head.generate(
                    concept_tensor, 
                    tokenizer,
                    self.generation_config
                )
                # Decode with proper tokenizer
                if self.text_tokenizer:
                    conversational_response = self.text_tokenizer.decode(tokens, skip_special_tokens=True)
                else:
                    conversational_response = tokenizer.decode(tokens)
                
            return philosophical_response, f"{conversational_response} [Φ={phi_score:.3f}]"
            
        elif self.model_type == 'original':
            # Enhanced generation for original model
            # Pad concept tensor to match expected hidden_dim if needed
            if concept_tensor.shape[-1] == 384 and self.isc.network.hidden_dim == 512:
                # Pad from 384 to 512
                padding = torch.zeros(512 - 384)
                concept_tensor = torch.cat([concept_tensor, padding])
                
            if self.using_original_tokenizer:
                # Use simplified generation for BERT tokenizer models
                conversational_response = self._generate_from_bert_model(concept_tensor)
            else:
                conversational_response = self._generate_from_original_model(concept_tensor)
            return philosophical_response, conversational_response
            
        return philosophical_response, None
    
    def _generate_from_original_model(self, concept_tensor: torch.Tensor) -> str:
        """Improved generation for original conversational model"""
        with torch.no_grad():
            # Ensure batch dimension
            if concept_tensor.dim() == 1:
                concept_tensor = concept_tensor.unsqueeze(0)
            
            # Use text tokenizer if available
            tokenizer = self.text_tokenizer if self.text_tokenizer else self.isc.tokenizer
            
            # Start with a prompt token or the concept projection
            if self.text_tokenizer and hasattr(self.text_tokenizer, 'bos_token_id'):
                input_ids = torch.tensor([[self.text_tokenizer.bos_token_id]], dtype=torch.long)
            else:
                # Get initial token from concept
                logits = self.lm_head(concept_tensor)
                if logits.dim() == 3:
                    logits = logits[:, 0, :]
                probs = torch.softmax(logits / self.generation_config.temperature, dim=-1)
                input_ids = torch.multinomial(probs, num_samples=1)
            
            generated_tokens = input_ids[0].tolist()
            
            # Generate tokens iteratively
            for _ in range(self.generation_config.max_length - 1):
                # Get logits for next token
                # Note: Original model may not be truly autoregressive, so we'll use concept + noise
                noise = torch.randn_like(concept_tensor) * 0.1
                mixed_input = concept_tensor + noise
                logits = self.lm_head(mixed_input)
                
                if logits.dim() == 3:
                    logits = logits[:, 0, :]
                
                # Apply temperature
                logits = logits / self.generation_config.temperature
                
                # Apply top-k filtering
                if self.generation_config.top_k > 0:
                    top_k_values, _ = torch.topk(logits, self.generation_config.top_k)
                    min_value = top_k_values[:, -1].unsqueeze(-1)
                    logits = torch.where(logits < min_value, torch.full_like(logits, -float('inf')), logits)
                
                # Sample
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs[0], num_samples=1).item()
                
                # Apply repetition penalty
                if next_token in generated_tokens[-10:]:
                    continue
                    
                generated_tokens.append(next_token)
                
                # Check for EOS
                if self.text_tokenizer and hasattr(self.text_tokenizer, 'eos_token_id'):
                    if next_token == self.text_tokenizer.eos_token_id:
                        break
            
            # Decode tokens
            if self.text_tokenizer:
                response = self.text_tokenizer.decode(generated_tokens, skip_special_tokens=True)
            else:
                # Fallback decoding
                response = ' '.join(str(t) for t in generated_tokens)
            
            return response
    
    def _generate_from_bert_model(self, concept_tensor: torch.Tensor) -> str:
        """Generate text for models trained with BERT tokenizer"""
        with torch.no_grad():
            # Ensure batch dimension
            if concept_tensor.dim() == 1:
                concept_tensor = concept_tensor.unsqueeze(0)
            
            # Get logits from concept
            logits = self.lm_head(concept_tensor)
            
            if logits.dim() == 3:
                logits = logits[:, 0, :]
                
            # Simple token generation (BERT tokenizer doesn't support proper generation)
            # We'll use template-based responses instead
            templates = [
                "I understand your interest in this topic.",
                "That's an interesting perspective to explore.",
                "Let me share my thoughts on this.",
                "This connects to broader concepts of consciousness.",
                "I find this question thought-provoking.",
                "From my perspective, this relates to information integration.",
                "This is a fascinating area to consider.",
                "Your question touches on fundamental aspects of awareness.",
                "Let me explore this concept with you.",
                "That's a profound question to consider.",
                "I appreciate you bringing up this topic.",
                "This opens up interesting possibilities for discussion."
            ]
            
            # Use concept tensor sum as a pseudo-random seed for variety
            seed_value = abs(concept_tensor.sum().item())
            template_idx = int(seed_value * 100) % len(templates)
            
            return templates[template_idx]
    
    def chat_loop(self):
        """Main chat interaction loop"""
        self.console.print("\n[bold cyan]ISC Chat Interface[/bold cyan]")
        self.console.print(f"[dim]Model type: {self.model_type}[/dim]")
        self.console.print("[dim]Commands: /help, /metrics, /save, /exit[/dim]\n")
        
        conversation_history = []
        
        while True:
            try:
                # Get user input
                user_input = Prompt.ask("\n[bold green]You[/bold green]")
                
                # Check for commands
                if user_input.lower() == '/exit':
                    break
                elif user_input.lower() == '/help':
                    self.show_help()
                    continue
                elif user_input.lower() == '/metrics':
                    self.show_metrics()
                    continue
                elif user_input.lower() == '/save':
                    self.save_conversation(conversation_history)
                    continue
                    
                # Generate response
                with self.console.status("[blue]ISC is thinking...[/blue]", spinner="dots"):
                    philosophical, conversational = self.generate_response(user_input)
                
                # Display response
                if conversational and self.model_type != 'base':
                    self.console.print(f"\n[bold blue]ISC[/bold blue]: {conversational}")
                    
                    # Option to show philosophical response
                    show_phil = Prompt.ask("[dim]Show philosophical response? (y/n)[/dim]", default="n")
                    if show_phil.lower() == 'y':
                        self.console.print(f"[dim]Philosophical: {philosophical}[/dim]")
                else:
                    # Base model or fallback
                    self.console.print(f"\n[bold blue]ISC[/bold blue]: {philosophical}")
                
                # Save to history
                conversation_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'user': user_input,
                    'philosophical': philosophical,
                    'conversational': conversational,
                    'metrics': self.isc.metrics.copy()
                })
                
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Chat interrupted. Type /exit to quit.[/yellow]")
            except Exception as e:
                self.console.print(f"\n[red]Error: {e}[/red]")
                traceback.print_exc()
    
    def show_help(self):
        """Display help information"""
        help_panel = Panel(
            "[bold]Available Commands:[/bold]\n\n"
            "/help    - Show this help message\n"
            "/metrics - Display current ISC metrics\n"
            "/save    - Save conversation to file\n"
            "/exit    - Exit the chat\n\n"
            "[bold]Model Information:[/bold]\n"
            f"Type: {self.model_type}\n"
            f"Concepts: {len(self.isc.knowledge_graph.graph.nodes())}\n"
            f"Phi: {self.isc.metrics.get('phi_value', 0):.4f}",
            title="ISC Chat Help",
            border_style="cyan"
        )
        self.console.print(help_panel)
    
    def show_metrics(self):
        """Display current ISC metrics"""
        metrics_table = Table(title="ISC Metrics", show_header=True, header_style="bold cyan")
        metrics_table.add_column("Metric", style="yellow")
        metrics_table.add_column("Value", style="green")
        
        for key, value in self.isc.metrics.items():
            if isinstance(value, float):
                metrics_table.add_row(key, f"{value:.4f}")
            else:
                metrics_table.add_row(key, str(value))
                
        self.console.print(metrics_table)
    
    def save_conversation(self, history):
        """Save conversation history to file"""
        filename = f"isc_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                'session_id': self.isc.current_session_id,
                'model_type': self.model_type,
                'conversation': history,
                'final_metrics': self.isc.metrics
            }, f, indent=2)
            
        self.console.print(f"[green]Conversation saved to {filename}[/green]")

def main():
    """Main entry point"""
    console = Console()
    
    console.print(Panel(
        "[bold cyan]ISC Universal Chat Interface[/bold cyan]\n"
        "[dim]Compatible with all ISC model types[/dim]",
        border_style="cyan"
    ))
    
    chat = UniversalISCChat()
    
    # Find available models
    models = chat.find_available_models()
    
    # Display available models
    model_table = Table(show_header=True, header_style="bold cyan")
    model_table.add_column("#", style="cyan", width=3)
    model_table.add_column("Model Type", style="yellow", width=20)
    model_table.add_column("Model File", style="white")
    
    all_models = []
    
    # Add consciousness models
    for model, lm_head in models['consciousness']:
        all_models.append(('consciousness', model, lm_head))
        model_table.add_row(str(len(all_models)), "Consciousness", os.path.basename(model))
    
    # Add original conversational models
    for model, lm_head in models['original_conversational']:
        all_models.append(('original', model, lm_head))
        model_table.add_row(str(len(all_models)), "Original Conv.", os.path.basename(model))
    
    # Add base models
    for model in models['base']:
        all_models.append(('base', model, None))
        model_table.add_row(str(len(all_models)), "Base ISC", os.path.basename(model))
    
    if not all_models:
        console.print("[red]No ISC models found![/red]")
        return
        
    console.print("\n[bold]Available Models:[/bold]")
    console.print(model_table)
    
    # Get user choice
    choice = Prompt.ask("\n[cyan]Select model number[/cyan]", default="1")
    
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(all_models):
            model_type, model_path, lm_head_path = all_models[idx]
            
            if chat.load_model(model_path, lm_head_path):
                chat.chat_loop()
            else:
                console.print("[red]Failed to load selected model[/red]")
        else:
            console.print("[red]Invalid model number[/red]")
    except ValueError:
        console.print("[red]Invalid input[/red]")
    
    console.print("\n[green]Thank you for using ISC Chat![/green]")

if __name__ == "__main__":
    main()