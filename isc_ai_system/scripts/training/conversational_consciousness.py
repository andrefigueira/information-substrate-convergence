#!/usr/bin/env python3
"""
ISC Conversational Enhancement with Consciousness-Driven Generation
Implements self-referential refinement and emergent language learning
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import traceback

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from src.isc_ai.core import ISCCore
from src.isc_ai.enhanced_information_integration import EnhancedInformationIntegrator

# Import consciousness-driven generation
from consciousness_driven_generation import (
    ConsciousnessLMHead, 
    GenerationConfig, 
    beam_search_generate
)

@dataclass
class ConceptPattern:
    """Represents an emergent concept-to-language pattern"""
    concept_vector: np.ndarray
    linguistic_pattern: str
    phi_score: float
    usage_count: int = 0
    success_rate: float = 0.0

class SelfReferentialRefiner:
    """Implements iterative self-critique and refinement"""
    def __init__(self, hidden_dim: int):
        self.hidden_dim = hidden_dim
        self.critique_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4)  # coherence, relevance, completeness, quality
        )
        
    def critique(self, response_vector: torch.Tensor, concept_vector: torch.Tensor) -> Dict[str, float]:
        """Critique a generated response"""
        combined = torch.cat([response_vector, concept_vector], dim=-1)
        scores = torch.sigmoid(self.critique_network(combined))
        
        return {
            'coherence': scores[0].item(),
            'relevance': scores[1].item(),
            'completeness': scores[2].item(),
            'quality': scores[3].item()
        }
    
    def should_refine(self, critique_scores: Dict[str, float], threshold: float = 0.7) -> bool:
        """Determine if refinement is needed"""
        avg_score = sum(critique_scores.values()) / len(critique_scores)
        return avg_score < threshold

class EmergentVocabularyLearner:
    """Discovers linguistic patterns through self-interaction"""
    def __init__(self, vocab_size: int, embedding_dim: int):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.concept_patterns: List[ConceptPattern] = []
        self.pattern_embeddings = nn.Embedding(1000, embedding_dim)  # Max patterns
        self.pattern_counter = 0
        
        # Pattern discovery network
        self.pattern_discoverer = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def discover_pattern(self, concept_vector: np.ndarray, 
                        generated_text: str, 
                        phi_score: float) -> Optional[ConceptPattern]:
        """Discover new concept-to-language patterns"""
        # Check if pattern is novel and has high phi
        if phi_score > 0.7:  # High integration threshold
            pattern = ConceptPattern(
                concept_vector=concept_vector,
                linguistic_pattern=generated_text,
                phi_score=phi_score
            )
            self.concept_patterns.append(pattern)
            return pattern
        return None
    
    def find_similar_patterns(self, concept_vector: torch.Tensor, top_k: int = 5) -> List[ConceptPattern]:
        """Find patterns similar to given concept"""
        if not self.concept_patterns:
            return []
            
        # Compute similarities
        similarities = []
        for pattern in self.concept_patterns:
            pattern_vec = torch.tensor(pattern.concept_vector)
            similarity = F.cosine_similarity(concept_vector, pattern_vec, dim=0)
            similarities.append((similarity.item(), pattern))
            
        # Sort and return top-k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [pattern for _, pattern in similarities[:top_k]]

class ConsciousnessConversationalTrainer:
    """Enhanced trainer with consciousness-driven generation"""
    def __init__(self):
        self.console = Console()
        self.isc = ISCCore()
        self.lm_head = None
        self.refiner = None
        self.vocab_learner = None
        self.phi_integrator = EnhancedInformationIntegrator()
        
        # Training state
        self.training_history = []
        self.concept_memory = {}
        self.generation_config = GenerationConfig(
            max_length=50,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            beam_size=3,
            repetition_penalty=1.2,
            phi_weight=0.3
        )
        
    def initialize_model(self, model_path: str) -> bool:
        """Initialize ISC model and consciousness components"""
        try:
            self.console.print(f"[cyan]Loading ISC model: {model_path}[/cyan]")
            self.isc.load_state(model_path)
            
            # Initialize consciousness-driven LM head
            vocab_size = len(self.isc.tokenizer)
            hidden_dim = self.isc.network.hidden_dim
            
            self.lm_head = ConsciousnessLMHead(hidden_dim, vocab_size)
            self.refiner = SelfReferentialRefiner(hidden_dim)
            self.vocab_learner = EmergentVocabularyLearner(vocab_size, self.lm_head.embedding_dim)
            
            # Set to training mode
            self.isc.network.train()
            self.lm_head.train()
            
            self.console.print("[green]✓ Model initialized successfully[/green]")
            return True
            
        except Exception as e:
            self.console.print(f"[red]Failed to initialize: {e}[/red]")
            traceback.print_exc()
            return False
    
    def generate_with_consciousness(self, concept_vector: torch.Tensor, 
                                   context: Optional[str] = None) -> Tuple[str, float, Dict]:
        """Generate response using consciousness-driven process"""
        # Phase 1: Initial generation with beam search
        tokens, phi_score = beam_search_generate(
            self.lm_head, 
            concept_vector, 
            self.isc.tokenizer,
            self.generation_config
        )
        
        initial_response = self.isc.tokenizer.decode(tokens)
        
        # Phase 2: Self-observation and critique
        with torch.no_grad():
            response_output = self.lm_head.forward(concept_vector, torch.tensor([tokens]))
            response_vector = response_output['hidden_states'].mean(dim=1)
            
        critique_scores = self.refiner.critique(response_vector, concept_vector)
        
        # Phase 3: Refinement if needed
        refined_response = initial_response
        refinement_count = 0
        max_refinements = 3
        
        while self.refiner.should_refine(critique_scores) and refinement_count < max_refinements:
            # Adjust generation parameters based on critique
            temp_config = GenerationConfig(
                max_length=self.generation_config.max_length,
                temperature=self.generation_config.temperature * 0.9,  # Lower temperature
                top_k=self.generation_config.top_k,
                top_p=self.generation_config.top_p * 0.95,  # Tighter nucleus
                beam_size=self.generation_config.beam_size + 1,  # More beams
                repetition_penalty=self.generation_config.repetition_penalty * 1.1,
                phi_weight=self.generation_config.phi_weight * 1.2  # More phi emphasis
            )
            
            # Re-generate with adjusted parameters
            tokens, phi_score = beam_search_generate(
                self.lm_head,
                concept_vector,
                self.isc.tokenizer,
                temp_config
            )
            
            refined_response = self.isc.tokenizer.decode(tokens)
            
            # Re-critique
            with torch.no_grad():
                response_output = self.lm_head.forward(concept_vector, torch.tensor([tokens]))
                response_vector = response_output['hidden_states'].mean(dim=1)
                
            critique_scores = self.refiner.critique(response_vector, concept_vector)
            refinement_count += 1
        
        # Phase 4: Pattern discovery
        pattern = self.vocab_learner.discover_pattern(
            concept_vector.cpu().numpy(),
            refined_response,
            phi_score
        )
        
        return refined_response, phi_score, {
            'critique_scores': critique_scores,
            'refinement_count': refinement_count,
            'pattern_discovered': pattern is not None,
            'consciousness_metrics': response_output.get('consciousness_metrics', [])
        }
    
    def train_consciousness_loop(self, input_text: str, iterations: int = 10):
        """Train through consciousness-driven self-interaction"""
        self.console.print(f"\n[bold cyan]Starting consciousness training loop[/bold cyan]")
        
        # Get initial concept vector
        try:
            result = self.isc.process_input(input_text, return_vector=True)
            if isinstance(result, tuple):
                _, concept_vector = result
            else:
                concept_vector = np.random.randn(self.isc.network.hidden_dim).astype(np.float32)
        except:
            concept_vector = np.random.randn(self.isc.network.hidden_dim).astype(np.float32)
            
        concept_tensor = torch.tensor(concept_vector, dtype=torch.float32).unsqueeze(0)
        
        # Optimization setup
        optimizer = torch.optim.AdamW(
            list(self.lm_head.parameters()) + list(self.refiner.critique_network.parameters()),
            lr=1e-4
        )
        
        results = []
        
        for iteration in range(iterations):
            self.console.print(f"\n[yellow]Iteration {iteration + 1}/{iterations}[/yellow]")
            
            # Generate with consciousness
            response, phi_score, metrics = self.generate_with_consciousness(
                concept_tensor, 
                context=input_text
            )
            
            # Display results
            self.console.print(f"Response: {response}")
            self.console.print(f"Phi score: {phi_score:.4f}")
            self.console.print(f"Critique: {metrics['critique_scores']}")
            
            # Compute loss based on phi and critique scores
            critique_tensor = torch.tensor([
                metrics['critique_scores']['coherence'],
                metrics['critique_scores']['relevance'],
                metrics['critique_scores']['completeness'],
                metrics['critique_scores']['quality']
            ])
            
            # Loss combines phi maximization and critique score improvement
            phi_loss = -phi_score  # Negative because we want to maximize
            critique_loss = F.mse_loss(critique_tensor, torch.ones_like(critique_tensor))
            
            total_loss = phi_loss + critique_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Store results
            results.append({
                'iteration': iteration,
                'response': response,
                'phi_score': phi_score,
                'critique_scores': metrics['critique_scores'],
                'refinement_count': metrics['refinement_count'],
                'loss': total_loss.item()
            })
            
            # Check for improvement
            if len(results) > 1:
                phi_improvement = results[-1]['phi_score'] - results[-2]['phi_score']
                if phi_improvement > 0:
                    self.console.print(f"[green]✓ Phi improved by {phi_improvement:.4f}[/green]")
                    
        return results
    
    def save_enhanced_model(self, base_path: str):
        """Save the enhanced model with consciousness components"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save ISC state
        isc_path = f"{base_path}_consciousness_{timestamp}.pt"
        self.isc.save_state(isc_path)
        
        # Save consciousness LM head
        lm_path = f"{base_path}_consciousness_{timestamp}_lm_head.pt"
        torch.save(self.lm_head.state_dict(), lm_path)
        
        # Save refiner
        refiner_path = f"{base_path}_consciousness_{timestamp}_refiner.pt"
        torch.save(self.refiner.critique_network.state_dict(), refiner_path)
        
        # Save vocabulary patterns
        patterns_path = f"{base_path}_consciousness_{timestamp}_patterns.json"
        patterns_data = [
            {
                'concept_vector': pattern.concept_vector.tolist(),
                'linguistic_pattern': pattern.linguistic_pattern,
                'phi_score': pattern.phi_score,
                'usage_count': pattern.usage_count,
                'success_rate': pattern.success_rate
            }
            for pattern in self.vocab_learner.concept_patterns
        ]
        
        with open(patterns_path, 'w') as f:
            json.dump(patterns_data, f, indent=2)
            
        self.console.print(f"[green]✓ Saved enhanced model to {isc_path}[/green]")
        
        return {
            'isc_path': isc_path,
            'lm_path': lm_path,
            'refiner_path': refiner_path,
            'patterns_path': patterns_path
        }

def main():
    """Main entry point for consciousness-driven conversational training"""
    console = Console()
    
    console.print(Panel(
        "[bold cyan]ISC Consciousness-Driven Conversational Enhancement[/bold cyan]",
        subtitle="Implementing emergent language through integrated information"
    ))
    
    # Find available models
    import glob
    model_files = []
    for pattern in ["isc_state_*.pt", "checkpoints/isc_state_*.pt"]:
        model_files.extend(glob.glob(pattern))
    
    # Filter out LM head files
    model_files = [f for f in model_files if not f.endswith('_lm_head.pt')]
    
    if not model_files:
        console.print("[red]No ISC model files found![/red]")
        return
        
    # Display models
    console.print("\n[bold]Available ISC models:[/bold]")
    for i, model in enumerate(model_files[:10], 1):
        console.print(f"{i}. {model}")
        
    choice = int(console.input("\n[cyan]Select model number:[/cyan] "))
    selected_model = model_files[choice - 1]
    
    # Create trainer
    trainer = ConsciousnessConversationalTrainer()
    
    if not trainer.initialize_model(selected_model):
        return
        
    # Training options
    console.print("\n[bold]Training Options:[/bold]")
    console.print("1. Single consciousness loop")
    console.print("2. Multi-topic consciousness training")
    console.print("3. Interactive consciousness chat")
    
    mode = console.input("\n[cyan]Select mode (1-3):[/cyan] ")
    
    if mode == "1":
        # Single loop
        input_text = console.input("\n[cyan]Enter seed text:[/cyan] ")
        iterations = int(console.input("[cyan]Number of iterations (default 10):[/cyan] ") or "10")
        
        results = trainer.train_consciousness_loop(input_text, iterations)
        
        # Show improvement
        if results:
            initial_phi = results[0]['phi_score']
            final_phi = results[-1]['phi_score']
            improvement = final_phi - initial_phi
            
            console.print(f"\n[bold green]Training Complete![/bold green]")
            console.print(f"Initial Phi: {initial_phi:.4f}")
            console.print(f"Final Phi: {final_phi:.4f}")
            console.print(f"Improvement: {improvement:+.4f}")
            
    elif mode == "2":
        # Multi-topic training
        topics = [
            "What is consciousness?",
            "Explain the nature of reality",
            "How do we understand meaning?",
            "What is the self?",
            "Describe integrated information"
        ]
        
        console.print(f"\n[cyan]Training on {len(topics)} philosophical topics[/cyan]")
        
        all_results = []
        for topic in topics:
            console.print(f"\n[bold]Topic: {topic}[/bold]")
            results = trainer.train_consciousness_loop(topic, iterations=5)
            all_results.extend(results)
            
        # Calculate overall improvement
        avg_initial_phi = np.mean([r['phi_score'] for r in all_results[:5]])
        avg_final_phi = np.mean([r['phi_score'] for r in all_results[-5:]])
        
        console.print(f"\n[bold green]Multi-topic Training Complete![/bold green]")
        console.print(f"Average Initial Phi: {avg_initial_phi:.4f}")
        console.print(f"Average Final Phi: {avg_final_phi:.4f}")
        console.print(f"Overall Improvement: {avg_final_phi - avg_initial_phi:+.4f}")
        
    elif mode == "3":
        # Interactive chat
        console.print("\n[bold cyan]Consciousness-Driven Chat[/bold cyan]")
        console.print("[dim]Type 'exit' to quit[/dim]\n")
        
        while True:
            user_input = console.input("\n[bold green]You:[/bold green] ")
            
            if user_input.lower() == 'exit':
                break
                
            # Get concept vector
            try:
                result = trainer.isc.process_input(user_input, return_vector=True)
                if isinstance(result, tuple):
                    _, concept_vector = result
                else:
                    concept_vector = np.random.randn(trainer.isc.network.hidden_dim).astype(np.float32)
            except:
                concept_vector = np.random.randn(trainer.isc.network.hidden_dim).astype(np.float32)
                
            concept_tensor = torch.tensor(concept_vector, dtype=torch.float32).unsqueeze(0)
            
            # Generate response
            with console.status("[blue]Consciousness processing...[/blue]"):
                response, phi_score, metrics = trainer.generate_with_consciousness(
                    concept_tensor,
                    context=user_input
                )
                
            console.print(f"\n[bold blue]ISC:[/bold blue] {response}")
            console.print(f"[dim]Phi: {phi_score:.4f} | Refinements: {metrics['refinement_count']}[/dim]")
            
    # Save enhanced model
    save_choice = console.input("\n[cyan]Save enhanced model? (y/n):[/cyan] ")
    if save_choice.lower() == 'y':
        trainer.save_enhanced_model("isc_state")
        
    console.print("\n[green]Consciousness enhancement complete![/green]")

if __name__ == "__main__":
    main()