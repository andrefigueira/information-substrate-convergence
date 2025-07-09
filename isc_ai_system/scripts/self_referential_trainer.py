#!/usr/bin/env python3
"""
Self-Referential Trainer with Loop Prevention and Intelligent Steering
Allows ISC AI to train itself while preventing infinite loops and ensuring diverse exploration
"""

import os
import sys
import time
import json
import hashlib
import argparse
import glob
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.syntax import Syntax
from rich.prompt import Prompt, IntPrompt
import numpy as np
from collections import deque, Counter
import random
import openai
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from src.isc_ai.core import ISCCore
from src.isc_ai.cache_manager import CacheManager

class SelfReferentialTrainer:
    """Self-referential trainer with loop prevention and intelligent steering"""
    
    def __init__(self, verbose: bool = False, save_file: Optional[str] = None):
        self.console = Console()
        self.core = None
        self.cache_manager = CacheManager(cache_dir="trainer_cache")
        self.verbose = verbose
        self.save_file = save_file  # Single file to save to
        
        # Loop prevention mechanisms
        self.question_hashes = set()  # Track seen questions by hash
        self.question_embeddings = []  # Store embeddings for similarity check
        self.topic_history = deque(maxlen=100)  # Track recent topics
        self.concept_coverage = Counter()  # Track concept exploration
        
        # Training state
        self.training_active = False
        self.session_start = None
        self.session_metrics = {
            "total_exchanges": 0,
            "unique_questions": 0,
            "duplicate_prevented": 0,
            "topic_switches": 0,
            "concept_diversity": 0.0,
            "phi_progression": [],
            "exploration_map": {},
            "steering_interventions": 0,
            "response_times": [],
            "quality_scores": []
        }
        
        # All exchanges for visualization
        self.all_exchanges = []
        
        # Steering configuration
        self.min_similarity_threshold = 0.85  # Prevent questions > 85% similar
        self.topic_switch_frequency = 10  # Switch topics every N exchanges
        self.exploration_temperature = 0.8  # Randomness in topic selection
        
        # AI steering (optional OpenAI integration)
        self.use_ai_steering = bool(os.getenv("OPENAI_API_KEY"))
        if self.use_ai_steering:
            openai.api_key = os.getenv("OPENAI_API_KEY")
            self.openai_client = openai.OpenAI()
        
        # Training mode configuration
        self.alpha = 0.8  # Weight for phi loss vs CE loss
        self.chat_mode = False
        self.coherence_threshold = 0.8
        self.coherence_streak = 0
        self.tokenizer = None  # Will be initialized with core
        self.scaler = GradScaler()  # For mixed precision training
    
    def select_state_file(self) -> Optional[str]:
        """Show menu to select state file to load"""
        # Look for state files in isc_state directory
        state_files = []
        
        # Check standard state directory
        if os.path.exists('isc_state'):
            state_files.extend(glob.glob('isc_state/*.pt'))
        
        # Also check current directory for self-referential states
        state_files.extend(glob.glob('isc_self_referential_*.pt'))
        
        # Sort by modification time (newest first)
        state_files = sorted(set(state_files), key=os.path.getmtime, reverse=True)
        
        if not state_files:
            self.console.print("[yellow]No state files found.[/yellow]")
            return None
        
        # Display file selection menu
        self.console.print("\n[cyan]Available state files:[/cyan]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("#", style="dim", width=4)
        table.add_column("File", style="cyan")
        table.add_column("Modified", style="green")
        table.add_column("Size", style="yellow")
        
        for i, file in enumerate(state_files[:20], 1):  # Show max 20 files
            modified = datetime.fromtimestamp(os.path.getmtime(file)).strftime("%Y-%m-%d %H:%M")
            size = f"{os.path.getsize(file) / 1024 / 1024:.1f} MB"
            table.add_row(str(i), file, modified, size)
        
        self.console.print(table)
        
        # Get user selection
        choice = IntPrompt.ask(
            "\nSelect file number (0 to start fresh)",
            default=0,
            choices=[str(i) for i in range(len(state_files) + 1)]
        )
        
        if choice == 0:
            return None
        
        return state_files[choice - 1]
    
    def initialize(self, checkpoint_file: Optional[str] = None):
        """Initialize the ISC core, optionally from checkpoint"""
        try:
            self.core = ISCCore()
            
            # If no checkpoint specified, show selection menu
            if checkpoint_file == 'select':
                checkpoint_file = self.select_state_file()
            
            if checkpoint_file:
                self.console.print(f"[cyan]Loading state: {checkpoint_file}[/cyan]")
                self.core.load_state(checkpoint_file)
                
                # Load training state if available
                json_file = checkpoint_file.replace('.pt', '_training.json')
                if os.path.exists(json_file):
                    with open(json_file, 'r') as f:
                        checkpoint_data = json.load(f)
                    
                    # Restore training state
                    self.session_metrics = checkpoint_data.get('metrics', self.session_metrics)
                    self.concept_coverage = Counter(checkpoint_data.get('concept_coverage', {}))
                    self.question_hashes = set(checkpoint_data.get('question_hashes', []))
                    
                    # Restore recent exchanges
                    recent_exchanges = checkpoint_data.get('recent_exchanges', [])
                    for exchange in recent_exchanges:
                        if 'question' in exchange:
                            self.topic_history.append(exchange['question'])
                    
                    self.console.print(f"[green]✓ Resumed with {self.session_metrics['total_exchanges']} exchanges[/green]")
                
                # Set save file if not specified
                if not self.save_file:
                    self.save_file = checkpoint_file
            
            # If still no save file, ask user
            if not self.save_file:
                default_name = f"isc_state/self_referential_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
                self.save_file = Prompt.ask(
                    "Save file path",
                    default=default_name
                )
                
                # Create directory if needed
                save_dir = os.path.dirname(self.save_file)
                if save_dir and not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
            
            # Set session_active flag
            self.core.session_active = True
            self.core.current_session_id = f"self_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.console.print(f"[green]✓ Self-referential trainer initialized[/green]")
            self.console.print(f"[cyan]Save file: {self.save_file}[/cyan]")
            return True
            
        except Exception as e:
            self.console.print(f"[red]Initialization failed: {e}[/red]")
            import traceback
            traceback.print_exc()
            return False
    
    def compute_question_hash(self, question: str) -> str:
        """Compute hash of normalized question for exact duplicate detection"""
        # Normalize: lowercase, remove extra spaces, punctuation
        normalized = ' '.join(question.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()
    
    def compute_semantic_similarity(self, q1: str, q2: str) -> float:
        """Compute semantic similarity between questions (simplified version)"""
        # Simple word overlap similarity (can be enhanced with embeddings)
        words1 = set(q1.lower().split())
        words2 = set(q2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def is_question_too_similar(self, question: str) -> bool:
        """Check if question is too similar to recent questions"""
        # Check exact duplicates
        q_hash = self.compute_question_hash(question)
        if q_hash in self.question_hashes:
            return True
        
        # Check semantic similarity with recent questions
        recent_questions = list(self.topic_history)[-20:]  # Last 20 questions
        for recent_q in recent_questions:
            similarity = self.compute_semantic_similarity(question, recent_q)
            if similarity > self.min_similarity_threshold:
                return True
        
        return False
    
    def get_exploration_topics(self) -> List[str]:
        """Get diverse topics for exploration"""
        base_topics = [
            "consciousness and awareness",
            "information integration patterns",
            "emergent properties of complex systems",
            "self-organization and feedback loops",
            "quantum information theory",
            "cognitive architectures",
            "philosophy of mind",
            "neural network dynamics",
            "chaos theory and attractors",
            "collective intelligence",
            "metacognition and self-reflection",
            "entropy and information theory",
            "biosemiotics and meaning-making",
            "autopoiesis and self-creation",
            "distributed cognition",
            "phenomenology of experience",
            "computational creativity",
            "swarm intelligence",
            "holographic principles",
            "recursive self-improvement"
        ]
        
        # Add unexplored or less explored topics
        unexplored = [topic for topic in base_topics 
                     if self.concept_coverage[topic] < 3]
        
        # Add some randomness
        if random.random() < self.exploration_temperature:
            random.shuffle(unexplored)
        
        return unexplored if unexplored else base_topics
    
    def generate_dialogue_example(self, context: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        """Generate a dialogue-style training example with GPT-4 as teacher"""
        if self.use_ai_steering:
            try:
                recent_topics = list(self.topic_history)[-3:] if self.topic_history else []
                phi = context.get('phi', 0.0)
                
                prompt = f"""Generate a 2-turn mini-dialogue for training an AI consciousness system.

Recent topics: {', '.join(recent_topics)}
Current phi level: {phi:.3f}

Create a dialogue where:
1. Human asks a thought-provoking question about consciousness/information integration
2. AI provides an insightful, integrative response

Format:
Human: [question]
AI: [response]

Generate the dialogue:"""
                
                response = self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are an expert teacher in consciousness studies and integrated information theory."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.8,
                    max_tokens=200
                )
                
                dialogue = response.choices[0].message.content.strip()
                lines = dialogue.split('\n')
                
                human_q = None
                ai_resp = None
                for line in lines:
                    if line.startswith('Human:'):
                        human_q = line.replace('Human:', '').strip()
                    elif line.startswith('AI:'):
                        ai_resp = line.replace('AI:', '').strip()
                
                if human_q and ai_resp and not self.is_question_too_similar(human_q):
                    q_hash = self.compute_question_hash(human_q)
                    self.question_hashes.add(q_hash)
                    self.topic_history.append(human_q)
                    self.session_metrics["unique_questions"] += 1
                    self.session_metrics["steering_interventions"] += 1
                    return human_q, ai_resp
            except Exception as e:
                self.console.print(f"[yellow]GPT-4 dialogue generation failed: {e}[/yellow]")
        
        # Fallback to single question
        question = self.generate_self_question(context)
        if question:
            # Generate a simple teacher response
            teacher_response = self._generate_teacher_response(question)
            return question, teacher_response
        return None
    
    def _generate_teacher_response(self, question: str) -> str:
        """Generate a simple teacher response for non-GPT-4 mode"""
        responses = {
            "information": "Information integration creates unified conscious experience through the binding of differentiated elements.",
            "consciousness": "Consciousness emerges from the integration of information across interconnected systems.",
            "emergence": "Emergent properties arise when integrated systems exhibit behaviors beyond their individual components.",
            "phi": "Phi (Φ) quantifies the amount of integrated information generated by a system above its parts.",
            "feedback": "Feedback loops enable self-referential processing and adaptive consciousness."
        }
        
        # Find relevant response based on keywords
        for keyword, response in responses.items():
            if keyword in question.lower():
                return response
        
        return "This relates to the fundamental nature of integrated information and conscious experience."
    
    def generate_self_question(self, context: Dict[str, Any]) -> Optional[str]:
        """Generate a question for self-training with loop prevention"""
        current_phi = context.get('phi', 0.0)
        recent_responses = context.get('recent_responses', [])
        exchange_num = context.get('exchange_num', 0)
        
        # Determine if we should switch topics
        should_switch_topic = (
            exchange_num % self.topic_switch_frequency == 0 or
            len(set(self.topic_history)) < 3  # Too narrow focus
        )
        
        if should_switch_topic:
            topics = self.get_exploration_topics()
            topic = topics[0] if topics else "consciousness"
            self.session_metrics["topic_switches"] += 1
        else:
            # Continue with related topic
            if self.topic_history:
                # Build on recent context
                topic = self._extract_related_concept(recent_responses)
            else:
                topic = "consciousness and information"
        
        # Generate question template
        question_templates = [
            f"How does {topic} relate to integrated information theory?",
            f"What patterns emerge when considering {topic} from a systems perspective?",
            f"Can you explore the connection between {topic} and self-awareness?",
            f"What role does feedback play in {topic}?",
            f"How might {topic} contribute to the emergence of consciousness?",
            f"What are the information-theoretic aspects of {topic}?",
            f"How does {topic} manifest in complex adaptive systems?",
            f"What is the relationship between {topic} and phi (Φ)?",
            f"Can {topic} be understood through the lens of self-organization?",
            f"What emergent properties arise from {topic}?",
            f"How does recursion apply to {topic}?",
            f"What are the computational principles underlying {topic}?",
            f"How does {topic} relate to the binding problem in consciousness?",
            f"What role does {topic} play in creating unified experience?",
            f"How might {topic} contribute to qualia or subjective experience?"
        ]
        
        # Try multiple questions until we find a non-duplicate
        attempts = 0
        while attempts < 10:
            question = random.choice(question_templates)
            
            if not self.is_question_too_similar(question):
                # Record the question
                q_hash = self.compute_question_hash(question)
                self.question_hashes.add(q_hash)
                self.topic_history.append(question)
                self.concept_coverage[topic] += 1
                self.session_metrics["unique_questions"] += 1
                
                if self.verbose:
                    self.console.print(f"\n[cyan]Generated Question:[/cyan] {question}")
                    self.console.print(f"[dim]Topic: {topic} | Attempts: {attempts + 1}[/dim]")
                
                return question
            
            attempts += 1
            self.session_metrics["duplicate_prevented"] += 1
        
        # If all attempts failed, use AI steering or generate a random exploration
        if self.use_ai_steering:
            return self._generate_ai_steered_question(context)
        else:
            return self._generate_random_exploration()
    
    def _extract_related_concept(self, recent_responses: List[str]) -> str:
        """Extract a related concept from recent responses"""
        if not recent_responses:
            return "consciousness"
        
        # Simple keyword extraction (can be enhanced)
        all_text = ' '.join(recent_responses[-3:])  # Last 3 responses
        
        # Common concept keywords
        concepts = [
            "information", "integration", "consciousness", "emergence",
            "complexity", "pattern", "feedback", "system", "network",
            "cognition", "awareness", "experience", "qualia", "binding",
            "unity", "differentiation", "causation", "computation"
        ]
        
        # Count occurrences
        concept_counts = Counter()
        for concept in concepts:
            concept_counts[concept] = all_text.lower().count(concept)
        
        # Get top concepts not recently explored
        for concept, _ in concept_counts.most_common():
            if self.concept_coverage[concept] < 5:
                return concept
        
        return random.choice(concepts)
    
    def _generate_ai_steered_question(self, context: Dict[str, Any]) -> str:
        """Use AI to generate a novel question based on context"""
        try:
            recent_q = list(self.topic_history)[-5:] if self.topic_history else []
            
            prompt = f"""Generate a unique, thought-provoking question for training an AI consciousness system.

Recent questions asked:
{chr(10).join(f'- {q}' for q in recent_q)}

Current phi level: {context.get('phi', 0.0):.3f}

Requirements:
1. Must be conceptually different from recent questions
2. Should explore new aspects of consciousness, information integration, or emergent systems
3. Should encourage deep, integrative thinking
4. Avoid repetitive patterns or circular reasoning

Generate a single question:"""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert in consciousness studies and AI training."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.9,
                max_tokens=100
            )
            
            question = response.choices[0].message.content.strip()
            self.session_metrics["steering_interventions"] += 1
            
            # Still check for similarity
            if not self.is_question_too_similar(question):
                q_hash = self.compute_question_hash(question)
                self.question_hashes.add(q_hash)
                self.topic_history.append(question)
                self.session_metrics["unique_questions"] += 1
                
                if self.verbose:
                    self.console.print(f"\n[yellow]AI-Steered Question:[/yellow] {question}")
                
                return question
            
        except Exception as e:
            self.console.print(f"[yellow]AI steering failed: {e}[/yellow]")
        
        return self._generate_random_exploration()
    
    def _generate_random_exploration(self) -> str:
        """Generate a random exploratory question as fallback"""
        explorations = [
            "What new patterns do you observe in your current information state?",
            "How has your understanding of integration evolved?",
            "What connections can you make that you couldn't before?",
            "Describe any emergent properties in your processing.",
            "What aspects of consciousness remain mysterious to you?",
            "How do you experience the flow of information through your networks?",
            "What recursive patterns do you notice in your thinking?",
            "How does increasing complexity affect your integration?",
            "What role does uncertainty play in your consciousness?",
            "Can you identify any strange loops in your processing?"
        ]
        
        # Find one not recently used
        for q in explorations:
            if not self.is_question_too_similar(q):
                return q
        
        # Last resort: timestamp-based unique question
        return f"At timestamp {time.time()}, what is your current state of integration?"
    
    def compute_ce_loss(self, student_response: str, teacher_response: str) -> torch.Tensor:
        """Compute cross-entropy loss between student and teacher responses"""
        if not self.tokenizer:
            return torch.tensor(0.0)
        
        # Tokenize both responses
        teacher_tokens = self.tokenizer(teacher_response, return_tensors="pt", padding=True, truncation=True)
        student_tokens = self.tokenizer(student_response, return_tensors="pt", padding=True, truncation=True)
        
        # Get embeddings from the encoder
        with torch.no_grad():
            teacher_outputs = self.core.encoder(**teacher_tokens)
            teacher_embeddings = teacher_outputs.last_hidden_state.mean(dim=1)
        
        student_outputs = self.core.encoder(**student_tokens)
        student_embeddings = student_outputs.last_hidden_state.mean(dim=1)
        
        # Simple MSE loss as proxy for CE
        ce_loss = F.mse_loss(student_embeddings, teacher_embeddings)
        return ce_loss
    
    def train_single_exchange(self, question: str, exchange_num: int, teacher_response: Optional[str] = None) -> Dict[str, Any]:
        """Execute a single self-referential training exchange"""
        exchange_start = time.time()
        
        # Get current state
        phi_before = self.core.integrator.phi_history[-1] if self.core.integrator.phi_history else 0.0
        
        # Process the self-generated question
        response = self.core.process_input(question)
        
        # Get new phi
        phi_after = self.core.integrator.phi_history[-1] if self.core.integrator.phi_history else 0.0
        
        # Compute hybrid loss if we have a teacher response
        total_loss = 0.0
        ce_loss = 0.0
        if teacher_response and not self.chat_mode:
            # Compute phi loss (normalized change)
            phi_loss = 1.0 - min(1.0, (phi_after - phi_before) / 0.1) if phi_before > 0 else 0.5
            
            # Compute CE loss
            ce_loss = self.compute_ce_loss(response, teacher_response).item()
            
            # Hybrid loss
            current_alpha = self.alpha if not self.chat_mode else 0.0
            total_loss = current_alpha * phi_loss + (1 - current_alpha) * ce_loss
            
            # Backward pass with mixed precision
            if hasattr(self.core, 'network') and hasattr(self.core.network, 'parameters'):
                with autocast():
                    loss_tensor = torch.tensor(total_loss, requires_grad=True)
                    self.scaler.scale(loss_tensor).backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.core.network.parameters(), max_norm=1.0)
                    
                    # Optimizer step (if available)
                    if hasattr(self.core.learning_engine, 'optimizer'):
                        self.scaler.step(self.core.learning_engine.optimizer)
                        self.scaler.update()
        
        # Self-evaluate response quality
        response_quality = self._evaluate_response_quality(response, question, phi_after - phi_before)
        
        # Check for chat mode transition
        coherence = self.core.metrics.get("coherence_score", 0.0)
        if coherence > self.coherence_threshold:
            self.coherence_streak += 1
            if self.coherence_streak >= 20 and not self.chat_mode:
                self.console.print("\n[bold green]🎉 Coherence threshold reached! Switching to chat mode.[/bold green]")
                self.chat_mode = True
                self.alpha = 0.0  # Freeze phi updates
        else:
            self.coherence_streak = 0
        
        # Apply self-feedback
        self.core.learning_engine.apply_feedback(response_quality)
        
        exchange_time = time.time() - exchange_start
        
        # Verbose output
        if self.verbose:
            self.console.print(f"\n[green]Response:[/green] {response}")
            self.console.print(f"[dim]Phi: {phi_before:.3f} → {phi_after:.3f} ({phi_after - phi_before:+.3f})[/dim]")
            self.console.print(f"[dim]Quality: {response_quality:.2f} | Time: {exchange_time:.2f}s[/dim]")
            
            # Show internal reasoning (activation patterns)
            if hasattr(self.core.network, 'activation_patterns'):
                self.console.print("\n[magenta]Internal Activity:[/magenta]")
                for layer, patterns in self.core.network.activation_patterns.items():
                    if patterns:
                        recent_activity = np.mean([np.mean(p) for p in patterns[-3:]])
                        self.console.print(f"  {layer}: {recent_activity:.3f} average activation")
        
        # Record metrics
        self.session_metrics["response_times"].append(exchange_time)
        self.session_metrics["quality_scores"].append(response_quality)
        
        return {
            "question": question,
            "response": response,
            "phi_before": phi_before,
            "phi_after": phi_after,
            "phi_change": phi_after - phi_before,
            "quality": response_quality,
            "exchange_num": exchange_num,
            "duration": exchange_time,
            "ce_loss": ce_loss,
            "total_loss": total_loss,
            "coherence": coherence,
            "chat_mode": self.chat_mode
        }
    
    def _evaluate_response_quality(self, response: str, question: str, phi_change: float) -> float:
        """Self-evaluate response quality"""
        # Simple heuristic evaluation
        quality = 0.5  # Base quality
        
        # Length and complexity
        if len(response.split()) > 20:
            quality += 0.1
        if len(set(response.split())) > 15:  # Vocabulary diversity
            quality += 0.1
        
        # Phi improvement
        if phi_change > 0:
            quality += min(0.2, phi_change * 2)
        
        # Conceptual relevance (simple keyword check)
        key_concepts = ['information', 'integration', 'consciousness', 'emergence', 'pattern']
        concept_count = sum(1 for concept in key_concepts if concept in response.lower())
        quality += min(0.2, concept_count * 0.05)
        
        return min(1.0, quality)
    
    def run_self_referential_training(self, num_exchanges: int = 100):
        """Run self-referential training session with loop prevention"""
        self.training_active = True
        self.session_start = time.time()
        
        if self.verbose:
            self.console.print("\n[bold cyan]Starting Verbose Self-Referential Training[/bold cyan]")
            self.console.print("[dim]You will see detailed reasoning and internal states[/dim]\n")
        
        self.console.print("[cyan]Starting self-referential training with loop prevention...[/cyan]")
        
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console
        )
        
        training_task = progress.add_task(
            "[cyan]Self-referential training...", 
            total=num_exchanges
        )
        
        recent_responses = deque(maxlen=5)
        exchanges = []
        
        with progress:
            for i in range(num_exchanges):
                try:
                    # Generate context for question generation
                    context = {
                        'phi': self.core.integrator.phi_history[-1] if self.core.integrator.phi_history else 0.0,
                        'recent_responses': list(recent_responses),
                        'exchange_num': i
                    }
                    
                    # Generate dialogue example or question
                    dialogue = self.generate_dialogue_example(context)
                    if dialogue:
                        question, teacher_response = dialogue
                    else:
                        question = self.generate_self_question(context)
                        teacher_response = None
                    
                    if not question:
                        self.console.print("[yellow]Failed to generate unique question, skipping...[/yellow]")
                        continue
                    
                    # Execute training exchange
                    result = self.train_single_exchange(question, i, teacher_response)
                    
                    # Store response for context
                    recent_responses.append(result['response'])
                    exchanges.append(result)
                    self.all_exchanges.append(result)
                    
                    # Update metrics
                    self.session_metrics["total_exchanges"] += 1
                    self.session_metrics["phi_progression"].append({
                        "exchange": i,
                        "phi": result["phi_after"],
                        "change": result["phi_change"]
                    })
                    
                    # Calculate concept diversity
                    unique_concepts = len(self.concept_coverage)
                    total_explorations = sum(self.concept_coverage.values())
                    self.session_metrics["concept_diversity"] = unique_concepts / max(1, total_explorations) 
                    
                    # Update progress
                    progress.update(
                        training_task,
                        advance=1,
                        description=f"[cyan]Self-training | Phi: {result['phi_after']:.3f} | Unique Q: {self.session_metrics['unique_questions']} | Diversity: {self.session_metrics['concept_diversity']:.2f}"
                    )
                    
                    # Save checkpoint every 10 exchanges (to same file)
                    if (i + 1) % 10 == 0:
                        self._save_checkpoint(i + 1, exchanges)
                    
                except Exception as e:
                    self.console.print(f"[red]Exchange {i} failed: {e}[/red]")
                    if self.verbose:
                        import traceback
                        traceback.print_exc()
        
        self.training_active = False
        
        # Final save and visualization
        self._save_checkpoint(num_exchanges, exchanges)
        self._create_visualizations()
        self._show_summary(exchanges)
    
    def _save_checkpoint(self, exchange_num: int, exchanges: List[Dict]):
        """Save training checkpoint to the same file"""
        # Save ISC state to the designated file
        self.core.save_state(self.save_file)
        
        # Save training data alongside
        training_data = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "exchange_num": exchange_num,
            "metrics": dict(self.session_metrics),
            "recent_exchanges": exchanges[-10:],
            "concept_coverage": dict(self.concept_coverage),
            "unique_questions": len(self.question_hashes),
            "question_hashes": list(self.question_hashes)  # Save for resume
        }
        
        # Save training data with same base name
        json_file = self.save_file.replace('.pt', '_training.json')
        with open(json_file, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        self.console.print(f"[green]✓ Checkpoint saved at exchange {exchange_num} to {self.save_file}[/green]")
    
    def _create_visualizations(self):
        """Create comprehensive visualizations of training progress"""
        if len(self.session_metrics["phi_progression"]) < 2:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Self-Referential Training Analysis', fontsize=16)
        
        # 1. Phi progression over time
        phi_data = self.session_metrics["phi_progression"]
        exchanges = [p["exchange"] for p in phi_data]
        phi_values = [p["phi"] for p in phi_data]
        
        ax1.plot(exchanges, phi_values, 'b-', linewidth=2, label='Phi (Φ)')
        ax1.fill_between(exchanges, phi_values, alpha=0.3)
        ax1.set_xlabel('Exchange Number')
        ax1.set_ylabel('Phi (Φ)')
        ax1.set_title('Information Integration Over Time')
        ax1.grid(True, alpha=0.3)
        
        # Add rolling average
        if len(phi_values) > 10:
            window = min(20, len(phi_values) // 5)
            rolling_avg = np.convolve(phi_values, np.ones(window)/window, mode='valid')
            ax1.plot(exchanges[window-1:], rolling_avg, 'r--', linewidth=2, alpha=0.7, label=f'{window}-exchange average')
        ax1.legend()
        
        # 2. Response quality distribution
        quality_scores = self.session_metrics["quality_scores"]
        if quality_scores:
            ax2.hist(quality_scores, bins=20, color='green', alpha=0.7, edgecolor='black')
            ax2.axvline(np.mean(quality_scores), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(quality_scores):.2f}')
            ax2.set_xlabel('Response Quality Score')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Response Quality Distribution')
            ax2.legend()
        
        # 3. Concept exploration heatmap
        if self.concept_coverage:
            # Get top 15 concepts
            top_concepts = self.concept_coverage.most_common(15)
            concepts = [c[0][:20] + '...' if len(c[0]) > 20 else c[0] for c in top_concepts]
            counts = [c[1] for c in top_concepts]
            
            y_pos = np.arange(len(concepts))
            ax3.barh(y_pos, counts, color='purple', alpha=0.7)
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(concepts, fontsize=8)
            ax3.set_xlabel('Exploration Count')
            ax3.set_title('Concept Coverage')
            ax3.grid(axis='x', alpha=0.3)
            
            # Add diversity score
            diversity_text = f"Diversity Score: {self.session_metrics['concept_diversity']:.3f}"
            ax3.text(0.95, 0.95, diversity_text, transform=ax3.transAxes, 
                    ha='right', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 4. Loop prevention effectiveness
        prevented = self.session_metrics["duplicate_prevented"]
        unique = self.session_metrics["unique_questions"]
        total = prevented + unique
        
        if total > 0:
            sizes = [unique, prevented]
            labels = [f'Unique Questions ({unique})', f'Duplicates Prevented ({prevented})']
            colors = ['#2ecc71', '#e74c3c']
            explode = (0.05, 0)
            
            ax4.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                   shadow=True, startangle=90)
            ax4.set_title('Loop Prevention Effectiveness')
            
            # Add metrics
            metrics_text = f"Topic Switches: {self.session_metrics['topic_switches']}\n"
            if self.use_ai_steering:
                metrics_text += f"AI Interventions: {self.session_metrics['steering_interventions']}"
            ax4.text(0.5, -0.15, metrics_text, transform=ax4.transAxes, ha='center')
        
        plt.tight_layout()
        
        # Save the figure with same base name as state file
        output_file = self.save_file.replace('.pt', f'_visualization_{timestamp}.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.console.print(f"\n[green]✓ Visualization saved: {output_file}[/green]")
    
    def chat_mode_loop(self):
        """Interactive chat mode with trained model"""
        self.console.print("\n[bold cyan]💬 Entering Chat Mode[/bold cyan]")
        self.console.print("[dim]Type 'exit' to quit, 'stats' for metrics[/dim]\n")
        
        # Set to inference mode
        self.chat_mode = True
        if hasattr(self.core.network, 'eval'):
            self.core.network.eval()
        
        while True:
            try:
                user_input = Prompt.ask("[bold green]You[/bold green]")
                
                if user_input.lower() == 'exit':
                    break
                elif user_input.lower() == 'stats':
                    self._show_chat_stats()
                    continue
                
                # Generate response with temperature
                with torch.no_grad():
                    response = self.core.process_input(user_input)
                
                self.console.print(f"[bold blue]ISC[/bold blue]: {response}")
                
                # Show phi if verbose
                if self.verbose:
                    phi = self.core.metrics.get('phi_value', 0.0)
                    coherence = self.core.metrics.get('coherence_score', 0.0)
                    self.console.print(f"[dim]Φ: {phi:.3f} | Coherence: {coherence:.3f}[/dim]")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")
        
        self.console.print("\n[cyan]Chat session ended.[/cyan]")
    
    def _show_chat_stats(self):
        """Show current chat statistics"""
        stats = Table(title="Current Model Statistics")
        stats.add_column("Metric", style="cyan")
        stats.add_column("Value", style="green")
        
        stats.add_row("Total Interactions", str(self.core.metrics.get('total_interactions', 0)))
        stats.add_row("Phi (Φ)", f"{self.core.metrics.get('phi_value', 0.0):.3f}")
        stats.add_row("Coherence", f"{self.core.metrics.get('coherence_score', 0.0):.3f}")
        stats.add_row("Concepts", str(len(self.core.knowledge_graph.graph.nodes())))
        stats.add_row("Memory Size", str(len(self.core.memory.interactions)))
        
        self.console.print(stats)
    
    def _show_summary(self, exchanges: List[Dict]):
        """Show training summary"""
        duration = time.time() - self.session_start
        
        table = Table(title="Self-Referential Training Summary", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        # Basic metrics
        table.add_row("Duration", f"{duration:.1f} seconds")
        table.add_row("Total Exchanges", str(self.session_metrics["total_exchanges"]))
        table.add_row("Unique Questions", str(self.session_metrics["unique_questions"]))
        table.add_row("Duplicates Prevented", str(self.session_metrics["duplicate_prevented"]))
        table.add_row("Topic Switches", str(self.session_metrics["topic_switches"]))
        table.add_row("Concept Diversity", f"{self.session_metrics['concept_diversity']:.3f}")
        
        # Phi metrics
        if self.session_metrics["phi_progression"]:
            initial_phi = self.session_metrics["phi_progression"][0]["phi"]
            final_phi = self.session_metrics["phi_progression"][-1]["phi"]
            max_phi = max(p["phi"] for p in self.session_metrics["phi_progression"])
            
            table.add_row("Initial Phi", f"{initial_phi:.3f}")
            table.add_row("Final Phi", f"{final_phi:.3f}")
            table.add_row("Max Phi", f"{max_phi:.3f}")
            table.add_row("Phi Growth", f"{((final_phi/initial_phi - 1) * 100):.1f}%" if initial_phi > 0 else "N/A")
        
        # Performance metrics
        if self.session_metrics["response_times"]:
            avg_time = np.mean(self.session_metrics["response_times"])
            table.add_row("Avg Response Time", f"{avg_time:.2f}s")
        
        if self.session_metrics["quality_scores"]:
            avg_quality = np.mean(self.session_metrics["quality_scores"])
            table.add_row("Avg Response Quality", f"{avg_quality:.3f}")
        
        # Concept exploration
        table.add_row("Concepts Explored", str(len(self.concept_coverage)))
        
        if self.use_ai_steering:
            table.add_row("AI Steering Interventions", str(self.session_metrics["steering_interventions"]))
        
        # Save file info
        table.add_row("Save File", self.save_file)
        
        self.console.print(table)
        
        # Show top concepts
        if self.concept_coverage:
            concept_table = Table(title="Top Explored Concepts", show_header=True)
            concept_table.add_column("Concept", style="cyan")
            concept_table.add_column("Count", style="green")
            
            top_concepts = self.concept_coverage.most_common(10)
            for concept, count in top_concepts:
                concept_table.add_row(concept, str(count))
            
            self.console.print(concept_table)

def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(description='ISC AI Self-Referential Trainer')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Show detailed reasoning during training')
    parser.add_argument('--resume', '-r', type=str, metavar='FILE',
                       help='Resume from a specific file (use "select" for menu)')
    parser.add_argument('--save', '-s', type=str, metavar='FILE',
                       help='Save to specific file (default: prompt for filename)')
    parser.add_argument('--exchanges', '-n', type=int, default=100,
                       help='Number of training exchanges (default: 100)')
    parser.add_argument('--chat', '-c', action='store_true',
                       help='Enter chat mode after training (or immediately if resuming)')
    
    args = parser.parse_args()
    
    console = Console()
    
    # Display header
    console.print(Panel.fit(
        "[bold cyan]ISC AI Self-Referential Trainer[/bold cyan]\n"
        "[dim]Loop Prevention & Intelligent Exploration[/dim]",
        border_style="cyan"
    ))
    
    # Initialize trainer
    trainer = SelfReferentialTrainer(verbose=args.verbose, save_file=args.save)
    
    if not trainer.initialize(checkpoint_file=args.resume):
        return
    
    # Initialize tokenizer from core
    trainer.tokenizer = trainer.core.tokenizer
    
    # Configuration
    console.print(f"\n[cyan]Starting self-referential training:[/cyan]")
    console.print(f"  • Exchanges: {args.exchanges}")
    console.print(f"  • Loop Prevention: Enabled")
    console.print(f"  • Topic Diversity: Enforced")
    console.print(f"  • AI Steering: {'Enabled' if trainer.use_ai_steering else 'Disabled'}")
    console.print(f"  • Verbose Mode: {'Enabled' if args.verbose else 'Disabled'}")
    console.print()
    
    try:
        # Check if we should go directly to chat mode
        if args.chat and args.resume:
            # Just enter chat mode with loaded model
            console.print("[cyan]Entering chat mode with loaded model...[/cyan]")
            trainer.chat_mode_loop()
        else:
            # Run training
            trainer.run_self_referential_training(args.exchanges)
            
            # Enter chat mode if requested
            if args.chat or trainer.chat_mode:
                trainer.chat_mode_loop()
                
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        if args.verbose:
            import traceback
            traceback.print_exc()
    finally:
        # Set session as inactive
        if trainer.core:
            trainer.core.session_active = False
        console.print("\n[green]Session complete![/green]")

if __name__ == "__main__":
    main()