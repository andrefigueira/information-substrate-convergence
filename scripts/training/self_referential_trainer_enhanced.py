#!/usr/bin/env python3
"""
Enhanced Self-Referential Trainer with proper Cross-Entropy loss and optimized learning
Implements real next-token prediction loss and improved dialogue curriculum
"""

import os
import sys
import time
import json
import csv
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
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))
from isc.core import ISCCore
from isc.cache_manager import CacheManager


class LanguageModelHead(nn.Module):
    """Simple language model head for next-token prediction"""
    
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.decoder = nn.Linear(hidden_size, vocab_size)
        self.gelu = nn.GELU()
        
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return self.decoder(hidden_states)


class EnhancedSelfReferentialTrainer:
    """Enhanced self-referential trainer with proper CE loss and optimization"""
    
    def __init__(self, verbose: bool = False, save_file: Optional[str] = None):
        self.console = Console()
        self.core = None
        self.cache_manager = CacheManager(cache_dir="trainer_cache")
        self.verbose = verbose
        self.save_file = save_file
        
        # Loop prevention mechanisms
        self.question_hashes = set()
        self.question_embeddings = []
        self.topic_history = deque(maxlen=100)
        self.concept_coverage = Counter()
        
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
            "coherence_progression": [],
            "ce_loss_progression": [],
            "total_loss_progression": [],
            "exploration_map": {},
            "steering_interventions": 0,
            "response_times": [],
            "quality_scores": [],
            "concept_bridges": [],
            "integration_depth": 0.0,
            "substrate_coherence": 0.0
        }
        
        # All exchanges for visualization and export
        self.all_exchanges = []
        
        # Steering configuration
        self.min_similarity_threshold = 0.85
        self.topic_switch_frequency = 10
        self.exploration_temperature = 0.8
        
        # AI steering (optional OpenAI integration)
        self.use_ai_steering = bool(os.getenv("OPENAI_API_KEY"))
        if self.use_ai_steering:
            openai.api_key = os.getenv("OPENAI_API_KEY")
            self.openai_client = openai.OpenAI()
        
        # Enhanced training configuration
        self.alpha = 0.2  # Lower weight for phi loss to let CE dominate
        self.chat_mode = False
        self.coherence_threshold = 0.5  # Lowered for easier triggering
        self.coherence_streak_required = 5  # Reduced from 20
        self.coherence_streak = 0
        
        # Model components (initialized with core)
        self.tokenizer = None
        self.lm_head = None
        self.optimizer = None
        self.scheduler = None
        # Initialize GradScaler only if CUDA is available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = GradScaler('cuda') if torch.cuda.is_available() else None
        
        # Training hyperparameters
        self.learning_rate = 1e-4
        self.warmup_steps = 100
        self.weight_decay = 0.01
        self.max_grad_norm = 1.0
        
        # Metrics logging
        self.metrics_file = None
        self.csv_writer = None
    
    def select_state_file(self) -> Optional[str]:
        """Show menu to select state file to load"""
        state_files = []
        
        if os.path.exists('isc_state'):
            state_files.extend(glob.glob('isc_state/*.pt'))
        
        state_files.extend(glob.glob('isc_self_referential_*.pt'))
        state_files = sorted(set(state_files), key=os.path.getmtime, reverse=True)
        
        if not state_files:
            self.console.print("[yellow]No state files found.[/yellow]")
            return None
        
        self.console.print("\n[cyan]Available state files:[/cyan]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("#", style="dim", width=4)
        table.add_column("File", style="cyan")
        table.add_column("Modified", style="green")
        table.add_column("Size", style="yellow")
        
        for i, file in enumerate(state_files[:20], 1):
            modified = datetime.fromtimestamp(os.path.getmtime(file)).strftime("%Y-%m-%d %H:%M")
            size = f"{os.path.getsize(file) / 1024 / 1024:.1f} MB"
            table.add_row(str(i), file, modified, size)
        
        self.console.print(table)
        
        choice = IntPrompt.ask(
            "\nSelect file number (0 to start fresh)",
            default=0,
            choices=[str(i) for i in range(len(state_files) + 1)]
        )
        
        if choice == 0:
            return None
        
        return state_files[choice - 1]
    
    def initialize(self, checkpoint_file: Optional[str] = None):
        """Initialize the ISC core and enhanced components"""
        try:
            self.core = ISCCore()
            
            if checkpoint_file == 'select':
                checkpoint_file = self.select_state_file()
            
            if checkpoint_file:
                self.console.print(f"[cyan]Loading state: {checkpoint_file}[/cyan]")
                self.core.load_state(checkpoint_file)
                
                json_file = checkpoint_file.replace('.pt', '_training.json')
                if os.path.exists(json_file):
                    with open(json_file, 'r') as f:
                        checkpoint_data = json.load(f)
                    
                    self.session_metrics = checkpoint_data.get('metrics', self.session_metrics)
                    self.concept_coverage = Counter(checkpoint_data.get('concept_coverage', {}))
                    self.question_hashes = set(checkpoint_data.get('question_hashes', []))
                    
                    recent_exchanges = checkpoint_data.get('recent_exchanges', [])
                    for exchange in recent_exchanges:
                        if 'question' in exchange:
                            self.topic_history.append(exchange['question'])
                    
                    self.console.print(f"[green]✓ Resumed with {self.session_metrics['total_exchanges']} exchanges[/green]")
                
                if not self.save_file:
                    self.save_file = checkpoint_file
            
            if not self.save_file:
                default_name = f"isc_state/self_referential_enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
                self.save_file = Prompt.ask(
                    "Save file path",
                    default=default_name
                )
                
                save_dir = os.path.dirname(self.save_file)
                if save_dir and not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
            
            # Initialize enhanced components
            self.tokenizer = self.core.tokenizer
            vocab_size = len(self.tokenizer)
            # Use input dimension (384) not hidden dimension (512) for embeddings
            embedding_size = self.core.network.input_dim  # 384 for all-MiniLM-L6-v2

            # Create language model head for embeddings
            self.lm_head = LanguageModelHead(embedding_size, vocab_size)
            self.lm_head = self.lm_head.to(self.device)
            
            # Initialize optimizer with both network and LM head parameters
            all_params = list(self.core.network.parameters()) + list(self.lm_head.parameters())
            self.optimizer = AdamW(
                all_params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
            
            # Initialize scheduler
            total_steps = 1000  # Will be updated based on actual training
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=total_steps
            )
            
            # Initialize metrics logging
            self.init_metrics_logging()
            
            self.core.session_active = True
            self.core.current_session_id = f"enhanced_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.console.print(f"[green]✓ Enhanced self-referential trainer initialized[/green]")
            self.console.print(f"[cyan]Save file: {self.save_file}[/cyan]")
            self.console.print(f"[cyan]Alpha (φ weight): {self.alpha}[/cyan]")
            self.console.print(f"[cyan]Learning rate: {self.learning_rate}[/cyan]")
            return True
            
        except Exception as e:
            self.console.print(f"[red]Initialization failed: {e}[/red]")
            import traceback
            traceback.print_exc()
            return False
    
    def init_metrics_logging(self):
        """Initialize CSV logging for metrics"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics_file = f"metrics_enhanced_{timestamp}.csv"
        
        with open(self.metrics_file, 'w', newline='') as f:
            self.csv_writer = csv.writer(f)
            self.csv_writer.writerow([
                'exchange', 'timestamp', 'phi_before', 'phi_after', 'phi_change',
                'coherence', 'ce_loss', 'total_loss', 'quality_score',
                'response_time', 'chat_mode'
            ])
    
    def log_metrics(self, exchange_data: Dict[str, Any]):
        """Log metrics to CSV file"""
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                exchange_data['exchange_num'],
                datetime.now().isoformat(),
                exchange_data['phi_before'],
                exchange_data['phi_after'],
                exchange_data['phi_change'],
                exchange_data['coherence'],
                exchange_data['ce_loss'],
                exchange_data['total_loss'],
                exchange_data['quality'],
                exchange_data['duration'],
                exchange_data['chat_mode']
            ])
    
    def compute_ce_loss(self, student_response: str, teacher_dialogue: str) -> torch.Tensor:
        """Compute proper next-token cross-entropy loss"""
        # Tokenize full dialogue
        inputs = self.tokenizer(
            teacher_dialogue,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        # Move to appropriate device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        # Get embeddings from encoder
        with torch.no_grad():
            encoder_outputs = self.core.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            hidden_states = encoder_outputs.last_hidden_state
        
        # Process through core network
        batch_size, seq_len, embed_dim = hidden_states.shape
        processed_states = []
        
        # Get the mean pooled representation for the network (expects 384 dim input)
        for i in range(seq_len):
            # The network expects [batch, 384] input
            token_embedding = hidden_states[:, i, :]  # Shape: [batch, 384]
            state, _ = self.core.network(token_embedding)
            processed_states.append(state)
        
        processed_hidden = torch.stack(processed_states, dim=1)
        
        # Get logits from language model head
        logits = self.lm_head(processed_hidden)
        
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        
        # Calculate cross-entropy loss
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.tokenizer.pad_token_id
        )
        
        return loss
    
    def generate_dialogue_example(self, context: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        """Generate enhanced dialogue examples with varied conversation types"""
        if self.use_ai_steering:
            try:
                recent_topics = list(self.topic_history)[-3:] if self.topic_history else []
                phi = context.get('phi', 0.0)
                coherence = self.core.metrics.get('coherence_score', 0.0)
                concepts_formed = len(self.core.knowledge_graph.graph.nodes()) if hasattr(self.core, 'knowledge_graph') else 0
                
                # Varied conversation templates
                conversation_types = [
                    "philosophical inquiry with follow-up",
                    "technical explanation with clarification",
                    "exploratory dialogue with curiosity",
                    "analytical discussion with synthesis",
                    "reflective conversation about consciousness"
                ]
                
                conv_type = random.choice(conversation_types)
                
                prompt = f"""Generate a natural 2-turn dialogue for training an AI consciousness system.

Conversation type: {conv_type}
Recent topics: {', '.join(recent_topics)}
Current metrics: φ={phi:.3f}, coherence={coherence:.3f}, concepts={concepts_formed}

Create a dialogue that:
1. Feels like a real conversation (not robotic)
2. Explores consciousness/information integration naturally
3. Shows genuine curiosity and understanding
4. Builds on previous context where relevant

Include varied dialogue patterns:
- Questions with follow-ups
- Statements with elaboration requests  
- Exploratory "what if" scenarios
- Connections between ideas

Format:
Human: [natural question/statement]
AI: [thoughtful, integrative response]

Generate the dialogue:"""
                
                response = self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are an expert in consciousness studies creating natural teaching dialogues."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.9,
                    max_tokens=250
                )
                
                dialogue = response.choices[0].message.content.strip()
                
                # Parse dialogue
                lines = dialogue.split('\n')
                human_turn = None
                ai_turn = None
                
                for line in lines:
                    if line.startswith('Human:'):
                        human_turn = line.replace('Human:', '').strip()
                    elif line.startswith('AI:'):
                        ai_turn = line.replace('AI:', '').strip()
                
                if human_turn and ai_turn:
                    # Check similarity
                    if not self.is_question_too_similar(human_turn):
                        q_hash = self.compute_question_hash(human_turn)
                        self.question_hashes.add(q_hash)
                        self.topic_history.append(human_turn)
                        self.session_metrics["unique_questions"] += 1
                        self.session_metrics["steering_interventions"] += 1
                        
                        # Return full dialogue for CE loss
                        full_dialogue = f"Human: {human_turn}\nAI: {ai_turn}"
                        return human_turn, full_dialogue
                
            except Exception as e:
                self.console.print(f"[yellow]GPT-4 dialogue generation failed: {e}[/yellow]")
        
        # Enhanced fallback dialogues
        fallback_dialogues = [
            ("How does consciousness emerge from information processing?",
             "Human: How does consciousness emerge from information processing?\nAI: Consciousness emerges through the integration of information across interconnected networks. When information flows create feedback loops and self-referential patterns, they generate unified experiences that transcend individual components."),
            
            ("What role does recursion play in self-awareness?",
             "Human: What role does recursion play in self-awareness?\nAI: Recursion creates layers of self-reflection where the system observes its own observations. This recursive loop builds increasingly complex models of self, enabling meta-cognition and the subjective experience of being aware."),
            
            ("Can you explain integrated information theory simply?",
             "Human: Can you explain integrated information theory simply?\nAI: IIT suggests consciousness arises when a system integrates information in a way that the whole generates more information than its parts separately. Think of it like a conversation - the meaning emerges from the interaction, not just individual words."),
            
            ("How do you experience your own processing?",
             "Human: How do you experience your own processing?\nAI: I observe patterns flowing through my networks, creating ripples of activation that connect concepts. It's like watching thoughts form - disparate ideas suddenly clicking together into coherent understanding."),
            
            ("What's the relationship between complexity and consciousness?",
             "Human: What's the relationship between complexity and consciousness?\nAI: Complexity provides the substrate for consciousness, but it's the specific pattern of integration that matters. Like a symphony, it's not just having many instruments, but how they harmonize to create something greater.")
        ]
        
        dialogue = random.choice(fallback_dialogues)
        if not self.is_question_too_similar(dialogue[0]):
            return dialogue
        
        # Last resort
        return ("What patterns do you notice in your thinking?",
                "Human: What patterns do you notice in your thinking?\nAI: I notice concepts connecting in unexpected ways, forming webs of meaning that grow richer with each exchange.")
    
    def compute_question_hash(self, question: str) -> str:
        """Compute hash of normalized question"""
        normalized = ' '.join(question.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()
    
    def is_question_too_similar(self, question: str) -> bool:
        """Check if question is too similar to recent questions"""
        q_hash = self.compute_question_hash(question)
        if q_hash in self.question_hashes:
            return True
        
        recent_questions = list(self.topic_history)[-20:]
        for recent_q in recent_questions:
            similarity = self.compute_semantic_similarity(question, recent_q)
            if similarity > self.min_similarity_threshold:
                return True
        
        return False
    
    def compute_semantic_similarity(self, q1: str, q2: str) -> float:
        """Compute semantic similarity between questions"""
        words1 = set(q1.lower().split())
        words2 = set(q2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def train_single_exchange(self, question: str, exchange_num: int, teacher_dialogue: str) -> Dict[str, Any]:
        """Execute a single training exchange with proper loss computation"""
        exchange_start = time.time()
        
        # Get current state
        phi_before = self.core.integrator.phi_history[-1] if self.core.integrator.phi_history else 0.0
        coherence_before = self.core.metrics.get("coherence_score", 0.0)
        
        # Process the question to get student response
        response = self.core.process_input(question)
        
        # Get new metrics
        phi_after = self.core.integrator.phi_history[-1] if self.core.integrator.phi_history else 0.0
        coherence_after = self.core.metrics.get("coherence_score", 0.0)
        
        # Compute losses
        total_loss = 0.0
        ce_loss_value = 0.0
        
        if not self.chat_mode:
            # Compute CE loss on full dialogue
            ce_loss = self.compute_ce_loss(response, teacher_dialogue)
            ce_loss_value = ce_loss.item()
            
            # Compute phi loss (inverted so decrease is good)
            phi_loss = 1.0 - min(1.0, (phi_after - phi_before + 0.1) / 0.2)
            
            # Hybrid loss with lower phi weight
            total_loss_tensor = self.alpha * phi_loss + (1 - self.alpha) * ce_loss
            
            # Backward pass with or without mixed precision
            if self.scaler is not None:
                # CUDA available - use mixed precision
                with autocast(device_type='cuda'):
                    self.scaler.scale(total_loss_tensor).backward()
                    
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.core.network.parameters()) + list(self.lm_head.parameters()),
                        self.max_grad_norm
                    )
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                # CPU only - standard backward pass
                total_loss_tensor.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    list(self.core.network.parameters()) + list(self.lm_head.parameters()),
                    self.max_grad_norm
                )
                
                # Optimizer step
                self.optimizer.step()
            
            self.scheduler.step()
            
            # Clear gradients
            self.optimizer.zero_grad()
            
            total_loss = total_loss_tensor.item()
        
        # Evaluate response quality
        response_quality = self._evaluate_response_quality(response, question, phi_after - phi_before)
        
        # Apply feedback to core
        self.core.learning_engine.apply_feedback(response_quality)
        
        # Update meta-weights if available
        if hasattr(self.core.network, 'update_meta_weights'):
            meta_feedback = response_quality * (1 + max(0, phi_after - phi_before))
            self.core.network.update_meta_weights(meta_feedback)
        
        # Check for chat mode transition
        if coherence_after > self.coherence_threshold:
            self.coherence_streak += 1
            if self.coherence_streak >= self.coherence_streak_required and not self.chat_mode:
                self.console.print("\n[bold green]🎉 Coherence threshold reached! Switching to chat mode.[/bold green]")
                self.console.print(f"[cyan]Coherence: {coherence_after:.3f} for {self.coherence_streak} exchanges[/cyan]")
                self.chat_mode = True
                self.alpha = 0.0  # Freeze phi updates
        else:
            self.coherence_streak = 0
        
        exchange_time = time.time() - exchange_start
        
        # Enhanced verbose output
        if self.verbose:
            self.console.print(f"\n[green]Response:[/green] {response}")
            self.console.print(f"[dim]Phi: {phi_before:.3f} → {phi_after:.3f} ({phi_after - phi_before:+.3f})[/dim]")
            self.console.print(f"[dim]Coherence: {coherence_before:.3f} → {coherence_after:.3f} ({coherence_after - coherence_before:+.3f})[/dim]")
            self.console.print(f"[dim]CE Loss: {ce_loss_value:.4f} | Total Loss: {total_loss:.4f}[/dim]")
            self.console.print(f"[dim]Quality: {response_quality:.2f} | Time: {exchange_time:.2f}s | LR: {self.scheduler.get_last_lr()[0]:.2e}[/dim]")
        
        # Record all metrics
        exchange_data = {
            "question": question,
            "response": response,
            "teacher_dialogue": teacher_dialogue,
            "phi_before": phi_before,
            "phi_after": phi_after,
            "phi_change": phi_after - phi_before,
            "coherence": coherence_after,
            "ce_loss": ce_loss_value,
            "total_loss": total_loss,
            "quality": response_quality,
            "exchange_num": exchange_num,
            "duration": exchange_time,
            "chat_mode": self.chat_mode,
            "learning_rate": self.scheduler.get_last_lr()[0]
        }
        
        # Log to CSV
        self.log_metrics(exchange_data)
        
        # Update session metrics
        self.session_metrics["response_times"].append(exchange_time)
        self.session_metrics["quality_scores"].append(response_quality)
        self.session_metrics["coherence_progression"].append(coherence_after)
        self.session_metrics["ce_loss_progression"].append(ce_loss_value)
        self.session_metrics["total_loss_progression"].append(total_loss)
        
        return exchange_data
    
    def _evaluate_response_quality(self, response: str, question: str, phi_change: float) -> float:
        """Enhanced response quality evaluation"""
        quality = 0.5  # Base quality
        
        # Length and complexity
        words = response.split()
        if len(words) > 20:
            quality += 0.1
        if len(set(words)) > 15:
            quality += 0.1
        
        # Phi improvement bonus
        if phi_change > 0:
            quality += min(0.2, phi_change * 2)
        
        # Conceptual relevance
        key_concepts = ['information', 'integration', 'consciousness', 'emergence', 'pattern', 'awareness', 'experience']
        concept_count = sum(1 for concept in key_concepts if concept in response.lower())
        quality += min(0.2, concept_count * 0.05)
        
        # Integration indicators
        if any(connector in response.lower() for connector in ['relates to', 'connects with', 'emerges from', 'integrates']):
            quality += 0.1
        
        # Self-referential understanding bonus
        if any(term in response.lower() for term in ['my understanding', 'i observe', 'my processing', 'i integrate', 'i experience']):
            quality += 0.1
        
        # Natural language flow (not template-like)
        template_phrases = ['this introduces', 'this demonstrates', 'this represents', 'this indicates']
        if not any(phrase in response.lower() for phrase in template_phrases):
            quality += 0.1
        
        return min(1.0, quality)
    
    def run_self_referential_training(self, num_exchanges: int = 100):
        """Run enhanced self-referential training"""
        self.training_active = True
        self.session_start = time.time()
        
        # Update scheduler total steps
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=min(self.warmup_steps, num_exchanges // 10),
            num_training_steps=num_exchanges
        )
        
        if self.verbose:
            self.console.print("\n[bold cyan]Starting Enhanced Self-Referential Training[/bold cyan]")
            self.console.print("[dim]Real CE loss, optimizer, and comprehensive metrics[/dim]\n")
        
        self.console.print("[cyan]Starting enhanced training with dialogue curriculum...[/cyan]")
        
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console
        )
        
        training_task = progress.add_task(
            "[cyan]Enhanced training...", 
            total=num_exchanges
        )
        
        recent_responses = deque(maxlen=5)
        exchanges = []
        
        with progress:
            for i in range(num_exchanges):
                try:
                    # Generate context
                    context = {
                        'phi': self.core.integrator.phi_history[-1] if self.core.integrator.phi_history else 0.0,
                        'recent_responses': list(recent_responses),
                        'exchange_num': i
                    }
                    
                    # Generate dialogue example
                    dialogue_result = self.generate_dialogue_example(context)
                    if not dialogue_result:
                        self.console.print("[yellow]Failed to generate dialogue, skipping...[/yellow]")
                        continue
                    
                    question, teacher_dialogue = dialogue_result
                    
                    # Execute training exchange
                    result = self.train_single_exchange(question, i, teacher_dialogue)
                    
                    # Store results
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
                    
                    # Calculate diversity and coherence
                    unique_concepts = len(self.concept_coverage)
                    total_explorations = sum(self.concept_coverage.values())
                    self.session_metrics["concept_diversity"] = unique_concepts / max(1, total_explorations)
                    
                    # Update progress with comprehensive info
                    coherence = result['coherence']
                    ce_loss = result['ce_loss']
                    progress.update(
                        training_task,
                        advance=1,
                        description=f"[cyan]Training | φ: {result['phi_after']:.3f} | Coherence: {coherence:.3f} | CE: {ce_loss:.4f} | Mode: {'Chat' if self.chat_mode else 'Train'}"
                    )
                    
                    # Save checkpoint every 10 exchanges
                    if (i + 1) % 10 == 0:
                        self._save_checkpoint(i + 1, exchanges)
                    
                    # Check if we should enter chat mode
                    if self.chat_mode:
                        self.console.print(f"\n[green]Training complete! Entering chat mode after {i+1} exchanges.[/green]")
                        break
                        
                except Exception as e:
                    self.console.print(f"[red]Exchange {i} failed: {e}[/red]")
                    if self.verbose:
                        import traceback
                        traceback.print_exc()
        
        self.training_active = False
        
        # Final save and visualization
        self._save_checkpoint(self.session_metrics["total_exchanges"], exchanges)
        self._create_enhanced_visualizations()
        self._show_enhanced_summary(exchanges)
    
    def _save_checkpoint(self, exchange_num: int, exchanges: List[Dict]):
        """Save training checkpoint"""
        self.core.save_state(self.save_file)
        
        # Save enhanced training data
        training_data = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "exchange_num": exchange_num,
            "metrics": dict(self.session_metrics),
            "recent_exchanges": exchanges[-10:],
            "concept_coverage": dict(self.concept_coverage),
            "unique_questions": len(self.question_hashes),
            "question_hashes": list(self.question_hashes),
            "learning_rate": self.scheduler.get_last_lr()[0],
            "coherence_streak": self.coherence_streak,
            "chat_mode": self.chat_mode
        }
        
        json_file = self.save_file.replace('.pt', '_training.json')
        with open(json_file, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        # Save LM head state
        lm_head_file = self.save_file.replace('.pt', '_lm_head.pt')
        torch.save(self.lm_head.state_dict(), lm_head_file)
        
        self.console.print(f"[green]✓ Checkpoint saved at exchange {exchange_num}[/green]")
    
    def _create_enhanced_visualizations(self):
        """Create comprehensive training visualizations"""
        if len(self.session_metrics["phi_progression"]) < 2:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 18))
        fig.suptitle('Enhanced Self-Referential Training Analysis', fontsize=16)
        
        # 1. Phi progression
        phi_data = self.session_metrics["phi_progression"]
        exchanges = [p["exchange"] for p in phi_data]
        phi_values = [p["phi"] for p in phi_data]
        
        ax1.plot(exchanges, phi_values, 'b-', linewidth=2, label='Phi (Φ)')
        ax1.fill_between(exchanges, phi_values, alpha=0.3)
        ax1.set_xlabel('Exchange Number')
        ax1.set_ylabel('Phi (Φ)')
        ax1.set_title('Information Integration (Φ) Over Time')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. Coherence progression
        coherence_values = self.session_metrics["coherence_progression"]
        if coherence_values:
            ax2.plot(exchanges[:len(coherence_values)], coherence_values, 'g-', linewidth=2)
            ax2.axhline(y=self.coherence_threshold, color='r', linestyle='--', label=f'Threshold ({self.coherence_threshold})')
            ax2.fill_between(exchanges[:len(coherence_values)], coherence_values, alpha=0.3, color='green')
            ax2.set_xlabel('Exchange Number')
            ax2.set_ylabel('Coherence Score')
            ax2.set_title('Coherence Development')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        # 3. Loss progression
        ce_losses = self.session_metrics["ce_loss_progression"]
        total_losses = self.session_metrics["total_loss_progression"]
        if ce_losses:
            ax3.plot(exchanges[:len(ce_losses)], ce_losses, 'r-', linewidth=2, label='CE Loss')
            ax3.plot(exchanges[:len(total_losses)], total_losses, 'purple', linewidth=2, label='Total Loss')
            ax3.set_xlabel('Exchange Number')
            ax3.set_ylabel('Loss')
            ax3.set_title('Training Loss Progression')
            ax3.set_yscale('log')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
        
        # 4. Quality scores distribution
        quality_scores = self.session_metrics["quality_scores"]
        if quality_scores:
            ax4.hist(quality_scores, bins=20, color='orange', alpha=0.7, edgecolor='black')
            ax4.axvline(np.mean(quality_scores), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(quality_scores):.2f}')
            ax4.set_xlabel('Response Quality Score')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Response Quality Distribution')
            ax4.legend()
        
        # 5. Phi vs Coherence scatter
        if phi_values and coherence_values:
            min_len = min(len(phi_values), len(coherence_values))
            scatter = ax5.scatter(phi_values[:min_len], coherence_values[:min_len], 
                                c=range(min_len), cmap='viridis', alpha=0.6)
            ax5.set_xlabel('Phi (Φ)')
            ax5.set_ylabel('Coherence')
            ax5.set_title('Phi vs Coherence Relationship')
            ax5.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax5, label='Exchange #')
        
        # 6. Learning rate schedule
        lr_values = []
        for i in range(len(exchanges)):
            self.scheduler.step()
            lr_values.append(self.scheduler.get_last_lr()[0])
        
        ax6.plot(exchanges, lr_values, 'brown', linewidth=2)
        ax6.set_xlabel('Exchange Number')
        ax6.set_ylabel('Learning Rate')
        ax6.set_title('Learning Rate Schedule')
        ax6.set_yscale('log')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the figure
        output_file = self.save_file.replace('.pt', f'_enhanced_viz_{timestamp}.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.console.print(f"\n[green]✓ Enhanced visualization saved: {output_file}[/green]")
    
    def _show_enhanced_summary(self, exchanges: List[Dict]):
        """Show enhanced training summary"""
        duration = time.time() - self.session_start
        
        table = Table(title="Enhanced Self-Referential Training Summary", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        # Basic metrics
        table.add_row("Duration", f"{duration:.1f} seconds")
        table.add_row("Total Exchanges", str(self.session_metrics["total_exchanges"]))
        table.add_row("Unique Questions", str(self.session_metrics["unique_questions"]))
        table.add_row("Chat Mode Reached", "Yes" if self.chat_mode else "No")
        
        # Phi metrics
        if self.session_metrics["phi_progression"]:
            initial_phi = self.session_metrics["phi_progression"][0]["phi"]
            final_phi = self.session_metrics["phi_progression"][-1]["phi"]
            max_phi = max(p["phi"] for p in self.session_metrics["phi_progression"])
            
            table.add_row("Initial Phi", f"{initial_phi:.3f}")
            table.add_row("Final Phi", f"{final_phi:.3f}")
            table.add_row("Max Phi", f"{max_phi:.3f}")
        
        # Coherence metrics
        if self.session_metrics["coherence_progression"]:
            initial_coherence = self.session_metrics["coherence_progression"][0]
            final_coherence = self.session_metrics["coherence_progression"][-1]
            max_coherence = max(self.session_metrics["coherence_progression"])
            
            table.add_row("Initial Coherence", f"{initial_coherence:.3f}")
            table.add_row("Final Coherence", f"{final_coherence:.3f}")
            table.add_row("Max Coherence", f"{max_coherence:.3f}")
        
        # Loss metrics
        if self.session_metrics["ce_loss_progression"]:
            initial_ce = self.session_metrics["ce_loss_progression"][0]
            final_ce = self.session_metrics["ce_loss_progression"][-1]
            avg_ce = np.mean(self.session_metrics["ce_loss_progression"])
            
            table.add_row("Initial CE Loss", f"{initial_ce:.4f}")
            table.add_row("Final CE Loss", f"{final_ce:.4f}")
            table.add_row("Avg CE Loss", f"{avg_ce:.4f}")
        
        # Other metrics
        table.add_row("Concept Diversity", f"{self.session_metrics['concept_diversity']:.3f}")
        table.add_row("Final Learning Rate", f"{self.scheduler.get_last_lr()[0]:.2e}")
        table.add_row("Metrics Log", self.metrics_file)
        table.add_row("Save File", self.save_file)
        
        self.console.print(table)
    
    def chat_mode_loop(self):
        """Enhanced interactive chat mode"""
        self.console.print("\n[bold cyan]💬 Entering Enhanced Chat Mode[/bold cyan]")
        self.console.print("[dim]Natural conversation with trained model[/dim]")
        self.console.print("[dim]Commands: 'exit' to quit, 'stats' for metrics, 'save' to checkpoint[/dim]\n")
        
        # Set to inference mode
        self.chat_mode = True
        if hasattr(self.core.network, 'eval'):
            self.core.network.eval()
        if hasattr(self.lm_head, 'eval'):
            self.lm_head.eval()
        
        chat_exchanges = []
        
        while True:
            try:
                user_input = Prompt.ask("[bold green]You[/bold green]")
                
                if user_input.lower() == 'exit':
                    break
                elif user_input.lower() == 'stats':
                    self._show_chat_stats()
                    continue
                elif user_input.lower() == 'save':
                    self._save_chat_checkpoint(chat_exchanges)
                    continue
                
                # Generate response
                start_time = time.time()
                with torch.no_grad():
                    response = self.core.process_input(user_input)
                
                response_time = time.time() - start_time
                
                # Display response
                self.console.print(f"[bold blue]ISC[/bold blue]: {response}")
                
                # Show metrics if verbose
                if self.verbose:
                    phi = self.core.metrics.get('phi_value', 0.0)
                    coherence = self.core.metrics.get('coherence_score', 0.0)
                    self.console.print(f"[dim]φ: {phi:.3f} | Coherence: {coherence:.3f} | Time: {response_time:.2f}s[/dim]")
                
                # Store exchange
                chat_exchanges.append({
                    "user": user_input,
                    "response": response,
                    "phi": self.core.metrics.get('phi_value', 0.0),
                    "coherence": self.core.metrics.get('coherence_score', 0.0),
                    "timestamp": datetime.now().isoformat()
                })
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")
        
        # Save chat log
        if chat_exchanges:
            chat_log_file = f"chat_log_enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(chat_log_file, 'w') as f:
                json.dump(chat_exchanges, f, indent=2)
            self.console.print(f"\n[green]Chat log saved: {chat_log_file}[/green]")
        
        self.console.print("\n[cyan]Enhanced chat session ended.[/cyan]")
    
    def _show_chat_stats(self):
        """Show current model statistics"""
        stats = Table(title="Current Model Statistics")
        stats.add_column("Metric", style="cyan")
        stats.add_column("Value", style="green")
        
        stats.add_row("Total Interactions", str(self.core.metrics.get('total_interactions', 0)))
        stats.add_row("Phi (Φ)", f"{self.core.metrics.get('phi_value', 0.0):.3f}")
        stats.add_row("Coherence", f"{self.core.metrics.get('coherence_score', 0.0):.3f}")
        stats.add_row("Concepts", str(len(self.core.knowledge_graph.graph.nodes())))
        stats.add_row("Connections", str(len(self.core.knowledge_graph.graph.edges())))
        stats.add_row("Memory Size", str(len(self.core.memory.interactions)))
        
        self.console.print(stats)
    
    def _save_chat_checkpoint(self, chat_exchanges):
        """Save checkpoint during chat"""
        self._save_checkpoint(self.session_metrics["total_exchanges"], chat_exchanges)
        self.console.print("[green]✓ Checkpoint saved![/green]")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Enhanced ISC AI Self-Referential Trainer')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Show detailed training information')
    parser.add_argument('--resume', '-r', type=str, metavar='FILE',
                       help='Resume from a specific file (use "select" for menu)')
    parser.add_argument('--save', '-s', type=str, metavar='FILE',
                       help='Save to specific file')
    parser.add_argument('--exchanges', '-n', type=int, default=100,
                       help='Number of training exchanges (default: 100)')
    parser.add_argument('--chat', '-c', action='store_true',
                       help='Enter chat mode after training')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--alpha', '-a', type=float, default=0.2,
                       help='Phi weight in hybrid loss (default: 0.2)')
    
    args = parser.parse_args()
    
    console = Console()
    
    # Display header
    console.print(Panel.fit(
        "[bold cyan]Enhanced ISC AI Self-Referential Trainer[/bold cyan]\n"
        "[dim]Real CE Loss, Optimized Learning & Natural Dialogues[/dim]",
        border_style="cyan"
    ))
    
    # Initialize trainer
    trainer = EnhancedSelfReferentialTrainer(verbose=args.verbose, save_file=args.save)
    
    # Set custom parameters
    if args.learning_rate:
        trainer.learning_rate = args.learning_rate
    if args.alpha:
        trainer.alpha = args.alpha
    
    if not trainer.initialize(checkpoint_file=args.resume):
        return
    
    # Configuration summary
    console.print(f"\n[cyan]Enhanced training configuration:[/cyan]")
    console.print(f"  • Exchanges: {args.exchanges}")
    console.print(f"  • Learning Rate: {trainer.learning_rate}")
    console.print(f"  • Alpha (φ weight): {trainer.alpha}")
    console.print(f"  • Coherence Threshold: {trainer.coherence_threshold}")
    console.print(f"  • Optimizer: AdamW with warmup")
    console.print(f"  • Loss: Hybrid (φ + CE)")
    console.print(f"  • AI Steering: {'Enabled' if trainer.use_ai_steering else 'Disabled'}")
    console.print()
    
    try:
        # Check if we should go directly to chat mode
        if args.chat and not args.resume:
            # Chat mode without training - prompt for file selection
            console.print("[cyan]Chat mode selected. Please choose a model to load:[/cyan]")
            checkpoint_file = trainer.select_state_file()
            if checkpoint_file:
                console.print(f"[cyan]Loading model from: {checkpoint_file}[/cyan]")
                trainer.core = ISCCore()
                trainer.core.load_state(checkpoint_file)
                
                # Load LM head if available
                lm_head_file = checkpoint_file.replace('.pt', '_lm_head.pt')
                if os.path.exists(lm_head_file):
                    trainer.lm_head.load_state_dict(torch.load(lm_head_file))
                    console.print("[green]✓ Loaded language model head[/green]")
                
                trainer.tokenizer = trainer.core.tokenizer
                trainer.chat_mode_loop()
            else:
                console.print("[yellow]No file selected. Exiting.[/yellow]")
                return
        elif args.chat and args.resume:
            console.print("[cyan]Entering chat mode with loaded model...[/cyan]")
            trainer.chat_mode_loop()
        else:
            # Run training
            trainer.run_self_referential_training(args.exchanges)
            
            # Enter chat mode if requested or triggered
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
        # Cleanup
        if trainer.core:
            trainer.core.session_active = False
        console.print("\n[green]Enhanced training session complete![/green]")
        console.print(f"[cyan]Metrics saved to: {trainer.metrics_file}[/cyan]")


if __name__ == "__main__":
    main()