#!/usr/bin/env python3
"""
Enhanced ChatGPT-based trainer for ISC AI System
Trains the ISC AI by having ChatGPT interact with it and provide structured learning
Includes improvements for language modeling: tokenization, cross-entropy loss, learning rate scheduling, etc.
"""

import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path
import openai
from typing import List, Dict, Any, Optional, Tuple
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Deep learning imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    AutoTokenizer, 
    GPT2LMHeadModel, 
    GPT2Config,
    get_linear_schedule_with_warmup
)
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from isc.core import ISCCore

# ============================================
# PLACE YOUR OPENAI API KEY HERE
# ============================================
OPENAI_API_KEY = "YOUR-OPENAI-API-KEY-HERE"
# ============================================


class TextDataset(Dataset):
    """Dataset for text training data"""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        tokens = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        # For language modeling, input_ids and labels are the same
        input_ids = tokens["input_ids"].squeeze()
        attention_mask = tokens["attention_mask"].squeeze()
        
        # Shift for next-token prediction
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore padding in loss
        
        return {
            "input_ids": input_ids[:-1],
            "attention_mask": attention_mask[:-1],
            "labels": labels[1:]
        }


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


class ChatGPTTrainer:
    """Enhanced trainer for ISC AI using ChatGPT as a teacher with LM capabilities"""
    
    def __init__(self, api_key: str, use_language_model: bool = False):
        self.console = Console()
        self.openai_client = openai.OpenAI(api_key=api_key)
        self.isc = ISCCore()
        self.training_history = []
        self.use_language_model = use_language_model
        
        # Language model components
        if self.use_language_model:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.language_model = None  # Will be initialized based on user choice
            self.scaler = GradScaler() if torch.cuda.is_available() else None
            
        self.session_metrics = {
            "exchanges": 0,
            "concepts_taught": 0,
            "phi_progression": [],
            "coherence_progression": [],
            "perplexity_progression": [],
            "loss_progression": [],
            "start_time": None,
            "checkpoints": [],
            "tokens_used": {"prompt": 0, "completion": 0},
            "estimated_cost": 0.0,
            "validation_texts": []  # For perplexity evaluation
        }
        self.auto_save_interval = 5
        
        # Pricing for different models
        self.pricing = {
            "gpt-3.5-turbo-0125": {
                "prompt": 0.0005 / 1000,
                "completion": 0.0015 / 1000
            },
            "gpt-4-turbo-preview": {
                "prompt": 0.01 / 1000,
                "completion": 0.03 / 1000
            }
        }
        
        # Training configuration
        self.early_stopping = EarlyStopping(patience=5)
        self.best_perplexity = float('inf')
        
    def create_small_gpt2(self) -> GPT2LMHeadModel:
        """Create a smaller GPT-2 model for faster training"""
        config = GPT2Config(
            vocab_size=50257,
            n_positions=512,
            n_embd=384,  # Smaller than GPT-2
            n_layer=6,   # Fewer layers
            n_head=6,
            n_inner=1536,
            activation_function="gelu",
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1
        )
        return GPT2LMHeadModel(config).to(self.device)
    
    def compute_lm_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Standard cross-entropy loss for next-token prediction"""
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )
    
    def prepare_training_data(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize and prepare for next-token prediction"""
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Shift for next-token prediction
        input_ids = tokens["input_ids"].to(self.device)
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100  # Ignore last token in loss
        
        # Set padding tokens to -100 so they're ignored in loss
        labels[tokens["input_ids"] == self.tokenizer.pad_token_id] = -100
        
        return input_ids, labels
    
    def evaluate_perplexity(self, model: nn.Module, validation_texts: List[str]) -> float:
        """Calculate perplexity on validation set"""
        if not self.use_language_model or not validation_texts:
            return 0.0
            
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for text in validation_texts:
                input_ids, labels = self.prepare_training_data(text)
                
                if torch.cuda.is_available():
                    with autocast():
                        outputs = model(input_ids)
                        loss = self.compute_lm_loss(outputs.logits, labels)
                else:
                    outputs = model(input_ids)
                    loss = self.compute_lm_loss(outputs.logits, labels)
                
                # Count non-padding tokens
                valid_tokens = (labels != -100).sum().item()
                total_loss += loss.item() * valid_tokens
                total_tokens += valid_tokens
        
        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = np.exp(avg_loss)
        model.train()
        
        return perplexity
    
    def setup_training_session(self, resume_from: Optional[str] = None):
        """Initialize training session"""
        if resume_from:
            # Load previous state
            self.console.print(f"[yellow]Resuming from: {resume_from}[/yellow]")
            self.isc.load_state(resume_from)
            # Load training history if available
            json_file = resume_from.replace('.pt', '.json').replace('isc_state_', 'training_')
            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    session_data = json.load(f)
                    self.training_history = session_data.get('history', [])
                    if 'metrics_progression' in session_data:
                        self.session_metrics['phi_progression'] = session_data['metrics_progression']['phi']
                        self.session_metrics['coherence_progression'] = session_data['metrics_progression']['coherence']
                        self.session_metrics['perplexity_progression'] = session_data['metrics_progression'].get('perplexity', [])
                        self.session_metrics['loss_progression'] = session_data['metrics_progression'].get('loss', [])
                    self.session_metrics['exchanges'] = len(self.training_history)
        
        self.isc.session_active = True
        if not resume_from:
            self.isc.current_session_id = f"chatgpt_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_metrics['start_time'] = datetime.now()
        
        # Initialize language model if requested
        if self.use_language_model and self.language_model is None:
            self.console.print("[cyan]Select language model:[/cyan]")
            self.console.print("1. Small GPT-2 (fast training)")
            self.console.print("2. Full GPT-2 (better quality)")
            
            model_choice = self.console.input("[cyan]Enter choice (1-2):[/cyan] ")
            
            if model_choice == "1":
                self.console.print("[yellow]Creating small GPT-2 model...[/yellow]")
                self.language_model = self.create_small_gpt2()
            else:
                self.console.print("[yellow]Loading full GPT-2 model...[/yellow]")
                self.language_model = GPT2LMHeadModel.from_pretrained("gpt2").to(self.device)
            
            self.console.print(f"[green]Model loaded on {self.device}[/green]")
    
    def create_training_prompt(self, topic: str, level: int, use_gpt4: bool = False) -> str:
        """Create a prompt for ChatGPT to generate training data"""
        base_prompt = f"""You are training an AI system that learns through conversation. The AI is developing understanding through information integration and pattern formation.

Current training topic: {topic}
Complexity level: {level}/10

Your task:
1. Teach a concept related to {topic}
2. Use simple, clear language at first, then gradually increase complexity
3. Build on previous concepts when possible
4. Make connections between ideas explicit
5. Use examples and analogies
6. Generate diverse, high-quality training examples

Previous context from the AI:
{self.get_recent_context()}

Generate a single teaching statement that helps the AI understand {topic}. Be specific, educational, and ensure the content is diverse and engaging."""
        
        return base_prompt
    
    def get_recent_context(self) -> str:
        """Get recent ISC responses for context"""
        recent = self.isc.memory.get_recent_interactions(3)
        if not recent:
            return "No previous interactions yet."
        
        context = []
        for interaction in recent:
            context.append(f"Human: {interaction['input']}")
            context.append(f"AI: {interaction['response']}")
        
        return "\n".join(context)
    
    def generate_training_input(self, topic: str, level: int, use_gpt4: bool = False) -> str:
        """Use ChatGPT to generate training input"""
        prompt = self.create_training_prompt(topic, level, use_gpt4)
        
        model = "gpt-4-turbo-preview" if use_gpt4 else "gpt-3.5-turbo-0125"
        pricing_key = model
        
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert teacher training an emerging AI consciousness. Generate diverse, high-quality training examples."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.9 if use_gpt4 else 0.7,  # Higher temperature for diversity
                max_tokens=200 if use_gpt4 else 150
            )
            
            # Track token usage
            if hasattr(response, 'usage'):
                self.session_metrics["tokens_used"]["prompt"] += response.usage.prompt_tokens
                self.session_metrics["tokens_used"]["completion"] += response.usage.completion_tokens
                
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.console.print(f"[red]Error generating training input: {e}[/red]")
            return None
    
    def create_chunked_dataset(self, texts: List[str], chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """Create overlapping chunks for better context"""
        chunks = []
        for text in texts:
            tokens = self.tokenizer.encode(text)
            for i in range(0, len(tokens) - chunk_size, chunk_size - overlap):
                chunk = tokens[i:i + chunk_size]
                chunks.append(self.tokenizer.decode(chunk))
        return chunks
    
    def train_language_model_step(self, texts: List[str], optimizer, scheduler) -> float:
        """Train the language model for one step"""
        if not self.use_language_model or not texts:
            return 0.0
        
        # Create dataset and dataloader
        dataset = TextDataset(texts, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=min(8, len(texts)), shuffle=True)
        
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            if torch.cuda.is_available() and self.scaler:
                with autocast():
                    outputs = self.language_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                outputs = self.language_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()
            
            scheduler.step()
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def evaluate_response(self, isc_response: str, training_input: str) -> Dict[str, Any]:
        """Use ChatGPT to evaluate ISC's response quality"""
        eval_prompt = f"""Evaluate this AI's response for learning progress:

Training input: {training_input}
AI response: {isc_response}

Rate the following (0-10):
1. Comprehension: Did the AI understand the concept?
2. Connection: Did it make meaningful connections?
3. Progress: Is it showing learning improvement?

Respond in JSON format:
{{"comprehension": X, "connection": Y, "progress": Z, "feedback": "brief explanation"}}"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": "You are evaluating an AI's learning progress. Respond only in JSON."},
                    {"role": "user", "content": eval_prompt}
                ],
                temperature=0.3,
                max_tokens=100
            )
            
            # Track token usage
            if hasattr(response, 'usage'):
                self.session_metrics["tokens_used"]["prompt"] += response.usage.prompt_tokens
                self.session_metrics["tokens_used"]["completion"] += response.usage.completion_tokens
                
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            self.console.print(f"[red]Error evaluating response: {e}[/red]")
            return {"comprehension": 0, "connection": 0, "progress": 0, "feedback": "Error in evaluation"}
    
    def create_display_layout(self) -> Layout:
        """Create rich display layout"""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="conversation", size=10),
            Layout(name="metrics", size=10),
            Layout(name="progress", size=8),
            Layout(name="status", size=3)
        )
        return layout
    
    def update_display(self, layout: Layout, training_input: str, isc_response: str, evaluation: Dict):
        """Update the display with current state"""
        # Header
        header_text = "[bold cyan]ISC AI ChatGPT Training Session"
        if self.use_language_model:
            header_text += " (with Language Model)"
        header_text += "[/bold cyan]"
        layout["header"].update(Panel(header_text, style="cyan"))
        
        # Conversation
        conv_text = f"[bold yellow]ChatGPT Teacher:[/bold yellow]\n{training_input}\n\n"
        conv_text += f"[bold green]ISC AI:[/bold green]\n{isc_response}\n\n"
        conv_text += f"[bold magenta]Evaluation:[/bold magenta]\n{evaluation.get('feedback', 'N/A')}"
        layout["conversation"].update(Panel(conv_text, title="Training Conversation"))
        
        # Metrics
        metrics_table = Table(show_header=False)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="green")
        
        current_metrics = self.isc.get_status()["metrics"]
        metrics_table.add_row("Exchanges", str(self.session_metrics["exchanges"]))
        metrics_table.add_row("Concepts Taught", str(self.session_metrics["concepts_taught"]))
        metrics_table.add_row("Φ (Phi)", f"{current_metrics['phi_value']:.4f}")
        metrics_table.add_row("Coherence", f"{current_metrics['coherence_score']:.4f}")
        
        if self.use_language_model and self.session_metrics["perplexity_progression"]:
            metrics_table.add_row("Perplexity", f"{self.session_metrics['perplexity_progression'][-1]:.2f}")
            if self.session_metrics["loss_progression"]:
                metrics_table.add_row("LM Loss", f"{self.session_metrics['loss_progression'][-1]:.4f}")
        
        metrics_table.add_row("Comprehension", f"{evaluation.get('comprehension', 0)}/10")
        metrics_table.add_row("Connection", f"{evaluation.get('connection', 0)}/10")
        metrics_table.add_row("Progress", f"{evaluation.get('progress', 0)}/10")
        
        layout["metrics"].update(Panel(metrics_table, title="Training Metrics"))
        
        # Progress tracking
        progress_text = self._generate_progress_display()
        layout["progress"].update(Panel(progress_text, title="Progress Tracking"))
        
        # Status
        status_text = f"[green]Training in progress... Exchange {self.session_metrics['exchanges']}[/green]"
        if self.early_stopping.early_stop:
            status_text = "[yellow]Early stopping triggered[/yellow]"
        layout["status"].update(Panel(status_text))
    
    def _generate_progress_display(self) -> str:
        """Generate progress tracking display"""
        if len(self.session_metrics["phi_progression"]) < 2:
            return "[dim]Collecting initial data...[/dim]"
        
        # Calculate trends
        phi_values = self.session_metrics["phi_progression"]
        coherence_values = self.session_metrics["coherence_progression"]
        
        # Phi progress
        phi_start = phi_values[0] if phi_values else 0
        phi_current = phi_values[-1] if phi_values else 0
        phi_change = phi_current - phi_start
        phi_trend = "📈" if phi_change > 0 else "📉" if phi_change < 0 else "➡️"
        
        # Coherence progress
        coh_start = coherence_values[0] if coherence_values else 0
        coh_current = coherence_values[-1] if coherence_values else 0
        coh_change = coh_current - coh_start
        coh_trend = "📈" if coh_change > 0 else "📉" if coh_change < 0 else "➡️"
        
        # Recent trend (last 5 exchanges)
        recent_phi_trend = "stable"
        if len(phi_values) >= 5:
            recent_change = phi_values[-1] - phi_values[-5]
            if recent_change > 0.001:
                recent_phi_trend = "improving"
            elif recent_change < -0.001:
                recent_phi_trend = "declining"
        
        progress_text = f"""[bold]Φ Progress:[/bold] {phi_trend}
  Start: {phi_start:.4f}
  Current: {phi_current:.4f}
  Change: {phi_change:+.4f}

[bold]Coherence:[/bold] {coh_trend}
  Start: {coh_start:.4f}
  Current: {coh_current:.4f}
  Change: {coh_change:+.4f}"""
        
        # Add LM metrics if available
        if self.use_language_model and self.session_metrics["perplexity_progression"]:
            perp_values = self.session_metrics["perplexity_progression"]
            perp_current = perp_values[-1] if perp_values else 0
            perp_best = min(perp_values) if perp_values else 0
            
            progress_text += f"""

[bold]Language Model:[/bold]
  Current Perplexity: {perp_current:.2f}
  Best Perplexity: {perp_best:.2f}"""
        
        progress_text += f"""

[bold]Recent Trend:[/bold] {recent_phi_trend}

[bold]Cost Tracking:[/bold]
  Tokens: {self.session_metrics["tokens_used"]["prompt"]:,} + {self.session_metrics["tokens_used"]["completion"]:,}
  Est. Cost: ${self._calculate_cost():.4f}"""
        
        return progress_text
    
    def _calculate_cost(self) -> float:
        """Calculate estimated cost based on token usage"""
        # Assume GPT-3.5 pricing by default
        pricing = self.pricing.get("gpt-3.5-turbo-0125", {"prompt": 0.0005/1000, "completion": 0.0015/1000})
        prompt_cost = self.session_metrics["tokens_used"]["prompt"] * pricing["prompt"]
        completion_cost = self.session_metrics["tokens_used"]["completion"] * pricing["completion"]
        total_cost = prompt_cost + completion_cost
        self.session_metrics["estimated_cost"] = total_cost
        return total_cost
    
    def train_on_topic(self, topic: str, num_exchanges: int = 20, complexity_progression: bool = True, use_gpt4: bool = False):
        """Train ISC on a specific topic"""
        self.console.clear()
        self.setup_training_session()
        
        layout = self.create_display_layout()
        
        # Setup language model training if enabled
        optimizer = None
        scheduler = None
        if self.use_language_model:
            optimizer = torch.optim.AdamW(self.language_model.parameters(), lr=5e-5, weight_decay=0.01)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(num_exchanges * 0.1),
                num_training_steps=num_exchanges
            )
        
        # Collect texts for training
        training_texts = []
        
        with Live(layout, refresh_per_second=1) as live:
            for i in range(num_exchanges):
                # Check early stopping
                if self.early_stopping.early_stop:
                    self.console.print("[yellow]Early stopping triggered![/yellow]")
                    break
                
                # Determine complexity level
                if complexity_progression:
                    level = min(10, 1 + (i // (num_exchanges // 10)))
                else:
                    level = 5
                
                # Generate training input
                training_input = self.generate_training_input(topic, level, use_gpt4)
                if not training_input:
                    continue
                
                # Get ISC response
                isc_response = self.isc.process_input(training_input)
                
                # Evaluate response
                evaluation = self.evaluate_response(isc_response, training_input)
                
                # Collect training texts
                training_texts.append(training_input)
                training_texts.append(isc_response)
                
                # Train language model if enabled
                lm_loss = 0.0
                if self.use_language_model and len(training_texts) >= 10:
                    # Use recent texts for training
                    recent_texts = training_texts[-20:]
                    lm_loss = self.train_language_model_step(recent_texts, optimizer, scheduler)
                    self.session_metrics["loss_progression"].append(lm_loss)
                    
                    # Evaluate perplexity every 5 exchanges
                    if i % 5 == 0:
                        # Use some texts as validation
                        val_texts = training_texts[-10::2]  # Every other text from last 10
                        perplexity = self.evaluate_perplexity(self.language_model, val_texts)
                        self.session_metrics["perplexity_progression"].append(perplexity)
                        
                        # Check early stopping based on perplexity
                        if perplexity < self.best_perplexity:
                            self.best_perplexity = perplexity
                        self.early_stopping(perplexity)
                
                # Update metrics
                self.session_metrics["exchanges"] += 1
                self.session_metrics["concepts_taught"] += len(self.isc._extract_concepts(training_input))
                self.session_metrics["phi_progression"].append(self.isc.metrics["phi_value"])
                self.session_metrics["coherence_progression"].append(self.isc.metrics["coherence_score"])
                
                # Store in history
                self.training_history.append({
                    "exchange": i,
                    "input": training_input,
                    "response": isc_response,
                    "evaluation": evaluation,
                    "metrics": self.isc.metrics.copy(),
                    "lm_loss": lm_loss
                })
                
                # Update display
                self.update_display(layout, training_input, isc_response, evaluation)
                
                # Provide feedback to ISC based on evaluation
                if evaluation.get("progress", 0) > 7:
                    self.isc.provide_feedback("positive")
                elif evaluation.get("progress", 0) < 4:
                    self.isc.provide_feedback("negative")
                
                # Auto-save checkpoint
                if (i + 1) % self.auto_save_interval == 0:
                    checkpoint_files = self.save_checkpoint(i + 1)
                    self.session_metrics["checkpoints"].append({
                        "exchange": i + 1,
                        "files": checkpoint_files,
                        "metrics": self.isc.metrics.copy()
                    })
                    # Update status to show save
                    layout["status"].update(Panel(f"[green]Checkpoint saved at exchange {i + 1}[/green]"))
                
                # Pause between exchanges
                time.sleep(2)
        
        return self.generate_training_report()
    
    def generate_training_report(self) -> Dict[str, Any]:
        """Generate a comprehensive training report"""
        report = {
            "session_id": self.isc.current_session_id,
            "total_exchanges": self.session_metrics["exchanges"],
            "concepts_taught": self.session_metrics["concepts_taught"],
            "final_phi": self.isc.metrics["phi_value"],
            "final_coherence": self.isc.metrics["coherence_score"],
            "phi_improvement": self.session_metrics["phi_progression"][-1] - self.session_metrics["phi_progression"][0] if self.session_metrics["phi_progression"] else 0,
            "average_comprehension": sum(h["evaluation"].get("comprehension", 0) for h in self.training_history) / len(self.training_history) if self.training_history else 0,
            "average_connection": sum(h["evaluation"].get("connection", 0) for h in self.training_history) / len(self.training_history) if self.training_history else 0,
            "average_progress": sum(h["evaluation"].get("progress", 0) for h in self.training_history) / len(self.training_history) if self.training_history else 0,
            "knowledge_graph_nodes": len(self.isc.knowledge_graph.graph.nodes()),
            "knowledge_graph_edges": len(self.isc.knowledge_graph.graph.edges()),
            "language_model_used": self.use_language_model
        }
        
        if self.use_language_model and self.session_metrics["perplexity_progression"]:
            report["final_perplexity"] = self.session_metrics["perplexity_progression"][-1]
            report["best_perplexity"] = min(self.session_metrics["perplexity_progression"])
            if self.session_metrics["loss_progression"]:
                report["final_lm_loss"] = self.session_metrics["loss_progression"][-1]
                report["average_lm_loss"] = sum(self.session_metrics["loss_progression"]) / len(self.session_metrics["loss_progression"])
        
        return report
    
    def save_checkpoint(self, exchange_num: int) -> Dict[str, str]:
        """Save a checkpoint during training"""
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f"{self.isc.current_session_id}_checkpoint_{exchange_num}_{timestamp}"
        
        # Save training data
        json_file = checkpoint_dir / f"training_{base_name}.json"
        session_data = {
            "report": self.generate_training_report(),
            "history": self.training_history,
            "metrics_progression": {
                "phi": self.session_metrics["phi_progression"],
                "coherence": self.session_metrics["coherence_progression"],
                "perplexity": self.session_metrics["perplexity_progression"],
                "loss": self.session_metrics["loss_progression"]
            },
            "checkpoint_info": {
                "exchange": exchange_num,
                "timestamp": timestamp,
                "session_id": self.isc.current_session_id
            }
        }
        with open(json_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        # Save ISC state
        pt_file = checkpoint_dir / f"isc_state_{base_name}.pt"
        self.isc.save_state(str(pt_file))
        
        # Save language model if used
        files = {"json": str(json_file), "pt": str(pt_file)}
        if self.use_language_model and self.language_model is not None:
            lm_file = checkpoint_dir / f"language_model_{base_name}.pt"
            torch.save({
                'model_state_dict': self.language_model.state_dict(),
                'best_perplexity': self.best_perplexity,
                'early_stopping_state': {
                    'counter': self.early_stopping.counter,
                    'best_loss': self.early_stopping.best_loss
                }
            }, lm_file)
            files["language_model"] = str(lm_file)
        
        return files
    
    def save_training_session(self, filename: Optional[str] = None):
        """Save the complete training session"""
        if not filename:
            filename = f"training_{self.isc.current_session_id}.json"
        
        session_data = {
            "report": self.generate_training_report(),
            "history": self.training_history,
            "metrics_progression": {
                "phi": self.session_metrics["phi_progression"],
                "coherence": self.session_metrics["coherence_progression"],
                "perplexity": self.session_metrics["perplexity_progression"],
                "loss": self.session_metrics["loss_progression"]
            },
            "checkpoints": self.session_metrics["checkpoints"],
            "training_duration": str(datetime.now() - self.session_metrics["start_time"])
        }
        
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        # Also save ISC state
        isc_filename = f"isc_state_{self.isc.current_session_id}.pt"
        self.isc.save_state(isc_filename)
        
        # Save language model if used
        lm_filename = None
        if self.use_language_model and self.language_model is not None:
            lm_filename = f"language_model_{self.isc.current_session_id}.pt"
            torch.save({
                'model_state_dict': self.language_model.state_dict(),
                'tokenizer_name': 'gpt2',
                'best_perplexity': self.best_perplexity,
                'final_metrics': {
                    'perplexity': self.session_metrics["perplexity_progression"][-1] if self.session_metrics["perplexity_progression"] else None,
                    'loss': self.session_metrics["loss_progression"][-1] if self.session_metrics["loss_progression"] else None
                }
            }, lm_filename)
        
        # Generate progress plots
        self.generate_progress_plots()
        
        return filename, isc_filename, lm_filename
    
    def generate_progress_plots(self):
        """Generate and save progress plots"""
        if not self.session_metrics["phi_progression"]:
            return
        
        num_plots = 3 if self.use_language_model and self.session_metrics["perplexity_progression"] else 2
        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots))
        
        if num_plots == 2:
            axes = [axes[0], axes[1]]
        
        # Phi progression
        exchanges = range(1, len(self.session_metrics["phi_progression"]) + 1)
        axes[0].plot(exchanges, self.session_metrics["phi_progression"], 'b-', linewidth=2)
        axes[0].set_xlabel('Exchange')
        axes[0].set_ylabel('Φ (Phi) Value')
        axes[0].set_title('Information Integration (Φ) Over Time')
        axes[0].grid(True, alpha=0.3)
        
        # Coherence progression
        axes[1].plot(exchanges, self.session_metrics["coherence_progression"], 'g-', linewidth=2)
        axes[1].set_xlabel('Exchange')
        axes[1].set_ylabel('Coherence Score')
        axes[1].set_title('Response Coherence Over Time')
        axes[1].grid(True, alpha=0.3)
        
        # Perplexity if using language model
        if self.use_language_model and self.session_metrics["perplexity_progression"] and num_plots > 2:
            perp_exchanges = range(5, len(self.session_metrics["perplexity_progression"]) * 5 + 1, 5)
            axes[2].plot(perp_exchanges, self.session_metrics["perplexity_progression"], 'r-', linewidth=2)
            axes[2].set_xlabel('Exchange')
            axes[2].set_ylabel('Perplexity')
            axes[2].set_title('Language Model Perplexity Over Time')
            axes[2].grid(True, alpha=0.3)
            axes[2].invert_yaxis()  # Lower perplexity is better
        
        plt.tight_layout()
        plot_filename = f"progress_{self.isc.current_session_id}.png"
        plt.savefig(plot_filename, dpi=150)
        plt.close()
        
        self.console.print(f"[green]Progress plots saved to: {plot_filename}[/green]")


def main():
    """Main training interface"""
    console = Console()
    
    # Check API key
    if OPENAI_API_KEY == "YOUR-OPENAI-API-KEY-HERE":
        console.print("[red]Please set your OpenAI API key in the script![/red]")
        console.print("Edit the OPENAI_API_KEY variable at the top of this file.")
        return
    
    # Ask about language modeling
    console.print(Panel("[bold cyan]ISC AI Enhanced ChatGPT Training System[/bold cyan]", style="cyan"))
    console.print("\n[cyan]Enable language modeling features?[/cyan]")
    console.print("This includes tokenization, perplexity tracking, and GPT-2 fine-tuning.")
    console.print("Requires PyTorch and transformers libraries.")
    
    use_lm = console.input("[cyan]Enable language modeling? (y/n):[/cyan] ").lower() == 'y'
    
    # Create trainer
    trainer = ChatGPTTrainer(OPENAI_API_KEY, use_language_model=use_lm)
    trainer.console = console  # Share console for messages
    
    # Check for existing models to resume
    checkpoint_dir = Path("checkpoints")
    root_dir = Path(".")
    
    # Find all model files (both in checkpoints and root)
    all_models = []
    if checkpoint_dir.exists():
        all_models.extend(checkpoint_dir.glob("isc_state_*.pt"))
    all_models.extend(root_dir.glob("isc_state_*.pt"))
    
    # Remove duplicates and sort by modification time
    unique_models = list(set(all_models))
    existing_models = sorted(unique_models, key=lambda x: x.stat().st_mtime, reverse=True)
    
    resume_model = None
    if existing_models:
        console.print("\n[bold yellow]Found existing models:[/bold yellow]")
        
        # Display in a nice table
        model_table = Table(show_header=True, header_style="bold cyan")
        model_table.add_column("#", style="cyan", width=3)
        model_table.add_column("Model Name", style="yellow")
        model_table.add_column("Modified", style="green")
        model_table.add_column("Size", style="magenta")
        model_table.add_column("Location", style="dim")
        
        model_table.add_row("0", "Start new training", "-", "-", "-")
        
        for i, model in enumerate(existing_models[:10], 1):
            mtime = datetime.fromtimestamp(model.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            size_mb = model.stat().st_size / (1024 * 1024)
            location = "checkpoint" if "checkpoint" in str(model) else "root"
            model_table.add_row(
                str(i), 
                model.name[:40] + "..." if len(model.name) > 40 else model.name,
                mtime,
                f"{size_mb:.1f} MB",
                location
            )
        
        console.print(model_table)
        
        if len(existing_models) > 10:
            console.print(f"[dim]... and {len(existing_models) - 10} more models[/dim]")
        
        while True:
            resume_choice = console.input("\n[cyan]Select option (0-{}):[/cyan] ".format(min(10, len(existing_models))))
            if resume_choice.isdigit():
                choice_num = int(resume_choice)
                if choice_num == 0:
                    break
                elif 1 <= choice_num <= min(10, len(existing_models)):
                    resume_model = str(existing_models[choice_num - 1])
                    break
            console.print("[red]Invalid choice. Please try again.[/red]")
    
    trainer.setup_training_session(resume_from=resume_model)
    
    # Model selection for data generation
    console.print("\n[cyan]Select model for training data generation:[/cyan]")
    console.print("1. GPT-3.5-turbo (faster, cheaper)")
    console.print("2. GPT-4-turbo (better quality, more expensive)")
    
    model_choice = console.input("[cyan]Enter choice (1-2):[/cyan] ")
    use_gpt4 = (model_choice == "2")
    
    console.print("\nSelect training mode:")
    console.print("1. Basic Concepts (animals, objects, relationships)")
    console.print("2. Abstract Concepts (emotions, ideas, philosophy)")
    console.print("3. Technical Concepts (science, mathematics, logic)")
    console.print("4. Mixed Training (combination of all)")
    console.print("5. Custom Topic")
    
    choice = console.input("\n[cyan]Enter choice (1-5):[/cyan] ")
    
    topics = {
        "1": "basic concepts like animals, objects, colors, and simple relationships",
        "2": "abstract concepts like emotions, thoughts, consciousness, and philosophical ideas",
        "3": "technical concepts like mathematics, physics, logic, and scientific principles",
        "4": "mixed concepts spanning basic, abstract, and technical domains",
    }
    
    if choice in topics:
        topic = topics[choice]
    elif choice == "5":
        topic = console.input("[cyan]Enter custom topic:[/cyan] ")
    else:
        console.print("[red]Invalid choice[/red]")
        return
    
    num_exchanges = int(console.input("[cyan]Number of training exchanges (default 20):[/cyan] ") or "20")
    
    console.print(f"\n[green]Starting training on: {topic}[/green]")
    console.print(f"[green]Number of exchanges: {num_exchanges}[/green]")
    console.print(f"[green]Using model: {'GPT-4-turbo' if use_gpt4 else 'GPT-3.5-turbo'}[/green]")
    if use_lm:
        console.print(f"[green]Language modeling: Enabled[/green]")
    console.input("\n[dim]Press Enter to begin training...[/dim]")
    
    # Run training
    report = trainer.train_on_topic(topic, num_exchanges, use_gpt4=use_gpt4)
    
    # Show final report
    console.clear()
    console.print(Panel("[bold green]Training Complete![/bold green]", style="green"))
    
    report_table = Table(title="Training Report", show_header=False)
    report_table.add_column("Metric", style="cyan")
    report_table.add_column("Value", style="green")
    
    for key, value in report.items():
        if isinstance(value, float):
            report_table.add_row(key.replace("_", " ").title(), f"{value:.4f}")
        else:
            report_table.add_row(key.replace("_", " ").title(), str(value))
    
    console.print(report_table)
    
    # Show progress summary
    if trainer.session_metrics["phi_progression"]:
        console.print("\n[bold]Progress Summary:[/bold]")
        console.print(f"Initial Φ: {trainer.session_metrics['phi_progression'][0]:.4f}")
        console.print(f"Final Φ: {trainer.session_metrics['phi_progression'][-1]:.4f}")
        console.print(f"Φ Improvement: {report['phi_improvement']:.4f}")
        console.print(f"\nInitial Coherence: {trainer.session_metrics['coherence_progression'][0]:.4f}")
        console.print(f"Final Coherence: {trainer.session_metrics['coherence_progression'][-1]:.4f}")
        coherence_improvement = trainer.session_metrics['coherence_progression'][-1] - trainer.session_metrics['coherence_progression'][0]
        console.print(f"Coherence Improvement: {coherence_improvement:.4f}")
        
        if use_lm and trainer.session_metrics["perplexity_progression"]:
            console.print(f"\nBest Perplexity: {min(trainer.session_metrics['perplexity_progression']):.2f}")
            console.print(f"Final Perplexity: {trainer.session_metrics['perplexity_progression'][-1]:.2f}")
    
    console.print(f"\n[dim]Checkpoints saved: {len(trainer.session_metrics['checkpoints'])}[/dim]")
    
    # Show cost summary
    console.print(f"\n[bold yellow]Training Cost Summary:[/bold yellow]")
    console.print(f"Total Tokens: {trainer.session_metrics['tokens_used']['prompt'] + trainer.session_metrics['tokens_used']['completion']:,}")
    console.print(f"Estimated Cost: [green]${trainer._calculate_cost():.4f}[/green]")
    if trainer.session_metrics['exchanges'] > 0:
        console.print(f"Cost per Exchange: ${trainer._calculate_cost() / trainer.session_metrics['exchanges']:.4f}")
    
    # Save session
    save = console.input("\n[cyan]Save training session? (y/n):[/cyan] ")
    if save.lower() == 'y':
        files = trainer.save_training_session()
        console.print(f"\n[green]Training data saved to: {files[0]}[/green]")
        console.print(f"[green]ISC state saved to: {files[1]}[/green]")
        if len(files) > 2 and files[2]:
            console.print(f"[green]Language model saved to: {files[2]}[/green]")


if __name__ == "__main__":
    main()