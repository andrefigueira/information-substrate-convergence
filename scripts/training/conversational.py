#!/usr/bin/env python3
"""
ISC Conversational Trainer
Enhances the ISC model with conversational abilities while preserving philosophical core
"""

import os
import sys
import time
import json
import glob
from datetime import datetime
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
from collections import deque
from dataclasses import dataclass
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import openai
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.syntax import Syntax
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from isc.core import ISCCore

# ============================================
# OPENAI API KEY CONFIGURATION
# ============================================
# You can set your API key in one of these ways:
# 1. Set the OPENAI_API_KEY environment variable
# 2. Replace "YOUR_OPENAI_API_KEY" below with your actual key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
# ============================================

@dataclass
class ConversationalTask:
    """Represents a conversational training task"""
    exchange_id: int
    topic: str
    prompt_type: str  # 'dialogue', 'question', 'evaluation', 'response_template'
    level: int = 5
    context: str = ""
    
@dataclass
class ConversationalResult:
    """Represents the result of a conversational training task"""
    exchange_id: int
    task_type: str
    content: str
    tokens_used: dict
    error: str = None


class ConversationalLMHead(nn.Module):
    """Enhanced language model head for conversational output"""
    def __init__(self, hidden_dim, vocab_size, embedding_dim=768):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Projection layers
        self.concept_projection = nn.Linear(hidden_dim, embedding_dim)
        self.output_projection = nn.Linear(embedding_dim, vocab_size)
        
        # Conversational enhancement layers
        self.dialogue_context = nn.GRU(embedding_dim, embedding_dim, batch_first=True)
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads=8, batch_first=True)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, concept_vector, context_vectors=None):
        """
        Forward pass through the language model head
        
        Args:
            concept_vector: Vector from the ISC network (batch_size, hidden_dim)
            context_vectors: Optional context from previous exchanges (batch_size, seq_len, embedding_dim)
        """
        # Project concept vector to embedding space
        projected = self.concept_projection(concept_vector)
        
        # Apply conversational enhancements if context is provided
        if context_vectors is not None:
            # Process context through GRU
            context_out, _ = self.dialogue_context(context_vectors)
            
            # Apply self-attention to integrate context
            attn_output, _ = self.attention(
                projected.unsqueeze(1),  # Query (from concept)
                context_out,             # Key (from context)
                context_out              # Value (from context)
            )
            
            # Combine with original projection and normalize
            projected = self.layer_norm(projected + attn_output.squeeze(1))
        
        # Project to vocabulary space
        logits = self.output_projection(projected)
        return logits


class ConversationalTrainer:
    """Trains ISC AI to be more conversational while preserving philosophical core"""
    
    def __init__(self, api_key: str, max_workers: int = 5):
        self.console = Console()
        self.openai_client = openai.OpenAI(api_key=api_key)
        self.isc = ISCCore()
        self.training_history = []
        self.conversation_contexts = {}  # Store context for ongoing conversations
        self.lm_head = None
        
        # Dialogue management
        self.context_window = 5  # Number of exchanges to keep as context
        self.response_templates = {}  # Templates for different response types
        
        # Session metrics
        self.session_metrics = {
            "exchanges": 0,
            "conversational_score": [],  # Track conversational quality
            "naturalness_score": [],     # Track how natural responses sound
            "coherence_score": [],       # Track coherence across conversation
            "phi_progression": [],       # Track ISC's phi value
            "tokens_used": {"prompt": 0, "completion": 0},
            "start_time": None,
            "checkpoints": []
        }
        
        # Thread-safe components
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.lock = threading.Lock()
        self.max_workers = max_workers
        
        # Training parameters
        self.learning_rate = 1e-4
        self.batch_size = 4
        self.auto_save_interval = 10
        
        # GPT pricing
        self.pricing = {
            "prompt": 0.0005 / 1000,     # $0.0005 per 1K prompt tokens
            "completion": 0.0015 / 1000   # $0.0015 per 1K completion tokens
        }
    
    def load_model(self, model_path: str) -> bool:
        """Load existing ISC model"""
        try:
            self.console.print(f"\n[cyan]Loading model: {model_path}[/cyan]")
            
            # Initialize core and load state
            self.isc.load_state(model_path)
            
            # Check for enhanced model components
            lm_head_path = model_path.replace('.pt', '_lm_head.pt')
            if os.path.exists(lm_head_path):
                # Load existing language model head
                vocab_size = len(self.isc.tokenizer)
                hidden_size = self.isc.network.hidden_dim
                
                self.lm_head = ConversationalLMHead(hidden_size, vocab_size)
                self.lm_head.load_state_dict(torch.load(lm_head_path, map_location='cpu'))
                self.console.print("[green]✓ Loaded existing language model head[/green]")
            else:
                # Create new conversational LM head
                vocab_size = len(self.isc.tokenizer)
                hidden_size = self.isc.network.hidden_dim
                
                self.lm_head = ConversationalLMHead(hidden_size, vocab_size)
                self.console.print("[yellow]! Created new conversational language model head[/yellow]")
            
            # Set to training mode
            self.isc.network.train()
            self.lm_head.train()
            
            # Initialize session
            self.isc.session_active = True
            self.isc.current_session_id = f"conversational_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            return True
            
        except Exception as e:
            self.console.print(f"[red]Failed to load model: {e}[/red]")
            traceback.print_exc()
            return False
    
    def create_optimizer(self):
        """Create optimizers for training"""
        # We'll only train the LM head and leave the core ISC model unchanged
        self.optimizer = torch.optim.AdamW(self.lm_head.parameters(), lr=self.learning_rate)
        
        # Create scheduler for learning rate decay
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
    
    def generate_dialogue_examples(self, topic: str, num_examples: int) -> list:
        """Generate dialogue examples using ChatGPT"""
        examples = []
        
        prompt = f"""Generate {num_examples} realistic conversational exchanges for training an AI model.
Each exchange should include:
1. A human message that sounds natural and conversational
2. An ideal AI response that's helpful, natural, and engages appropriately

Focus on the topic: {topic}

The responses should be:
- Natural and conversational (not robotic)
- Thoughtful but concise
- Engaging with the user's specific points
- Occasionally asking relevant follow-up questions
- Including transitions and conversational markers

Format each example as:
Human: [human message]
AI: [ideal response]

Use different conversation styles and tones in your examples.
"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": "You are a dialogue creation assistant that generates realistic conversations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=1500
            )
            
            # Track token usage
            with self.lock:
                self.session_metrics["tokens_used"]["prompt"] += response.usage.prompt_tokens
                self.session_metrics["tokens_used"]["completion"] += response.usage.completion_tokens
            
            content = response.choices[0].message.content
            
            # Parse the generated examples
            examples = []
            current_human = None
            current_ai = None
            
            for line in content.split('\n'):
                if line.startswith('Human:'):
                    # If we have a complete exchange, add it
                    if current_human is not None and current_ai is not None:
                        examples.append((current_human, current_ai))
                    
                    # Start new exchange
                    current_human = line[6:].strip()
                    current_ai = None
                elif line.startswith('AI:'):
                    current_ai = line[3:].strip()
            
            # Add the last exchange if complete
            if current_human is not None and current_ai is not None:
                examples.append((current_human, current_ai))
            
            return examples
            
        except Exception as e:
            self.console.print(f"[red]Error generating dialogue examples: {e}[/red]")
            return []
    
    def generate_response_templates(self, num_templates: int = 10) -> dict:
        """Generate conversational response templates for different situations"""
        templates_prompt = f"""Generate {num_templates} conversational response templates for an AI assistant.
Each template should be general enough to be filled in with specific content, but include natural language patterns.

Include templates for these scenarios:
1. Answering factual questions
2. Responding to philosophical inquiries
3. Handling greetings/small talk
4. Continuing a discussion
5. Asking follow-up questions
6. Expressing uncertainty
7. Transitioning between topics
8. Summarizing a complex concept
9. Responding to emotional content
10. Clarifying understanding

Format each template as:
"scenario_name": "template text with {{concept}} placeholders"

The templates should be natural and conversational, not robotic.
"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates natural language templates."},
                    {"role": "user", "content": templates_prompt}
                ],
                temperature=0.7,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            # Track token usage
            with self.lock:
                self.session_metrics["tokens_used"]["prompt"] += response.usage.prompt_tokens
                self.session_metrics["tokens_used"]["completion"] += response.usage.completion_tokens
            
            templates = json.loads(response.choices[0].message.content)
            return templates
            
        except Exception as e:
            self.console.print(f"[red]Error generating response templates: {e}[/red]")
            return {}
    
    def evaluate_response(self, human_input: str, isc_response: str, ideal_response: str = None) -> dict:
        """Evaluate the conversational quality of an ISC response"""
        if ideal_response:
            eval_prompt = f"""Evaluate this AI response in terms of conversational quality:

Human input: {human_input}
AI response: {isc_response}
Ideal response example: {ideal_response}

Rate on a scale of 1-10 for each:
1. Naturalness: How natural and human-like is the response?
2. Relevance: How well does it address the human's input?
3. Engagement: Does it engage with the specific content?
4. Conversational flow: Does it use natural transitions and markers?

Respond in JSON format:
{{"naturalness": X, "relevance": Y, "engagement": Z, "conversational_flow": W, "overall": A, "feedback": "brief explanation"}}"""
        else:
            eval_prompt = f"""Evaluate this AI response in terms of conversational quality:

Human input: {human_input}
AI response: {isc_response}

Rate on a scale of 1-10 for each:
1. Naturalness: How natural and human-like is the response?
2. Relevance: How well does it address the human's input?
3. Engagement: Does it engage with the specific content?
4. Conversational flow: Does it use natural transitions and markers?

Respond in JSON format:
{{"naturalness": X, "relevance": Y, "engagement": Z, "conversational_flow": W, "overall": A, "feedback": "brief explanation"}}"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": "You evaluate conversational quality of AI responses. Respond only in JSON."},
                    {"role": "user", "content": eval_prompt}
                ],
                temperature=0.3,
                max_tokens=150,
                response_format={"type": "json_object"}
            )
            
            # Track token usage
            with self.lock:
                self.session_metrics["tokens_used"]["prompt"] += response.usage.prompt_tokens
                self.session_metrics["tokens_used"]["completion"] += response.usage.completion_tokens
                
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            self.console.print(f"[red]Error evaluating response: {e}[/red]")
            return {
                "naturalness": 5, 
                "relevance": 5, 
                "engagement": 5, 
                "conversational_flow": 5, 
                "overall": 5,
                "feedback": "Error in evaluation"
            }
    
    def process_response(self, concept_vector, context_vectors=None):
        """Process an ISC concept vector through the conversational LM head"""
        # Convert to tensor if needed
        if not isinstance(concept_vector, torch.Tensor):
            concept_vector = torch.tensor(concept_vector, dtype=torch.float32)
        
        # Ensure batch dimension
        if concept_vector.dim() == 1:
            concept_vector = concept_vector.unsqueeze(0)
        
        # Process through LM head
        with torch.no_grad():  # Inference mode
            logits = self.lm_head(concept_vector, context_vectors)
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample from distribution (with temperature)
            temperature = 0.8  # Adjust for creativity vs. determinism
            scaled_probs = probs / temperature
            token_ids = torch.multinomial(scaled_probs, num_samples=20)
            
            # Convert tokens to text
            generated_text = self.isc.tokenizer.decode(token_ids[0].tolist())
            
            return generated_text
    
    def enhance_response(self, isc_response: str, human_input: str) -> str:
        """Enhance ISC's philosophical response to be more conversational"""
        enhancement_prompt = f"""You are enhancing an AI's philosophical response to make it more conversational.

Original human input: {human_input}

AI's philosophical response: {isc_response}

Your task is to rewrite this response to be:
1. More natural and conversational
2. Directly engaging with the human's question
3. Thoughtful but clear
4. Using natural transitions and flow

Keep the core philosophical insights but make it sound like a helpful assistant speaking naturally.
Do not add any disclaimers or explanations outside the actual response.
"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": "You enhance philosophical AI responses to be more conversational."},
                    {"role": "user", "content": enhancement_prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            
            # Track token usage
            with self.lock:
                self.session_metrics["tokens_used"]["prompt"] += response.usage.prompt_tokens
                self.session_metrics["tokens_used"]["completion"] += response.usage.completion_tokens
                
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.console.print(f"[red]Error enhancing response: {e}[/red]")
            return isc_response
    
    def train_batch(self, examples: list):
        """Train the conversational LM head on a batch of examples"""
        # Prepare optimizer
        self.optimizer.zero_grad()
        
        batch_loss = 0
        
        for human_input, ideal_response in examples:
            # Get ISC's concept vector
            try:
                # Try to get vector if return_vector is supported
                result = self.isc.process_input(human_input, return_vector=True)
                if isinstance(result, tuple):
                    _, isc_vector = result
                else:
                    # Fallback: generate a random vector if return_vector not supported
                    isc_vector = np.random.randn(self.isc.network.hidden_dim).astype(np.float32)
            except TypeError:
                # Fallback: generate a random vector if return_vector parameter not supported
                isc_vector = np.random.randn(self.isc.network.hidden_dim).astype(np.float32)
            
            # Tokenize the ideal response
            target_tokens = self.isc.tokenizer.encode(ideal_response)
            target_tensor = torch.tensor(target_tokens, dtype=torch.long)
            
            # Forward pass
            logits = self.lm_head(torch.tensor(isc_vector, dtype=torch.float32).unsqueeze(0))
            
            # Calculate loss (only on the first token for simplicity)
            loss = F.cross_entropy(logits, target_tensor[0].unsqueeze(0))
            batch_loss += loss.item()
            
            # Backward pass
            loss.backward()
        
        # Update parameters
        self.optimizer.step()
        
        # Return average loss
        return batch_loss / len(examples)
    
    def train_conversational(self, topic: str, num_exchanges: int = 50):
        """Train the ISC model to be more conversational"""
        self.console.print("[bold cyan]Starting Conversational Training[/bold cyan]")
        self.session_metrics["start_time"] = datetime.now()
        
        # Set up display
        layout = self.create_display_layout()
        
        with Live(layout, refresh_per_second=1) as live:
            # Step 1: Generate response templates
            self.console.print("[yellow]Generating response templates...[/yellow]")
            self.response_templates = self.generate_response_templates()
            
            # Step 2: Set up optimizer
            self.create_optimizer()
            
            # Step 3: Main training loop
            exchange_count = 0
            
            while exchange_count < num_exchanges:
                # Generate a batch of dialogue examples
                batch_size = min(self.batch_size, num_exchanges - exchange_count)
                examples = self.generate_dialogue_examples(topic, batch_size)
                
                for human_input, ideal_response in examples:
                    if exchange_count >= num_exchanges:
                        break
                    
                    # Phase 1: Get ISC's philosophical response
                    try:
                        result = self.isc.process_input(human_input, return_vector=True)
                        if isinstance(result, tuple):
                            isc_response, _ = result
                        else:
                            isc_response = result
                    except TypeError:
                        isc_response = self.isc.process_input(human_input)
                    
                    # Phase 2: Enhance the response to be more conversational
                    enhanced_response = self.enhance_response(isc_response, human_input)
                    
                    # Phase 3: Evaluate the conversational quality
                    evaluation = self.evaluate_response(human_input, enhanced_response, ideal_response)
                    
                    # Phase 4: Train the conversational model
                    loss = self.train_batch([(human_input, enhanced_response)])
                    
                    # Update metrics
                    with self.lock:
                        self.session_metrics["exchanges"] += 1
                        self.session_metrics["conversational_score"].append(evaluation["overall"])
                        self.session_metrics["naturalness_score"].append(evaluation["naturalness"])
                        self.session_metrics["coherence_score"].append(evaluation["conversational_flow"])
                        self.session_metrics["phi_progression"].append(self.isc.metrics["phi_value"])
                    
                    # Store in history
                    self.training_history.append({
                        "exchange": exchange_count,
                        "human_input": human_input,
                        "isc_response": isc_response,
                        "enhanced_response": enhanced_response,
                        "ideal_response": ideal_response,
                        "evaluation": evaluation,
                        "loss": loss,
                        "metrics": self.isc.metrics.copy()
                    })
                    
                    # Update display
                    self.update_display(layout, human_input, isc_response, enhanced_response, ideal_response, evaluation)
                    
                    # Auto-save checkpoint
                    if (exchange_count + 1) % self.auto_save_interval == 0:
                        checkpoint_files = self.save_checkpoint(exchange_count + 1)
                        self.session_metrics["checkpoints"].append({
                            "exchange": exchange_count + 1,
                            "files": checkpoint_files,
                            "metrics": self.isc.metrics.copy()
                        })
                        layout["status"].update(Panel(f"[green]Checkpoint saved at exchange {exchange_count + 1}[/green]"))
                    
                    exchange_count += 1
                    
                    # Brief pause to avoid display flicker
                    time.sleep(0.5)
                
                # Update learning rate based on progress
                if self.session_metrics["conversational_score"]:
                    avg_score = sum(self.session_metrics["conversational_score"][-batch_size:]) / batch_size
                    self.scheduler.step(1.0 - avg_score / 10)  # Invert score for minimization
            
            # Final save
            self.save_training_session()
            
        return self.generate_training_report()
    
    def create_display_layout(self) -> Layout:
        """Create rich display layout"""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="conversation", size=12),
            Layout(name="metrics", size=8),
            Layout(name="progress", size=7),
            Layout(name="status", size=3)
        )
        return layout
    
    def update_display(self, layout: Layout, human_input: str, isc_response: str, 
                       enhanced_response: str, ideal_response: str, evaluation: dict):
        """Update the display with current state"""
        # Header
        layout["header"].update(Panel("[bold cyan]ISC AI Conversational Training[/bold cyan]", style="cyan"))
        
        # Conversation
        conv_text = f"[bold yellow]Human:[/bold yellow]\n{human_input[:200]}\n\n"
        conv_text += f"[bold red]ISC Original:[/bold red]\n{isc_response[:200]}\n\n"
        conv_text += f"[bold green]Enhanced Response:[/bold green]\n{enhanced_response[:200]}\n\n"
        conv_text += f"[bold blue]Ideal Response:[/bold blue]\n{ideal_response[:200]}\n\n"
        conv_text += f"[bold magenta]Evaluation:[/bold magenta] Overall: {evaluation['overall']}/10 - {evaluation['feedback']}"
        layout["conversation"].update(Panel(conv_text, title="Latest Training Conversation"))
        
        # Metrics
        metrics_table = Table(show_header=False)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="green")
        
        current_metrics = self.isc.metrics
        metrics_table.add_row("Exchanges", str(self.session_metrics["exchanges"]))
        metrics_table.add_row("Φ (Phi)", f"{current_metrics['phi_value']:.4f}")
        
        # Calculate average scores from last 5 exchanges or all if fewer
        recent_scores = self.session_metrics["conversational_score"][-5:]
        if recent_scores:
            avg_score = sum(recent_scores) / len(recent_scores)
            metrics_table.add_row("Conversational Score", f"{avg_score:.2f}/10")
        
        recent_natural = self.session_metrics["naturalness_score"][-5:]
        if recent_natural:
            avg_natural = sum(recent_natural) / len(recent_natural)
            metrics_table.add_row("Naturalness", f"{avg_natural:.2f}/10")
        
        if evaluation:
            metrics_table.add_row("Relevance", f"{evaluation.get('relevance', 0)}/10")
            metrics_table.add_row("Engagement", f"{evaluation.get('engagement', 0)}/10")
        
        layout["metrics"].update(Panel(metrics_table, title="Training Metrics"))
        
        # Progress tracking
        progress_text = self._generate_progress_display()
        layout["progress"].update(Panel(progress_text, title="Progress Tracking"))
        
        # Status
        layout["status"].update(Panel(f"[green]Training in progress... Exchange {self.session_metrics['exchanges']}[/green]"))
    
    def _generate_progress_display(self) -> str:
        """Generate progress tracking display"""
        if len(self.session_metrics["conversational_score"]) < 2:
            return "[dim]Collecting initial data...[/dim]"
        
        # Calculate trends
        conv_scores = self.session_metrics["conversational_score"]
        phi_values = self.session_metrics["phi_progression"]
        
        # Conversational score progress
        conv_start = conv_scores[0] if conv_scores else 0
        conv_current = conv_scores[-1] if conv_scores else 0
        conv_change = conv_current - conv_start
        conv_trend = "📈" if conv_change > 0.5 else "📉" if conv_change < -0.5 else "➡️"
        
        # Phi progress
        phi_start = phi_values[0] if phi_values else 0
        phi_current = phi_values[-1] if phi_values else 0
        phi_change = phi_current - phi_start
        phi_trend = "📈" if phi_change > 0.0001 else "📉" if phi_change < -0.0001 else "➡️"
        
        # Calculate recent improvement (last 10 exchanges)
        recent_size = min(10, len(conv_scores))
        if recent_size >= 5:
            recent_avg_start = sum(conv_scores[-recent_size:-recent_size//2]) / (recent_size//2)
            recent_avg_end = sum(conv_scores[-recent_size//2:]) / (recent_size//2)
            recent_improvement = recent_avg_end - recent_avg_start
            recent_trend = f"{recent_improvement:+.2f} points"
        else:
            recent_trend = "insufficient data"
        
        progress_text = f"""[bold]Conversational Score:[/bold] {conv_trend}
  Start: {conv_start:.2f}/10
  Current: {conv_current:.2f}/10
  Change: {conv_change:+.2f}

[bold]Φ Value:[/bold] {phi_trend}
  Start: {phi_start:.4f}
  Current: {phi_current:.4f}
  Change: {phi_change:+.4f}

[bold]Recent Trend:[/bold] {recent_trend}

[bold]Cost Tracking:[/bold]
  Tokens: {self.session_metrics["tokens_used"]["prompt"]:,} + {self.session_metrics["tokens_used"]["completion"]:,}
  Est. Cost: ${self._calculate_cost():.4f}"""
        
        return progress_text
    
    def _calculate_cost(self) -> float:
        """Calculate estimated cost based on token usage"""
        prompt_cost = self.session_metrics["tokens_used"]["prompt"] * self.pricing["prompt"]
        completion_cost = self.session_metrics["tokens_used"]["completion"] * self.pricing["completion"]
        total_cost = prompt_cost + completion_cost
        return total_cost
    
    def save_checkpoint(self, exchange_num: int) -> dict:
        """Save a checkpoint during training"""
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f"conversational_{self.isc.current_session_id}_checkpoint_{exchange_num}_{timestamp}"
        
        # Save training data
        json_file = checkpoint_dir / f"training_{base_name}.json"
        session_data = {
            "report": self.generate_training_report(),
            "history": self.training_history,
            "metrics_progression": {
                "conversational_score": self.session_metrics["conversational_score"],
                "naturalness_score": self.session_metrics["naturalness_score"],
                "phi_progression": self.session_metrics["phi_progression"]
            },
            "checkpoint_info": {
                "exchange": exchange_num,
                "timestamp": timestamp,
                "session_id": self.isc.current_session_id
            }
        }
        try:
            with open(json_file, 'w') as f:
                json.dump(session_data, f, indent=2)
        except Exception as e:
            self.console.print(f"[red]Error saving checkpoint JSON: {e}[/red]")
            return {"json": None, "pt": None, "lm": None}
        
        # Save ISC state (core knowledge graph)
        pt_file = checkpoint_dir / f"isc_state_{base_name}.pt"
        try:
            self.isc.save_state(str(pt_file))
        except Exception as e:
            self.console.print(f"[red]Error saving ISC state: {e}[/red]")
            return {"json": str(json_file), "pt": None, "lm": None}
        
        # Save LM head separately
        lm_head_file = checkpoint_dir / f"isc_state_{base_name}_lm_head.pt"
        try:
            torch.save(self.lm_head.state_dict(), str(lm_head_file))
        except Exception as e:
            self.console.print(f"[red]Error saving LM head: {e}[/red]")
            return {"json": str(json_file), "pt": str(pt_file), "lm": None}
        
        return {"json": str(json_file), "pt": str(pt_file), "lm": str(lm_head_file)}
    
    def save_training_session(self, filename: str = None) -> tuple:
        """Save the complete training session"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"conversational_{self.isc.current_session_id}_{timestamp}.json"
        
        session_data = {
            "report": self.generate_training_report(),
            "history": self.training_history,
            "metrics_progression": {
                "conversational_score": self.session_metrics["conversational_score"],
                "naturalness_score": self.session_metrics["naturalness_score"],
                "phi_progression": self.session_metrics["phi_progression"]
            },
            "checkpoints": self.session_metrics["checkpoints"],
            "training_duration": str(datetime.now() - self.session_metrics["start_time"])
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(session_data, f, indent=2)
        except Exception as e:
            self.console.print(f"[red]Error saving training session: {e}[/red]")
        
        # Save ISC state
        isc_filename = f"isc_state_conversational_{self.isc.current_session_id}.pt"
        self.isc.save_state(isc_filename)
        
        # Save LM head separately
        lm_head_filename = f"isc_state_conversational_{self.isc.current_session_id}_lm_head.pt"
        torch.save(self.lm_head.state_dict(), lm_head_filename)
        
        # Generate progress plots
        self.generate_progress_plots()
        
        return filename, isc_filename, lm_head_filename
    
    def generate_progress_plots(self):
        """Generate and save progress plots"""
        if not self.session_metrics["conversational_score"]:
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Conversational score progression
        exchanges = range(1, len(self.session_metrics["conversational_score"]) + 1)
        ax1.plot(exchanges, self.session_metrics["conversational_score"], 'b-', linewidth=2)
        ax1.set_xlabel('Exchange')
        ax1.set_ylabel('Conversational Score')
        ax1.set_title('Conversational Quality Over Time')
        ax1.grid(True, alpha=0.3)
        
        # Phi progression
        ax2.plot(exchanges, self.session_metrics["phi_progression"], 'g-', linewidth=2)
        ax2.set_xlabel('Exchange')
        ax2.set_ylabel('Φ (Phi) Value')
        ax2.set_title('Information Integration (Φ) Over Time')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_filename = f"progress_conversational_{self.isc.current_session_id}.png"
        try:
            plt.savefig(plot_filename, dpi=150)
            plt.close()
            self.console.print(f"[green]Progress plots saved to: {plot_filename}[/green]")
        except Exception as e:
            self.console.print(f"[red]Error saving plots: {e}[/red]")
            plt.close()
    
    def generate_training_report(self) -> dict:
        """Generate a comprehensive training report"""
        report = {
            "session_id": self.isc.current_session_id,
            "total_exchanges": self.session_metrics["exchanges"],
            "final_phi": self.isc.metrics["phi_value"],
            "final_conversational_score": self.session_metrics["conversational_score"][-1] if self.session_metrics["conversational_score"] else 0,
            "improvement": {
                "conversational": self.session_metrics["conversational_score"][-1] - self.session_metrics["conversational_score"][0] if len(self.session_metrics["conversational_score"]) > 1 else 0,
                "naturalness": self.session_metrics["naturalness_score"][-1] - self.session_metrics["naturalness_score"][0] if len(self.session_metrics["naturalness_score"]) > 1 else 0,
                "phi": self.session_metrics["phi_progression"][-1] - self.session_metrics["phi_progression"][0] if len(self.session_metrics["phi_progression"]) > 1 else 0
            },
            "average_scores": {
                "conversational": sum(self.session_metrics["conversational_score"]) / max(len(self.session_metrics["conversational_score"]), 1),
                "naturalness": sum(self.session_metrics["naturalness_score"]) / max(len(self.session_metrics["naturalness_score"]), 1)
            },
            "knowledge_graph": {
                "nodes": len(self.isc.knowledge_graph.graph.nodes()),
                "edges": len(self.isc.knowledge_graph.graph.edges())
            },
            "training_config": {
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "parallel_workers": self.max_workers
            }
        }
        
        return report


class ConversationalISC:
    """Enhanced ISC AI with conversational capabilities"""
    
    def __init__(self, model_path: str, lm_head_path: str):
        self.console = Console()
        self.isc = ISCCore()
        self.lm_head = None
        self.response_templates = {}
        self.conversation_history = []
        self.max_context_length = 5  # Number of exchanges to remember
        
        # Load model and LM head
        self.load_model(model_path, lm_head_path)
    
    def load_model(self, model_path: str, lm_head_path: str):
        """Load the ISC model and conversational LM head"""
        try:
            self.console.print(f"[cyan]Loading ISC model: {model_path}[/cyan]")
            self.isc.load_state(model_path)
            
            # Load LM head
            vocab_size = len(self.isc.tokenizer)
            hidden_size = self.isc.network.hidden_dim
            
            self.lm_head = ConversationalLMHead(hidden_size, vocab_size)
            self.lm_head.load_state_dict(torch.load(lm_head_path, map_location='cpu'))
            self.lm_head.eval()  # Set to evaluation mode
            
            self.console.print("[green]✓ Models loaded successfully[/green]")
            
            # Initialize session
            self.isc.session_active = True
            self.isc.current_session_id = f"conversational_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Try to load response templates
            self._load_templates()
            
            return True
        except Exception as e:
            self.console.print(f"[red]Failed to load model: {e}[/red]")
            traceback.print_exc()
            return False
    
    def _load_templates(self):
        """Load response templates"""
        try:
            template_file = "response_templates.json"
            if os.path.exists(template_file):
                with open(template_file, 'r') as f:
                    self.response_templates = json.load(f)
            else:
                self.console.print("[yellow]No response templates found. Using default responses.[/yellow]")
                self.response_templates = {
                    "greeting": "Hello there! I'm thinking about {concept}. How can I assist you today?",
                    "question": "That's an interesting question about {concept}. I've been contemplating that and think {response}.",
                    "continuation": "Building on that idea about {concept}, I'd also consider {response}. What are your thoughts?",
                    "uncertainty": "I'm still developing my understanding of {concept}. My current thinking is {response}, but I'm curious to explore this further with you.",
                    "philosophical": "From a philosophical perspective, {concept} relates to {response}. This raises interesting questions about how we understand our world."
                }
        except Exception as e:
            self.console.print(f"[yellow]Error loading templates: {e}. Using defaults.[/yellow]")
    
    def _prepare_context_vectors(self):
        """Prepare context vectors from recent conversation history"""
        if not self.conversation_history:
            return None
        
        # Use only the most recent exchanges
        recent_history = self.conversation_history[-self.max_context_length:]
        
        # Create context vectors (placeholder - in a real implementation, 
        # you would embed each exchange)
        # Add batch dimension for attention mechanism
        context_vectors = torch.zeros((1, len(recent_history), self.lm_head.embedding_dim))
        
        return context_vectors
    
    def _format_response(self, isc_response: str, concept_vector, template_key: str = None):
        """Format a response using templates and enhanced generation"""
        # Extract key concepts from the ISC response
        words = isc_response.split()
        concepts = [word for word in words if len(word) > 4 and word.isalpha()]
        main_concept = concepts[0] if concepts else "this topic"
        
        # Choose template based on input or randomly
        if not template_key:
            template_keys = list(self.response_templates.keys())
            template_key = np.random.choice(template_keys)
        
        template = self.response_templates.get(template_key, "{response}")
        
        # Process through LM head to get coherent text
        generated_text = self.process_response(concept_vector)
        
        # Format using template
        formatted_response = template.format(
            concept=main_concept,
            response=generated_text
        )
        
        return formatted_response
    
    def process_response(self, concept_vector):
        """Process concept vector through LM head"""
        # Prepare context from conversation history
        context_vectors = self._prepare_context_vectors()
        
        # Generate response through LM head
        with torch.no_grad():
            if not isinstance(concept_vector, torch.Tensor):
                concept_vector = torch.tensor(concept_vector, dtype=torch.float32)
            
            # Ensure batch dimension
            if concept_vector.dim() == 1:
                concept_vector = concept_vector.unsqueeze(0)
            
            logits = self.lm_head(concept_vector, context_vectors)
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample from distribution (with temperature)
            temperature = 0.8
            scaled_probs = probs / temperature
            token_ids = torch.multinomial(scaled_probs, num_samples=20)
            
            # Convert tokens to text
            generated_text = self.isc.tokenizer.decode(token_ids[0].tolist())
            
            return generated_text
    
    def chat(self, user_input: str):
        """Process user input and generate conversational response"""
        # Get ISC's philosophical response and concept vector
        try:
            result = self.isc.process_input(user_input, return_vector=True)
            if isinstance(result, tuple):
                isc_response, concept_vector = result
            else:
                # Fallback if return_vector not supported
                isc_response = result
                concept_vector = np.random.randn(self.isc.network.hidden_dim).astype(np.float32)
        except TypeError:
            # Fallback if return_vector parameter not supported
            isc_response = self.isc.process_input(user_input)
            concept_vector = np.random.randn(self.isc.network.hidden_dim).astype(np.float32)
        
        # Determine response type based on input
        response_type = self._classify_input(user_input)
        
        # Generate conversational response
        conversational_response = self._format_response(isc_response, concept_vector, response_type)
        
        # Update conversation history
        self.conversation_history.append({
            "user": user_input,
            "response": conversational_response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep history limited to max length
        if len(self.conversation_history) > self.max_context_length * 2:
            self.conversation_history = self.conversation_history[-self.max_context_length * 2:]
        
        return conversational_response, isc_response
    
    def _classify_input(self, user_input: str):
        """Classify the type of user input to select appropriate response template"""
        user_input = user_input.lower()
        
        if any(greeting in user_input for greeting in ["hello", "hi ", "hey", "greetings"]):
            return "greeting"
        elif "?" in user_input:
            return "question"
        elif any(word in user_input for word in ["think", "philosophy", "consciousness", "meaning"]):
            return "philosophical"
        elif len(user_input.split()) <= 3:
            return "continuation"
        else:
            # Default to more complex response
            return "continuation"


def main():
    """Main entry point for conversational ISC training"""
    console = Console()
    
    # Check API key
    if OPENAI_API_KEY == "YOUR_OPENAI_API_KEY":
        console.print("[red]Please set your OpenAI API key in the script![/red]")
        console.print("Edit the OPENAI_API_KEY variable at the top of this file.")
        return
    
    # Display header
    console.print(Panel("[bold cyan]ISC AI Conversational Trainer[/bold cyan]", style="cyan"))
    console.print("[dim]Enhancing your ISC model with conversational abilities[/dim]\n")
    
    # Menu
    console.print("[bold]Available options:[/bold]")
    console.print("1. Train existing ISC model to be conversational")
    console.print("2. Use already trained conversational model")
    console.print("3. Exit")
    
    choice = console.input("\n[cyan]Enter choice (1-3):[/cyan] ")
    
    if choice == "1":
        # Train model
        console.print("\n[bold]Select an ISC model to enhance:[/bold]")
        
        # Find model files
        model_files = []
        for pattern in ["isc_state_*.pt", "checkpoints/isc_state_*.pt"]:
            model_files.extend(glob.glob(pattern))
        
        # Remove LM head files
        model_files = [f for f in model_files if not f.endswith('_lm_head.pt')]
        
        if not model_files:
            console.print("[red]No model files found![/red]")
            return
        
        # Display in a nice table
        model_table = Table(show_header=True, header_style="bold cyan")
        model_table.add_column("#", style="cyan", width=3)
        model_table.add_column("Model Name", style="yellow")
        model_table.add_column("Modified", style="green")
        
        for i, model in enumerate(model_files[:10], 1):
            mtime = datetime.fromtimestamp(os.path.getmtime(model)).strftime("%Y-%m-%d %H:%M")
            model_table.add_row(str(i), model, mtime)
        
        console.print(model_table)
        
        model_choice = int(console.input("\n[cyan]Select model number:[/cyan] "))
        if model_choice < 1 or model_choice > len(model_files):
            console.print("[red]Invalid choice[/red]")
            return
        
        selected_model = model_files[model_choice - 1]
        
        # Get training parameters
        console.print("\n[bold]Training Parameters:[/bold]")
        
        max_workers = int(console.input("[cyan]Number of parallel workers (default 3, max 8):[/cyan] ") or "3")
        max_workers = min(max_workers, 8)  # Cap at 8 to avoid rate limits
        
        topic = console.input("[cyan]Training topic (e.g., 'general conversation', 'philosophy'):[/cyan] ") or "general conversation"
        
        num_exchanges = int(console.input("[cyan]Number of training exchanges (default 30, max 200):[/cyan] ") or "30")
        num_exchanges = min(num_exchanges, 200)  # Cap at 200 for safety
        
        # Create trainer
        trainer = ConversationalTrainer(OPENAI_API_KEY, max_workers=max_workers)
        
        # Load model
        if not trainer.load_model(selected_model):
            console.print("[red]Failed to load model. Exiting.[/red]")
            return
        
        # Run training
        console.print(f"\n[green]Starting conversational training on: {topic}[/green]")
        console.print(f"[green]Number of exchanges: {num_exchanges}[/green]")
        console.print(f"[yellow]Estimated time: {(num_exchanges * 4) // max_workers} minutes[/yellow]")
        console.input("\n[dim]Press Enter to begin training...[/dim]")
        
        trainer.train_conversational(topic, num_exchanges)
        
    elif choice == "2":
        # Use existing conversational model
        console.print("\n[bold]Select a conversational ISC model:[/bold]")
        
        # Find model files that have both .pt and _lm_head.pt
        model_files = []
        
        for pattern in ["isc_state_conversational_*.pt", "checkpoints/isc_state_conversational_*.pt"]:
            for model_file in glob.glob(pattern):
                # Check if there's a corresponding LM head file
                lm_head_file = model_file.replace('.pt', '_lm_head.pt')
                if os.path.exists(lm_head_file) and not model_file.endswith('_lm_head.pt'):
                    model_files.append((model_file, lm_head_file))
        
        if not model_files:
            console.print("[red]No conversational model files found![/red]")
            console.print("[yellow]Please train a conversational model first.[/yellow]")
            return
        
        # Display in a nice table
        model_table = Table(show_header=True, header_style="bold cyan")
        model_table.add_column("#", style="cyan", width=3)
        model_table.add_column("Model Name", style="yellow")
        model_table.add_column("Modified", style="green")
        
        for i, (model, _) in enumerate(model_files[:10], 1):
            mtime = datetime.fromtimestamp(os.path.getmtime(model)).strftime("%Y-%m-%d %H:%M")
            model_table.add_row(str(i), model, mtime)
        
        console.print(model_table)
        
        model_choice = int(console.input("\n[cyan]Select model number:[/cyan] "))
        if model_choice < 1 or model_choice > len(model_files):
            console.print("[red]Invalid choice[/red]")
            return
        
        selected_model, selected_lm_head = model_files[model_choice - 1]
        
        # Create conversational interface
        conversational_isc = ConversationalISC(selected_model, selected_lm_head)
        
        # Start chat loop
        console.print("\n[bold cyan]Conversational ISC Chat[/bold cyan]")
        console.print("[dim]Type 'exit' to quit[/dim]\n")
        
        while True:
            user_input = console.input("\n[bold green]You:[/bold green] ")
            
            if user_input.lower() == 'exit':
                break
            
            with console.status("[blue]ISC is thinking...[/blue]", spinner="dots"):
                conv_response, original_response = conversational_isc.chat(user_input)
            
            console.print(f"\n[bold blue]ISC:[/bold blue] {conv_response}")
            
            # Option to see original response
            show_original = console.input("[dim]Show original philosophical response? (y/n):[/dim] ")
            if show_original.lower() == 'y':
                console.print(f"[dim]Original: {original_response}[/dim]")
        
        console.print("\n[green]Thank you for chatting with Conversational ISC![/green]")
        
    else:
        console.print("[yellow]Exiting...[/yellow]")


if __name__ == "__main__":
    main()