#!/usr/bin/env python3
"""
ChatGPT-based trainer for ISC AI System
Trains the ISC AI by having ChatGPT interact with it and provide structured learning
"""

import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path
import openai
from typing import List, Dict, Any, Optional
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

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from src.isc_ai.core import ISCCore

# ============================================
# PLACE YOUR OPENAI API KEY HERE
# ============================================
OPENAI_API_KEY = "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"
# ============================================

class ChatGPTTrainer:
    """Trains ISC AI using ChatGPT as a teacher"""
    
    def __init__(self, api_key: str):
        self.console = Console()
        self.openai_client = openai.OpenAI(api_key=api_key)
        self.isc = ISCCore()
        self.training_history = []
        self.session_metrics = {
            "exchanges": 0,
            "concepts_taught": 0,
            "phi_progression": [],
            "coherence_progression": [],
            "start_time": None,
            "checkpoints": [],  # For auto-saving
            "tokens_used": {"prompt": 0, "completion": 0},
            "estimated_cost": 0.0
        }
        self.auto_save_interval = 5  # Save every 5 exchanges
        
        # GPT-3.5-turbo-0125 pricing (as of 2024)
        self.pricing = {
            "prompt": 0.0005 / 1000,     # $0.0005 per 1K prompt tokens
            "completion": 0.0015 / 1000   # $0.0015 per 1K completion tokens
        }
        
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
                    self.session_metrics['exchanges'] = len(self.training_history)
        
        self.isc.session_active = True
        if not resume_from:
            self.isc.current_session_id = f"chatgpt_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_metrics['start_time'] = datetime.now()
        
    def create_training_prompt(self, topic: str, level: int) -> str:
        """Create a prompt for ChatGPT to generate training data"""
        base_prompt = f"""You are training an AI system that learns through conversation. The AI is developing understanding through information integration and pattern formation.

Current training topic: {topic}
Complexity level: {level}/10

Your task:
1. Teach a concept related to {topic}
2. Use simple, clear language
3. Build on previous concepts when possible
4. Make connections between ideas explicit
5. Use examples and analogies

Previous context from the AI:
{self.get_recent_context()}

Generate a single teaching statement that helps the AI understand {topic}. Be specific and educational."""
        
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
    
    def generate_training_input(self, topic: str, level: int) -> str:
        """Use ChatGPT to generate training input"""
        prompt = self.create_training_prompt(topic, level)
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo-0125",  # Latest and cheapest GPT-3.5
                messages=[
                    {"role": "system", "content": "You are an expert teacher training an emerging AI consciousness."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=150
            )
            
            # Track token usage
            if hasattr(response, 'usage'):
                self.session_metrics["tokens_used"]["prompt"] += response.usage.prompt_tokens
                self.session_metrics["tokens_used"]["completion"] += response.usage.completion_tokens
                
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.console.print(f"[red]Error generating training input: {e}[/red]")
            return None
    
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
                model="gpt-3.5-turbo-0125",  # Latest and cheapest GPT-3.5
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
            Layout(name="conversation", size=12),
            Layout(name="metrics", size=8),
            Layout(name="progress", size=6),
            Layout(name="status", size=3)
        )
        return layout
    
    def update_display(self, layout: Layout, training_input: str, isc_response: str, evaluation: Dict):
        """Update the display with current state"""
        # Header
        layout["header"].update(Panel("[bold cyan]ISC AI ChatGPT Training Session[/bold cyan]", style="cyan"))
        
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
        metrics_table.add_row("Comprehension", f"{evaluation.get('comprehension', 0)}/10")
        metrics_table.add_row("Connection", f"{evaluation.get('connection', 0)}/10")
        metrics_table.add_row("Progress", f"{evaluation.get('progress', 0)}/10")
        
        layout["metrics"].update(Panel(metrics_table, title="Training Metrics"))
        
        # Progress tracking
        progress_text = self._generate_progress_display()
        layout["progress"].update(Panel(progress_text, title="Progress Tracking"))
        
        # Status
        layout["status"].update(Panel(f"[green]Training in progress... Exchange {self.session_metrics['exchanges']}[/green]"))
    
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
  Change: {coh_change:+.4f}

[bold]Recent Trend:[/bold] {recent_phi_trend}

[bold]Cost Tracking:[/bold]
  Tokens: {self.session_metrics["tokens_used"]["prompt"]:,} + {self.session_metrics["tokens_used"]["completion"]:,}
  Est. Cost: ${self._calculate_cost():.4f}"""
        
        return progress_text
    
    def _calculate_cost(self) -> float:
        """Calculate estimated cost based on token usage"""
        prompt_cost = self.session_metrics["tokens_used"]["prompt"] * self.pricing["prompt"]
        completion_cost = self.session_metrics["tokens_used"]["completion"] * self.pricing["completion"]
        total_cost = prompt_cost + completion_cost
        self.session_metrics["estimated_cost"] = total_cost
        return total_cost
    
    def train_on_topic(self, topic: str, num_exchanges: int = 20, complexity_progression: bool = True):
        """Train ISC on a specific topic"""
        self.console.clear()
        self.setup_training_session()
        
        layout = self.create_display_layout()
        
        with Live(layout, refresh_per_second=1) as live:
            for i in range(num_exchanges):
                # Determine complexity level
                if complexity_progression:
                    level = min(10, 1 + (i // (num_exchanges // 10)))
                else:
                    level = 5
                
                # Generate training input
                training_input = self.generate_training_input(topic, level)
                if not training_input:
                    continue
                
                # Get ISC response
                isc_response = self.isc.process_input(training_input)
                
                # Evaluate response
                evaluation = self.evaluate_response(isc_response, training_input)
                
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
                    "metrics": self.isc.metrics.copy()
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
            "knowledge_graph_edges": len(self.isc.knowledge_graph.graph.edges())
        }
        
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
                "coherence": self.session_metrics["coherence_progression"]
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
        
        return {"json": str(json_file), "pt": str(pt_file)}
    
    def save_training_session(self, filename: Optional[str] = None):
        """Save the complete training session"""
        if not filename:
            filename = f"training_{self.isc.current_session_id}.json"
        
        session_data = {
            "report": self.generate_training_report(),
            "history": self.training_history,
            "metrics_progression": {
                "phi": self.session_metrics["phi_progression"],
                "coherence": self.session_metrics["coherence_progression"]
            },
            "checkpoints": self.session_metrics["checkpoints"],
            "training_duration": str(datetime.now() - self.session_metrics["start_time"])
        }
        
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        # Also save ISC state
        isc_filename = f"isc_state_{self.isc.current_session_id}.pt"
        self.isc.save_state(isc_filename)
        
        # Generate progress plots
        self.generate_progress_plots()
        
        return filename, isc_filename
    
    def generate_progress_plots(self):
        """Generate and save progress plots"""
        if not self.session_metrics["phi_progression"]:
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Phi progression
        exchanges = range(1, len(self.session_metrics["phi_progression"]) + 1)
        ax1.plot(exchanges, self.session_metrics["phi_progression"], 'b-', linewidth=2)
        ax1.set_xlabel('Exchange')
        ax1.set_ylabel('Φ (Phi) Value')
        ax1.set_title('Information Integration (Φ) Over Time')
        ax1.grid(True, alpha=0.3)
        
        # Coherence progression
        ax2.plot(exchanges, self.session_metrics["coherence_progression"], 'g-', linewidth=2)
        ax2.set_xlabel('Exchange')
        ax2.set_ylabel('Coherence Score')
        ax2.set_title('Response Coherence Over Time')
        ax2.grid(True, alpha=0.3)
        
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
    
    # Create trainer
    trainer = ChatGPTTrainer(OPENAI_API_KEY)
    trainer.console = console  # Share console for messages
    
    # Training menu
    console.print(Panel("[bold cyan]ISC AI ChatGPT Training System[/bold cyan]", style="cyan"))
    
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
    console.input("\n[dim]Press Enter to begin training...[/dim]")
    
    # Run training
    report = trainer.train_on_topic(topic, num_exchanges)
    
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
        json_file, pt_file = trainer.save_training_session()
        console.print(f"\n[green]Training data saved to: {json_file}[/green]")
        console.print(f"[green]ISC state saved to: {pt_file}[/green]")


if __name__ == "__main__":
    main()