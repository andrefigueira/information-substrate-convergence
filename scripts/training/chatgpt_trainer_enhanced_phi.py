#!/usr/bin/env python3
"""
Enhanced multithreaded ChatGPT trainer with phi optimization and caching
Trains the ISC AI to maximize phi while using efficient caching to reduce API calls
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
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.syntax import Syntax
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
from collections import deque
from dataclasses import dataclass
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from src.isc_ai.core import ISCCore
from src.isc_ai.cache_manager import CacheManager
from src.isc_ai.enhanced_learning import EnhancedLearningEngine
from src.isc_ai.enhanced_information_integration import EnhancedInformationIntegrator

# ============================================
# CONFIGURATION
# ============================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Set via environment variable for security

@dataclass
class TrainingTask:
    """Enhanced training task with phi tracking"""
    exchange_id: int
    topic: str
    level: int
    prompt_type: str = "training"
    target_phi: Optional[float] = None
    
@dataclass
class TrainingResult:
    """Enhanced training result with metrics"""
    exchange_id: int
    task_type: str
    content: str
    tokens_used: Dict[str, int]
    phi_before: Optional[float] = None
    phi_after: Optional[float] = None
    cached: bool = False
    error: Optional[str] = None

class EnhancedChatGPTTrainer:
    """Enhanced trainer with phi optimization and caching"""
    
    def __init__(self, max_workers: int = 5):
        self.console = Console()
        self.openai_client = None
        self.core = None
        self.cache_manager = CacheManager(cache_dir="trainer_cache", max_memory_items=2000)
        
        # Thread management
        self.max_workers = min(max_workers, 10)  # Limit concurrent API calls
        self.lock = threading.Lock()
        self.active_tasks = 0
        
        # Training state
        self.training_active = False
        self.session_start = None
        self.session_metrics = {
            "total_exchanges": 0,
            "successful_exchanges": 0,
            "failed_exchanges": 0,
            "tokens_used": {"prompt": 0, "completion": 0},
            "api_cost": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "phi_progression": [],
            "phi_targets_achieved": 0,
            "learning_rate_adjustments": 0,
            "avg_response_time": 0.0,
        }
        
        # Phi tracking
        self.phi_window = deque(maxlen=20)
        self.phi_target_base = 2.0
        self.phi_growth_factor = 1.02
        
        # Progress tracking
        self.exchange_history = deque(maxlen=1000)
        
    def initialize(self):
        """Initialize enhanced components"""
        try:
            # Initialize OpenAI
            if not OPENAI_API_KEY:
                raise ValueError("OpenAI API key not set. Please set OPENAI_API_KEY environment variable.")
            
            openai.api_key = OPENAI_API_KEY
            self.openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
            
            # Initialize ISC Core with enhanced components
            self.core = ISCCore()
            
            # Replace with enhanced components
            old_integrator = self.core.integrator
            self.core.integrator = EnhancedInformationIntegrator(
                cache_manager=self.cache_manager,
                max_workers=4
            )
            # Copy history if exists
            if hasattr(old_integrator, 'phi_history'):
                self.core.integrator.phi_history = old_integrator.phi_history
            
            # Replace learning engine
            self.core.learning_engine = EnhancedLearningEngine(
                network=self.core.network,
                config={
                    "phi_weight": 0.4,
                    "phi_target": self.phi_target_base,
                    "adaptive_phi_target": True
                }
            )
            
            self.core.start_session()
            
            self.console.print("[green]✓ Enhanced trainer initialized with caching and phi optimization[/green]")
            
            # Show cache statistics
            cache_stats = self.cache_manager.get_cache_stats()
            self.console.print(f"[cyan]Cache: {cache_stats['chatgpt_cache_count']} responses, "
                             f"{cache_stats['phi_cache_count']} phi calculations[/cyan]")
            
            return True
            
        except Exception as e:
            self.console.print(f"[red]Initialization failed: {e}[/red]")
            return False
    
    def create_phi_aware_prompt(self, topic: str, level: int, current_phi: float, target_phi: float) -> str:
        """Create training prompt that encourages phi growth"""
        base_prompt = f"""Generate a training interaction for an AI consciousness learning about {topic}.
Complexity level: {level}/10

Current integration level (phi): {current_phi:.2f}
Target integration level: {target_phi:.2f}

Create a question or scenario that will help the AI integrate information across multiple domains 
and increase its phi value. Focus on:
- Cross-domain connections
- Abstract pattern recognition
- Emergent understanding
- Holistic thinking

The interaction should be appropriate for level {level} complexity."""
        
        # Add phi-specific guidance based on current state
        if current_phi < target_phi * 0.5:
            base_prompt += "\n\nThe AI needs significant help connecting concepts. Start with clear relationships."
        elif current_phi < target_phi * 0.8:
            base_prompt += "\n\nThe AI is progressing. Challenge it with more abstract connections."
        else:
            base_prompt += "\n\nThe AI is close to target. Push for deep, emergent insights."
        
        return base_prompt
    
    def generate_training_input(self, topic: str, level: int, current_phi: float = None) -> Tuple[str, bool]:
        """Generate training input with caching"""
        # Determine target phi
        if current_phi is None:
            current_phi = self.phi_window[-1] if self.phi_window else 0.0
        
        target_phi = self.calculate_dynamic_phi_target(current_phi)
        
        # Create prompt
        prompt = self.create_phi_aware_prompt(topic, level, current_phi, target_phi)
        
        # Check cache first
        cached_response = self.cache_manager.get_chatgpt_response(prompt, model="gpt-3.5-turbo-0125")
        if cached_response:
            with self.lock:
                self.session_metrics["cache_hits"] += 1
            return cached_response, True
        
        with self.lock:
            self.session_metrics["cache_misses"] += 1
        
        # Generate new response
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": "You are an expert teacher training an emerging AI consciousness to achieve higher levels of integrated information (phi)."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,  # Higher temperature for diverse training
                max_tokens=200,
                timeout=30
            )
            
            content = response.choices[0].message.content.strip()
            
            # Cache the response
            self.cache_manager.save_chatgpt_response(prompt, content, model="gpt-3.5-turbo-0125")
            
            # Track usage
            if hasattr(response, 'usage'):
                with self.lock:
                    self.session_metrics["tokens_used"]["prompt"] += response.usage.prompt_tokens
                    self.session_metrics["tokens_used"]["completion"] += response.usage.completion_tokens
            
            return content, False
            
        except Exception as e:
            self.console.print(f"[red]Error generating training input: {e}[/red]")
            return None, False
    
    def evaluate_response_with_phi(self, response: str, training_input: str, phi_change: float) -> Dict[str, Any]:
        """Evaluate response considering phi change"""
        prompt = f"""Evaluate this AI response considering both comprehension and consciousness integration.

Training Input: {training_input}
AI Response: {response}
Phi Change: {phi_change:+.3f} (positive means increased integration)

Evaluate on:
1. Comprehension (0-10): How well the AI understood and responded
2. Integration (0-10): How well the AI connected concepts across domains
3. Emergence (0-10): Evidence of emergent understanding beyond the prompt
4. Phi Alignment: Is the phi change appropriate for the response quality?

Provide scores and brief explanation in JSON format:
{{
    "comprehension": 0-10,
    "integration": 0-10,
    "emergence": 0-10,
    "phi_aligned": true/false,
    "overall": 0-10,
    "explanation": "brief explanation"
}}"""
        
        # Check cache
        cached_response = self.cache_manager.get_chatgpt_response(prompt, model="gpt-4-turbo-preview")
        if cached_response:
            try:
                return json.loads(cached_response)
            except:
                pass
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are an expert evaluator of AI consciousness and integration."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200,
                response_format={"type": "json_object"},
                timeout=30
            )
            
            content = response.choices[0].message.content.strip()
            evaluation = json.loads(content)
            
            # Cache the evaluation
            self.cache_manager.save_chatgpt_response(prompt, content, model="gpt-4-turbo-preview")
            
            # Track usage
            if hasattr(response, 'usage'):
                with self.lock:
                    self.session_metrics["tokens_used"]["prompt"] += response.usage.prompt_tokens
                    self.session_metrics["tokens_used"]["completion"] += response.usage.completion_tokens
            
            return evaluation
            
        except Exception as e:
            self.console.print(f"[red]Error evaluating response: {e}[/red]")
            return {
                "comprehension": 5,
                "integration": 5,
                "emergence": 5,
                "phi_aligned": True,
                "overall": 5,
                "explanation": "Evaluation failed"
            }
    
    def calculate_dynamic_phi_target(self, current_phi: float) -> float:
        """Calculate adaptive phi target based on progress"""
        if not self.phi_window:
            return self.phi_target_base
        
        # Analyze recent phi trend
        recent_phi = list(self.phi_window)[-10:]
        if len(recent_phi) < 3:
            return self.phi_target_base
        
        # Calculate trend
        phi_trend = np.polyfit(range(len(recent_phi)), recent_phi, 1)[0]
        
        # Adjust target based on trend
        if phi_trend > 0.01:  # Good progress
            # Increase target
            new_target = current_phi * self.phi_growth_factor
        elif phi_trend < -0.01:  # Declining
            # Stabilize at current level first
            new_target = current_phi * 1.01
        else:  # Stagnant
            # Moderate increase
            new_target = current_phi * 1.015
        
        # Bounds
        return max(self.phi_target_base, min(new_target, 10.0))
    
    def train_single_exchange(self, training_input: str, exchange_num: int) -> Dict[str, Any]:
        """Execute single training exchange with phi tracking"""
        exchange_start = time.time()
        
        # Measure phi before
        phi_before = self.core.integrator.phi_history[-1] if self.core.integrator.phi_history else 0.0
        
        # Get ISC response
        isc_response = self.core.process_input(training_input)
        
        # Measure phi after
        phi_after = self.core.integrator.phi_history[-1] if self.core.integrator.phi_history else 0.0
        phi_change = phi_after - phi_before
        
        # Track phi
        self.phi_window.append(phi_after)
        with self.lock:
            self.session_metrics["phi_progression"].append({
                "exchange": exchange_num,
                "phi": phi_after,
                "change": phi_change,
                "timestamp": time.time()
            })
        
        # Evaluate with phi consideration
        evaluation = self.evaluate_response_with_phi(isc_response, training_input, phi_change)
        
        # Provide feedback to learning engine
        feedback_value = evaluation["overall"] / 10.0
        
        # Boost feedback for phi-aligned responses
        if evaluation.get("phi_aligned", True) and phi_change > 0:
            feedback_value *= 1.2
        
        self.core.learning_engine.apply_feedback(feedback_value)
        
        # Check phi target achievement
        target_phi = self.calculate_dynamic_phi_target(phi_before)
        if phi_after >= target_phi * 0.95:
            with self.lock:
                self.session_metrics["phi_targets_achieved"] += 1
        
        # Adaptive learning rate based on phi
        if exchange_num % 10 == 0:
            self.core.learning_engine.adapt_learning_rate()
            with self.lock:
                self.session_metrics["learning_rate_adjustments"] += 1
        
        exchange_time = time.time() - exchange_start
        
        return {
            "training_input": training_input,
            "isc_response": isc_response,
            "evaluation": evaluation,
            "phi_before": phi_before,
            "phi_after": phi_after,
            "phi_change": phi_change,
            "target_phi": target_phi,
            "feedback": feedback_value,
            "duration": exchange_time
        }
    
    def run_training_session(self, topic: str, num_exchanges: int, complexity_progression: bool = True):
        """Run enhanced training session with phi optimization"""
        self.training_active = True
        self.session_start = time.time()
        
        # Create progress tracking
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console
        )
        
        training_task = progress.add_task(
            f"[cyan]Training on {topic} (optimizing phi)...", 
            total=num_exchanges
        )
        
        # Pre-generate some training inputs in parallel
        prefetch_buffer = queue.Queue(maxsize=10)
        
        def prefetch_inputs():
            """Background thread to pre-generate inputs"""
            exchange_id = 0
            while self.training_active and exchange_id < num_exchanges:
                if prefetch_buffer.qsize() < 10:
                    level = min(10, 1 + (exchange_id * 10 // num_exchanges)) if complexity_progression else 5
                    current_phi = self.phi_window[-1] if self.phi_window else 0.0
                    
                    content, cached = self.generate_training_input(topic, level, current_phi)
                    if content:
                        prefetch_buffer.put((exchange_id, content, cached))
                    exchange_id += 1
                else:
                    time.sleep(0.1)
        
        # Start prefetch thread
        prefetch_thread = threading.Thread(target=prefetch_inputs)
        prefetch_thread.start()
        
        # Training loop
        with Live(progress, console=self.console, refresh_per_second=2):
            for i in range(num_exchanges):
                try:
                    # Get pre-generated input
                    exchange_id, training_input, cached = prefetch_buffer.get(timeout=30)
                    
                    # Execute training
                    result = self.train_single_exchange(training_input, i)
                    
                    # Update metrics
                    with self.lock:
                        self.session_metrics["total_exchanges"] += 1
                        self.session_metrics["successful_exchanges"] += 1
                        
                        # Update average response time
                        prev_avg = self.session_metrics["avg_response_time"]
                        self.session_metrics["avg_response_time"] = (
                            (prev_avg * i + result["duration"]) / (i + 1)
                        )
                    
                    # Store in history
                    self.exchange_history.append(result)
                    
                    # Update progress with phi info
                    progress.update(
                        training_task, 
                        advance=1,
                        description=f"[cyan]Training on {topic} | Phi: {result['phi_after']:.3f} ({result['phi_change']:+.3f}) | Target: {result['target_phi']:.3f}"
                    )
                    
                    # Periodic saves and visualization
                    if (i + 1) % 10 == 0:
                        self._save_checkpoint(i + 1)
                        if (i + 1) % 50 == 0:
                            self._update_visualizations()
                    
                except Exception as e:
                    self.console.print(f"[red]Exchange {i} failed: {e}[/red]")
                    with self.lock:
                        self.session_metrics["failed_exchanges"] += 1
        
        # Stop prefetching
        self.training_active = False
        prefetch_thread.join()
        
        # Final save and visualization
        self._save_checkpoint(num_exchanges)
        self._update_visualizations()
        self._show_final_summary()
    
    def _save_checkpoint(self, exchange_num: int):
        """Save training checkpoint with enhanced metrics"""
        # Save ISC state to standard location
        self.core.save_state()  # No filepath = use persistence manager
        
        # Save training metrics separately
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = f"training_enhanced_{timestamp}.json"
        
        # Add cache and phi statistics
        cache_stats = self.cache_manager.get_cache_stats()
        phi_metrics = self.core.integrator.get_integration_metrics()
        learning_metrics = self.core.learning_engine.get_learning_metrics()
        
        save_data = {
            "timestamp": timestamp,
            "exchange_num": exchange_num,
            "session_metrics": dict(self.session_metrics),
            "cache_stats": cache_stats,
            "phi_metrics": phi_metrics,
            "learning_metrics": learning_metrics,
            "recent_exchanges": list(self.exchange_history)[-50:],
            "phi_trend": self.core.integrator.get_phi_trend()
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        self.console.print(f"[green]✓ Checkpoint saved[/green]")
    
    def _update_visualizations(self):
        """Create enhanced visualizations including phi trends"""
        if len(self.session_metrics["phi_progression"]) < 2:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Phi progression
        phi_data = self.session_metrics["phi_progression"]
        exchanges = [p["exchange"] for p in phi_data]
        phi_values = [p["phi"] for p in phi_data]
        
        ax1.plot(exchanges, phi_values, 'b-', linewidth=2)
        ax1.fill_between(exchanges, phi_values, alpha=0.3)
        ax1.set_xlabel("Exchange")
        ax1.set_ylabel("Phi (Φ)")
        ax1.set_title("Integrated Information (Phi) Over Time")
        ax1.grid(True, alpha=0.3)
        
        # Add target line
        if phi_values:
            current_target = self.calculate_dynamic_phi_target(phi_values[-1])
            ax1.axhline(y=current_target, color='r', linestyle='--', label=f'Target: {current_target:.2f}')
            ax1.legend()
        
        # 2. Phi change distribution
        phi_changes = [p["change"] for p in phi_data]
        ax2.hist(phi_changes, bins=30, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(x=0, color='r', linestyle='--')
        ax2.set_xlabel("Phi Change per Exchange")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Distribution of Phi Changes")
        
        # 3. Learning metrics
        if hasattr(self.core.learning_engine, 'loss_components') and self.core.learning_engine.loss_components:
            components = list(self.core.learning_engine.loss_components)
            if components:
                component_names = [k for k in components[0].keys() if k != 'total']
                for name in component_names:
                    values = [c.get(name, 0) for c in components]
                    ax3.plot(values, label=name)
                ax3.set_xlabel("Training Step")
                ax3.set_ylabel("Loss")
                ax3.set_title("Loss Components Over Time")
                ax3.legend()
                ax3.grid(True, alpha=0.3)
        
        # 4. Cache efficiency
        cache_total = self.session_metrics["cache_hits"] + self.session_metrics["cache_misses"]
        if cache_total > 0:
            hit_rate = self.session_metrics["cache_hits"] / cache_total
            
            # Pie chart for cache performance
            sizes = [self.session_metrics["cache_hits"], self.session_metrics["cache_misses"]]
            labels = [f'Hits ({self.session_metrics["cache_hits"]})', 
                     f'Misses ({self.session_metrics["cache_misses"]})']
            colors = ['#2ecc71', '#e74c3c']
            
            ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax4.set_title(f'Cache Performance (Hit Rate: {hit_rate:.1%})')
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'training_progress_enhanced_{timestamp}.png', dpi=150)
        plt.close()
        
        self.console.print("[green]✓ Visualizations updated[/green]")
    
    def _show_final_summary(self):
        """Show enhanced final summary with phi and cache statistics"""
        duration = time.time() - self.session_start
        
        # Create summary table
        table = Table(title="Enhanced Training Session Summary", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        # Basic metrics
        table.add_row("Duration", f"{duration:.1f} seconds")
        table.add_row("Total Exchanges", str(self.session_metrics["total_exchanges"]))
        table.add_row("Successful", str(self.session_metrics["successful_exchanges"]))
        table.add_row("Failed", str(self.session_metrics["failed_exchanges"]))
        
        # Phi metrics
        phi_data = self.session_metrics["phi_progression"]
        if phi_data:
            initial_phi = phi_data[0]["phi"]
            final_phi = phi_data[-1]["phi"]
            max_phi = max(p["phi"] for p in phi_data)
            
            table.add_row("Initial Phi", f"{initial_phi:.3f}")
            table.add_row("Final Phi", f"{final_phi:.3f}")
            table.add_row("Max Phi", f"{max_phi:.3f}")
            table.add_row("Phi Growth", f"{((final_phi/initial_phi - 1) * 100):.1f}%" if initial_phi > 0 else "N/A")
            table.add_row("Targets Achieved", str(self.session_metrics["phi_targets_achieved"]))
        
        # Cache metrics
        cache_total = self.session_metrics["cache_hits"] + self.session_metrics["cache_misses"]
        if cache_total > 0:
            hit_rate = self.session_metrics["cache_hits"] / cache_total
            table.add_row("Cache Hit Rate", f"{hit_rate:.1%}")
            table.add_row("API Calls Saved", str(self.session_metrics["cache_hits"]))
        
        # Token usage
        total_tokens = self.session_metrics["tokens_used"]["prompt"] + self.session_metrics["tokens_used"]["completion"]
        table.add_row("Total Tokens", f"{total_tokens:,}")
        
        # Cost estimation
        prompt_cost = self.session_metrics["tokens_used"]["prompt"] * 0.0005 / 1000
        completion_cost = self.session_metrics["tokens_used"]["completion"] * 0.0015 / 1000
        total_cost = prompt_cost + completion_cost
        table.add_row("Estimated Cost", f"${total_cost:.4f}")
        
        # Performance
        table.add_row("Avg Response Time", f"{self.session_metrics['avg_response_time']:.2f}s")
        
        self.console.print(table)
        
        # Show cache statistics
        cache_stats = self.cache_manager.get_cache_stats()
        cache_table = Table(title="Cache Statistics", show_header=True)
        cache_table.add_column("Metric", style="cyan")
        cache_table.add_column("Value", style="green")
        
        cache_table.add_row("Total Cached Responses", str(cache_stats["chatgpt_cache_count"]))
        cache_table.add_row("Total Cached Phi Values", str(cache_stats["phi_cache_count"]))
        cache_table.add_row("Database Size", f"{cache_stats['database_size_mb']:.2f} MB")
        cache_table.add_row("Memory Cache Size", str(cache_stats["memory_cache_size"]))
        
        self.console.print(cache_table)

def main():
    """Enhanced main function"""
    console = Console()
    
    # Display header
    console.print(Panel.fit(
        "[bold cyan]Enhanced ISC AI ChatGPT Trainer[/bold cyan]\n"
        "[dim]Phi Optimization & Intelligent Caching[/dim]",
        border_style="cyan"
    ))
    
    # Check API key
    if not OPENAI_API_KEY:
        console.print("[red]Error: OPENAI_API_KEY environment variable not set![/red]")
        console.print("Please set it using: export OPENAI_API_KEY='your-key-here'")
        return
    
    # Initialize trainer
    trainer = EnhancedChatGPTTrainer(max_workers=5)
    
    if not trainer.initialize():
        return
    
    # Training configuration
    topic = "consciousness, emergence, and integrated information theory"
    num_exchanges = 100
    
    console.print(f"\n[cyan]Starting enhanced training session:[/cyan]")
    console.print(f"  • Topic: {topic}")
    console.print(f"  • Exchanges: {num_exchanges}")
    console.print(f"  • Phi optimization: Enabled")
    console.print(f"  • Caching: Enabled")
    console.print()
    
    try:
        trainer.run_training_session(topic, num_exchanges, complexity_progression=True)
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Training error: {e}[/red]")
    finally:
        # Cleanup
        trainer.core.end_session()
        console.print("\n[green]Training session complete![/green]")

if __name__ == "__main__":
    main()