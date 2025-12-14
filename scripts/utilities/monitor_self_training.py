#!/usr/bin/env python3
"""
Monitor and analyze self-referential training sessions
Provides real-time monitoring and post-session analysis
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from collections import Counter
import glob

class SelfTrainingMonitor:
    """Monitor for self-referential training sessions"""
    
    def __init__(self):
        self.console = Console()
    
    def analyze_training_session(self, session_file: str) -> Dict[str, Any]:
        """Analyze a single training session"""
        with open(session_file, 'r') as f:
            data = json.load(f)
        
        metrics = data.get('metrics', {})
        exchanges = data.get('recent_exchanges', [])
        concept_coverage = data.get('concept_coverage', {})
        
        # Calculate loop detection metrics
        loop_metrics = self._detect_loops(exchanges)
        
        # Calculate diversity metrics
        diversity_metrics = self._calculate_diversity(concept_coverage, exchanges)
        
        # Calculate phi progression metrics
        phi_metrics = self._analyze_phi_progression(metrics.get('phi_progression', []))
        
        return {
            'session_file': session_file,
            'timestamp': data.get('timestamp'),
            'total_exchanges': metrics.get('total_exchanges', 0),
            'unique_questions': metrics.get('unique_questions', 0),
            'duplicates_prevented': metrics.get('duplicate_prevented', 0),
            'loop_metrics': loop_metrics,
            'diversity_metrics': diversity_metrics,
            'phi_metrics': phi_metrics,
            'concept_coverage': concept_coverage
        }
    
    def _detect_loops(self, exchanges: List[Dict]) -> Dict[str, Any]:
        """Detect potential loops in training"""
        if not exchanges:
            return {'loops_detected': 0, 'repetition_score': 0.0}
        
        questions = [e.get('question', '') for e in exchanges]
        
        # Check for exact repetitions
        question_counts = Counter(questions)
        repeated_questions = sum(1 for count in question_counts.values() if count > 1)
        
        # Check for semantic loops (simplified n-gram analysis)
        bigrams = []
        for i in range(len(questions) - 1):
            bigrams.append((questions[i], questions[i+1]))
        
        bigram_counts = Counter(bigrams)
        repeated_patterns = sum(1 for count in bigram_counts.values() if count > 1)
        
        # Calculate repetition score
        total_questions = len(questions)
        repetition_score = (repeated_questions + repeated_patterns) / max(1, total_questions)
        
        return {
            'loops_detected': repeated_questions + repeated_patterns,
            'repetition_score': repetition_score,
            'repeated_questions': repeated_questions,
            'repeated_patterns': repeated_patterns
        }
    
    def _calculate_diversity(self, concept_coverage: Dict[str, int], 
                           exchanges: List[Dict]) -> Dict[str, Any]:
        """Calculate diversity metrics"""
        # Concept diversity
        num_concepts = len(concept_coverage)
        total_explorations = sum(concept_coverage.values())
        concept_entropy = 0.0
        
        if total_explorations > 0:
            probabilities = [count/total_explorations for count in concept_coverage.values()]
            concept_entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        
        # Response diversity (vocabulary)
        all_words = []
        for exchange in exchanges:
            response = exchange.get('response', '')
            all_words.extend(response.lower().split())
        
        vocabulary_size = len(set(all_words))
        
        return {
            'num_concepts_explored': num_concepts,
            'concept_entropy': concept_entropy,
            'vocabulary_size': vocabulary_size,
            'exploration_balance': concept_entropy / np.log2(max(1, num_concepts)) if num_concepts > 1 else 0
        }
    
    def _analyze_phi_progression(self, phi_progression: List[Dict]) -> Dict[str, Any]:
        """Analyze phi progression patterns"""
        if not phi_progression:
            return {
                'initial_phi': 0.0,
                'final_phi': 0.0,
                'max_phi': 0.0,
                'growth_rate': 0.0,
                'stability': 0.0
            }
        
        phi_values = [p['phi'] for p in phi_progression]
        
        # Calculate growth rate
        if len(phi_values) > 1:
            growth_rate = (phi_values[-1] - phi_values[0]) / len(phi_values)
        else:
            growth_rate = 0.0
        
        # Calculate stability (inverse of variance)
        if len(phi_values) > 1:
            stability = 1.0 / (1.0 + np.var(phi_values))
        else:
            stability = 1.0
        
        return {
            'initial_phi': phi_values[0] if phi_values else 0.0,
            'final_phi': phi_values[-1] if phi_values else 0.0,
            'max_phi': max(phi_values) if phi_values else 0.0,
            'growth_rate': growth_rate,
            'stability': stability,
            'trend': 'increasing' if growth_rate > 0.001 else 'decreasing' if growth_rate < -0.001 else 'stable'
        }
    
    def show_session_summary(self, analysis: Dict[str, Any]):
        """Display session summary"""
        self.console.print(Panel.fit(
            f"[bold cyan]Self-Training Session Analysis[/bold cyan]\n"
            f"[dim]{analysis['timestamp']}[/dim]",
            border_style="cyan"
        ))
        
        # Basic metrics table
        basic_table = Table(title="Basic Metrics", show_header=True)
        basic_table.add_column("Metric", style="cyan")
        basic_table.add_column("Value", style="green")
        
        basic_table.add_row("Total Exchanges", str(analysis['total_exchanges']))
        basic_table.add_row("Unique Questions", str(analysis['unique_questions']))
        basic_table.add_row("Duplicates Prevented", str(analysis['duplicates_prevented']))
        
        self.console.print(basic_table)
        
        # Loop detection table
        loop_table = Table(title="Loop Detection", show_header=True)
        loop_table.add_column("Metric", style="cyan")
        loop_table.add_column("Value", style="yellow")
        
        loop_metrics = analysis['loop_metrics']
        loop_table.add_row("Loops Detected", str(loop_metrics['loops_detected']))
        loop_table.add_row("Repetition Score", f"{loop_metrics['repetition_score']:.3f}")
        loop_table.add_row("Repeated Questions", str(loop_metrics['repeated_questions']))
        loop_table.add_row("Repeated Patterns", str(loop_metrics['repeated_patterns']))
        
        # Color code based on severity
        if loop_metrics['repetition_score'] > 0.1:
            loop_table.border_style = "red"
        elif loop_metrics['repetition_score'] > 0.05:
            loop_table.border_style = "yellow"
        else:
            loop_table.border_style = "green"
        
        self.console.print(loop_table)
        
        # Diversity metrics table
        diversity_table = Table(title="Diversity Metrics", show_header=True)
        diversity_table.add_column("Metric", style="cyan")
        diversity_table.add_column("Value", style="green")
        
        diversity_metrics = analysis['diversity_metrics']
        diversity_table.add_row("Concepts Explored", str(diversity_metrics['num_concepts_explored']))
        diversity_table.add_row("Concept Entropy", f"{diversity_metrics['concept_entropy']:.3f}")
        diversity_table.add_row("Vocabulary Size", str(diversity_metrics['vocabulary_size']))
        diversity_table.add_row("Exploration Balance", f"{diversity_metrics['exploration_balance']:.3f}")
        
        self.console.print(diversity_table)
        
        # Phi progression table
        phi_table = Table(title="Phi Progression", show_header=True)
        phi_table.add_column("Metric", style="cyan")
        phi_table.add_column("Value", style="green")
        
        phi_metrics = analysis['phi_metrics']
        phi_table.add_row("Initial Phi", f"{phi_metrics['initial_phi']:.3f}")
        phi_table.add_row("Final Phi", f"{phi_metrics['final_phi']:.3f}")
        phi_table.add_row("Max Phi", f"{phi_metrics['max_phi']:.3f}")
        phi_table.add_row("Growth Rate", f"{phi_metrics['growth_rate']:.4f}")
        phi_table.add_row("Stability", f"{phi_metrics['stability']:.3f}")
        phi_table.add_row("Trend", phi_metrics['trend'])
        
        self.console.print(phi_table)
        
        # Top concepts
        if analysis['concept_coverage']:
            concept_table = Table(title="Top Explored Concepts", show_header=True)
            concept_table.add_column("Concept", style="cyan")
            concept_table.add_column("Count", style="green")
            
            top_concepts = Counter(analysis['concept_coverage']).most_common(10)
            for concept, count in top_concepts:
                concept_table.add_row(concept, str(count))
            
            self.console.print(concept_table)
    
    def generate_visualizations(self, analysis: Dict[str, Any], output_dir: str = "."):
        """Generate visualization plots"""
        # This would create plots similar to the training script
        # but focused on loop detection and diversity metrics
        pass
    
    def compare_sessions(self, session_files: List[str]):
        """Compare multiple training sessions"""
        analyses = []
        for session_file in session_files:
            try:
                analysis = self.analyze_training_session(session_file)
                analyses.append(analysis)
            except Exception as e:
                self.console.print(f"[red]Error analyzing {session_file}: {e}[/red]")
        
        if not analyses:
            self.console.print("[red]No valid sessions to compare[/red]")
            return
        
        # Create comparison table
        comparison_table = Table(title="Session Comparison", show_header=True)
        comparison_table.add_column("Session", style="cyan")
        comparison_table.add_column("Exchanges", style="green")
        comparison_table.add_column("Unique Q", style="green")
        comparison_table.add_column("Loops", style="yellow")
        comparison_table.add_column("Rep. Score", style="yellow")
        comparison_table.add_column("Concepts", style="green")
        comparison_table.add_column("Final Phi", style="green")
        comparison_table.add_column("Trend", style="green")
        
        for analysis in analyses:
            timestamp = analysis['timestamp'].split('_')[0] if analysis['timestamp'] else 'Unknown'
            comparison_table.add_row(
                timestamp,
                str(analysis['total_exchanges']),
                str(analysis['unique_questions']),
                str(analysis['loop_metrics']['loops_detected']),
                f"{analysis['loop_metrics']['repetition_score']:.3f}",
                str(analysis['diversity_metrics']['num_concepts_explored']),
                f"{analysis['phi_metrics']['final_phi']:.3f}",
                analysis['phi_metrics']['trend']
            )
        
        self.console.print(comparison_table)
        
        # Identify best session
        best_session = max(analyses, key=lambda x: (
            x['unique_questions'] / max(1, x['total_exchanges']) *  # Question uniqueness
            x['diversity_metrics']['exploration_balance'] *          # Concept diversity
            (1 - x['loop_metrics']['repetition_score']) *           # Loop avoidance
            x['phi_metrics']['final_phi']                           # Phi achievement
        ))
        
        self.console.print(f"\n[green]Best session: {best_session['timestamp']}[/green]")

def main():
    """Main entry point"""
    console = Console()
    monitor = SelfTrainingMonitor()
    
    # Display header
    console.print(Panel.fit(
        "[bold cyan]Self-Training Session Monitor[/bold cyan]\n"
        "[dim]Analyze loop patterns and training diversity[/dim]",
        border_style="cyan"
    ))
    
    # Find training session files
    session_files = sorted(glob.glob("self_training_*.json"), reverse=True)
    
    if not session_files:
        console.print("[yellow]No self-training sessions found.[/yellow]")
        console.print("Run 'make train-self' to create a training session.")
        return
    
    # Show available sessions
    console.print(f"\n[cyan]Found {len(session_files)} training sessions:[/cyan]")
    for i, session_file in enumerate(session_files[:10]):  # Show last 10
        console.print(f"  {i+1}. {session_file}")
    
    if len(session_files) == 1:
        # Analyze single session
        console.print("\n[cyan]Analyzing session...[/cyan]\n")
        analysis = monitor.analyze_training_session(session_files[0])
        monitor.show_session_summary(analysis)
    else:
        # Ask user what to do
        console.print("\n[cyan]Options:[/cyan]")
        console.print("  1. Analyze latest session")
        console.print("  2. Compare all sessions")
        console.print("  3. Select specific session")
        
        try:
            choice = input("\nEnter choice (1-3): ").strip()
            
            if choice == "1":
                analysis = monitor.analyze_training_session(session_files[0])
                monitor.show_session_summary(analysis)
            elif choice == "2":
                monitor.compare_sessions(session_files)
            elif choice == "3":
                session_num = int(input("Enter session number: ")) - 1
                if 0 <= session_num < len(session_files):
                    analysis = monitor.analyze_training_session(session_files[session_num])
                    monitor.show_session_summary(analysis)
                else:
                    console.print("[red]Invalid session number[/red]")
        except (ValueError, KeyboardInterrupt):
            console.print("\n[yellow]Cancelled[/yellow]")

if __name__ == "__main__":
    main()