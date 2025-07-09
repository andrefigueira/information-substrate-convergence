"""
Command Line Interface for the ISC AI System
"""

import click
import sys
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress
from rich.syntax import Syntax
from rich.markdown import Markdown
import torch
import json
import os
from datetime import datetime
import uuid
import plotext as plt
from pathlib import Path
import glob
import time
import termios
import tty
from typing import Optional

from .core import ISCCore
from .visualizer import SystemVisualizer
from .storage import StorageManager


class ISCCommandInterface:
    """
    Command line interface for interacting with the ISC AI system.
    """
    
    def __init__(self):
        self.console = Console()
        self.core = ISCCore()
        self.visualizer = SystemVisualizer()
        self.storage = StorageManager("isc_storage")
        self.session_id = None
        self.verbose = False
        
        # Command registry
        self.commands = {
            "/start": self.cmd_start,
            "/end": self.cmd_end,
            "/save": self.cmd_save,
            "/load": self.cmd_load,
            "/status": self.cmd_status,
            "/help": self.cmd_help,
            "/verbose": self.cmd_verbose,
            "/feedback": self.cmd_feedback,
            "/reset": self.cmd_reset,
            "/introspect": self.cmd_introspect,
            "/metrics": self.cmd_metrics,
            "/connections": self.cmd_connections,
            "/history": self.cmd_history,
            "/concepts": self.cmd_concepts,
            "/predict": self.cmd_predict,
            "/explain": self.cmd_explain,
            "/exit": self.cmd_exit,
            "/quit": self.cmd_exit,
            # Storage commands
            "/save_graph": self.cmd_save_graph,
            "/load_graph": self.cmd_load_graph,
            "/query": self.cmd_query_graph,
            "/update_graph": self.cmd_update_graph,
            "/export": self.cmd_export_graph,
            "/import": self.cmd_import_graph,
            "/backup": self.cmd_backup,
            "/restore": self.cmd_restore,
            "/storage": self.cmd_storage_info,
            "/versions": self.cmd_list_versions,
        }
    
    def run(self):
        """Main interaction loop."""
        self._show_welcome()
        
        while True:
            try:
                # Get user input
                user_input = self.console.input("\n[bold cyan]You:[/bold cyan] ").strip()
                
                if not user_input:
                    continue
                
                # Check if it's a command
                if user_input.startswith("/"):
                    self._handle_command(user_input)
                else:
                    # Process as conversation
                    if not self.core.session_active:
                        self.console.print("[yellow]No active session. Use /start to begin.[/yellow]")
                        continue
                    
                    # Show processing indicator if verbose
                    if self.verbose:
                        with self.console.status("[bold green]Processing...") as status:
                            response = self.core.process_input(user_input)
                            self._show_verbose_info()
                    else:
                        response = self.core.process_input(user_input)
                    
                    # Display response
                    self.console.print(f"\n[bold green]ISC:[/bold green] {response}")
                    
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Use /exit to quit.[/yellow]")
            except Exception as e:
                self.console.print(f"[red]Error: {str(e)}[/red]")
    
    def _show_welcome(self):
        """Display welcome message."""
        welcome = Panel(
            "[bold cyan]ISC AI System v0.1.0[/bold cyan]\n\n"
            "An interactive AI based on Informational Substrate Convergence\n\n"
            "Type [bold]/help[/bold] for commands or [bold]/start[/bold] to begin a session.",
            title="Welcome",
            border_style="cyan"
        )
        self.console.print(welcome)
    
    def _handle_command(self, command_str: str):
        """Handle command execution."""
        parts = command_str.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        if command in self.commands:
            self.commands[command](args)
        else:
            self.console.print(f"[red]Unknown command: {command}[/red]")
            self.console.print("Type /help for available commands.")
    
    def _show_verbose_info(self):
        """Display verbose processing information."""
        if not self.verbose:
            return
        
        # Get current metrics
        metrics = self.core.metrics
        
        # Create info panel
        info = Table(title="Processing Information", show_header=False)
        info.add_column("Metric", style="cyan")
        info.add_column("Value", style="green")
        
        info.add_row("Φ (Phi)", f"{metrics['phi_value']:.4f}")
        info.add_row("Coherence", f"{metrics['coherence_score']:.4f}")
        info.add_row("Concepts", str(metrics['concepts_formed']))
        
        self.console.print(info)
    
    # Command implementations
    
    def cmd_start(self, args: str):
        """Start a new conversation session."""
        if self.core.session_active:
            self.console.print("[yellow]Session already active. Use /end to close it first.[/yellow]")
            return
        
        self.session_id = str(uuid.uuid4())
        self.core.session_active = True
        self.core.current_session_id = self.session_id
        
        self.console.print(Panel(
            f"[green]Session started![/green]\n"
            f"Session ID: {self.session_id[:8]}...\n\n"
            "I'm ready to learn from our conversation.",
            border_style="green"
        ))
    
    def cmd_end(self, args: str):
        """End current session."""
        if not self.core.session_active:
            self.console.print("[yellow]No active session.[/yellow]")
            return
        
        # Show session summary
        interactions = len(self.core.memory.get_session_history(self.session_id))
        
        self.console.print(Panel(
            f"[yellow]Session ended.[/yellow]\n"
            f"Total interactions: {interactions}\n"
            f"Final Φ: {self.core.metrics['phi_value']:.4f}\n"
            f"Concepts formed: {self.core.metrics['concepts_formed']}",
            border_style="yellow"
        ))
        
        self.core.session_active = False
        self.core.current_session_id = None
    
    def cmd_save(self, args: str):
        """Save system state."""
        filename = args.strip() or f"isc_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        
        try:
            result = self.core.save_state(filename)
            self.console.print(f"[green]{result}[/green]")
        except Exception as e:
            self.console.print(f"[red]Failed to save: {str(e)}[/red]")
    
    def cmd_load(self, args: str):
        """Load system state."""
        filename = args.strip()
        
        if not filename:
            # Show file browser
            filename = self._file_browser()
            if not filename:
                return
        
        if not os.path.exists(filename):
            self.console.print(f"[red]File not found: {filename}[/red]")
            return
        
        try:
            result = self.core.load_state(filename)
            self.console.print(f"[green]{result}[/green]")
        except Exception as e:
            self.console.print(f"[red]Failed to load: {str(e)}[/red]")
    
    def cmd_status(self, args: str):
        """Show system status."""
        status = self.core.get_status()
        
        # Create status table
        table = Table(title="System Status", show_header=True)
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        
        table.add_row("Session", "Active" if status["session_active"] else "Inactive")
        table.add_row("Total Interactions", str(status["metrics"]["total_interactions"]))
        table.add_row("Φ Value", f"{status['metrics']['phi_value']:.4f}")
        table.add_row("Coherence", f"{status['metrics']['coherence_score']:.4f}")
        table.add_row("Concepts", str(status["total_concepts"]))
        table.add_row("Connections", str(status["total_connections"]))
        table.add_row("Memory Size", str(status["memory_size"]))
        table.add_row("Network Parameters", f"{status['network_parameters']:,}")
        
        self.console.print(table)
    
    def cmd_help(self, args: str):
        """Show help information."""
        help_text = """
[bold cyan]Available Commands:[/bold cyan]

[bold]Basic Commands:[/bold]
  /start              - Begin a new conversation session
  /end                - End current session
  /save [filename]    - Save conversation and system state
  /load <filename>    - Load a previous conversation and state
  /status             - Show current system metrics and state
  /help               - Show this help message
  /exit, /quit        - Exit the program

[bold]Interaction Commands:[/bold]
  /verbose [on|off]   - Toggle detailed output
  /feedback <pos|neg> - Provide feedback on last response
  /reset              - Reset system to initial state
  /introspect         - System explains its understanding

[bold]Information Commands:[/bold]
  /metrics            - Show learning metrics
  /connections        - Display concept connections
  /history            - Show conversation history
  /concepts           - List formed concepts
  /predict            - Predict next input
  /explain <concept>  - Explain understanding of concept

[bold]Storage Commands:[/bold]
  /save_graph [tag]   - Save knowledge graph to local storage
  /load_graph [ver]   - Load a specific graph version
  /query <query>      - Query the graph (e.g. "find node learning")
  /update_graph       - Add/remove nodes and edges
  /export <format>    - Export graph (json/graphml/text/etc)
  /import <file>      - Import graph from file
  /backup             - Create full backup of storage
  /restore <path>     - Restore from backup
  /storage            - Show storage information
  /versions           - List all saved versions
"""
        self.console.print(Markdown(help_text))
    
    def cmd_verbose(self, args: str):
        """Toggle verbose mode."""
        if args.lower() == "on":
            self.verbose = True
            self.console.print("[green]Verbose mode ON[/green]")
        elif args.lower() == "off":
            self.verbose = False
            self.console.print("[yellow]Verbose mode OFF[/yellow]")
        else:
            self.verbose = not self.verbose
            status = "ON" if self.verbose else "OFF"
            self.console.print(f"[cyan]Verbose mode {status}[/cyan]")
    
    def cmd_feedback(self, args: str):
        """Provide feedback."""
        if args.lower() in ["positive", "pos", "+"]:
            response = self.core.provide_feedback("positive")
        elif args.lower() in ["negative", "neg", "-"]:
            response = self.core.provide_feedback("negative")
        else:
            self.console.print("[red]Please specify 'positive' or 'negative'[/red]")
            return
        
        self.console.print(f"[cyan]{response}[/cyan]")
    
    def cmd_reset(self, args: str):
        """Reset the system."""
        confirm = self.console.input("[yellow]Reset system? This will clear learning but keep history. (y/n): [/yellow]")
        
        if confirm.lower() == 'y':
            self.core = ISCCore()
            self.console.print("[green]System reset complete.[/green]")
        else:
            self.console.print("[yellow]Reset cancelled.[/yellow]")
    
    def cmd_introspect(self, args: str):
        """Show system introspection."""
        introspection = self.core.introspect()
        self.console.print(Panel(introspection, title="System Introspection", border_style="cyan"))
    
    def cmd_metrics(self, args: str):
        """Show detailed metrics."""
        # Integration metrics
        integration_metrics = self.core.integrator.get_integration_metrics()
        
        # Learning metrics
        learning_metrics = self.core.learning_engine.get_learning_metrics()
        
        # Create metrics table
        table = Table(title="Detailed Metrics", show_header=True)
        table.add_column("Category", style="cyan")
        table.add_column("Metric", style="yellow")
        table.add_column("Value", style="green")
        
        # Integration metrics
        table.add_row("Integration", "Current Φ", f"{integration_metrics['current_phi']:.4f}")
        table.add_row("Integration", "Average Φ", f"{integration_metrics['average_phi']:.4f}")
        table.add_row("Integration", "Max Φ", f"{integration_metrics['max_phi']:.4f}")
        table.add_row("Integration", "Φ Trend", f"{integration_metrics['phi_trend']:.4f}")
        
        # Learning metrics
        table.add_row("Learning", "Average Loss", f"{learning_metrics['average_loss']:.4f}")
        table.add_row("Learning", "Prediction Accuracy", f"{learning_metrics['prediction_accuracy']:.4f}")
        table.add_row("Learning", "Learning Progress", f"{learning_metrics['learning_progress']:.4f}")
        table.add_row("Learning", "Experience Count", str(learning_metrics['experience_count']))
        
        self.console.print(table)
        
        # Show Phi history as ASCII plot
        if integration_metrics['current_phi'] > 0:
            self._plot_phi_history()
    
    def _plot_phi_history(self):
        """Plot Phi history using plotext."""
        history = self.core.integrator.phi_history
        if len(history) > 1:
            plt.clear_figure()
            plt.plot(history)
            plt.title("Φ (Phi) History")
            plt.xlabel("Interaction")
            plt.ylabel("Φ Value")
            plt.show()
    
    def cmd_connections(self, args: str):
        """Show concept connections."""
        ascii_graph = self.core.knowledge_graph.visualize_ascii()
        self.console.print(Panel(ascii_graph, title="Knowledge Graph", border_style="cyan"))
    
    def cmd_history(self, args: str):
        """Show conversation history."""
        recent = self.core.memory.get_recent_interactions(10)
        
        if not recent:
            self.console.print("[yellow]No conversation history yet.[/yellow]")
            return
        
        table = Table(title="Recent Conversation History", show_header=True)
        table.add_column("#", style="cyan", width=3)
        table.add_column("You", style="yellow", width=40)
        table.add_column("ISC", style="green", width=40)
        
        for i, interaction in enumerate(recent, 1):
            user_text = interaction['input'][:37] + "..." if len(interaction['input']) > 40 else interaction['input']
            ai_text = interaction['response'][:37] + "..." if len(interaction['response']) > 40 else interaction['response']
            table.add_row(str(i), user_text, ai_text)
        
        self.console.print(table)
    
    def cmd_concepts(self, args: str):
        """List formed concepts."""
        concepts = list(self.core.knowledge_graph.graph.nodes())
        
        if not concepts:
            self.console.print("[yellow]No concepts formed yet.[/yellow]")
            return
        
        # Get central concepts
        central = self.core.knowledge_graph.get_central_concepts(10)
        
        table = Table(title="Formed Concepts", show_header=True)
        table.add_column("Concept", style="cyan")
        table.add_column("Connections", style="green")
        table.add_column("Central", style="yellow")
        
        for concept in concepts[:20]:  # Show top 20
            connections = len(list(self.core.knowledge_graph.graph.neighbors(concept)))
            is_central = "✓" if concept in central else ""
            table.add_row(concept, str(connections), is_central)
        
        self.console.print(table)
        
        if len(concepts) > 20:
            self.console.print(f"[dim]... and {len(concepts) - 20} more concepts[/dim]")
    
    def cmd_predict(self, args: str):
        """Show prediction."""
        prediction = self.core.predict_next_input()
        self.console.print(Panel(prediction, title="Input Prediction", border_style="cyan"))
    
    def cmd_explain(self, args: str):
        """Explain a concept."""
        concept = args.strip()
        
        if not concept:
            self.console.print("[red]Please specify a concept to explain.[/red]")
            return
        
        explanation = self.core.explain_concept(concept)
        self.console.print(Panel(explanation, title=f"Concept: {concept}", border_style="cyan"))
    
    def cmd_exit(self, args: str):
        """Exit the program."""
        if self.core.session_active:
            self.cmd_end("")
        
        self.console.print("[cyan]Thank you for interacting with ISC AI. Goodbye![/cyan]")
        sys.exit(0)
    
    # Storage Commands
    
    def cmd_save_graph(self, args: str):
        """Save the knowledge graph to local storage."""
        parts = args.strip().split(maxsplit=1)
        version_tag = parts[0] if parts else None
        description = parts[1] if len(parts) > 1 else ""
        
        # Update storage with current graph
        self.storage.graph_db.graph = self.core.knowledge_graph.graph
        
        # Save
        result = self.storage.save(version_tag, description)
        
        # Display results
        table = Table(title="Graph Saved", show_header=False)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Version", result["version_id"])
        table.add_row("Nodes", str(result["stats"]["graph_stats"]["nodes"]))
        table.add_row("Edges", str(result["stats"]["graph_stats"]["edges"]))
        table.add_row("Storage Size", f"{result['stats']['total_size_mb']} MB")
        
        self.console.print(table)
        
        if result["exports"]:
            self.console.print("\n[cyan]Exported to:[/cyan]")
            for fmt, path in result["exports"].items():
                self.console.print(f"  - {fmt}: {path}")
    
    def cmd_load_graph(self, args: str):
        """Load a graph version from storage."""
        version = args.strip() or None
        
        try:
            result = self.storage.load(version)
            
            # Update core knowledge graph
            self.core.knowledge_graph.graph = self.storage.graph_db.graph
            
            self.console.print(Panel(
                f"[green]Graph loaded successfully![/green]\n"
                f"Version: {result['version']}\n"
                f"Nodes: {result['nodes']}\n"
                f"Edges: {result['edges']}",
                border_style="green"
            ))
        except Exception as e:
            self.console.print(f"[red]Failed to load graph: {str(e)}[/red]")
    
    def cmd_query_graph(self, args: str):
        """Query the knowledge graph."""
        query = args.strip()
        
        if not query:
            self.console.print("[yellow]Query syntax:[/yellow]")
            self.console.print(self.storage.query_engine.explain_query_syntax())
            return
        
        results = self.storage.query(query)
        
        if not results:
            self.console.print("[yellow]No results found.[/yellow]")
            return
        
        # Display results based on type
        for result in results[:10]:  # Limit to 10 results
            result_type = result.get('type', 'unknown')
            
            if result_type == 'node':
                self.console.print(f"\n[cyan]Node:[/cyan] {result['id']}")
                self.console.print(f"  Degree: {result['degree']}")
                if result.get('neighbors'):
                    self.console.print(f"  Neighbors: {', '.join(result['neighbors'])}")
            
            elif result_type == 'path':
                self.console.print(f"\n[cyan]Path {result.get('path_id', '')}:[/cyan]")
                self.console.print(f"  {' → '.join(result['nodes'])}")
                self.console.print(f"  Length: {result['length']}, Weight: {result.get('total_weight', 'N/A')}")
            
            elif result_type == 'neighbors':
                self.console.print(f"\n[cyan]Neighbors of {result['node']}:[/cyan]")
                for neighbor in result['neighbors'][:10]:
                    if isinstance(neighbor, dict):
                        self.console.print(f"  - {neighbor['node']} (weight: {neighbor['weight']:.3f})")
                    else:
                        self.console.print(f"  - {neighbor}")
            
            elif result_type == 'statistics':
                self.console.print("\n[cyan]Graph Statistics:[/cyan]")
                stats = result.get('basic_stats', {})
                for key, value in stats.items():
                    self.console.print(f"  {key}: {value}")
            
            elif result_type == 'error':
                self.console.print(f"[red]Error: {result.get('message', 'Unknown error')}[/red]")
            
            else:
                # Generic display
                self.console.print(f"\n[cyan]{result_type}:[/cyan]")
                for key, value in result.items():
                    if key != 'type' and not isinstance(value, (dict, list)):
                        self.console.print(f"  {key}: {value}")
    
    def cmd_update_graph(self, args: str):
        """Update the graph (add/remove nodes/edges)."""
        if not args:
            self.console.print("[yellow]Update syntax:[/yellow]")
            self.console.print("  add node <name> [attributes]")
            self.console.print("  add edge <source> <target> [weight]")
            self.console.print("  remove node <name>")
            self.console.print("  remove edge <source> <target>")
            return
        
        parts = args.split()
        if len(parts) < 3:
            self.console.print("[red]Invalid update command.[/red]")
            return
        
        action = parts[0]
        entity = parts[1]
        
        updates = {
            "nodes_to_add": [],
            "nodes_to_remove": [],
            "edges_to_add": [],
            "edges_to_remove": []
        }
        
        if action == "add" and entity == "node":
            node_id = parts[2]
            self.storage.graph_db.add_node(node_id)
            self.console.print(f"[green]Added node: {node_id}[/green]")
        
        elif action == "add" and entity == "edge":
            if len(parts) >= 4:
                source, target = parts[2], parts[3]
                weight = float(parts[4]) if len(parts) > 4 else 1.0
                self.storage.graph_db.add_edge(source, target, weight=weight)
                self.console.print(f"[green]Added edge: {source} → {target} (weight: {weight})[/green]")
        
        elif action == "remove" and entity == "node":
            node_id = parts[2]
            self.storage.graph_db.remove_node(node_id)
            self.console.print(f"[yellow]Removed node: {node_id}[/yellow]")
        
        elif action == "remove" and entity == "edge":
            if len(parts) >= 4:
                source, target = parts[2], parts[3]
                self.storage.graph_db.remove_edge(source, target)
                self.console.print(f"[yellow]Removed edge: {source} → {target}[/yellow]")
    
    def cmd_export_graph(self, args: str):
        """Export graph to file."""
        parts = args.strip().split()
        format_type = parts[0] if parts else "text"
        
        valid_formats = ["json", "graphml", "text", "adjacency", "edgelist", "pickle"]
        if format_type not in valid_formats:
            self.console.print(f"[red]Invalid format. Choose from: {', '.join(valid_formats)}[/red]")
            return
        
        try:
            path = self.storage.export(format_type)
            self.console.print(f"[green]Exported to: {path}[/green]")
        except Exception as e:
            self.console.print(f"[red]Export failed: {str(e)}[/red]")
    
    def cmd_import_graph(self, args: str):
        """Import graph from file."""
        if not args:
            self.console.print("[red]Please specify a file path.[/red]")
            return
        
        from pathlib import Path
        file_path = Path(args.strip())
        
        if not file_path.exists():
            self.console.print(f"[red]File not found: {file_path}[/red]")
            return
        
        try:
            result = self.storage.import_graph(file_path)
            self.console.print(Panel(
                f"[green]Graph imported successfully![/green]\n"
                f"From: {result['imported_from']}\n"
                f"Format: {result['format']}\n"
                f"Nodes: {result['nodes']}\n"
                f"Edges: {result['edges']}\n"
                f"Saved as: {result['saved_as']}",
                border_style="green"
            ))
        except Exception as e:
            self.console.print(f"[red]Import failed: {str(e)}[/red]")
    
    def cmd_backup(self, args: str):
        """Create a backup of the storage system."""
        try:
            backup_path = self.storage.backup()
            self.console.print(Panel(
                f"[green]Backup created successfully![/green]\n"
                f"Location: {backup_path}",
                border_style="green"
            ))
        except Exception as e:
            self.console.print(f"[red]Backup failed: {str(e)}[/red]")
    
    def cmd_restore(self, args: str):
        """Restore from a backup."""
        if not args:
            self.console.print("[red]Please specify backup path.[/red]")
            return
        
        from pathlib import Path
        backup_path = Path(args.strip())
        
        confirm = self.console.input(
            f"[yellow]Restore from {backup_path}? Current data will be backed up first. (y/n): [/yellow]"
        )
        
        if confirm.lower() == 'y':
            try:
                result = self.storage.restore(backup_path)
                self.console.print(Panel(
                    f"[green]Restored successfully![/green]\n"
                    f"From: {result['restored_from']}\n"
                    f"Backup of current data: {result['restore_backup']}",
                    border_style="green"
                ))
            except Exception as e:
                self.console.print(f"[red]Restore failed: {str(e)}[/red]")
    
    def cmd_storage_info(self, args: str):
        """Show storage system information."""
        info = self.storage.get_info()
        
        # Basic info
        table = Table(title="Storage Information", show_header=True)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Base Directory", info["base_directory"])
        table.add_row("Current Version", info.get("current_version", "None"))
        table.add_row("Total Size", f"{info['storage_stats']['total_size_mb']} MB")
        table.add_row("File Count", str(info['storage_stats']['file_count']))
        
        # Graph stats
        graph_stats = info.get("graph_stats", {})
        table.add_row("Graph Nodes", str(graph_stats.get("nodes", 0)))
        table.add_row("Graph Edges", str(graph_stats.get("edges", 0)))
        table.add_row("Graph Density", f"{graph_stats.get('density', 0):.4f}")
        
        self.console.print(table)
        
        # Storage breakdown
        if info["storage_stats"].get("size_by_directory"):
            self.console.print("\n[cyan]Storage Breakdown:[/cyan]")
            for dir_name, size in info["storage_stats"]["size_by_directory"].items():
                size_mb = round(size / (1024 * 1024), 2)
                self.console.print(f"  {dir_name}: {size_mb} MB")
    
    def cmd_list_versions(self, args: str):
        """List saved graph versions."""
        versions = self.storage.graph_db.list_versions()
        
        if not versions:
            self.console.print("[yellow]No saved versions found.[/yellow]")
            return
        
        table = Table(title="Saved Versions", show_header=True)
        table.add_column("Version", style="cyan")
        table.add_column("Timestamp", style="yellow")
        table.add_column("Nodes", style="green")
        table.add_column("Edges", style="green")
        table.add_column("Description", style="white")
        
        for v in versions[:20]:  # Show last 20
            table.add_row(
                v["version_id"],
                v["timestamp"][:19],  # Trim microseconds
                str(v["node_count"]),
                str(v["edge_count"]),
                v["description"][:40] + "..." if len(v["description"]) > 40 else v["description"]
            )
        
        self.console.print(table)
        
        if len(versions) > 20:
            self.console.print(f"[dim]... and {len(versions) - 20} more versions[/dim]")
    
    def _file_browser(self) -> Optional[str]:
        """Interactive file browser with arrow navigation."""
        # Find all .pt files in current directory and subdirectories
        files = []
        for ext in ['*.pt', '*.pth']:
            files.extend(glob.glob(f"**/{ext}", recursive=True))
            files.extend(glob.glob(ext))
        
        # Remove duplicates and sort by modification time
        files = sorted(set(files), key=lambda x: os.path.getmtime(x), reverse=True)
        
        if not files:
            self.console.print("[yellow]No saved state files found in current directory.[/yellow]")
            return None
        
        # Add file info
        file_info = []
        for f in files:
            stat = os.stat(f)
            size_mb = stat.st_size / (1024 * 1024)
            mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            file_info.append({
                'path': f,
                'size': size_mb,
                'modified': mtime,
                'name': os.path.basename(f)
            })
        
        # Interactive selection
        selected = 0
        
        while True:
            # Clear and show file list
            self.console.clear()
            self.console.print("[bold cyan]Select a file to load:[/bold cyan]")
            self.console.print("[dim]Use ↑/↓ arrows to navigate, Enter to select, q to cancel[/dim]\n")
            
            # Display files
            for i, info in enumerate(file_info):
                if i == selected:
                    style = "[bold white on blue]"
                    end_style = "[/bold white on blue]"
                else:
                    style = ""
                    end_style = ""
                
                self.console.print(
                    f"{style}{i+1:2d}. {info['name']:40s} {info['size']:6.1f}MB  {info['modified']}{end_style}"
                )
            
            # Get user input
            key = self._get_key()
            
            if key == 'q':
                return None
            elif key == '\r':  # Enter
                return file_info[selected]['path']
            elif key == '\x1b[A':  # Up arrow
                selected = max(0, selected - 1)
            elif key == '\x1b[B':  # Down arrow
                selected = min(len(file_info) - 1, selected + 1)
            elif key.isdigit() and 1 <= int(key) <= len(file_info):
                # Direct number selection
                return file_info[int(key) - 1]['path']
    
    def _get_key(self) -> str:
        """Get a single keypress."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            key = sys.stdin.read(1)
            # Check for escape sequences
            if key == '\x1b':
                key += sys.stdin.read(2)
            return key
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


@click.command()
@click.option('--verbose', '-v', is_flag=True, help='Start in verbose mode')
@click.option('--load', '-l', help='Load state from file on startup')
def main(verbose, load):
    """ISC AI System - Interactive command line interface."""
    interface = ISCCommandInterface()
    
    if verbose:
        interface.verbose = True
    
    if load:
        interface.cmd_load(load)
    
    try:
        interface.run()
    except KeyboardInterrupt:
        interface.console.print("\n[cyan]Goodbye![/cyan]")
        sys.exit(0)


if __name__ == "__main__":
    main()