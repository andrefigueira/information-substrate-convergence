#!/usr/bin/env python3
"""
ISC Lite - Fast terminal interface using the reasoning engine

Uses the lightweight ImprovedEmergentReasoner instead of the full
NeuromorphicSubstrate for faster startup and response times.
"""

import sys
import time
import os
from pathlib import Path

# Suppress all warnings before any imports
import warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

console = Console()


def print_banner():
    banner = """
[bold #E67E22]╭─────────────────────────────────────────────────────────────╮[/]
[bold #E67E22]│[/]  [bold white]ISC Lite[/] [dim]Fast Reasoning Mode[/]                             [bold #E67E22]│[/]
[bold #E67E22]╰─────────────────────────────────────────────────────────────╯[/]
"""
    console.print(banner)


def create_response_panel(response: str, phi: float, time_ms: int, confidence: float) -> Panel:
    content = Text(response)
    conf_color = "#2ECC71" if confidence > 0.7 else "#F39C12" if confidence > 0.4 else "#E74C3C"
    subtitle = f"[dim]{time_ms}ms[/] · [#1ABC9C]φ={phi:.3f}[/] · [{conf_color}]{confidence:.0%} conf[/]"

    return Panel(
        content,
        title="[bold #9B59B6]ISC[/]",
        title_align="left",
        subtitle=subtitle,
        subtitle_align="right",
        border_style="#9B59B6",
        padding=(0, 1),
    )


def create_user_panel(message: str) -> Panel:
    return Panel(
        Text(message),
        title="[bold #3498DB]You[/]",
        title_align="left",
        border_style="#3498DB",
        padding=(0, 1),
    )


def print_help():
    help_text = """
[bold]Commands:[/]
  [#E67E22]/status[/]    Show system status
  [#E67E22]/train[/]     Train on example problems
  [#E67E22]/clear[/]     Clear the screen
  [#E67E22]/help[/]      Show this help
  [#E67E22]/quit[/]      Exit

[bold]Try asking:[/]
  • All cats are mammals. Felix is a cat. Is Felix a mammal?
  • If it rains, the ground is wet. It rained. Is the ground wet?
  • No fish can fly. Nemo is a fish. Can Nemo fly?
"""
    console.print(Panel(help_text, title="[bold]Help[/]", border_style="dim"))


def print_status(substrate, reasoner):
    stats = {
        'phi': substrate.global_phi,
        'nodes': len(substrate.nodes),
        'emergent': len([n for n in substrate.nodes if n.startswith('emergent')]),
        'traces': len(reasoner.experience_buffer),
    }

    table = Table(title="System Status", box=box.ROUNDED, border_style="#E67E22")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Phi (φ)", f"[#1ABC9C]{stats['phi']:.4f}[/]")
    table.add_row("Total Nodes", str(stats['nodes']))
    table.add_row("Emergent Nodes", f"[#9B59B6]{stats['emergent']}[/]")
    table.add_row("Experience", str(stats['traces']))

    console.print(table)


def train_examples(reasoner):
    """Train on some example problems"""
    examples = [
        ('All cats are mammals. Felix is a cat.', 'Is Felix a mammal?', 'yes'),
        ('All dogs bark. Rover is a dog.', 'Does Rover bark?', 'yes'),
        ('No fish can walk. Nemo is a fish.', 'Can Nemo walk?', 'no'),
        ('If it rains, the ground is wet. It rained.', 'Is the ground wet?', 'yes'),
        ('All birds have feathers. Tweety is a bird.', 'Does Tweety have feathers?', 'yes'),
    ]

    console.print("[dim]Training on examples...[/]")

    for premise, question, expected in examples:
        result = reasoner.reason(premise, question)
        reasoner.learn_from_feedback(premise, question, expected, result['answer'], True)

    console.print(f"[green]Trained on {len(examples)} examples[/]")


def main():
    print_banner()

    # Fast initialization
    with console.status("[bold #E67E22]Initializing...", spinner="dots"):
        from src.isc.improved_emergent_reasoning import (
            ImprovedPhiDrivenSubstrate,
            ImprovedEmergentReasoner
        )

        import io
        import contextlib

        with contextlib.redirect_stdout(io.StringIO()):
            substrate = ImprovedPhiDrivenSubstrate(initial_nodes=100)
            reasoner = ImprovedEmergentReasoner(substrate)

    # Show initial status
    emergent = len([n for n in substrate.nodes if n.startswith('emergent')])

    init_info = Table(show_header=False, box=None, padding=(0, 2))
    init_info.add_column()
    init_info.add_column()
    init_info.add_column()
    init_info.add_row(
        f"[#1ABC9C]φ={substrate.global_phi:.3f}[/]",
        f"[dim]{len(substrate.nodes)} nodes[/]",
        f"[#9B59B6]{emergent} emergent[/]"
    )
    console.print(init_info)
    console.print()
    console.print("[dim]Type [bold]/help[/] for commands, [bold]/train[/] to learn examples[/]")
    console.print()

    # Chat loop
    while True:
        try:
            console.print("[bold #3498DB]>[/] ", end="")
            user_input = input().strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith('/'):
                cmd = user_input.lower()

                if cmd in ['/quit', '/exit', '/q']:
                    console.print("\n[dim]Goodbye![/]")
                    break
                elif cmd == '/help':
                    print_help()
                    continue
                elif cmd in ['/status', '/stats']:
                    print_status(substrate, reasoner)
                    continue
                elif cmd == '/train':
                    train_examples(reasoner)
                    continue
                elif cmd == '/clear':
                    console.clear()
                    print_banner()
                    continue
                else:
                    console.print(f"[yellow]Unknown command: {cmd}[/]")
                    continue

            # Parse input - try to split into premise and question
            if '?' in user_input:
                parts = user_input.rsplit('?', 1)
                if len(parts) == 2 and parts[1].strip() == '':
                    # Find last sentence boundary before the question
                    premise_part = parts[0]
                    for sep in ['. ', '! ']:
                        if sep in premise_part:
                            idx = premise_part.rfind(sep)
                            premise = premise_part[:idx+1].strip()
                            question = premise_part[idx+2:].strip() + '?'
                            break
                    else:
                        premise = premise_part.strip()
                        question = "Is this true?"
                else:
                    premise = user_input
                    question = "What do you think?"
            else:
                premise = user_input
                question = "What do you think?"

            # Show user message
            console.print(create_user_panel(user_input))

            # Process
            start_time = time.time()

            with console.status("[#9B59B6]Reasoning...", spinner="dots"):
                import io
                import contextlib
                with contextlib.redirect_stdout(io.StringIO()):
                    result = reasoner.reason(premise, question)

            elapsed_ms = int((time.time() - start_time) * 1000)

            # Format response
            answer = result.get('answer', 'uncertain')
            confidence = result.get('confidence', 0.5)
            phi = substrate.global_phi

            # Make response more conversational
            if answer == 'yes':
                response = f"Yes, that's correct."
            elif answer == 'no':
                response = f"No, that's not the case."
            elif answer == 'uncertain':
                response = "I'm not certain about that."
            else:
                response = str(answer)

            # Add reasoning context
            emergent_used = result.get('emergent_nodes_used', 0)
            if emergent_used > 0:
                response += f" (Used {emergent_used} emergent patterns)"

            console.print(create_response_panel(response, phi, elapsed_ms, confidence))
            console.print()

        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted. Type /quit to exit.[/]")
        except EOFError:
            console.print("\n[dim]Goodbye![/]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/]")


if __name__ == "__main__":
    main()
