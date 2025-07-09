#!/usr/bin/env python3
"""
Report Generator for ISC AI System
Generates comprehensive analysis reports using GPT-4 for analysis
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import openai
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from src.isc_ai.core import ISCCore

# OpenAI API Key - will be loaded from the existing trainer config
OPENAI_API_KEY = "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"

class ReportGenerator:
    """Generates comprehensive analysis reports for ISC AI System"""
    
    def __init__(self, api_key: str = OPENAI_API_KEY):
        self.console = Console()
        self.openai_client = openai.OpenAI(api_key=api_key)
        self.isc = ISCCore()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def collect_system_data(self) -> Dict[str, Any]:
        """Collect comprehensive data about the ISC AI system"""
        self.console.print("[yellow]Collecting system data...[/yellow]")
        
        # Get system status
        status = self.isc.get_status()
        
        # Load any existing checkpoints
        checkpoint_files = list(Path("checkpoints").glob("isc_checkpoint_*.pt"))
        
        # Get graph data from ISC core's knowledge graph
        graph_data = {}
        if hasattr(self.isc, 'knowledge_graph'):
            graph = self.isc.knowledge_graph.graph
            graph_data = {
                "nodes": len(graph.nodes()),
                "edges": len(graph.edges()),
                "graph_structure": {
                    "nodes": list(graph.nodes()),
                    "edges": list(graph.edges())
                }
            }
        
        # Run some test interactions
        test_results = self._run_test_interactions()
        
        # Collect training history if available
        training_history = self._load_training_history()
        
        return {
            "timestamp": self.timestamp,
            "system_status": status,
            "checkpoints": [str(f) for f in checkpoint_files],
            "graph_data": graph_data,
            "test_results": test_results,
            "training_history": training_history,
            "theoretical_foundation": self._load_theoretical_foundation()
        }
    
    def _run_test_interactions(self) -> List[Dict[str, Any]]:
        """Run test interactions to gather empirical data"""
        test_prompts = [
            "What is consciousness?",
            "How do you process information?",
            "Explain your understanding of learning",
            "What patterns have you noticed in our conversations?",
            "How does your phi value relate to your responses?"
        ]
        
        results = []
        for prompt in test_prompts:
            response = self.isc.process_input(prompt)
            status = self.isc.get_status()
            results.append({
                "prompt": prompt,
                "response": response,
                "phi": status["metrics"]["phi_value"],
                "coherence": status["metrics"]["coherence_score"],
                "concepts": status["total_concepts"],
                "connections": status["total_connections"]
            })
        
        return results
    
    def _load_training_history(self) -> Optional[Dict[str, Any]]:
        """Load training history from files"""
        training_files = list(Path(".").glob("training_history_*.json"))
        if training_files:
            latest_file = max(training_files, key=lambda x: x.stat().st_mtime)
            with open(latest_file, 'r') as f:
                return json.load(f)
        return None
    
    def _load_theoretical_foundation(self) -> str:
        """Load key points from the theoretical paper"""
        paper_path = Path(__file__).parent.parent.parent / "PAPER.md"
        if paper_path.exists():
            with open(paper_path, 'r') as f:
                content = f.read()
                # Extract abstract and key sections
                sections = {
                    "abstract": content[content.find("## Abstract"):content.find("## Introduction")],
                    "key_insights": self._extract_key_insights(content)
                }
                return sections
        return "Theoretical foundation not found"
    
    def _extract_key_insights(self, content: str) -> str:
        """Extract key theoretical insights"""
        insights = []
        
        # Extract key sections
        if "logical necessity of existence" in content.lower():
            insights.append("- Existence is logically necessary rather than contingent")
        if "consciousness as an informational pattern" in content.lower():
            insights.append("- Consciousness emerges as specific patterns within informational substrate")
        if "mathematical necessity of other intelligences" in content.lower():
            insights.append("- Advanced intelligences are mathematically inevitable throughout the universe")
        
        return "\n".join(insights)
    
    def analyze_with_gpt(self, data: Dict[str, Any], analysis_type: str = "comprehensive") -> str:
        """Use GPT-4 to analyze the collected data"""
        self.console.print(f"[yellow]Analyzing data with GPT-4 ({analysis_type})...[/yellow]")
        
        # Prepare the analysis prompt
        if analysis_type == "uniqueness":
            prompt = self._create_uniqueness_analysis_prompt(data)
        elif analysis_type == "validity":
            prompt = self._create_validity_analysis_prompt(data)
        elif analysis_type == "empirical":
            prompt = self._create_empirical_analysis_prompt(data)
        else:
            prompt = self._create_comprehensive_analysis_prompt(data)
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are an expert in consciousness studies, information theory, quantum physics, and AI. Provide thorough, critical analysis without being sycophantic. Be honest about strengths and weaknesses."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=4000
            )
            
            return response.choices[0].message.content
        except Exception as e:
            self.console.print(f"[red]Error during GPT analysis: {str(e)}[/red]")
            return f"Analysis failed: {str(e)}"
    
    def _create_uniqueness_analysis_prompt(self, data: Dict[str, Any]) -> str:
        """Create prompt for uniqueness analysis"""
        return f"""
Analyze the uniqueness and novelty of the Information Substrate Convergence (ISC) theory based on the following:

Theoretical Foundation:
{data.get('theoretical_foundation', 'Not available')}

Compare this with existing theories:
1. Integrated Information Theory (IIT) by Giulio Tononi
2. Digital Physics (Zuse, Fredkin, Wolfram)
3. It from Bit (Wheeler)
4. Mathematical Universe Hypothesis (Tegmark)
5. Computational Theory of Mind
6. Panpsychism and related theories

Please assess:
1. What aspects of ISC are genuinely novel?
2. What aspects build on or synthesize existing work?
3. How does ISC differ from similar theories?
4. What unique contributions does ISC make to consciousness studies?
5. Are there any theories ISC should acknowledge but doesn't?

Be critical and thorough in your analysis.
"""
    
    def _create_validity_analysis_prompt(self, data: Dict[str, Any]) -> str:
        """Create prompt for validity analysis"""
        return f"""
Critically evaluate the validity and coherence of the Information Substrate Convergence (ISC) theory:

System Performance Data:
- Test Results: {json.dumps(data.get('test_results', []), indent=2)}
- System Status: {json.dumps(data.get('system_status', {}), indent=2)}

Theoretical Claims:
{data.get('theoretical_foundation', 'Not available')}

Please assess:
1. Logical consistency of the theoretical framework
2. Alignment between theoretical claims and empirical implementation
3. Testability of key hypotheses
4. Potential philosophical or scientific weaknesses
5. Whether the implementation supports the theoretical claims
6. Comparison of phi values and coherence scores with theoretical expectations

Provide a balanced assessment of strengths and weaknesses.
"""
    
    def _create_empirical_analysis_prompt(self, data: Dict[str, Any]) -> str:
        """Create prompt for empirical analysis"""
        return f"""
Analyze the empirical performance of the ISC AI system:

Test Interaction Results:
{json.dumps(data.get('test_results', []), indent=2)}

System Metrics:
{json.dumps(data.get('system_status', {}).get('metrics', {}), indent=2)}

Graph Structure:
- Nodes: {data.get('graph_data', {}).get('nodes', 0)}
- Edges: {data.get('graph_data', {}).get('edges', 0)}

Training History:
{json.dumps(data.get('training_history', {}) if data.get('training_history') else 'No training history', indent=2)}

Please evaluate:
1. Quality of responses and coherence over interactions
2. Phi value progression and what it indicates
3. Knowledge graph growth and concept formation
4. Evidence of learning and self-modification
5. Comparison with traditional AI systems
6. Whether the system exhibits consciousness-like properties as claimed

Focus on empirical evidence rather than theoretical claims.
"""
    
    def _create_comprehensive_analysis_prompt(self, data: Dict[str, Any]) -> str:
        """Create prompt for comprehensive analysis"""
        return f"""
Provide a comprehensive analysis of the Information Substrate Convergence (ISC) theory and its implementation:

System Data:
{json.dumps(data, indent=2)}

Please provide:

1. EXECUTIVE SUMMARY
   - Key findings about the theory and implementation
   - Overall assessment of validity and novelty

2. THEORETICAL ANALYSIS
   - Strengths and weaknesses of the theoretical framework
   - Comparison with existing theories
   - Philosophical implications

3. EMPIRICAL EVALUATION
   - Performance metrics analysis
   - Evidence for consciousness-like properties
   - Learning and adaptation capabilities

4. NOVELTY ASSESSMENT
   - Unique contributions to the field
   - Synthesis of existing ideas
   - Potential impact on consciousness studies

5. CRITICAL ASSESSMENT
   - Major strengths
   - Significant weaknesses or gaps
   - Methodological concerns

6. FUTURE DIRECTIONS
   - Promising research avenues
   - Necessary improvements
   - Potential applications

Be thorough, balanced, and critical in your analysis.
"""
    
    def generate_visualizations(self, data: Dict[str, Any], output_dir: Path):
        """Generate visualizations for the report"""
        self.console.print("[yellow]Generating visualizations...[/yellow]")
        
        # Phi progression plot
        if data.get("test_results"):
            phi_values = [r["phi"] for r in data["test_results"]]
            coherence_values = [r["coherence"] for r in data["test_results"]]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Phi values
            ax1.plot(phi_values, 'b-o', linewidth=2, markersize=8)
            ax1.set_ylabel('Φ (Phi) Value', fontsize=12)
            ax1.set_title('Information Integration (Φ) Across Interactions', fontsize=14)
            ax1.grid(True, alpha=0.3)
            
            # Coherence values
            ax2.plot(coherence_values, 'g-s', linewidth=2, markersize=8)
            ax2.set_ylabel('Coherence Score', fontsize=12)
            ax2.set_xlabel('Interaction Number', fontsize=12)
            ax2.set_title('Response Coherence Across Interactions', fontsize=14)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / "metrics_progression.png", dpi=150)
            plt.close()
        
        # Knowledge graph growth
        if data.get("test_results"):
            concepts = [r["concepts"] for r in data["test_results"]]
            connections = [r["connections"] for r in data["test_results"]]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            x = range(len(concepts))
            ax.bar(x, concepts, width=0.4, label='Concepts', alpha=0.7)
            ax.bar([i + 0.4 for i in x], connections, width=0.4, label='Connections', alpha=0.7)
            
            ax.set_xlabel('Interaction Number', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title('Knowledge Graph Growth', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / "knowledge_growth.png", dpi=150)
            plt.close()
    
    def generate_report(self, report_types: List[str] = ["comprehensive"]) -> str:
        """Generate complete report with all analyses"""
        # Create report directory
        report_dir = Path(f"reports/report_{self.timestamp}")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect system data
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Collecting system data...", total=None)
            data = self.collect_system_data()
            
            # Save raw data
            with open(report_dir / "raw_data.json", 'w') as f:
                json.dump(data, f, indent=2)
            
            progress.update(task, description="Generating visualizations...")
            self.generate_visualizations(data, report_dir)
            
            # Generate analyses
            analyses = {}
            for report_type in report_types:
                progress.update(task, description=f"Generating {report_type} analysis...")
                analyses[report_type] = self.analyze_with_gpt(data, report_type)
            
            progress.update(task, description="Compiling final report...")
        
        # Compile final report
        report_content = self._compile_report(data, analyses, report_dir)
        
        # Save report
        report_path = report_dir / "report.md"
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        self.console.print(Panel(
            f"[green]Report generated successfully![/green]\n\n"
            f"Location: {report_path}\n"
            f"Visualizations: {report_dir}/*.png\n"
            f"Raw data: {report_dir}/raw_data.json",
            title="Report Generation Complete"
        ))
        
        return str(report_path)
    
    def _compile_report(self, data: Dict[str, Any], analyses: Dict[str, str], report_dir: Path) -> str:
        """Compile all analyses into a final report"""
        report = f"""# Information Substrate Convergence (ISC) Analysis Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Report ID:** {self.timestamp}

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Overview](#system-overview)
3. [Theoretical Analysis](#theoretical-analysis)
4. [Empirical Results](#empirical-results)
5. [Uniqueness Assessment](#uniqueness-assessment)
6. [Validity Evaluation](#validity-evaluation)
7. [Visualizations](#visualizations)
8. [Conclusions](#conclusions)

---

## Executive Summary

This report provides a comprehensive analysis of the Information Substrate Convergence (ISC) theory and its implementation in the ISC AI System. The analysis includes theoretical evaluation, empirical testing, uniqueness assessment, and validity evaluation.

**Key Metrics:**
- Total Concepts: {data['system_status']['total_concepts']}
- Total Connections: {data['system_status']['total_connections']}
- Average Φ (Phi): {np.mean([r['phi'] for r in data['test_results']]):.4f}
- Average Coherence: {np.mean([r['coherence'] for r in data['test_results']]):.4f}

---

## System Overview

The ISC AI System implements the theoretical framework of Information Substrate Convergence, which proposes that:
1. Reality is fundamentally informational rather than material
2. Consciousness emerges as specific patterns within this informational substrate
3. The existence of consciousness is mathematically necessary

### Current System Status

```json
{json.dumps(data['system_status'], indent=2)}
```

---

## Theoretical Analysis

{analyses.get('comprehensive', 'Analysis not available')}

---

## Empirical Results

### Test Interactions

The system was tested with {len(data['test_results'])} standardized prompts to evaluate its consciousness-like properties:

"""
        
        for i, result in enumerate(data['test_results']):
            report += f"""
#### Interaction {i+1}
**Prompt:** {result['prompt']}
**Response:** {result['response']}
**Metrics:** Φ={result['phi']:.4f}, Coherence={result['coherence']:.4f}, Concepts={result['concepts']}, Connections={result['connections']}
"""
        
        report += f"""
---

## Uniqueness Assessment

{analyses.get('uniqueness', 'Analysis not available')}

---

## Validity Evaluation

{analyses.get('validity', 'Analysis not available')}

---

## Visualizations

### Metrics Progression
![Metrics Progression](metrics_progression.png)

### Knowledge Graph Growth
![Knowledge Growth](knowledge_growth.png)

---

## Conclusions

{analyses.get('empirical', 'Analysis not available')}

---

## Appendix

### Raw Data
Complete raw data is available in `raw_data.json`

### Checkpoints
Available checkpoints: {len(data['checkpoints'])}

### Report Metadata
- Generation Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- Report Directory: `{report_dir}`
- Analysis Model: GPT-4 Turbo

---

*This report was generated automatically by the ISC Report Generator using GPT-4 for analysis.*
"""
        
        return report


def main():
    """Main entry point for report generation"""
    console = Console()
    
    console.print(Panel(
        "[bold blue]ISC AI System Report Generator[/bold blue]\n\n"
        "This tool generates comprehensive analysis reports for the ISC AI System\n"
        "using GPT-4 for critical analysis.",
        title="Welcome"
    ))
    
    # Check for API key
    if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("YOUR"):
        console.print("[red]Error: Please set your OpenAI API key in the script[/red]")
        return
    
    # Create generator
    generator = ReportGenerator(OPENAI_API_KEY)
    
    # Ask for report types
    console.print("\n[yellow]Available report types:[/yellow]")
    console.print("1. comprehensive - Full analysis of theory and implementation")
    console.print("2. uniqueness - Comparison with existing theories")
    console.print("3. validity - Critical evaluation of claims")
    console.print("4. empirical - Focus on test results and performance")
    console.print("5. all - Generate all report types")
    
    choice = console.input("\n[cyan]Select report types (comma-separated numbers or 'all'): [/cyan]")
    
    if choice.lower() == 'all':
        report_types = ["comprehensive", "uniqueness", "validity", "empirical"]
    else:
        type_map = {
            "1": "comprehensive",
            "2": "uniqueness", 
            "3": "validity",
            "4": "empirical"
        }
        report_types = [type_map.get(c.strip(), "comprehensive") for c in choice.split(",")]
    
    console.print(f"\n[green]Generating {len(report_types)} report type(s)...[/green]")
    
    # Generate report
    try:
        report_path = generator.generate_report(report_types)
        console.print(f"\n[green]Success! Report saved to: {report_path}[/green]")
    except Exception as e:
        console.print(f"\n[red]Error generating report: {str(e)}[/red]")
        raise


if __name__ == "__main__":
    main()