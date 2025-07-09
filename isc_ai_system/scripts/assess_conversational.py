#!/usr/bin/env python3
"""
ISC Conversational Model Assessment Tool
Comprehensive evaluation and reporting for conversational ISC models
"""

import os
import sys
import json
import glob
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import shutil
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
from dataclasses import dataclass, asdict
import openai
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.markdown import Markdown
import pandas as pd
from collections import defaultdict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from scripts.conversational import ConversationalISC, ConversationalLMHead
try:
    from src.isc_ai.core import ISCCore
except ImportError:
    # Mock ISCCore for testing
    class ISCCore:
        pass

# Get API key from environment
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")

@dataclass
class ConversationMetrics:
    """Metrics for a single conversation exchange"""
    exchange_id: int
    human_input: str
    model_response: str
    response_time: float
    
    # Quality metrics (0-10 scale)
    coherence: float
    relevance: float
    naturalness: float
    engagement: float
    completeness: float
    
    # Linguistic metrics
    response_length: int
    vocabulary_diversity: float
    sentence_complexity: float
    
    # Conversational flow
    topic_consistency: float
    context_utilization: float
    
    # ISC-specific metrics
    philosophical_depth: float
    conceptual_integration: float
    
    def overall_score(self) -> float:
        """Calculate weighted overall score"""
        weights = {
            'coherence': 0.20,
            'relevance': 0.20,
            'naturalness': 0.15,
            'engagement': 0.15,
            'completeness': 0.10,
            'topic_consistency': 0.10,
            'philosophical_depth': 0.10
        }
        
        score = (
            weights['coherence'] * self.coherence +
            weights['relevance'] * self.relevance +
            weights['naturalness'] * self.naturalness +
            weights['engagement'] * self.engagement +
            weights['completeness'] * self.completeness +
            weights['topic_consistency'] * self.topic_consistency +
            weights['philosophical_depth'] * self.philosophical_depth
        )
        return score


@dataclass
class ModelAssessment:
    """Complete assessment of a conversational model"""
    model_path: str
    assessment_date: str
    total_exchanges: int
    
    # Aggregate metrics
    avg_coherence: float
    avg_relevance: float
    avg_naturalness: float
    avg_engagement: float
    avg_completeness: float
    avg_response_time: float
    avg_response_length: float
    
    # Performance trends
    coherence_trend: float  # Positive = improving
    relevance_trend: float
    naturalness_trend: float
    
    # Conversation quality
    topic_diversity: float
    context_consistency: float
    philosophical_sophistication: float
    
    # Comparison metrics (if baseline exists)
    improvement_from_baseline: Optional[float] = None
    strengths: List[str] = None
    weaknesses: List[str] = None
    
    # LLM assessment
    llm_summary: str = ""
    llm_recommendations: List[str] = None


class ConversationalAssessor:
    """Comprehensive assessment tool for conversational ISC models"""
    
    def __init__(self, api_key: str = None):
        self.console = Console()
        self.api_key = api_key or OPENAI_API_KEY
        self.openai_client = openai.OpenAI(api_key=self.api_key)
        
        # Test conversations for evaluation
        self.test_conversations = [
            # Basic interactions
            ("Hello! How are you today?", "greeting"),
            ("What do you think about consciousness?", "philosophical"),
            ("Can you explain quantum mechanics?", "educational"),
            ("I'm feeling a bit overwhelmed lately.", "emotional"),
            
            # Complex queries
            ("How does information theory relate to consciousness?", "complex_philosophical"),
            ("What's the meaning of life from your perspective?", "existential"),
            ("Tell me about emergence in complex systems.", "technical"),
            
            # Conversational flow
            ("That's interesting. Can you elaborate?", "continuation"),
            ("I don't quite understand. Can you simplify?", "clarification"),
            ("What do you think about that?", "opinion"),
            
            # Context testing
            ("Going back to what we discussed earlier...", "context_reference"),
            ("How does this relate to consciousness?", "topic_connection"),
            
            # Edge cases
            ("", "empty_input"),
            ("asdfghjkl", "nonsense"),
            ("Why?", "minimal"),
        ]
        
        self.metrics_history = []
        self.console.print(Panel("[bold cyan]ISC Conversational Model Assessor[/bold cyan]", style="cyan"))
    
    def load_model(self, model_path: str, lm_head_path: str) -> ConversationalISC:
        """Load a conversational ISC model"""
        try:
            self.console.print(f"[cyan]Loading model: {model_path}[/cyan]")
            conv_model = ConversationalISC(model_path, lm_head_path)
            return conv_model
        except Exception as e:
            self.console.print(f"[red]Error loading model: {e}[/red]")
            return None
    
    def evaluate_response(self, human_input: str, model_response: str, 
                         conversation_type: str, context: List[Dict] = None) -> Dict:
        """Evaluate a single response using GPT-4"""
        
        eval_prompt = f"""Evaluate this AI response on multiple dimensions. Be critical but fair.

Human input: {human_input}
AI response: {model_response}
Conversation type: {conversation_type}
Previous context: {json.dumps(context[-3:]) if context else 'None'}

Rate each aspect on a scale of 0-10 and provide brief reasoning:

1. Coherence: Is the response internally consistent and well-structured?
2. Relevance: Does it appropriately address the human's input?
3. Naturalness: Does it sound like natural human conversation?
4. Engagement: Is it engaging and does it move the conversation forward?
5. Completeness: Does it fully address the query without being overly verbose?
6. Topic Consistency: Does it stay on topic or transition smoothly?
7. Context Utilization: Does it appropriately use conversation history?
8. Philosophical Depth: For an ISC model, does it show philosophical insight?
9. Conceptual Integration: Does it integrate concepts in meaningful ways?

Also provide:
- Vocabulary diversity score (0-10)
- Sentence complexity score (0-10)
- Overall impression and specific strengths/weaknesses

Format as JSON with scores and brief explanations."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": "You are an expert in conversational AI evaluation. Be thorough and objective."},
                    {"role": "user", "content": eval_prompt}
                ],
                temperature=0.3,
                max_tokens=800,
                response_format={"type": "json_object"}
            )
            
            evaluation = json.loads(response.choices[0].message.content)
            return evaluation
            
        except Exception as e:
            self.console.print(f"[red]Error in evaluation: {e}[/red]")
            # Return default scores on error
            return {
                "coherence": {"score": 5, "reason": "Error in evaluation"},
                "relevance": {"score": 5, "reason": "Error in evaluation"},
                "naturalness": {"score": 5, "reason": "Error in evaluation"},
                "engagement": {"score": 5, "reason": "Error in evaluation"},
                "completeness": {"score": 5, "reason": "Error in evaluation"},
                "topic_consistency": {"score": 5, "reason": "Error in evaluation"},
                "context_utilization": {"score": 5, "reason": "Error in evaluation"},
                "philosophical_depth": {"score": 5, "reason": "Error in evaluation"},
                "conceptual_integration": {"score": 5, "reason": "Error in evaluation"},
                "vocabulary_diversity": {"score": 5, "reason": "Error in evaluation"},
                "sentence_complexity": {"score": 5, "reason": "Error in evaluation"},
                "overall_impression": "Evaluation error occurred"
            }
    
    def run_conversation_test(self, model: ConversationalISC, 
                            test_input: str, test_type: str,
                            context: List[Dict] = None) -> ConversationMetrics:
        """Run a single conversation test and collect metrics"""
        
        # Time the response
        start_time = time.time()
        conv_response, isc_response = model.chat(test_input)
        response_time = time.time() - start_time
        
        # Evaluate the response
        evaluation = self.evaluate_response(test_input, conv_response, test_type, context)
        
        # Calculate linguistic metrics
        response_length = len(conv_response.split())
        vocabulary = set(conv_response.lower().split())
        vocabulary_diversity = len(vocabulary) / max(response_length, 1)
        
        # Create metrics object
        metrics = ConversationMetrics(
            exchange_id=len(self.metrics_history) + 1,
            human_input=test_input,
            model_response=conv_response,
            response_time=response_time,
            coherence=evaluation.get("coherence", {}).get("score", 5),
            relevance=evaluation.get("relevance", {}).get("score", 5),
            naturalness=evaluation.get("naturalness", {}).get("score", 5),
            engagement=evaluation.get("engagement", {}).get("score", 5),
            completeness=evaluation.get("completeness", {}).get("score", 5),
            response_length=response_length,
            vocabulary_diversity=vocabulary_diversity,
            sentence_complexity=evaluation.get("sentence_complexity", {}).get("score", 5),
            topic_consistency=evaluation.get("topic_consistency", {}).get("score", 5),
            context_utilization=evaluation.get("context_utilization", {}).get("score", 5),
            philosophical_depth=evaluation.get("philosophical_depth", {}).get("score", 5),
            conceptual_integration=evaluation.get("conceptual_integration", {}).get("score", 5)
        )
        
        return metrics, evaluation
    
    def assess_model(self, model_path: str, lm_head_path: str) -> ModelAssessment:
        """Run comprehensive assessment on a model"""
        
        self.console.print("\n[bold]Starting Model Assessment[/bold]")
        
        # Load model
        model = self.load_model(model_path, lm_head_path)
        if not model:
            return None
        
        # Reset metrics
        self.metrics_history = []
        all_evaluations = []
        
        # Run tests with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console
        ) as progress:
            
            task = progress.add_task("Running conversation tests...", 
                                   total=len(self.test_conversations))
            
            context = []
            for test_input, test_type in self.test_conversations:
                # Skip empty input for actual testing
                if not test_input.strip():
                    test_input = "Hello"
                
                metrics, evaluation = self.run_conversation_test(
                    model, test_input, test_type, context
                )
                
                self.metrics_history.append(metrics)
                all_evaluations.append(evaluation)
                
                # Update context
                context.append({
                    "human": test_input,
                    "assistant": metrics.model_response
                })
                
                # Keep context window limited
                if len(context) > 5:
                    context = context[-5:]
                
                progress.update(task, advance=1)
                time.sleep(0.5)  # Rate limiting
        
        # Calculate aggregate metrics
        assessment = self._calculate_assessment(model_path, self.metrics_history, all_evaluations)
        
        # Generate LLM assessment
        assessment.llm_summary, assessment.llm_recommendations = self._generate_llm_assessment(
            assessment, self.metrics_history, all_evaluations
        )
        
        return assessment
    
    def _calculate_assessment(self, model_path: str, 
                            metrics_list: List[ConversationMetrics],
                            evaluations: List[Dict]) -> ModelAssessment:
        """Calculate aggregate assessment from metrics"""
        
        if not metrics_list:
            return None
        
        # Calculate averages
        avg_coherence = np.mean([m.coherence for m in metrics_list])
        avg_relevance = np.mean([m.relevance for m in metrics_list])
        avg_naturalness = np.mean([m.naturalness for m in metrics_list])
        avg_engagement = np.mean([m.engagement for m in metrics_list])
        avg_completeness = np.mean([m.completeness for m in metrics_list])
        avg_response_time = np.mean([m.response_time for m in metrics_list])
        avg_response_length = np.mean([m.response_length for m in metrics_list])
        
        # Calculate trends (linear regression slope)
        def calculate_trend(values):
            if len(values) < 2:
                return 0
            x = np.arange(len(values))
            slope, _ = np.polyfit(x, values, 1)
            return slope
        
        coherence_trend = calculate_trend([m.coherence for m in metrics_list])
        relevance_trend = calculate_trend([m.relevance for m in metrics_list])
        naturalness_trend = calculate_trend([m.naturalness for m in metrics_list])
        
        # Calculate topic diversity (unique topics discussed)
        unique_topics = len(set(m.human_input[:20] for m in metrics_list))
        topic_diversity = unique_topics / len(metrics_list)
        
        # Context consistency (how well it maintains context)
        context_scores = [m.context_utilization for m in metrics_list]
        context_consistency = np.mean(context_scores) if context_scores else 5.0
        
        # Philosophical sophistication
        phil_scores = [m.philosophical_depth for m in metrics_list]
        philosophical_sophistication = np.mean(phil_scores) if phil_scores else 5.0
        
        # Identify strengths and weaknesses
        metric_avgs = {
            'Coherence': avg_coherence,
            'Relevance': avg_relevance,
            'Naturalness': avg_naturalness,
            'Engagement': avg_engagement,
            'Completeness': avg_completeness,
            'Context Usage': context_consistency,
            'Philosophy': philosophical_sophistication
        }
        
        sorted_metrics = sorted(metric_avgs.items(), key=lambda x: x[1], reverse=True)
        strengths = [f"{name} ({score:.1f}/10)" for name, score in sorted_metrics[:3]]
        weaknesses = [f"{name} ({score:.1f}/10)" for name, score in sorted_metrics[-3:] if score < 7]
        
        return ModelAssessment(
            model_path=model_path,
            assessment_date=datetime.now().isoformat(),
            total_exchanges=len(metrics_list),
            avg_coherence=avg_coherence,
            avg_relevance=avg_relevance,
            avg_naturalness=avg_naturalness,
            avg_engagement=avg_engagement,
            avg_completeness=avg_completeness,
            avg_response_time=avg_response_time,
            avg_response_length=avg_response_length,
            coherence_trend=coherence_trend,
            relevance_trend=relevance_trend,
            naturalness_trend=naturalness_trend,
            topic_diversity=topic_diversity,
            context_consistency=context_consistency,
            philosophical_sophistication=philosophical_sophistication,
            strengths=strengths,
            weaknesses=weaknesses
        )
    
    def _generate_llm_assessment(self, assessment: ModelAssessment, 
                               metrics_list: List[ConversationMetrics],
                               evaluations: List[Dict]) -> Tuple[str, List[str]]:
        """Generate LLM-based assessment summary and recommendations"""
        
        # Prepare assessment data
        assessment_data = {
            "aggregate_metrics": {
                "coherence": assessment.avg_coherence,
                "relevance": assessment.avg_relevance,
                "naturalness": assessment.avg_naturalness,
                "engagement": assessment.avg_engagement,
                "completeness": assessment.avg_completeness,
                "philosophical_sophistication": assessment.philosophical_sophistication
            },
            "trends": {
                "coherence_trend": assessment.coherence_trend,
                "relevance_trend": assessment.relevance_trend,
                "naturalness_trend": assessment.naturalness_trend
            },
            "performance": {
                "avg_response_time": assessment.avg_response_time,
                "avg_response_length": assessment.avg_response_length,
                "topic_diversity": assessment.topic_diversity,
                "context_consistency": assessment.context_consistency
            },
            "sample_exchanges": [
                {
                    "input": m.human_input,
                    "response": m.model_response[:200] + "...",
                    "score": m.overall_score()
                }
                for m in sorted(metrics_list, key=lambda x: x.overall_score(), reverse=True)[:3]
            ],
            "worst_exchanges": [
                {
                    "input": m.human_input,
                    "response": m.model_response[:200] + "...",
                    "score": m.overall_score(),
                    "issues": evaluations[i].get("overall_impression", "")
                }
                for i, m in enumerate(sorted(metrics_list, key=lambda x: x.overall_score())[:3])
            ]
        }
        
        assessment_prompt = f"""Based on this comprehensive assessment data for an ISC conversational AI model, provide:

1. A detailed summary (200-300 words) covering:
   - Overall performance and capabilities
   - Key strengths and standout features
   - Main areas needing improvement
   - Comparison to typical conversational AI standards
   - Assessment of philosophical depth and ISC-specific qualities

2. Specific, actionable recommendations (5-7 items) for improvement

Assessment Data:
{json.dumps(assessment_data, indent=2)}

Focus on practical insights and be specific about what makes this model unique as an ISC-based system.

Format response as JSON with 'summary' and 'recommendations' fields."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": "You are an expert in conversational AI assessment, particularly for philosophically-oriented systems."},
                    {"role": "user", "content": assessment_prompt}
                ],
                temperature=0.4,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get("summary", ""), result.get("recommendations", [])
            
        except Exception as e:
            self.console.print(f"[red]Error generating LLM assessment: {e}[/red]")
            return "Error generating assessment", ["Unable to generate recommendations"]
    
    def generate_plots(self, metrics_list: List[ConversationMetrics], output_dir: Path):
        """Generate visualization plots"""
        
        if not metrics_list:
            return
        
        # Set style
        if HAS_SEABORN:
            sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # 1. Overall scores over time
        fig, ax = plt.subplots(figsize=(12, 6))
        exchanges = range(1, len(metrics_list) + 1)
        overall_scores = [m.overall_score() for m in metrics_list]
        
        ax.plot(exchanges, overall_scores, 'b-', linewidth=2, marker='o')
        ax.set_xlabel('Exchange Number')
        ax.set_ylabel('Overall Score (0-10)')
        ax.set_title('Conversation Quality Over Time')
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(list(exchanges), overall_scores, 1)
        p = np.poly1d(z)
        ax.plot(exchanges, p(exchanges), "r--", alpha=0.8, label=f'Trend (slope: {z[0]:.3f})')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'overall_scores.png', dpi=150)
        plt.close()
        
        # 2. Multi-metric comparison
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Radar chart
        categories = ['Coherence', 'Relevance', 'Naturalness', 'Engagement', 
                     'Completeness', 'Philosophy', 'Context']
        
        avg_scores = [
            np.mean([m.coherence for m in metrics_list]),
            np.mean([m.relevance for m in metrics_list]),
            np.mean([m.naturalness for m in metrics_list]),
            np.mean([m.engagement for m in metrics_list]),
            np.mean([m.completeness for m in metrics_list]),
            np.mean([m.philosophical_depth for m in metrics_list]),
            np.mean([m.context_utilization for m in metrics_list])
        ]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
        avg_scores = np.concatenate((avg_scores, [avg_scores[0]]))
        angles = np.concatenate((angles, [angles[0]]))
        
        ax = plt.subplot(111, projection='polar')
        ax.plot(angles, avg_scores, 'o-', linewidth=2)
        ax.fill(angles, avg_scores, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 10)
        ax.set_title('Model Performance Across Dimensions', size=16, y=1.1)
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'performance_radar.png', dpi=150)
        plt.close()
        
        # 3. Response time distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        response_times = [m.response_time for m in metrics_list]
        ax1.hist(response_times, bins=20, edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Response Time (seconds)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Response Time Distribution')
        ax1.axvline(np.mean(response_times), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(response_times):.2f}s')
        ax1.legend()
        
        # Response length vs quality
        response_lengths = [m.response_length for m in metrics_list]
        overall_scores = [m.overall_score() for m in metrics_list]
        
        ax2.scatter(response_lengths, overall_scores, alpha=0.6)
        ax2.set_xlabel('Response Length (words)')
        ax2.set_ylabel('Overall Score')
        ax2.set_title('Response Length vs Quality')
        
        # Add regression line
        z = np.polyfit(response_lengths, overall_scores, 1)
        p = np.poly1d(z)
        ax2.plot(sorted(response_lengths), p(sorted(response_lengths)), 
                "r--", alpha=0.8, label=f'Correlation: {np.corrcoef(response_lengths, overall_scores)[0,1]:.3f}')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'response_analysis.png', dpi=150)
        plt.close()
        
        # 4. Metric trends
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        metrics_to_plot = [
            ('coherence', 'Coherence'),
            ('relevance', 'Relevance'),
            ('naturalness', 'Naturalness'),
            ('philosophical_depth', 'Philosophical Depth')
        ]
        
        for idx, (metric_name, title) in enumerate(metrics_to_plot):
            values = [getattr(m, metric_name) for m in metrics_list]
            axes[idx].plot(exchanges, values, marker='o', linewidth=1.5)
            axes[idx].set_title(f'{title} Over Time')
            axes[idx].set_xlabel('Exchange')
            axes[idx].set_ylabel('Score (0-10)')
            axes[idx].grid(True, alpha=0.3)
            
            # Add moving average
            if len(values) > 3:
                moving_avg = pd.Series(values).rolling(window=3).mean()
                axes[idx].plot(exchanges, moving_avg, 'r--', alpha=0.8, label='Moving Avg')
                axes[idx].legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'metric_trends.png', dpi=150)
        plt.close()
        
        self.console.print("[green]✓ Generated visualization plots[/green]")
    
    def generate_report(self, assessment: ModelAssessment, 
                       metrics_list: List[ConversationMetrics],
                       output_dir: Path) -> str:
        """Generate comprehensive markdown report"""
        
        report = f"""# ISC Conversational Model Assessment Report

**Model:** `{Path(assessment.model_path).name}`  
**Assessment Date:** {datetime.fromisoformat(assessment.assessment_date).strftime('%Y-%m-%d %H:%M')}  
**Total Exchanges Tested:** {assessment.total_exchanges}

## Executive Summary

{assessment.llm_summary}

## Overall Performance Metrics

| Metric | Score | Trend | Rating |
|--------|-------|-------|--------|
| **Coherence** | {assessment.avg_coherence:.2f}/10 | {self._trend_indicator(assessment.coherence_trend)} | {self._rating(assessment.avg_coherence)} |
| **Relevance** | {assessment.avg_relevance:.2f}/10 | {self._trend_indicator(assessment.relevance_trend)} | {self._rating(assessment.avg_relevance)} |
| **Naturalness** | {assessment.avg_naturalness:.2f}/10 | {self._trend_indicator(assessment.naturalness_trend)} | {self._rating(assessment.avg_naturalness)} |
| **Engagement** | {assessment.avg_engagement:.2f}/10 | - | {self._rating(assessment.avg_engagement)} |
| **Completeness** | {assessment.avg_completeness:.2f}/10 | - | {self._rating(assessment.avg_completeness)} |
| **Philosophical Depth** | {assessment.philosophical_sophistication:.2f}/10 | - | {self._rating(assessment.philosophical_sophistication)} |

### Performance Characteristics

- **Average Response Time:** {assessment.avg_response_time:.2f} seconds
- **Average Response Length:** {assessment.avg_response_length:.0f} words
- **Topic Diversity:** {assessment.topic_diversity:.2%}
- **Context Consistency:** {assessment.context_consistency:.2f}/10

## Strengths and Weaknesses

### Key Strengths
"""
        for strength in assessment.strengths:
            report += f"- ✅ {strength}\n"
        
        report += "\n### Areas for Improvement\n"
        for weakness in assessment.weaknesses:
            report += f"- ⚠️ {weakness}\n"
        
        report += "\n## Detailed Analysis\n\n"
        
        # Best and worst performances
        sorted_metrics = sorted(metrics_list, key=lambda x: x.overall_score(), reverse=True)
        
        report += "### Best Performances\n\n"
        for i, metric in enumerate(sorted_metrics[:3], 1):
            report += f"**Exchange {i}:**\n"
            report += f"- Human: *\"{metric.human_input}\"*\n"
            report += f"- Model: *\"{metric.model_response[:150]}...\"*\n"
            report += f"- Overall Score: {metric.overall_score():.2f}/10\n\n"
        
        report += "### Areas Needing Attention\n\n"
        for i, metric in enumerate(sorted_metrics[-3:], 1):
            report += f"**Weak Exchange {i}:**\n"
            report += f"- Human: *\"{metric.human_input}\"*\n"
            report += f"- Model: *\"{metric.model_response[:150]}...\"*\n"
            report += f"- Overall Score: {metric.overall_score():.2f}/10\n"
            report += f"- Main Issues: Low {self._identify_weak_areas(metric)}\n\n"
        
        # Recommendations
        report += "## Recommendations for Improvement\n\n"
        for i, rec in enumerate(assessment.llm_recommendations, 1):
            report += f"{i}. {rec}\n"
        
        # Visualizations
        report += """
## Performance Visualizations

### Overall Quality Trend
![Overall Scores](overall_scores.png)

### Multi-Dimensional Performance
![Performance Radar](performance_radar.png)

### Response Characteristics
![Response Analysis](response_analysis.png)

### Individual Metric Trends
![Metric Trends](metric_trends.png)

## Technical Details

### Test Coverage
"""
        # Test type distribution
        test_types = defaultdict(int)
        for _, test_type in self.test_conversations:
            test_types[test_type] += 1
        
        report += "| Test Type | Count |\n|-----------|-------|\n"
        for test_type, count in sorted(test_types.items()):
            report += f"| {test_type.replace('_', ' ').title()} | {count} |\n"
        
        report += f"""
### Response Time Analysis
- **Mean:** {np.mean([m.response_time for m in metrics_list]):.3f}s
- **Median:** {np.median([m.response_time for m in metrics_list]):.3f}s
- **Std Dev:** {np.std([m.response_time for m in metrics_list]):.3f}s
- **95th Percentile:** {np.percentile([m.response_time for m in metrics_list], 95):.3f}s

## Conclusion

This assessment evaluated the conversational capabilities of the ISC model across {assessment.total_exchanges} test exchanges. """
        
        # Overall rating
        overall_avg = np.mean([
            assessment.avg_coherence,
            assessment.avg_relevance,
            assessment.avg_naturalness,
            assessment.avg_engagement,
            assessment.avg_completeness
        ])
        
        if overall_avg >= 8:
            conclusion = "The model demonstrates **excellent** conversational abilities with strong philosophical grounding."
        elif overall_avg >= 7:
            conclusion = "The model shows **good** conversational performance with room for targeted improvements."
        elif overall_avg >= 6:
            conclusion = "The model has **adequate** conversational skills but would benefit from further training."
        else:
            conclusion = "The model requires **significant improvement** in conversational capabilities."
        
        report += conclusion
        
        report += f"""

---
*Generated by ISC Conversational Assessment Tool*  
*Assessment ID: {datetime.now().strftime('%Y%m%d_%H%M%S')}*
"""
        
        return report
    
    def _trend_indicator(self, trend: float) -> str:
        """Convert trend to visual indicator"""
        if trend > 0.1:
            return "📈 Improving"
        elif trend < -0.1:
            return "📉 Declining"
        else:
            return "➡️ Stable"
    
    def _rating(self, score: float) -> str:
        """Convert score to rating"""
        if score >= 8:
            return "⭐ Excellent"
        elif score >= 7:
            return "✅ Good"
        elif score >= 6:
            return "⚠️ Fair"
        else:
            return "❌ Needs Work"
    
    def _identify_weak_areas(self, metric: ConversationMetrics) -> str:
        """Identify the weakest aspects of a response"""
        aspects = {
            'coherence': metric.coherence,
            'relevance': metric.relevance,
            'naturalness': metric.naturalness,
            'engagement': metric.engagement,
            'completeness': metric.completeness
        }
        
        weak_areas = [name for name, score in aspects.items() if score < 6]
        return ', '.join(weak_areas) if weak_areas else 'multiple aspects'
    
    def compare_models(self, assessments: List[ModelAssessment]) -> str:
        """Generate comparison report for multiple models"""
        if len(assessments) < 2:
            return "Need at least 2 assessments for comparison"
        
        comparison = "# Model Comparison Report\n\n"
        comparison += f"Comparing {len(assessments)} models:\n\n"
        
        # Create comparison table
        comparison += "| Model | Overall | Coherence | Relevance | Naturalness | Philosophy | Response Time |\n"
        comparison += "|-------|---------|-----------|-----------|-------------|------------|---------------|\n"
        
        for assessment in assessments:
            model_name = Path(assessment.model_path).name
            overall = np.mean([
                assessment.avg_coherence,
                assessment.avg_relevance,
                assessment.avg_naturalness,
                assessment.avg_engagement,
                assessment.avg_completeness
            ])
            
            comparison += f"| {model_name[:20]}... | {overall:.2f} | "
            comparison += f"{assessment.avg_coherence:.2f} | "
            comparison += f"{assessment.avg_relevance:.2f} | "
            comparison += f"{assessment.avg_naturalness:.2f} | "
            comparison += f"{assessment.philosophical_sophistication:.2f} | "
            comparison += f"{assessment.avg_response_time:.2f}s |\n"
        
        # Find best model
        best_idx = np.argmax([
            np.mean([a.avg_coherence, a.avg_relevance, a.avg_naturalness, 
                    a.avg_engagement, a.avg_completeness])
            for a in assessments
        ])
        
        comparison += f"\n**Best Overall Model:** {Path(assessments[best_idx].model_path).name}\n"
        
        return comparison


def main():
    """Main entry point"""
    console = Console()
    
    # Check API key
    if OPENAI_API_KEY == "YOUR_OPENAI_API_KEY":
        console.print("[red]Please set your OpenAI API key![/red]")
        console.print("Set the OPENAI_API_KEY environment variable")
        return
    
    # Display header
    console.print(Panel("[bold cyan]ISC Conversational Model Assessment[/bold cyan]", style="cyan"))
    console.print("[dim]Comprehensive evaluation and reporting tool[/dim]\n")
    
    # Find conversational models
    console.print("[bold]Searching for conversational models...[/bold]")
    
    model_pairs = []
    search_paths = [
        ".",
        "checkpoints",
        "../checkpoints"
    ]
    
    for search_path in search_paths:
        for model_file in glob.glob(f"{search_path}/isc_state_conversational_*.pt"):
            if not model_file.endswith('_lm_head.pt'):
                lm_head_file = model_file.replace('.pt', '_lm_head.pt')
                if os.path.exists(lm_head_file):
                    model_pairs.append((model_file, lm_head_file))
    
    if not model_pairs:
        console.print("[red]No conversational models found![/red]")
        console.print("[yellow]Please train a conversational model first.[/yellow]")
        return
    
    # Display models
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("#", style="cyan", width=3)
    table.add_column("Model", style="yellow")
    table.add_column("Modified", style="green")
    table.add_column("Size", style="blue")
    
    for i, (model, lm_head) in enumerate(model_pairs[:10], 1):
        mtime = datetime.fromtimestamp(os.path.getmtime(model)).strftime("%Y-%m-%d %H:%M")
        size_mb = os.path.getsize(model) / (1024 * 1024)
        table.add_row(str(i), Path(model).name, mtime, f"{size_mb:.1f} MB")
    
    console.print(table)
    
    # Select model
    choice = console.input("\n[cyan]Select model to assess (number):[/cyan] ")
    
    try:
        model_idx = int(choice) - 1
        if model_idx < 0 or model_idx >= len(model_pairs):
            raise ValueError
    except:
        console.print("[red]Invalid selection[/red]")
        return
    
    selected_model, selected_lm_head = model_pairs[model_idx]
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f"conversational_reports/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run assessment
    console.print(f"\n[green]Assessing model: {Path(selected_model).name}[/green]")
    console.print(f"[yellow]This will take approximately 2-3 minutes...[/yellow]\n")
    
    assessor = ConversationalAssessor()
    assessment = assessor.assess_model(selected_model, selected_lm_head)
    
    if not assessment:
        console.print("[red]Assessment failed![/red]")
        return
    
    # Generate visualizations
    console.print("\n[cyan]Generating visualizations...[/cyan]")
    assessor.generate_plots(assessor.metrics_history, output_dir)
    
    # Generate report
    console.print("[cyan]Generating report...[/cyan]")
    report = assessor.generate_report(assessment, assessor.metrics_history, output_dir)
    
    # Save report
    report_file = output_dir / "assessment_report.md"
    with open(report_file, 'w') as f:
        f.write(report)
    
    # Save raw data
    data_file = output_dir / "assessment_data.json"
    with open(data_file, 'w') as f:
        json.dump({
            'assessment': asdict(assessment),
            'metrics': [asdict(m) for m in assessor.metrics_history],
            'model_info': {
                'model_path': selected_model,
                'lm_head_path': selected_lm_head,
                'assessment_date': timestamp
            }
        }, f, indent=2)
    
    # Copy model info for reference
    model_info_file = output_dir / "model_info.txt"
    with open(model_info_file, 'w') as f:
        f.write(f"Model: {selected_model}\n")
        f.write(f"LM Head: {selected_lm_head}\n")
        f.write(f"Assessment Date: {timestamp}\n")
        f.write(f"Model Size: {os.path.getsize(selected_model) / (1024*1024):.1f} MB\n")
    
    # Display summary
    console.print("\n[bold green]✓ Assessment Complete![/bold green]")
    console.print(f"\nReport saved to: [cyan]{report_file}[/cyan]")
    console.print(f"Visualizations saved to: [cyan]{output_dir}[/cyan]")
    
    # Show quick summary
    console.print("\n[bold]Quick Summary:[/bold]")
    summary_table = Table(show_header=False)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    
    overall_score = np.mean([
        assessment.avg_coherence,
        assessment.avg_relevance,
        assessment.avg_naturalness,
        assessment.avg_engagement,
        assessment.avg_completeness
    ])
    
    summary_table.add_row("Overall Score", f"{overall_score:.2f}/10")
    summary_table.add_row("Best Aspect", assessment.strengths[0] if assessment.strengths else "N/A")
    summary_table.add_row("Needs Work", assessment.weaknesses[0] if assessment.weaknesses else "N/A")
    summary_table.add_row("Response Time", f"{assessment.avg_response_time:.2f}s")
    
    console.print(summary_table)
    
    # Option to view report
    view = console.input("\n[cyan]View report in console? (y/n):[/cyan] ")
    if view.lower() == 'y':
        console.print("\n" + "="*80 + "\n")
        console.print(Markdown(report))


if __name__ == "__main__":
    main()