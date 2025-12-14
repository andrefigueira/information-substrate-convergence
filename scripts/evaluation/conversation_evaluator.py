#!/usr/bin/env python3
"""
Conversational Quality Measurement System for Neuromorphic ISC AI
Tracks and measures improvement in conversational abilities over time
"""

import json
import os
import sys
import time
import csv
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path

class ConversationEvaluator:
    """Measures and tracks conversational quality metrics"""

    def __init__(self):
        self.test_questions = [
            # Basic conversation
            "Hello, how are you today?",
            "What's your favorite hobby?",
            "Tell me about yourself.",

            # Emotional intelligence
            "I'm feeling sad today. Can you help?",
            "I'm excited about my new job!",
            "I'm worried about my presentation tomorrow.",

            # Knowledge and explanation
            "Can you explain how computers work?",
            "What do you think about climate change?",
            "How do you learn new things?",

            # Creative and abstract
            "If you could travel anywhere, where would you go?",
            "What's the meaning of life?",
            "Tell me a short story about a robot.",

            # Problem solving
            "I lost my keys. What should I do?",
            "How can I improve my communication skills?",
            "I can't decide what to eat for dinner.",

            # Social interaction
            "What makes a good friend?",
            "How do you handle disagreements?",
            "What's your opinion on social media?"
        ]

        self.metrics_file = "conversation_metrics.csv"
        self.detailed_log = "conversation_evaluations.json"

        # Initialize CSV if it doesn't exist
        if not os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'model_concepts', 'training_sessions', 'total_examples',
                    'avg_response_length', 'relevance_score', 'naturalness_score',
                    'engagement_score', 'coherence_score', 'overall_score',
                    'improvement_notes'
                ])

    def evaluate_response_quality(self, question: str, response: str) -> Dict[str, float]:
        """Evaluate quality of a single response"""

        # Response length (normalized)
        response_length = len(response.split())
        length_score = min(1.0, response_length / 50)  # Optimal around 50 words

        # Basic relevance check (keywords from question appear in response)
        question_words = set(question.lower().split())
        response_words = set(response.lower().split())
        relevance_score = len(question_words & response_words) / max(len(question_words), 1)

        # Naturalness (avoid overly technical language)
        technical_terms = ['substrate', 'neuromorphic', 'phi', 'integration', 'activation', 'propagate']
        tech_count = sum(1 for term in technical_terms if term in response.lower())
        naturalness_score = max(0.0, 1.0 - (tech_count * 0.2))

        # Engagement (check for questions, personal elements, conversational markers)
        engagement_markers = ['?', '!', 'I think', 'I feel', 'you might', 'perhaps', 'interesting']
        engagement_count = sum(1 for marker in engagement_markers if marker in response)
        engagement_score = min(1.0, engagement_count * 0.3)

        # Coherence (response length vs complexity)
        if response_length > 10 and response_length < 200:
            coherence_score = 1.0
        elif response_length < 5:
            coherence_score = 0.3
        else:
            coherence_score = 0.7

        return {
            'length_score': length_score,
            'relevance_score': relevance_score,
            'naturalness_score': naturalness_score,
            'engagement_score': engagement_score,
            'coherence_score': coherence_score
        }

    def run_conversation_test(self) -> Dict[str, any]:
        """Run full conversation test and return metrics"""

        print("🧪 Running conversational quality assessment...")

        # Load neuromorphic core
        sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))
        try:
            from isc.neuromorphic_core import NeuromorphicISCCore
            core = NeuromorphicISCCore()
        except Exception as e:
            print(f"⚠ Could not load neuromorphic core: {e}")
            print("💡 Try running without transformers dependencies")
            return None
        status = core.get_status()

        results = {
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'concepts': status['substrate']['node_count'],
                'edges': status['substrate']['edge_count'],
                'training_sessions': status['metrics'].get('training_sessions', 0),
                'total_examples': status['metrics'].get('total_training_examples', 0),
                'phi_value': status['metrics']['phi_value']
            },
            'responses': [],
            'aggregate_scores': {}
        }

        print(f"📊 Testing model with {results['model_info']['concepts']} concepts...")

        # Test each question
        for i, question in enumerate(self.test_questions, 1):
            print(f"  [{i}/{len(self.test_questions)}] Testing: {question[:40]}...")

            try:
                start_time = time.time()
                response = core.process_input(question)
                response_time = time.time() - start_time

                # Evaluate response quality
                quality_scores = self.evaluate_response_quality(question, response)

                result = {
                    'question': question,
                    'response': response,
                    'response_time': response_time,
                    'quality_scores': quality_scores
                }

                results['responses'].append(result)

                # Brief pause to avoid overwhelming the system
                time.sleep(0.1)

            except Exception as e:
                print(f"    ⚠ Error testing question: {e}")
                continue

        # Calculate aggregate scores
        if results['responses']:
            score_sums = {}
            for response in results['responses']:
                for score_name, score_value in response['quality_scores'].items():
                    score_sums[score_name] = score_sums.get(score_name, 0) + score_value

            num_responses = len(results['responses'])
            results['aggregate_scores'] = {
                score_name: score_sum / num_responses
                for score_name, score_sum in score_sums.items()
            }

            # Overall score (weighted average)
            overall_score = (
                results['aggregate_scores']['relevance_score'] * 0.3 +
                results['aggregate_scores']['naturalness_score'] * 0.25 +
                results['aggregate_scores']['engagement_score'] * 0.25 +
                results['aggregate_scores']['coherence_score'] * 0.2
            )
            results['aggregate_scores']['overall_score'] = overall_score

            avg_response_length = sum(len(r['response'].split()) for r in results['responses']) / num_responses
            results['avg_response_length'] = avg_response_length

        print(f"✅ Assessment complete! Overall score: {results['aggregate_scores'].get('overall_score', 0):.3f}")

        return results

    def save_results(self, results: Dict) -> None:
        """Save results to files"""

        # Save detailed results to JSON
        try:
            if os.path.exists(self.detailed_log):
                with open(self.detailed_log, 'r') as f:
                    all_results = json.load(f)
            else:
                all_results = []

            all_results.append(results)

            with open(self.detailed_log, 'w') as f:
                json.dump(all_results, f, indent=2)

        except Exception as e:
            print(f"⚠ Could not save detailed log: {e}")

        # Save summary to CSV
        try:
            with open(self.metrics_file, 'a', newline='') as f:
                writer = csv.writer(f)

                agg = results.get('aggregate_scores', {})
                model = results.get('model_info', {})

                writer.writerow([
                    results.get('timestamp', ''),
                    model.get('concepts', 0),
                    model.get('training_sessions', 0),
                    model.get('total_examples', 0),
                    results.get('avg_response_length', 0),
                    agg.get('relevance_score', 0),
                    agg.get('naturalness_score', 0),
                    agg.get('engagement_score', 0),
                    agg.get('coherence_score', 0),
                    agg.get('overall_score', 0),
                    self.generate_improvement_notes(results)
                ])

        except Exception as e:
            print(f"⚠ Could not save CSV metrics: {e}")

    def generate_improvement_notes(self, results: Dict) -> str:
        """Generate notes about areas for improvement"""
        notes = []
        agg = results.get('aggregate_scores', {})

        if agg.get('naturalness_score', 0) < 0.5:
            notes.append("Too technical")
        if agg.get('engagement_score', 0) < 0.3:
            notes.append("Low engagement")
        if agg.get('relevance_score', 0) < 0.4:
            notes.append("Poor relevance")
        if results.get('avg_response_length', 0) < 10:
            notes.append("Responses too short")
        elif results.get('avg_response_length', 0) > 100:
            notes.append("Responses too long")

        return "; ".join(notes) if notes else "Good progress"

    def show_progress_report(self) -> None:
        """Show progress over time"""

        if not os.path.exists(self.metrics_file):
            print("❌ No metrics data found. Run evaluation first.")
            return

        print("\n📈 CONVERSATIONAL PROGRESS REPORT")
        print("=" * 50)

        try:
            with open(self.metrics_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            if not rows:
                print("❌ No evaluation data found.")
                return

            # Show latest results
            latest = rows[-1]
            print(f"🧠 Current Model: {latest['model_concepts']} concepts")
            print(f"🎓 Training Sessions: {latest['training_sessions']}")
            print(f"📚 Total Examples: {latest['total_examples']}")
            print(f"⭐ Overall Score: {float(latest['overall_score']):.3f}/1.0")
            print()

            # Show breakdown
            print("📊 Quality Breakdown:")
            print(f"  Relevance:    {float(latest['relevance_score']):.3f}")
            print(f"  Naturalness:  {float(latest['naturalness_score']):.3f}")
            print(f"  Engagement:   {float(latest['engagement_score']):.3f}")
            print(f"  Coherence:    {float(latest['coherence_score']):.3f}")
            print(f"  Avg Length:   {float(latest['avg_response_length']):.1f} words")
            print()

            # Show improvement over time
            if len(rows) > 1:
                first = rows[0]
                improvement = float(latest['overall_score']) - float(first['overall_score'])
                concept_growth = int(latest['model_concepts']) - int(first['model_concepts'])

                print("📈 Progress Since First Evaluation:")
                print(f"  Score Change:    {improvement:+.3f}")
                print(f"  Concept Growth:  +{concept_growth} concepts")
                print(f"  Notes: {latest['improvement_notes']}")

            else:
                print("💡 Run more evaluations to see progress trends!")

        except Exception as e:
            print(f"⚠ Error reading metrics: {e}")

    def compare_with_target(self) -> None:
        """Compare current performance with target conversational AI"""

        print("\n🎯 COMPARISON WITH TARGET PERFORMANCE")
        print("=" * 50)

        target_scores = {
            'relevance_score': 0.8,
            'naturalness_score': 0.9,
            'engagement_score': 0.7,
            'coherence_score': 0.8,
            'overall_score': 0.8
        }

        if not os.path.exists(self.metrics_file):
            print("❌ No current data to compare")
            return

        try:
            with open(self.metrics_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            if not rows:
                print("❌ No evaluation data found")
                return

            latest = rows[-1]

            print("Target vs Current Performance:")
            for metric, target in target_scores.items():
                current = float(latest[metric])
                gap = target - current
                status = "✅" if gap <= 0 else "🔴" if gap > 0.3 else "🟡"

                print(f"  {metric:15s}: {current:.3f} | Target: {target:.3f} | Gap: {gap:+.3f} {status}")

            overall_gap = target_scores['overall_score'] - float(latest['overall_score'])

            if overall_gap <= 0:
                print("\n🎉 CONGRATULATIONS! Your AI meets conversational targets!")
            elif overall_gap < 0.2:
                print(f"\n🟡 Close to target! Need {overall_gap:.3f} improvement")
            else:
                print(f"\n🔴 More training needed. Gap: {overall_gap:.3f}")

        except Exception as e:
            print(f"⚠ Error comparing performance: {e}")

def main():
    """Main CLI interface"""

    evaluator = ConversationEvaluator()

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "test":
            results = evaluator.run_conversation_test()
            if results:
                evaluator.save_results(results)
            else:
                print("❌ Test failed")

        elif command == "report":
            evaluator.show_progress_report()

        elif command == "compare":
            evaluator.compare_with_target()

        elif command == "full":
            # Run full evaluation cycle
            results = evaluator.run_conversation_test()
            if results:
                evaluator.save_results(results)
                evaluator.show_progress_report()
                evaluator.compare_with_target()
            else:
                print("❌ Evaluation failed")

        else:
            print("❌ Unknown command. Use: test, report, compare, or full")

    else:
        print("🧪 Conversational Quality Evaluator")
        print("Commands:")
        print("  test     - Run conversation test")
        print("  report   - Show progress report")
        print("  compare  - Compare with targets")
        print("  full     - Run complete evaluation")

if __name__ == "__main__":
    main()