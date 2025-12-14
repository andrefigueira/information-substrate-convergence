#!/usr/bin/env python3
"""
Training Progress Tracker
Monitors improvement as model grows bigger
"""

import json
import csv
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt

def get_current_model_stats():
    """Get current model statistics"""
    try:
        with open('models/neuromorphic_state.json', 'r') as f:
            state = json.load(f)

        return {
            'concepts': state['metrics']['substrate_nodes'],
            'training_sessions': state['metrics']['training_sessions'],
            'total_examples': state['metrics']['total_training_examples'],
            'conversations': state['metrics']['total_interactions'],
            'phi_value': state['metrics']['phi_value']
        }
    except Exception as e:
        print(f"⚠ Could not read model state: {e}")
        return None

def run_evaluation_and_track():
    """Run evaluation and track progress"""

    print("📊 Training Progress Tracker")
    print("=" * 40)

    # Get current model stats
    stats = get_current_model_stats()
    if not stats:
        return

    print(f"📈 Current Model:")
    print(f"  Concepts: {stats['concepts']}")
    print(f"  Training Sessions: {stats['training_sessions']}")
    print(f"  Total Examples: {stats['total_examples']}")
    print(f"  Conversations: {stats['conversations']}")
    print(f"  Phi (φ): {stats['phi_value']:.6f}")

    # Run conversation quality test
    print(f"\n🧪 Running conversation quality test...")
    os.system('python simple_evaluator.py')

    # Find the latest evaluation file
    eval_files = [f for f in os.listdir('.') if f.startswith('ai_evaluation_') and f.endswith('.csv')]
    if not eval_files:
        print("❌ No evaluation results found")
        return

    latest_eval = sorted(eval_files)[-1]

    # Read the evaluation results
    try:
        with open(latest_eval, 'r') as f:
            reader = csv.DictReader(f)
            results = list(reader)

        avg_score = sum(float(row['score']) for row in results) / len(results)
        print(f"✅ Conversation Score: {avg_score:.3f}/1.0")

    except Exception as e:
        print(f"⚠ Could not read evaluation: {e}")
        avg_score = 0

    # Update progress tracking file
    progress_file = 'training_progress.csv'

    # Initialize if doesn't exist
    if not os.path.exists(progress_file):
        with open(progress_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'concepts', 'training_sessions', 'total_examples',
                'conversations', 'phi_value', 'conversation_score', 'notes'
            ])

    # Add current data
    with open(progress_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            stats['concepts'],
            stats['training_sessions'],
            stats['total_examples'],
            stats['conversations'],
            stats['phi_value'],
            avg_score,
            f"Evaluated {len(results)} questions"
        ])

    print(f"💾 Progress saved to {progress_file}")

    # Show progress over time
    show_progress_chart()

def show_progress_chart():
    """Show progress visualization"""

    if not os.path.exists('training_progress.csv'):
        print("❌ No progress data found")
        return

    try:
        import matplotlib.pyplot as plt
        import pandas as pd

        df = pd.read_csv('training_progress.csv')

        if len(df) < 2:
            print("💡 Need more data points for progress chart")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

        # Concepts over time
        ax1.plot(df.index, df['concepts'], 'b-o')
        ax1.set_title('Model Size (Concepts)')
        ax1.set_ylabel('Concepts')

        # Conversation score over time
        ax2.plot(df.index, df['conversation_score'], 'g-o')
        ax2.set_title('Conversation Quality')
        ax2.set_ylabel('Score (0-1)')
        ax2.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='Target')
        ax2.legend()

        # Training examples over time
        ax3.plot(df.index, df['total_examples'], 'orange', marker='o')
        ax3.set_title('Training Examples')
        ax3.set_ylabel('Examples')

        # Phi value over time
        ax4.plot(df.index, df['phi_value'], 'purple', marker='o')
        ax4.set_title('Phi (φ) Value')
        ax4.set_ylabel('Information Integration')

        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
        plt.close()

        print("📊 Progress chart saved to training_progress.png")

    except ImportError:
        print("📊 Install matplotlib/pandas for progress charts: pip install matplotlib pandas")

        # Text-based progress
        with open('training_progress.csv', 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if len(rows) >= 2:
            first = rows[0]
            latest = rows[-1]

            print("\n📈 Progress Summary:")
            print(f"  Concepts: {first['concepts']} → {latest['concepts']} (+{int(latest['concepts']) - int(first['concepts'])})")
            print(f"  Score: {float(first['conversation_score']):.3f} → {float(latest['conversation_score']):.3f} ({float(latest['conversation_score']) - float(first['conversation_score']):+.3f})")
            print(f"  Examples: {first['total_examples']} → {latest['total_examples']} (+{int(latest['total_examples']) - int(first['total_examples'])})")

    except Exception as e:
        print(f"⚠ Could not create progress chart: {e}")

def show_recommendations():
    """Show training recommendations based on current performance"""

    print("\n🎯 TRAINING RECOMMENDATIONS")
    print("=" * 40)

    try:
        with open('training_progress.csv', 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            print("❌ No progress data available")
            return

        latest = rows[-1]
        score = float(latest['conversation_score'])
        concepts = int(latest['concepts'])

        if score < 0.3:
            print("🔴 POOR CONVERSATIONAL QUALITY")
            print("  - Run extensive ChatGPT training: make ai-train-quick")
            print("  - Focus on natural conversation patterns")
            print("  - Target: 2000+ concepts")

        elif score < 0.6:
            print("🟡 MODERATE CONVERSATIONAL QUALITY")
            print("  - Add more diverse training: make ai-train-chatgpt")
            print("  - Focus on engagement and naturalness")
            print("  - Current concepts look good, need better patterns")

        elif score < 0.8:
            print("🟢 GOOD CONVERSATIONAL QUALITY")
            print("  - Fine-tune with specific scenarios")
            print("  - Add personality training")
            print("  - Almost at target performance!")

        else:
            print("🎉 EXCELLENT CONVERSATIONAL QUALITY")
            print("  - Maintain current performance")
            print("  - Add specialized knowledge domains")
            print("  - Consider advanced training techniques")

        print(f"\n📊 Current Status: {score:.3f}/1.0 with {concepts} concepts")

    except Exception as e:
        print(f"⚠ Could not generate recommendations: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "recommend":
        show_recommendations()
    else:
        run_evaluation_and_track()
        show_recommendations()