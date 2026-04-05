#!/usr/bin/env python3
"""
Train the ISC system until it's conversational.
Uses the large training dataset and improved generation logic.
"""

import sys
from pathlib import Path
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from isc.neuromorphic_core import NeuromorphicISCCore
from large_training_dataset import TRAINING_DATA


def reset_language_patterns(core):
    """Reset language patterns to start fresh"""
    core.substrate.language_patterns = defaultdict(list)
    core.substrate.phrase_transitions = defaultdict(lambda: defaultdict(float))
    core.substrate.response_templates = []
    core.substrate.word_concept_map = defaultdict(set)
    print("🔄 Language patterns reset")


def train_batch(core, data, batch_name=""):
    """Train on a batch of data"""
    print(f"\n🎓 Training on {len(data)} pairs {batch_name}...")
    core.train_on_conversation(data, context=f"conversational_{batch_name}")


def test_responses(core, test_inputs):
    """Test responses and return results"""
    results = []
    for query in test_inputs:
        response = core.process_input(query)
        results.append((query, response))
    return results


def evaluate_quality(results):
    """Simple quality evaluation"""
    scores = []
    for query, response in results:
        score = 0
        # Length check (not too short, not too long)
        words = response.split()
        if 5 <= len(words) <= 50:
            score += 1
        # Ends properly
        if response.endswith(('.', '!', '?')):
            score += 1
        # Starts with capital
        if response[0].isupper():
            score += 1
        # No obvious repetition
        word_list = [w.lower() for w in words]
        if len(word_list) == len(set(word_list)) or len(word_list) - len(set(word_list)) <= 2:
            score += 1
        # Contains meaningful content (more than just stopwords)
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                    'should', 'may', 'might', 'must', 'to', 'of', 'in', 'for', 'on', 'with'}
        content_words = [w for w in word_list if w.lower() not in stopwords]
        if len(content_words) >= 3:
            score += 1
        scores.append(score)
    return sum(scores) / len(scores) if scores else 0


def main():
    print("=" * 70)
    print("🧠 ISC Conversational Training - Full Training Run")
    print("=" * 70)

    # Initialize core
    core = NeuromorphicISCCore()
    core.start_session()

    # Reset language patterns to start fresh with improved logic
    reset_language_patterns(core)

    # Get initial stats
    initial_stats = core.get_status()
    print(f"\n📊 Initial state:")
    print(f"   Substrate: {initial_stats['substrate']['node_count']} concepts")

    # Train in batches for better learning
    batch_size = 50
    batches = [TRAINING_DATA[i:i+batch_size] for i in range(0, len(TRAINING_DATA), batch_size)]

    for i, batch in enumerate(batches):
        train_batch(core, batch, f"batch_{i+1}")

    # Get stats after training
    final_stats = core.get_status()
    lang_stats = core.substrate.get_language_stats()

    print(f"\n" + "=" * 70)
    print("📊 Training Complete - Statistics")
    print("=" * 70)
    print(f"   Substrate: {final_stats['substrate']['node_count']} concepts")
    print(f"   Language patterns: {lang_stats['total_patterns']} phrases")
    print(f"   Response templates: {lang_stats['response_templates']}")
    print(f"   Phrase transitions: {lang_stats['phrase_transitions']}")

    # Test responses
    print(f"\n" + "=" * 70)
    print("🧪 Testing Responses")
    print("=" * 70)

    test_inputs = [
        # Greetings
        "Hello!",
        "Hi there",
        "How are you?",
        # Identity
        "What are you?",
        "Who are you?",
        "How do you work?",
        # Feelings
        "Do you have feelings?",
        "Are you conscious?",
        # Philosophical
        "What is consciousness?",
        "What is the meaning of life?",
        # Emotional support
        "I'm feeling sad",
        "I'm stressed",
        # Requests
        "Help me understand",
        "Tell me more",
        # General
        "That's interesting",
        "I agree",
        # Farewells
        "Thanks",
        "Goodbye",
    ]

    results = test_responses(core, test_inputs)

    for query, response in results:
        print(f"\n👤 User: {query}")
        print(f"🤖 ISC: {response}")

    # Evaluate quality
    quality_score = evaluate_quality(results)
    print(f"\n" + "=" * 70)
    print(f"📈 Quality Score: {quality_score:.2f}/5.0")
    print("=" * 70)

    if quality_score >= 4.0:
        print("✅ System is conversational!")
    elif quality_score >= 3.0:
        print("⚠️ System is mostly conversational, could use more training")
    else:
        print("❌ System needs more training")

    # Save state
    core.save_state(core.persistent_state_path)
    print(f"\n💾 State saved to {core.persistent_state_path}")

    return core, results, quality_score


if __name__ == "__main__":
    core, results, score = main()
