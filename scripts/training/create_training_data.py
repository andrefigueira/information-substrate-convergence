#!/usr/bin/env python3
"""
ChatGPT-style Training Data Creator for Neuromorphic ISC AI
"""

import json
import sys
from datetime import datetime
from pathlib import Path

def create_training_session():
    """Interactive training data creator"""
    print("🎓 Neuromorphic AI Training Data Creator")
    print("=" * 50)
    print("Create conversational training data for your AI.")
    print("Type 'done' when finished, 'help' for commands.\n")

    training_data = {
        "context": "",
        "metadata": {
            "created": datetime.now().isoformat(),
            "purpose": "Interactive training session",
            "training_type": "conversational"
        },
        "conversations": []
    }

    # Get context
    context = input("📝 Training context/topic (optional): ").strip()
    if context:
        training_data["context"] = context

    print("\n💬 Enter conversation pairs:")
    print("Format: [User] question → [AI] response")
    print("Commands: 'done', 'help', 'preview', 'save'\n")

    conversation_count = 0

    while True:
        try:
            # Get user input
            user_input = input(f"\n[{conversation_count + 1}] User: ").strip()

            if user_input.lower() == 'done':
                break
            elif user_input.lower() == 'help':
                print_help()
                continue
            elif user_input.lower() == 'preview':
                preview_data(training_data)
                continue
            elif user_input.lower() == 'save':
                save_data(training_data)
                continue
            elif not user_input:
                continue

            # Get AI response
            ai_response = input(f"[{conversation_count + 1}] AI: ").strip()
            if not ai_response:
                print("⚠ AI response cannot be empty. Try again.")
                continue

            # Optional context for this pair
            pair_context = input(f"[{conversation_count + 1}] Context (optional): ").strip()

            # Add conversation pair
            conversation_pair = {
                "input": user_input,
                "output": ai_response,
                "context": pair_context or context or "general"
            }

            training_data["conversations"].append(conversation_pair)
            conversation_count += 1

            print(f"✓ Added conversation pair {conversation_count}")

        except KeyboardInterrupt:
            print("\n\n🛑 Interrupted. Current progress:")
            break
        except Exception as e:
            print(f"⚠ Error: {e}")

    # Final save
    if training_data["conversations"]:
        save_data(training_data)
    else:
        print("❌ No conversations to save.")

def print_help():
    """Print help information"""
    print("\n📖 Commands:")
    print("  done     - Finish and save training data")
    print("  help     - Show this help")
    print("  preview  - Preview current training data")
    print("  save     - Save current progress")
    print("  [empty]  - Skip current input")

def preview_data(training_data):
    """Preview current training data"""
    print(f"\n📊 Current Training Data:")
    print(f"Context: {training_data.get('context', 'None')}")
    print(f"Conversations: {len(training_data['conversations'])}")

    if training_data["conversations"]:
        print("\nLast few conversations:")
        for i, conv in enumerate(training_data["conversations"][-3:], 1):
            print(f"  {len(training_data['conversations']) - 3 + i}. User: {conv['input'][:50]}...")
            print(f"     AI: {conv['output'][:50]}...")

def save_data(training_data):
    """Save training data to file"""
    if not training_data["conversations"]:
        print("❌ No conversations to save.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"training_data_{timestamp}.json"

    try:
        with open(filename, 'w') as f:
            json.dump(training_data, f, indent=2)

        print(f"💾 Training data saved to: {filename}")
        print(f"📊 Total conversations: {len(training_data['conversations'])}")

        # Ask if user wants to train immediately
        train_now = input("🎓 Train AI with this data now? (y/N): ").strip().lower()
        if train_now in ['y', 'yes']:
            train_immediately(filename)

    except Exception as e:
        print(f"❌ Save failed: {e}")

def train_immediately(filename):
    """Train AI immediately with the created data"""
    try:
        sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))
        from isc.neuromorphic_core import NeuromorphicISCCore

        print("🧠 Initializing neuromorphic core...")
        core = NeuromorphicISCCore()
        core.verbose = True

        print(f"📚 Loading training data from {filename}...")
        core.load_training_from_file(filename)

        print("✓ Training completed!")
        print("🔄 Enabling continuous learning...")
        core.continuous_learning_mode(True)

        print("\n🎉 Your AI is now trained and ready!")
        print("Run 'make ai' to start chatting with your trained AI.")

    except Exception as e:
        print(f"⚠ Training failed: {e}")
        print("You can manually train later with: make ai-train-file")

def batch_import():
    """Import training data from text file"""
    print("📁 Batch Import Mode")
    print("Format: user_input|ai_response|context (one per line)")

    filepath = input("📂 Text file path: ").strip()
    if not Path(filepath).exists():
        print("❌ File not found.")
        return

    training_data = {
        "context": "Batch imported training data",
        "metadata": {
            "created": datetime.now().isoformat(),
            "purpose": "Batch import",
            "training_type": "conversational",
            "source_file": filepath
        },
        "conversations": []
    }

    try:
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split('|')
                if len(parts) >= 2:
                    user_input = parts[0].strip()
                    ai_response = parts[1].strip()
                    context = parts[2].strip() if len(parts) > 2 else "general"

                    training_data["conversations"].append({
                        "input": user_input,
                        "output": ai_response,
                        "context": context
                    })
                else:
                    print(f"⚠ Skipping malformed line {line_num}: {line[:50]}...")

        if training_data["conversations"]:
            save_data(training_data)
        else:
            print("❌ No valid conversations found.")

    except Exception as e:
        print(f"❌ Import failed: {e}")

if __name__ == "__main__":
    print("🤖 Neuromorphic ISC AI Training Data Creator")
    print("Choose mode:")
    print("1. Interactive training data creation")
    print("2. Batch import from text file")

    try:
        choice = input("\nChoice (1-2): ").strip()

        if choice == "1":
            create_training_session()
        elif choice == "2":
            batch_import()
        else:
            print("❌ Invalid choice.")

    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")