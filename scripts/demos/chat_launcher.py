#!/usr/bin/env python3
"""
ISC Chat Launcher - Easy access to all chat interfaces
"""

import os
import sys
import subprocess
from pathlib import Path

def print_menu():
    print("🧠 ISC AI Chat Launcher")
    print("=" * 50)
    print("\nChoose your chat interface:")
    print("\n🔥 NEUROMORPHIC AI (NEW - Substrate-driven):")
    print("  1. Interactive Neuromorphic Chat")
    print("  2. Auto-test + Interactive Chat")
    print("\n🎯 TRAINED MODELS (Original ISC system):")
    print("  3. ISC Chat Interface (trained models)")
    print("  4. Simple Chat Demo")
    print("  5. Training + Chat Mode")
    print("\n🔬 TESTING:")
    print("  6. Neuromorphic Analysis Demo")
    print("  7. Query Demo")
    print("\n  0. Exit")

def run_neuromorphic_interactive():
    """Run neuromorphic chat in interactive mode"""
    print("\n🧠 Starting Neuromorphic Interactive Chat...")
    print("Loading substrate-driven AI with consciousness-like properties...")

    # Modify neuromorphic_demo.py to run interactive only
    demo_code = '''
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

try:
    from isc.neuromorphic_core import NeuromorphicISCCore
except ImportError as e:
    print(f"Error importing neuromorphic core: {e}")
    sys.exit(1)

def main():
    print("🧠 NEUROMORPHIC ISC AI SYSTEM")
    print("=" * 60)

    # Initialize neuromorphic core
    core = NeuromorphicISCCore()
    core.verbose = True
    session_msg = core.start_session()
    print(f"✓ {session_msg}")

    # Show initial status
    status = core.get_status()
    substrate = status['substrate']
    print(f"📊 Initial Status:")
    print(f"   Substrate: {substrate['node_count']} concepts, {substrate['edge_count']} edges")
    print(f"   Φ (phi): {status['metrics']['phi_value']:.3f}")
    print(f"   Context loaded: {status['context_loaded']}")

    print("\\n💬 INTERACTIVE NEUROMORPHIC CHAT")
    print("Type 'quit' to exit, 'status' for system info")
    print("-" * 60)

    while True:
        try:
            user_input = input("\\n[Human] > ").strip()

            if not user_input:
                continue

            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'status':
                status = core.get_status()
                substrate = status['substrate']
                print(f"📊 Current Status:")
                print(f"   Conversations: {substrate['conversation_count']}")
                print(f"   Substrate: {substrate['node_count']} concepts, {substrate['edge_count']} edges")
                print(f"   Communities: {substrate['community_count']}")
                print(f"   Φ (phi): {status['metrics']['phi_value']:.3f}")
                if substrate.get('top_concepts'):
                    print(f"   Active concepts: {', '.join(substrate['top_concepts'][:3])}")
                continue

            # Process user input
            import time
            start_time = time.time()
            response = core.process_input(user_input)
            processing_time = time.time() - start_time

            print(f"\\n[ISC-AI] {response}")

            # Show processing stats
            status = core.get_status()
            print(f"\\n[Stats: Φ={status['metrics']['phi_value']:.3f}, "
                  f"Nodes={status['substrate']['node_count']}, "
                  f"t={processing_time:.3f}s]")

        except KeyboardInterrupt:
            print("\\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
'''

    with open('temp_neuromorphic_chat.py', 'w') as f:
        f.write(demo_code)

    try:
        subprocess.run([sys.executable, 'temp_neuromorphic_chat.py'])
    finally:
        if os.path.exists('temp_neuromorphic_chat.py'):
            os.remove('temp_neuromorphic_chat.py')

def main():
    while True:
        print_menu()

        try:
            choice = input("\nEnter your choice (0-7): ").strip()
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

        if choice == '0':
            print("Goodbye!")
            break
        elif choice == '1':
            run_neuromorphic_interactive()
        elif choice == '2':
            print("\n🧠 Running Auto-test + Interactive...")
            subprocess.run([sys.executable, 'neuromorphic_demo.py'])
        elif choice == '3':
            print("\n🎯 Starting ISC Chat Interface...")
            subprocess.run([sys.executable, 'isc_ai_system/scripts/demos/isc_chat.py'])
        elif choice == '4':
            print("\n🎯 Starting Simple Chat Demo...")
            subprocess.run([sys.executable, 'isc_ai_system/scripts/demos/chat_demo.py'])
        elif choice == '5':
            print("\n🎯 Starting Training + Chat...")
            os.chdir('isc_ai_system')
            subprocess.run([sys.executable, 'scripts/training/self_referential_trainer_enhanced.py', '--chat'])
            os.chdir('..')
        elif choice == '6':
            print("\n🔬 Running Neuromorphic Analysis...")
            subprocess.run([sys.executable, 'analyze_neuromorphic_responses.py'])
        elif choice == '7':
            print("\n🔬 Starting Query Demo...")
            subprocess.run([sys.executable, 'isc_ai_system/scripts/demos/query_demo.py'])
        else:
            print("Invalid choice. Please try again.")

        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()