#!/usr/bin/env python3
"""
Interactive Teaching Mode for ISC.

Talk with the system, and when you don't like a response,
you can teach it a better one. The system learns in real-time.

Commands:
  /teach - Teach a better response for the last query
  /stats - Show language learning statistics
  /save  - Save current state
  /quit  - Exit

Example:
  You: Hello!
  ISC: [some response you don't like]
  You: /teach
  Better response: Hi there! Great to see you. How can I help?
  ISC: Got it! I'll learn from that.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from isc.neuromorphic_core import NeuromorphicISCCore


class InteractiveTeacher:
    def __init__(self):
        print("=" * 60)
        print("🧠 ISC Interactive Teaching Mode")
        print("=" * 60)
        print("\nCommands:")
        print("  /teach  - Teach a better response for the last query")
        print("  /stats  - Show language learning statistics")
        print("  /save   - Save current state")
        print("  /quit   - Exit")
        print("\nLoading ISC system...")

        self.core = NeuromorphicISCCore()
        self.core.start_session()

        self.last_query = None
        self.last_response = None

        stats = self.core.get_status()
        lang_stats = self.core.substrate.get_language_stats()

        print(f"\n📊 System loaded:")
        print(f"   Concepts: {stats['substrate']['node_count']}")
        print(f"   Language patterns: {lang_stats['total_patterns']}")
        print(f"   Response templates: {lang_stats['response_templates']}")
        print(f"\n" + "=" * 60)
        print("Start chatting! Type /teach after any response to improve it.")
        print("=" * 60 + "\n")

    def handle_command(self, cmd):
        """Handle special commands"""
        cmd = cmd.strip().lower()

        if cmd == '/quit' or cmd == '/exit':
            print("\n👋 Saving and exiting...")
            self.core.save_state(self.core.persistent_state_path)
            print("💾 State saved. Goodbye!")
            return False

        elif cmd == '/save':
            self.core.save_state(self.core.persistent_state_path)
            print("💾 State saved!")
            return True

        elif cmd == '/stats':
            stats = self.core.get_status()
            lang_stats = self.core.substrate.get_language_stats()
            print(f"\n📊 Statistics:")
            print(f"   Concepts: {stats['substrate']['node_count']}")
            print(f"   Edges: {stats['substrate']['edge_count']}")
            print(f"   Phi: {stats['metrics']['phi_value']:.4f}")
            print(f"   Language patterns: {lang_stats['total_patterns']}")
            print(f"   Response templates: {lang_stats['response_templates']}")
            print(f"   Phrase transitions: {lang_stats['phrase_transitions']}\n")
            return True

        elif cmd == '/teach':
            if not self.last_query:
                print("❌ No previous query to teach from. Say something first!")
                return True

            print(f"\n📝 Teaching mode for: \"{self.last_query}\"")
            print(f"   (Previous response: \"{self.last_response[:50]}...\")" if len(self.last_response) > 50 else f"   (Previous response: \"{self.last_response}\")")

            better = input("Better response: ").strip()
            if better:
                # Learn from the correction
                self.core.substrate.learn_language_patterns(
                    self.last_query,
                    better,
                    context="interactive_teaching"
                )

                # Also train as a conversation pair
                self.core.train_on_conversation(
                    [(self.last_query, better)],
                    context="interactive_correction"
                )

                print(f"✅ Learned! I'll respond better next time.\n")
            else:
                print("❌ No response provided, skipping.\n")
            return True

        elif cmd == '/help':
            print("\nCommands:")
            print("  /teach  - Teach a better response for the last query")
            print("  /stats  - Show language learning statistics")
            print("  /save   - Save current state")
            print("  /quit   - Exit\n")
            return True

        return None  # Not a command

    def chat(self, user_input):
        """Process chat input and return response"""
        self.last_query = user_input
        response = self.core.process_input(user_input)
        self.last_response = response
        return response

    def run(self):
        """Main interaction loop"""
        try:
            while True:
                try:
                    user_input = input("You: ").strip()
                except EOFError:
                    break

                if not user_input:
                    continue

                # Check for commands
                if user_input.startswith('/'):
                    result = self.handle_command(user_input)
                    if result is False:  # Quit command
                        break
                    elif result is True:  # Command handled
                        continue
                    # If None, treat as regular input

                # Regular chat
                response = self.chat(user_input)
                print(f"ISC: {response}\n")

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Saving...")
            self.core.save_state(self.core.persistent_state_path)
            print("💾 State saved. Goodbye!")


def main():
    teacher = InteractiveTeacher()
    teacher.run()


if __name__ == "__main__":
    main()
