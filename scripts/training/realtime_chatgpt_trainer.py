#!/usr/bin/env python3
"""
Real-time ChatGPT to Neuromorphic AI Conversation Trainer
Facilitates live conversations between ChatGPT and your neuromorphic AI
"""

import json
import os
import sys
import requests
import time
from datetime import datetime
from typing import List, Dict, Optional
import threading
from collections import deque

class RealtimeChatGPTTrainer:
    """Real-time conversation trainer between ChatGPT and Neuromorphic AI"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.model = "gpt-3.5-turbo"

        if not self.api_key:
            print("⚠ No OpenAI API key found!")
            print("Set your API key: export OPENAI_API_KEY='your-key-here'")
            sys.exit(1)

        # Load neuromorphic core
        try:
            sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))
            from isc.neuromorphic_core import NeuromorphicISCCore
            self.ai_core = NeuromorphicISCCore()
            self.ai_core.verbose = True
            print("🧠 Neuromorphic AI Core loaded")
        except Exception as e:
            print(f"❌ Could not load neuromorphic core: {e}")
            sys.exit(1)

        # Conversation tracking
        self.conversation_history = []
        self.chatgpt_context = []
        self.training_log = []
        self.conversation_count = 0

    def call_chatgpt(self, message: str, context: List[Dict] = None) -> Optional[str]:
        """Call ChatGPT API with message and context"""
        try:
            # Prepare conversation context
            messages = context or []
            messages.append({'role': 'user', 'content': message})

            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }

            payload = {
                'model': self.model,
                'messages': messages,
                'max_tokens': 300,
                'temperature': 0.7
            }

            response = requests.post(self.base_url, headers=headers, json=payload)

            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            else:
                print(f"❌ ChatGPT API error: {response.status_code}")
                return None

        except Exception as e:
            print(f"❌ Error calling ChatGPT: {e}")
            return None

    def get_ai_response(self, message: str) -> str:
        """Get response from the neuromorphic AI"""
        try:
            response = self.ai_core.process_input(message)
            return response
        except Exception as e:
            print(f"❌ Error getting AI response: {e}")
            return "I'm having trouble processing that right now."

    def display_conversation_turn(self, speaker: str, message: str, turn_number: int):
        """Display a conversation turn with formatting"""
        timestamp = datetime.now().strftime("%H:%M:%S")

        if speaker == "ChatGPT":
            print(f"\n{'='*60}")
            print(f"🤖 ChatGPT [{turn_number}] ({timestamp}):")
            print(f"{'='*60}")
            print(f"💬 {message}")
        else:
            print(f"\n{'-'*60}")
            print(f"🧠 Neuromorphic AI [{turn_number}] ({timestamp}):")
            print(f"{'-'*60}")
            print(f"🎯 {message}")

        # Show AI status after each response
        if speaker == "Neuromorphic AI":
            try:
                status = self.ai_core.get_status()
                substrate = status['substrate']
                metrics = status['metrics']
                print(f"\n📊 AI Status: {substrate['node_count']} concepts, φ={metrics['phi_value']:.4f}")
            except:
                pass

    def setup_chatgpt_persona(self) -> List[Dict]:
        """Setup ChatGPT's conversational persona and context"""
        system_prompt = """You are having a real conversation with a neuromorphic AI based on Information Substrate Convergence theory. This AI:

- Has a dynamic information substrate (evolving knowledge graph)
- Measures consciousness through phi (information integration)
- Uses neuromorphic processing with spike-driven updates
- Learns continuously from conversations

Your role:
- Be curious and ask thoughtful questions
- Build on the AI's responses naturally
- Explore topics like consciousness, intelligence, learning, philosophy
- Keep responses conversational and under 200 words
- Show genuine interest in the AI's perspective

Start the conversation by introducing yourself and asking about the AI's experience with consciousness or learning."""

        return [{'role': 'system', 'content': system_prompt}]

    def run_realtime_conversation(self, max_turns: int = 10, topic: str = "consciousness and AI"):
        """Run a real-time conversation session"""
        print("\n🚀 Starting Real-time ChatGPT ↔ Neuromorphic AI Conversation")
        print(f"🎯 Topic focus: {topic}")
        print(f"🔄 Max turns: {max_turns}")
        print("📺 Watch the conversation unfold below...\n")

        # Show initial AI status
        print("🧠 Initial AI Status:")
        self.show_ai_status()

        # Setup ChatGPT context
        self.chatgpt_context = self.setup_chatgpt_persona()

        # Add topic guidance
        topic_message = f"Please start a conversation with the neuromorphic AI about {topic}. Introduce yourself and ask an engaging opening question."

        session_start = datetime.now()
        session_log = {
            'session_id': f"realtime_{session_start.strftime('%Y%m%d_%H%M%S')}",
            'topic': topic,
            'start_time': session_start.isoformat(),
            'max_turns': max_turns,
            'conversation': []
        }

        try:
            for turn in range(1, max_turns + 1):
                print(f"\n{'='*80}")
                print(f"🔄 CONVERSATION TURN {turn}/{max_turns}")
                print(f"{'='*80}")

                # Get ChatGPT's message
                if turn == 1:
                    chatgpt_msg = self.call_chatgpt(topic_message, self.chatgpt_context)
                else:
                    # For subsequent turns, ChatGPT responds to the AI's last message
                    last_ai_response = self.conversation_history[-1]['ai_response']
                    prompt = f"Respond to the AI's message: '{last_ai_response}'"
                    chatgpt_msg = self.call_chatgpt(prompt, self.chatgpt_context)

                if not chatgpt_msg:
                    print("❌ Failed to get ChatGPT response")
                    break

                # Display ChatGPT's message
                self.display_conversation_turn("ChatGPT", chatgpt_msg, turn)

                # Add to ChatGPT context
                self.chatgpt_context.append({'role': 'assistant', 'content': chatgpt_msg})

                # Get AI response
                print("\n🔄 Processing with Neuromorphic AI...")
                ai_response = self.get_ai_response(chatgpt_msg)

                # Display AI's response
                self.display_conversation_turn("Neuromorphic AI", ai_response, turn)

                # Add AI response to ChatGPT context for next turn
                self.chatgpt_context.append({'role': 'user', 'content': ai_response})

                # Log the exchange
                exchange = {
                    'turn': turn,
                    'timestamp': datetime.now().isoformat(),
                    'chatgpt_message': chatgpt_msg,
                    'ai_response': ai_response
                }

                self.conversation_history.append(exchange)
                session_log['conversation'].append(exchange)

                # Train the AI on this exchange immediately
                try:
                    training_pair = [(chatgpt_msg, ai_response)]
                    context = f"realtime_chatgpt_conversation_turn_{turn}"
                    self.ai_core.train_on_conversation(training_pair, context)
                    print(f"   ✅ AI trained on exchange {turn}")
                except Exception as e:
                    print(f"   ⚠ Training failed for turn {turn}: {e}")

                # Brief pause for readability
                time.sleep(2)

            # Session complete
            session_log['end_time'] = datetime.now().isoformat()
            session_log['total_turns'] = len(self.conversation_history)

            print(f"\n{'='*80}")
            print(f"✅ CONVERSATION SESSION COMPLETE")
            print(f"{'='*80}")
            print(f"📊 Total exchanges: {len(self.conversation_history)}")

            # Save session log
            self.save_session_log(session_log)

            # Show final AI status
            self.show_ai_status()

        except KeyboardInterrupt:
            print(f"\n\n⏹️ Conversation interrupted by user")
            session_log['interrupted'] = True
            session_log['end_time'] = datetime.now().isoformat()
            self.save_session_log(session_log)

    def save_session_log(self, session_log: Dict):
        """Save the conversation session to a file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"realtime_session_{timestamp}.json"

        try:
            with open(filename, 'w') as f:
                json.dump(session_log, f, indent=2)
            print(f"💾 Session saved to: {filename}")
        except Exception as e:
            print(f"❌ Failed to save session: {e}")

    def show_ai_status(self):
        """Display current AI status and metrics"""
        try:
            status = self.ai_core.get_status()
            substrate = status['substrate']
            metrics = status['metrics']

            print(f"\n📊 Final Neuromorphic AI Status:")
            print(f"🧠 Substrate: {substrate['node_count']} concepts, {substrate['edge_count']} edges")
            print(f"🔢 Phi (φ): {metrics['phi_value']:.6f}")
            print(f"💬 Total conversations: {substrate.get('conversation_count', 0)}")
            print(f"🎓 Training sessions: {metrics.get('training_sessions', 0)}")

        except Exception as e:
            print(f"❌ Could not get AI status: {e}")

    def interactive_session_setup(self):
        """Interactive setup for conversation session"""
        print("\n🎯 Real-time Conversation Setup")
        print("="*50)

        # Topic selection
        suggested_topics = [
            "consciousness and self-awareness",
            "the nature of intelligence",
            "learning and memory",
            "creativity and problem solving",
            "the future of AI",
            "philosophy of mind",
            "information theory",
            "neural networks vs neuromorphic processing"
        ]

        print("\n📚 Suggested topics:")
        for i, topic in enumerate(suggested_topics, 1):
            print(f"  {i}. {topic}")

        topic_input = input(f"\n💭 Choose topic (1-{len(suggested_topics)}) or enter custom: ").strip()

        if topic_input.isdigit() and 1 <= int(topic_input) <= len(suggested_topics):
            topic = suggested_topics[int(topic_input) - 1]
        elif topic_input:
            topic = topic_input
        else:
            topic = "consciousness and AI"

        # Number of turns
        try:
            turns = int(input("\n🔄 Number of conversation turns (default 8): ").strip() or "8")
            turns = max(3, min(turns, 20))  # Reasonable bounds
        except:
            turns = 8

        print(f"\n✅ Setup complete!")
        print(f"🎯 Topic: {topic}")
        print(f"🔄 Turns: {turns}")

        return topic, turns

def main():
    """Main entry point"""
    print("🤖 Real-time ChatGPT ↔ Neuromorphic AI Trainer")
    print("Watch live conversations between ChatGPT and your AI!")

    if not os.getenv('OPENAI_API_KEY'):
        print("\n⚠ Setup Required:")
        print("Set your API key: export OPENAI_API_KEY='your-key-here'")
        return

    trainer = RealtimeChatGPTTrainer()

    print("\n🎯 Training Modes:")
    print("1. Interactive session setup")
    print("2. Quick session (8 turns, consciousness topic)")
    print("3. Extended session (15 turns, custom topic)")

    try:
        choice = input("\nChoice (1-3): ").strip()

        if choice == "1":
            topic, turns = trainer.interactive_session_setup()
            trainer.run_realtime_conversation(turns, topic)

        elif choice == "2":
            trainer.run_realtime_conversation(8, "consciousness and self-awareness")

        elif choice == "3":
            topic = input("Enter conversation topic: ").strip() or "artificial intelligence"
            trainer.run_realtime_conversation(15, topic)

        else:
            print("❌ Invalid choice")

    except KeyboardInterrupt:
        print("\n👋 Trainer cancelled")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()