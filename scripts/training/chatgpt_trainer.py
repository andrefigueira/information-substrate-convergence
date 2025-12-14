#!/usr/bin/env python3
"""
ChatGPT-powered Training Data Generator for Neuromorphic ISC AI
Connects to OpenAI API to generate conversational training data
"""

import json
import os
import sys
import requests
import time
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

class ChatGPTTrainer:
    """Generate training data using ChatGPT API"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.model = "gpt-3.5-turbo"  # or "gpt-4" if you have access

        if not self.api_key:
            print("⚠ No OpenAI API key found!")
            print("Set your API key: export OPENAI_API_KEY='your-key-here'")
            print("Or get one at: https://platform.openai.com/api-keys")
            sys.exit(1)

        # Load neuromorphic core for training
        try:
            sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))
            from isc.neuromorphic_core import NeuromorphicISCCore
            self.core = NeuromorphicISCCore()
            self.core.verbose = True
            print("🧠 Neuromorphic core loaded")
        except Exception as e:
            print(f"⚠ Could not load neuromorphic core: {e}")
            self.core = None

    def generate_conversation_with_chatgpt(self, topic: str, context: str = "") -> Optional[Dict]:
        """Generate a conversation pair using ChatGPT"""

        system_prompt = f"""You are helping train a neuromorphic AI system based on Information Substrate Convergence theory. Generate realistic conversation pairs about {topic}.

Context: {context}

The AI being trained:
- Has a dynamic information substrate (knowledge graph)
- Measures consciousness through phi (information integration)
- Uses neuromorphic processing with spike-driven updates
- Learns through substrate evolution and community detection

Generate a realistic user question about {topic} and provide a thoughtful response that demonstrates deep understanding while being conversational. The response should be informative but not robotic.

Format your response as:
USER: [realistic user question]
AI: [thoughtful, conversational response]"""

        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }

            payload = {
                'model': self.model,
                'messages': [
                    {'role': 'system', 'content': system_prompt}
                ],
                'max_tokens': 500,
                'temperature': 0.8
            }

            response = requests.post(self.base_url, headers=headers, json=payload)

            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']

                # Parse the response to extract user/AI parts
                lines = content.strip().split('\n')
                user_input = ""
                ai_response = ""

                for line in lines:
                    if line.startswith('USER:'):
                        user_input = line[5:].strip()
                    elif line.startswith('AI:'):
                        ai_response = line[3:].strip()

                if user_input and ai_response:
                    return {
                        'input': user_input,
                        'output': ai_response,
                        'context': f"{topic}_{context}".replace(' ', '_'),
                        'generated_by': 'chatgpt',
                        'model': self.model,
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    print(f"⚠ Could not parse ChatGPT response: {content}")
                    return None

            else:
                print(f"⚠ ChatGPT API error: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            print(f"⚠ Error calling ChatGPT API: {e}")
            return None

    def generate_training_batch(self, topics: List[str], conversations_per_topic: int = 5) -> List[Dict]:
        """Generate a batch of training conversations"""
        training_data = []

        print(f"🎓 Generating training data with ChatGPT...")
        print(f"📚 Topics: {', '.join(topics)}")
        print(f"💬 {conversations_per_topic} conversations per topic")

        for topic in topics:
            print(f"\n🔍 Generating conversations for: {topic}")

            for i in range(conversations_per_topic):
                print(f"  [{i+1}/{conversations_per_topic}] Generating conversation...", end=' ')

                conversation = self.generate_conversation_with_chatgpt(topic)

                if conversation:
                    training_data.append(conversation)
                    print("✓")
                else:
                    print("❌")

                # Rate limiting - be nice to the API
                time.sleep(1)

        print(f"\n🎉 Generated {len(training_data)} training conversations")
        return training_data

    def save_and_train(self, training_data: List[Dict], filename: Optional[str] = None):
        """Save training data and train the neuromorphic AI"""
        if not training_data:
            print("❌ No training data to save")
            return

        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chatgpt_training_{timestamp}.json"

        # Create training file format
        training_file = {
            "context": "ChatGPT-generated training data for neuromorphic ISC AI",
            "metadata": {
                "created": datetime.now().isoformat(),
                "purpose": "ChatGPT automated training",
                "training_type": "chatgpt_generated",
                "total_conversations": len(training_data),
                "model_used": self.model
            },
            "conversations": training_data
        }

        # Save to file
        try:
            with open(filename, 'w') as f:
                json.dump(training_file, f, indent=2)
            print(f"💾 Training data saved to: {filename}")
        except Exception as e:
            print(f"❌ Failed to save training data: {e}")
            return

        # Train the neuromorphic AI
        if self.core:
            try:
                print(f"🧠 Training neuromorphic AI with {len(training_data)} conversations...")

                # Convert to format expected by core
                conversation_pairs = [(conv['input'], conv['output']) for conv in training_data]
                context = f"chatgpt_generated_{len(training_data)}_conversations"

                self.core.train_on_conversation(conversation_pairs, context)
                print("✅ Training completed successfully!")

                # Show updated status
                status = self.core.get_status()
                substrate = status['substrate']
                metrics = status['metrics']

                print(f"\n📊 Updated AI Status:")
                print(f"🧠 Substrate: {substrate['node_count']} concepts, {substrate['edge_count']} edges")
                print(f"🎓 Training sessions: {metrics.get('training_sessions', 0)}")
                print(f"📚 Total examples: {metrics.get('total_training_examples', 0)}")
                print(f"🔢 Phi (φ): {metrics['phi_value']:.3f}")

            except Exception as e:
                print(f"⚠ Training failed: {e}")
                print(f"💡 You can manually train later with: make ai-train-file {filename}")

    def interactive_training_session(self):
        """Interactive training session with topic selection"""
        print("🤖 ChatGPT-Powered AI Training Session")
        print("=" * 50)

        # Predefined topic suggestions
        suggested_topics = [
            "consciousness and awareness",
            "artificial intelligence and machine learning",
            "philosophy and ethics",
            "science and technology",
            "creativity and problem solving",
            "human psychology and behavior",
            "future of technology",
            "quantum computing and physics",
            "neuroscience and cognition",
            "climate and environment"
        ]

        print("🎯 Suggested topics:")
        for i, topic in enumerate(suggested_topics, 1):
            print(f"  {i}. {topic}")

        print(f"\n💡 Or enter custom topics (comma-separated)")

        # Get user input
        user_input = input("\n📝 Topics (numbers or custom text): ").strip()

        if not user_input:
            print("❌ No topics provided")
            return

        # Parse topics
        selected_topics = []

        if user_input.replace(',', '').replace(' ', '').isdigit():
            # User selected numbers
            numbers = [int(x.strip()) for x in user_input.split(',') if x.strip().isdigit()]
            for num in numbers:
                if 1 <= num <= len(suggested_topics):
                    selected_topics.append(suggested_topics[num - 1])
        else:
            # User provided custom topics
            selected_topics = [topic.strip() for topic in user_input.split(',') if topic.strip()]

        if not selected_topics:
            print("❌ No valid topics selected")
            return

        # Get number of conversations per topic
        try:
            conv_count = int(input(f"\n💬 Conversations per topic (default 3): ").strip() or "3")
            conv_count = max(1, min(conv_count, 10))  # Limit to reasonable range
        except:
            conv_count = 3

        # Generate training data
        training_data = self.generate_training_batch(selected_topics, conv_count)

        # Save and train
        if training_data:
            self.save_and_train(training_data)
        else:
            print("❌ No training data generated")

def main():
    """Main entry point"""
    print("🤖 ChatGPT Training Data Generator")
    print("Connects to OpenAI API to generate training conversations")

    # Check for API key
    if not os.getenv('OPENAI_API_KEY'):
        print("\n⚠ Setup Required:")
        print("1. Get an API key from: https://platform.openai.com/api-keys")
        print("2. Set it in your environment: export OPENAI_API_KEY='your-key-here'")
        print("3. Or add it to your shell profile for persistence")
        return

    trainer = ChatGPTTrainer()

    print("\n🎯 Training Modes:")
    print("1. Interactive topic selection")
    print("2. Quick training (common topics)")
    print("3. Custom batch training")

    try:
        choice = input("\nChoice (1-3): ").strip()

        if choice == "1":
            trainer.interactive_training_session()

        elif choice == "2":
            # Quick training with common topics
            quick_topics = [
                "consciousness and AI",
                "technology and society",
                "science and discovery",
                "creativity and innovation"
            ]
            training_data = trainer.generate_training_batch(quick_topics, 3)
            trainer.save_and_train(training_data)

        elif choice == "3":
            topics_input = input("📝 Enter topics (comma-separated): ").strip()
            if topics_input:
                topics = [t.strip() for t in topics_input.split(',')]
                conv_count = int(input("💬 Conversations per topic (default 3): ").strip() or "3")
                training_data = trainer.generate_training_batch(topics, conv_count)
                trainer.save_and_train(training_data)
            else:
                print("❌ No topics provided")
        else:
            print("❌ Invalid choice")

    except KeyboardInterrupt:
        print("\n\n👋 Training cancelled")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()