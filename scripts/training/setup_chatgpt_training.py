#!/usr/bin/env python3
"""
Setup script for ChatGPT-powered AI training
"""

import os
import sys
from pathlib import Path

def setup_chatgpt_training():
    """Guide user through ChatGPT training setup"""

    print("🤖 ChatGPT-Powered AI Training Setup")
    print("=" * 50)

    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')

    if not api_key:
        print("❌ OpenAI API Key Not Found")
        print("\n📋 Setup Steps:")
        print("1. Visit: https://platform.openai.com/api-keys")
        print("2. Create a new API key")
        print("3. Set it in your environment:")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        print("\n💡 For permanent setup, add to your shell profile:")
        print("   echo 'export OPENAI_API_KEY=\"your-key\"' >> ~/.bashrc")
        print("   echo 'export OPENAI_API_KEY=\"your-key\"' >> ~/.zshrc")

        # Ask if they want to set it now
        print("\n🔧 Set API key now?")
        try:
            key_input = input("Enter your OpenAI API key (or press Enter to skip): ").strip()
        except (EOFError, KeyboardInterrupt):
            key_input = ""
            print("Skipping API key input (non-interactive mode)")

        if key_input:
            # Write to shell profile
            shell = os.getenv('SHELL', '').split('/')[-1]
            if shell in ['zsh', 'bash']:
                profile_file = f"~/.{shell}rc"
                profile_path = Path(profile_file).expanduser()

                try:
                    with open(profile_path, 'a') as f:
                        f.write(f'\n# OpenAI API Key for neuromorphic AI training\n')
                        f.write(f'export OPENAI_API_KEY="{key_input}"\n')

                    print(f"✅ API key added to {profile_file}")
                    print("🔄 Restart your terminal or run: source ~/.zshrc")

                    # Set for current session
                    os.environ['OPENAI_API_KEY'] = key_input
                    api_key = key_input

                except Exception as e:
                    print(f"⚠ Could not write to {profile_file}: {e}")
                    print(f"💡 Manually add: export OPENAI_API_KEY='{key_input}'")

                    # Set for current session anyway
                    os.environ['OPENAI_API_KEY'] = key_input
                    api_key = key_input

        if not api_key:
            print("\n❌ Cannot proceed without API key")
            return False

    else:
        print("✅ OpenAI API Key Found")
        # Mask the key for display
        masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
        print(f"🔑 Key: {masked_key}")

    print("\n🎯 Available Training Modes:")
    print("1. make ai-train-quick     - Quick training with common topics")
    print("2. make ai-train-chatgpt   - Interactive topic selection")
    print("3. make ai-continuous      - Enable learning from all chats")

    print("\n🚀 Recommended First Steps:")
    print("1. Run: make ai-train-quick")
    print("2. Run: make ai-continuous")
    print("3. Run: make ai")

    # Ask if they want to run quick training now (only in interactive mode)
    try:
        run_training = input("\n🎓 Run quick training now? (y/N): ").strip().lower() in ['y', 'yes']
    except (EOFError, KeyboardInterrupt):
        run_training = False
        print("\n💡 Skipping interactive training (non-interactive mode)")

    if run_training:
        try:
            from chatgpt_trainer import ChatGPTTrainer

            trainer = ChatGPTTrainer()
            topics = ['consciousness and AI', 'technology and philosophy', 'creativity and intelligence']

            print(f"\n🧠 Generating training conversations...")
            training_data = trainer.generate_training_batch(topics, 2)

            if training_data:
                trainer.save_and_train(training_data)
                print("\n🎉 Quick training completed!")
                print("💬 Your AI is now ready with ChatGPT-generated knowledge")
                print("🚀 Run 'make ai' to start chatting!")
            else:
                print("❌ Training failed")

        except Exception as e:
            print(f"⚠ Training error: {e}")
            print("💡 Try manually: make ai-train-quick")

    return True

if __name__ == "__main__":
    setup_chatgpt_training()