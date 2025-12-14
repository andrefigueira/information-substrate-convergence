#!/usr/bin/env python3
"""
Quick ChatGPT training script
"""

import os
import sys

def quick_training():
    if not os.getenv('OPENAI_API_KEY'):
        print('⚠ Set API key: export OPENAI_API_KEY=your-key')
        return

    sys.path.append('.')
    from chatgpt_trainer import ChatGPTTrainer

    trainer = ChatGPTTrainer()
    topics = ['consciousness and AI', 'technology and philosophy', 'future of intelligence']
    data = trainer.generate_training_batch(topics, 2)
    trainer.save_and_train(data)

if __name__ == "__main__":
    quick_training()