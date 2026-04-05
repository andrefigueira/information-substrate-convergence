#!/usr/bin/env python3
"""
Train the Neural Language Model on conversation data.

This trains a small LSTM model conditioned on substrate embeddings,
allowing the ISC system to generate fluent responses while keeping
understanding in the substrate.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import torch
import numpy as np
from sentence_transformers import SentenceTransformer

from isc.neural_language_model import (
    SubstrateConditionedLM,
    Vocabulary,
    NeuralLanguageModelTrainer,
    create_model_and_trainer
)
from large_training_dataset import TRAINING_DATA


def main():
    print("=" * 60)
    print("Neural Language Model Training")
    print("=" * 60)

    # Initialize embedding model
    print("\nLoading sentence transformer for substrate embeddings...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    substrate_dim = 384  # MiniLM embedding dimension

    # Build vocabulary from training data
    print("Building vocabulary...")
    vocab = Vocabulary(min_freq=1)

    for query, response in TRAINING_DATA:
        vocab.add_sentence(response)

    vocab.build()
    print(f"  Vocabulary size: {vocab.n_words} words")

    # Create model and trainer
    print("\nInitializing model...")
    model = SubstrateConditionedLM(
        vocab_size=vocab.n_words,
        embed_dim=256,
        hidden_dim=512,
        substrate_dim=substrate_dim,
        n_layers=2,
        dropout=0.3
    )
    trainer = NeuralLanguageModelTrainer(model, vocab, learning_rate=0.001)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")
    print(f"  Device: {trainer.device}")

    # Prepare training data with embeddings
    print("\nPreparing training data with substrate embeddings...")
    training_data = []

    for i, (query, response) in enumerate(TRAINING_DATA):
        # Get substrate embedding from query
        embedding = encoder.encode([query])[0]
        training_data.append((query, response, embedding))

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(TRAINING_DATA)} pairs")

    print(f"  Total training pairs: {len(training_data)}")

    # Training loop
    n_epochs = 100
    batch_size = 8  # Smaller batches for better convergence
    best_loss = float('inf')

    print(f"\nTraining for {n_epochs} epochs...")
    print("-" * 40)

    for epoch in range(n_epochs):
        loss = trainer.train_epoch(training_data, batch_size=batch_size)

        if loss < best_loss:
            best_loss = loss
            # Save best model
            save_path = Path(__file__).parent.parent.parent / "models" / "neural_lm.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            trainer.save(str(save_path))

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1:3d}: loss = {loss:.4f} (best: {best_loss:.4f})")

            # Test generation
            test_queries = ["Hello!", "How are you?", "What is consciousness?"]
            print("  Sample generations:")

            model.eval()
            for test_q in test_queries:
                test_emb = torch.tensor(encoder.encode([test_q])[0])
                generated = model.generate(test_emb, vocab, temperature=0.8)
                print(f"    Q: {test_q}")
                print(f"    A: {generated}")

    print("-" * 40)
    print(f"\nTraining complete!")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Model saved to: models/neural_lm.pt")

    # Final test
    print("\n" + "=" * 60)
    print("Final Generation Tests")
    print("=" * 60)

    model.eval()
    test_queries = [
        "Hello!",
        "How are you doing?",
        "What are you?",
        "Are you conscious?",
        "I'm feeling sad",
        "What is the meaning of life?",
        "Help me think",
        "Goodbye"
    ]

    for query in test_queries:
        emb = torch.tensor(encoder.encode([query])[0])
        response = model.generate(emb, vocab, temperature=0.7, top_k=40, top_p=0.9)
        print(f"\nQ: {query}")
        print(f"A: {response}")


if __name__ == "__main__":
    main()
