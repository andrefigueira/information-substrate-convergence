"""
Neural Language Model for ISC
A small LSTM-based model that generates fluent responses conditioned on substrate state.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from typing import List, Dict, Optional, Tuple
from collections import Counter
import pickle


class Vocabulary:
    """Handles word to index mapping"""

    def __init__(self, min_freq: int = 1):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        self.idx2word = {0: '<PAD>', 1: '<UNK>', 2: '<SOS>', 3: '<EOS>'}
        self.word_freq = Counter()
        self.min_freq = min_freq
        self.n_words = 4

    def add_sentence(self, sentence: str):
        """Add words from a sentence to vocabulary"""
        words = sentence.lower().split()
        self.word_freq.update(words)

    def build(self):
        """Build vocabulary from collected words"""
        for word, freq in self.word_freq.items():
            if freq >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = self.n_words
                self.idx2word[self.n_words] = word
                self.n_words += 1

    def encode(self, sentence: str) -> List[int]:
        """Convert sentence to list of indices"""
        words = sentence.lower().split()
        return [self.word2idx.get(w, self.word2idx['<UNK>']) for w in words]

    def decode(self, indices: List[int]) -> str:
        """Convert list of indices back to sentence"""
        words = []
        for idx in indices:
            if idx == self.word2idx['<EOS>']:
                break
            if idx not in [self.word2idx['<PAD>'], self.word2idx['<SOS>']]:
                words.append(self.idx2word.get(idx, '<UNK>'))
        return ' '.join(words)

    def save(self, path: str):
        """Save vocabulary to file"""
        data = {
            'word2idx': self.word2idx,
            'idx2word': {int(k): v for k, v in self.idx2word.items()},
            'n_words': self.n_words
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    def load(self, path: str):
        """Load vocabulary from file"""
        with open(path, 'r') as f:
            data = json.load(f)
        self.word2idx = data['word2idx']
        self.idx2word = {int(k): v for k, v in data['idx2word'].items()}
        self.n_words = data['n_words']


class SubstrateConditionedLM(nn.Module):
    """
    LSTM Language Model conditioned on substrate concept embeddings.

    Architecture:
    - Substrate embedding projection
    - Word embedding layer
    - LSTM layers
    - Output projection
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        substrate_dim: int = 384,  # Matches sentence-transformer dim
        n_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.substrate_dim = substrate_dim
        self.n_layers = n_layers

        # Word embeddings
        self.word_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Substrate context projection
        self.substrate_proj = nn.Sequential(
            nn.Linear(substrate_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # LSTM - input is word embed + projected substrate context
        self.lstm = nn.LSTM(
            input_size=embed_dim + hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better training"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(
        self,
        input_ids: torch.Tensor,
        substrate_embedding: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            input_ids: [batch_size, seq_len] word indices
            substrate_embedding: [batch_size, substrate_dim] context from substrate
            hidden: Optional LSTM hidden state

        Returns:
            logits: [batch_size, seq_len, vocab_size]
            hidden: Updated hidden state
        """
        batch_size, seq_len = input_ids.shape

        # Get word embeddings
        word_embeds = self.word_embed(input_ids)  # [batch, seq, embed_dim]

        # Project substrate context
        substrate_ctx = self.substrate_proj(substrate_embedding)  # [batch, hidden_dim]

        # Expand substrate context to match sequence length
        substrate_ctx = substrate_ctx.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq, hidden_dim]

        # Concatenate word embeddings with substrate context
        lstm_input = torch.cat([word_embeds, substrate_ctx], dim=-1)  # [batch, seq, embed_dim + hidden_dim]

        # LSTM forward
        if hidden is None:
            lstm_out, hidden = self.lstm(lstm_input)
        else:
            lstm_out, hidden = self.lstm(lstm_input, hidden)

        # Project to vocabulary
        logits = self.output_proj(lstm_out)  # [batch, seq, vocab_size]

        return logits, hidden

    def generate(
        self,
        substrate_embedding: torch.Tensor,
        vocab: Vocabulary,
        max_length: int = 50,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2
    ) -> str:
        """
        Generate a response given substrate context.

        Args:
            substrate_embedding: [substrate_dim] context vector
            vocab: Vocabulary object
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repeating tokens (1.0 = no penalty)

        Returns:
            Generated text string
        """
        self.eval()
        device = next(self.parameters()).device

        # Prepare substrate embedding
        if substrate_embedding.dim() == 1:
            substrate_embedding = substrate_embedding.unsqueeze(0)
        substrate_embedding = substrate_embedding.to(device)

        # Start with SOS token
        current_token = torch.tensor([[vocab.word2idx['<SOS>']]], device=device)
        hidden = None
        generated_tokens = []

        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                logits, hidden = self.forward(current_token, substrate_embedding, hidden)
                logits = logits[:, -1, :] / temperature  # [1, vocab_size]

                # Apply repetition penalty
                if repetition_penalty != 1.0 and generated_tokens:
                    for token_id in set(generated_tokens):
                        if logits[0, token_id] > 0:
                            logits[0, token_id] /= repetition_penalty
                        else:
                            logits[0, token_id] *= repetition_penalty

                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')

                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Check for EOS
                if next_token.item() == vocab.word2idx['<EOS>']:
                    break

                generated_tokens.append(next_token.item())
                current_token = next_token

        # Decode tokens to text
        text = vocab.decode(generated_tokens)

        # Post-process
        if text:
            text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
            if not text.endswith(('.', '!', '?')):
                text += '.'

        return text


class NeuralLanguageModelTrainer:
    """Handles training of the neural language model"""

    def __init__(
        self,
        model: SubstrateConditionedLM,
        vocab: Vocabulary,
        learning_rate: float = 0.001,
        device: str = None
    ):
        self.model = model
        self.vocab = vocab
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding

    def prepare_batch(
        self,
        responses: List[str],
        substrate_embeddings: List[np.ndarray],
        max_len: int = 64
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare a training batch"""
        batch_size = len(responses)

        # Encode responses with SOS and EOS
        encoded = []
        for resp in responses:
            tokens = [self.vocab.word2idx['<SOS>']] + self.vocab.encode(resp) + [self.vocab.word2idx['<EOS>']]
            encoded.append(tokens[:max_len])

        # Pad sequences
        max_seq_len = max(len(seq) for seq in encoded)
        input_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
        target_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long)

        for i, seq in enumerate(encoded):
            input_ids[i, :len(seq)-1] = torch.tensor(seq[:-1])
            target_ids[i, :len(seq)-1] = torch.tensor(seq[1:])

        # Stack substrate embeddings
        substrate_batch = torch.tensor(np.array(substrate_embeddings), dtype=torch.float32)

        return input_ids.to(self.device), target_ids.to(self.device), substrate_batch.to(self.device)

    def train_step(
        self,
        responses: List[str],
        substrate_embeddings: List[np.ndarray]
    ) -> float:
        """Single training step"""
        self.model.train()

        input_ids, target_ids, substrate_batch = self.prepare_batch(responses, substrate_embeddings)

        self.optimizer.zero_grad()

        logits, _ = self.model(input_ids, substrate_batch)

        # Reshape for loss calculation
        logits = logits.view(-1, self.vocab.n_words)
        target_ids = target_ids.view(-1)

        loss = self.criterion(logits, target_ids)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        return loss.item()

    def train_epoch(
        self,
        training_data: List[Tuple[str, str, np.ndarray]],
        batch_size: int = 16
    ) -> float:
        """Train for one epoch"""
        total_loss = 0
        n_batches = 0

        # Shuffle data
        import random
        random.shuffle(training_data)

        # Process in batches
        for i in range(0, len(training_data), batch_size):
            batch = training_data[i:i+batch_size]
            responses = [item[1] for item in batch]
            embeddings = [item[2] for item in batch]

            loss = self.train_step(responses, embeddings)
            total_loss += loss
            n_batches += 1

        avg_loss = total_loss / n_batches if n_batches > 0 else 0
        self.scheduler.step(avg_loss)

        return avg_loss

    def save(self, path: str):
        """Save model and vocabulary"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'vocab_n_words': self.vocab.n_words,
            'model_config': {
                'vocab_size': self.model.vocab_size,
                'embed_dim': self.model.embed_dim,
                'hidden_dim': self.model.hidden_dim,
                'substrate_dim': self.model.substrate_dim,
                'n_layers': self.model.n_layers
            }
        }, path)

        # Save vocabulary separately
        vocab_path = path.replace('.pt', '_vocab.json')
        self.vocab.save(vocab_path)

    def load(self, path: str):
        """Load model and vocabulary"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load vocabulary
        vocab_path = path.replace('.pt', '_vocab.json')
        if os.path.exists(vocab_path):
            self.vocab.load(vocab_path)


def create_model_and_trainer(
    vocab_size: int,
    substrate_dim: int = 384,
    embed_dim: int = 256,
    hidden_dim: int = 512,
    n_layers: int = 2,
    learning_rate: float = 0.001
) -> Tuple[SubstrateConditionedLM, Vocabulary, NeuralLanguageModelTrainer]:
    """Factory function to create model, vocabulary, and trainer"""
    vocab = Vocabulary(min_freq=1)
    model = SubstrateConditionedLM(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        substrate_dim=substrate_dim,
        n_layers=n_layers
    )
    trainer = NeuralLanguageModelTrainer(model, vocab, learning_rate)

    return model, vocab, trainer
