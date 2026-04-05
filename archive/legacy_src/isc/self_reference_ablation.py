"""
Rigorous Self-Reference Ablation Study

This module tests whether self-reference provides measurable advantages by:
1. Training both architectures on identical tasks
2. Evaluating on held-out test problems
3. Measuring generalization to novel problem types
4. Using proper statistical controls

The key insight: untrained random networks don't capture the self-reference advantage.
Self-reference should matter for LEARNING and GENERALIZATION, not static properties.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import stats
import random
from collections import defaultdict


@dataclass
class AblationResult:
    """Result of ablation study"""
    metric: str
    self_ref_score: float
    baseline_score: float
    difference: float
    p_value: float
    effect_size: float
    significant: bool
    favors: str  # "self_ref", "baseline", or "neither"


class SelfRefEncoder(nn.Module):
    """Self-referential encoder with observer layers"""

    def __init__(self, input_dim: int = 64, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.observers = nn.ModuleList()
        self.meta_weights = nn.ParameterList()

        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            self.layers.append(nn.Linear(in_d, hidden_dim))
            self.observers.append(nn.Linear(hidden_dim, hidden_dim))
            self.meta_weights.append(nn.Parameter(torch.ones(hidden_dim)))

        self.output = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer, observer, meta in zip(self.layers, self.observers, self.meta_weights):
            h = layer(h)
            # Self-observation and meta-modulation
            observed = observer(h)
            h = h * meta + 0.1 * observed
            h = F.relu(h)
        return self.output(h)


class BaselineEncoder(nn.Module):
    """Baseline encoder without self-reference (matched capacity)"""

    def __init__(self, input_dim: int = 64, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        # Add extra capacity to match self-ref parameter count
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            self.layers.append(nn.Linear(in_d, hidden_dim))
            # Extra layer to match observer params
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.output = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i in range(0, len(self.layers), 2):
            h = F.relu(self.layers[i](h))
            h = F.relu(self.layers[i + 1](h))
        return self.output(h)


class SequencePredictionTask:
    """
    Task: Given a sequence pattern, predict the next element.
    Tests: Pattern recognition, generalization to longer sequences.
    """

    def __init__(self, seq_len: int = 8, vocab_size: int = 10):
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def generate_pattern(self, pattern_type: str, length: int) -> List[int]:
        """Generate different pattern types"""
        if pattern_type == "repeat":
            # A, B, A, B, A, B...
            base = random.sample(range(self.vocab_size), 2)
            return [base[i % 2] for i in range(length)]

        elif pattern_type == "increment":
            # 1, 2, 3, 4, 5...
            start = random.randint(0, self.vocab_size - length)
            return [(start + i) % self.vocab_size for i in range(length)]

        elif pattern_type == "fibonacci_mod":
            # Fibonacci mod vocab_size
            seq = [1, 1]
            for i in range(length - 2):
                seq.append((seq[-1] + seq[-2]) % self.vocab_size)
            return seq[:length]

        elif pattern_type == "reverse_copy":
            # A, B, C, C, B, A
            half = length // 2
            first = [random.randint(0, self.vocab_size - 1) for _ in range(half)]
            return first + first[::-1]

        else:
            return [random.randint(0, self.vocab_size - 1) for _ in range(length)]

    def create_dataset(self, n_samples: int, pattern_types: List[str]
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create training/test data"""
        X, y = [], []

        for _ in range(n_samples):
            pattern_type = random.choice(pattern_types)
            seq = self.generate_pattern(pattern_type, self.seq_len + 1)

            # Input: sequence[:-1], Target: sequence[-1]
            x_seq = seq[:-1]
            target = seq[-1]

            # One-hot encode
            x_onehot = np.zeros((self.seq_len, self.vocab_size))
            for i, val in enumerate(x_seq):
                x_onehot[i, val] = 1

            X.append(x_onehot.flatten())
            y.append(target)

        return torch.FloatTensor(np.array(X)), torch.LongTensor(y)


class AnalogicalReasoningTask:
    """
    Task: A is to B as C is to ?
    Tests: Relational reasoning, abstraction.
    """

    def __init__(self, embed_dim: int = 16):
        self.embed_dim = embed_dim
        # Create concept embeddings
        self.concepts = {}
        concept_names = ['dog', 'puppy', 'cat', 'kitten', 'big', 'small',
                        'hot', 'cold', 'up', 'down', 'good', 'bad',
                        'man', 'woman', 'king', 'queen', 'boy', 'girl']
        for name in concept_names:
            self.concepts[name] = np.random.randn(embed_dim)

        # Define relations
        self.relations = {
            'young_of': [('dog', 'puppy'), ('cat', 'kitten'), ('man', 'boy'), ('woman', 'girl')],
            'opposite': [('big', 'small'), ('hot', 'cold'), ('up', 'down'), ('good', 'bad')],
            'gender': [('man', 'woman'), ('king', 'queen'), ('boy', 'girl')]
        }

    def create_dataset(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create analogy problems: A:B::C:?"""
        X, y = [], []

        for _ in range(n_samples):
            # Pick a relation
            rel_type = random.choice(list(self.relations.keys()))
            pairs = self.relations[rel_type]

            if len(pairs) < 2:
                continue

            # Pick two pairs
            pair1, pair2 = random.sample(pairs, 2)

            # A:B::C:D
            A, B = pair1
            C, D = pair2

            # Input: concat(A, B, C)
            x = np.concatenate([self.concepts[A], self.concepts[B], self.concepts[C]])

            # Output: D embedding
            y_vec = self.concepts[D]

            X.append(x)
            y.append(y_vec)

        return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y))


class SelfReferenceAblationStudy:
    """Run rigorous ablation study"""

    def __init__(self, n_trials: int = 10, epochs: int = 50):
        self.n_trials = n_trials
        self.epochs = epochs
        self.results = []

    def train_and_evaluate(self, model: nn.Module, train_loader: DataLoader,
                           test_loader: DataLoader, criterion: nn.Module,
                           task_type: str = "classification") -> Dict[str, float]:
        """Train model and return test metrics"""
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Training
        model.train()
        train_losses = []
        for epoch in range(self.epochs):
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = model(X_batch)

                if task_type == "classification":
                    # Add classification head
                    logits = nn.Linear(output.shape[-1], y_batch.max().item() + 1)(output)
                    loss = criterion(logits, y_batch)
                else:
                    loss = criterion(output, y_batch)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            train_losses.append(epoch_loss / len(train_loader))

        # Evaluation
        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                output = model(X_batch)

                if task_type == "classification":
                    logits = nn.Linear(output.shape[-1], y_batch.max().item() + 1)(output)
                    loss = criterion(logits, y_batch)
                    pred = logits.argmax(dim=1)
                    correct += (pred == y_batch).sum().item()
                    total += y_batch.size(0)
                else:
                    loss = criterion(output, y_batch)

                test_loss += loss.item()

        accuracy = correct / total if total > 0 else 0
        final_train_loss = train_losses[-1] if train_losses else 0
        convergence_speed = self._measure_convergence(train_losses)

        return {
            'test_loss': test_loss / len(test_loader),
            'train_loss': final_train_loss,
            'accuracy': accuracy,
            'convergence_speed': convergence_speed,
            'final_improvement': train_losses[0] - train_losses[-1] if len(train_losses) > 1 else 0
        }

    def _measure_convergence(self, losses: List[float]) -> float:
        """Measure how quickly the model converges"""
        if len(losses) < 2:
            return 0

        # Find epoch where loss drops below 50% of initial
        initial = losses[0]
        target = initial * 0.5

        for i, loss in enumerate(losses):
            if loss < target:
                return 1.0 / (i + 1)  # Faster = higher score

        return 1.0 / len(losses)

    def run_sequence_task(self) -> List[AblationResult]:
        """Run ablation on sequence prediction task"""
        task = SequencePredictionTask(seq_len=8, vocab_size=10)
        input_dim = 8 * 10  # seq_len * vocab_size

        self_ref_scores = defaultdict(list)
        baseline_scores = defaultdict(list)

        for trial in range(self.n_trials):
            torch.manual_seed(trial * 100)
            random.seed(trial * 100)
            np.random.seed(trial * 100)

            # Training data: simple patterns
            X_train, y_train = task.create_dataset(500, ["repeat", "increment"])
            train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

            # Test data: includes harder patterns (generalization)
            X_test, y_test = task.create_dataset(100, ["repeat", "increment", "fibonacci_mod"])
            test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

            # Train self-referential model
            self_ref = nn.Sequential(
                SelfRefEncoder(input_dim=input_dim, hidden_dim=64, num_layers=3),
                nn.Linear(64, 10)
            )
            sr_results = self.train_and_evaluate(
                self_ref, train_loader, test_loader,
                nn.CrossEntropyLoss(), "classification"
            )
            for k, v in sr_results.items():
                self_ref_scores[k].append(v)

            # Train baseline model
            baseline = nn.Sequential(
                BaselineEncoder(input_dim=input_dim, hidden_dim=64, num_layers=3),
                nn.Linear(64, 10)
            )
            bl_results = self.train_and_evaluate(
                baseline, train_loader, test_loader,
                nn.CrossEntropyLoss(), "classification"
            )
            for k, v in bl_results.items():
                baseline_scores[k].append(v)

        # Compute statistics
        results = []
        for metric in ['accuracy', 'test_loss', 'convergence_speed']:
            sr = np.array(self_ref_scores[metric])
            bl = np.array(baseline_scores[metric])

            t_stat, p_value = stats.ttest_ind(sr, bl)
            effect_size = (np.mean(sr) - np.mean(bl)) / np.sqrt((np.var(sr) + np.var(bl)) / 2)

            # Determine which is better (higher accuracy, lower loss, higher convergence)
            if metric == 'test_loss':
                favors = "self_ref" if np.mean(sr) < np.mean(bl) else "baseline"
            else:
                favors = "self_ref" if np.mean(sr) > np.mean(bl) else "baseline"

            results.append(AblationResult(
                metric=f"sequence_{metric}",
                self_ref_score=float(np.mean(sr)),
                baseline_score=float(np.mean(bl)),
                difference=float(np.mean(sr) - np.mean(bl)),
                p_value=float(p_value),
                effect_size=float(effect_size),
                significant=p_value < 0.05,
                favors=favors if p_value < 0.05 else "neither"
            ))

        return results

    def run_full_ablation(self) -> Dict[str, Any]:
        """Run complete ablation study"""
        print("Running Sequence Prediction Ablation...")
        sequence_results = self.run_sequence_task()

        # Aggregate results
        self_ref_wins = sum(1 for r in sequence_results if r.favors == "self_ref")
        baseline_wins = sum(1 for r in sequence_results if r.favors == "baseline")
        ties = sum(1 for r in sequence_results if r.favors == "neither")

        summary = {
            'total_metrics': len(sequence_results),
            'self_ref_wins': self_ref_wins,
            'baseline_wins': baseline_wins,
            'ties': ties,
            'detailed_results': [
                {
                    'metric': r.metric,
                    'self_ref': r.self_ref_score,
                    'baseline': r.baseline_score,
                    'p_value': r.p_value,
                    'effect_size': r.effect_size,
                    'significant': r.significant,
                    'favors': r.favors
                }
                for r in sequence_results
            ]
        }

        # Overall verdict
        if self_ref_wins > baseline_wins and self_ref_wins >= 2:
            summary['verdict'] = "SELF-REFERENCE PROVIDES ADVANTAGE"
            summary['confidence'] = self_ref_wins / len(sequence_results)
        elif baseline_wins > self_ref_wins:
            summary['verdict'] = "NO SELF-REFERENCE ADVANTAGE"
            summary['confidence'] = 1 - (baseline_wins / len(sequence_results))
        else:
            summary['verdict'] = "INCONCLUSIVE"
            summary['confidence'] = 0.5

        return summary


def run_ablation():
    """Run and report ablation study"""
    print("=" * 60)
    print("SELF-REFERENCE ABLATION STUDY")
    print("=" * 60)
    print()

    study = SelfReferenceAblationStudy(n_trials=10, epochs=30)
    results = study.run_full_ablation()

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()

    for r in results['detailed_results']:
        sig = "*" if r['significant'] else ""
        print(f"{r['metric']:25} | SR={r['self_ref']:.4f} | BL={r['baseline']:.4f} | "
              f"p={r['p_value']:.4f}{sig} | d={r['effect_size']:.2f} | {r['favors']}")

    print()
    print(f"Self-Ref Wins: {results['self_ref_wins']}/{results['total_metrics']}")
    print(f"Baseline Wins: {results['baseline_wins']}/{results['total_metrics']}")
    print(f"Ties: {results['ties']}/{results['total_metrics']}")
    print()
    print(f"VERDICT: {results['verdict']} (confidence={results['confidence']:.2f})")

    return results


if __name__ == "__main__":
    run_ablation()
