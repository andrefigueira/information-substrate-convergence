"""
Elastic Weight Consolidation (EWC) for Continuous Learning

Based on: "Overcoming catastrophic forgetting in neural networks"
Kirkpatrick et al., PNAS 2017

EWC prevents catastrophic forgetting by:
1. Computing Fisher Information Matrix after learning a task
2. Penalizing changes to important parameters when learning new tasks
3. Important parameters = high Fisher Information = large gradient variance

Loss = L_new + (λ/2) * Σ F_i * (θ_i - θ*_i)²

Where:
- L_new: loss on new task
- F_i: Fisher information for parameter i
- θ_i: current parameter value
- θ*_i: optimal parameter value for previous tasks
- λ: regularization strength
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from copy import deepcopy
from collections import defaultdict
import json
from pathlib import Path


@dataclass
class TaskMemory:
    """Stores Fisher information and optimal parameters for a learned task"""
    task_id: str
    fisher_information: Dict[str, torch.Tensor]  # param_name -> Fisher info
    optimal_params: Dict[str, torch.Tensor]  # param_name -> optimal values
    num_samples: int
    performance_metric: float  # Task performance at time of consolidation


class EWCModule:
    """
    Elastic Weight Consolidation wrapper for any PyTorch model.

    Usage:
        ewc = EWCModule(model, lambda_ewc=400)

        # Train on task A
        train_model(model, task_a_data)
        ewc.consolidate_task("task_a", task_a_dataloader)

        # Train on task B with EWC penalty
        for batch in task_b_data:
            loss = criterion(model(x), y)
            ewc_loss = ewc.compute_ewc_loss()
            total_loss = loss + ewc_loss
            total_loss.backward()
    """

    def __init__(
        self,
        model: nn.Module,
        lambda_ewc: float = 400.0,
        gamma: float = 0.9,  # Decay for old task importance
        max_tasks: int = 10
    ):
        """
        Args:
            model: The neural network to protect
            lambda_ewc: Regularization strength (higher = more protection)
            gamma: Decay factor for old task importance (0.9 = 10% decay per task)
            max_tasks: Maximum number of tasks to remember
        """
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.gamma = gamma
        self.max_tasks = max_tasks

        self.task_memories: List[TaskMemory] = []
        self.device = next(model.parameters()).device

    def compute_fisher_information(
        self,
        dataloader: DataLoader,
        criterion: Optional[nn.Module] = None,
        num_samples: int = 1000
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Fisher Information Matrix (diagonal approximation).

        Fisher Information measures how much the loss changes when parameters change.
        High Fisher = parameter is important for this task.

        F_i = E[(∂log p(y|x,θ) / ∂θ_i)²]

        In practice, we approximate this with squared gradients:
        F_i ≈ (1/N) * Σ (∂L/∂θ_i)²
        """
        self.model.eval()
        fisher = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        samples_processed = 0

        for batch in dataloader:
            if samples_processed >= num_samples:
                break

            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                if len(batch) == 3:  # (input_ids, targets, substrate_embedding)
                    input_ids, targets, substrate_emb = batch
                    input_ids = input_ids.to(self.device)
                    targets = targets.to(self.device)
                    substrate_emb = substrate_emb.to(self.device)

                    self.model.zero_grad()
                    output, _ = self.model(input_ids, substrate_emb)
                    output = output.view(-1, output.size(-1))
                    targets = targets.view(-1)

                elif len(batch) == 2:  # (inputs, targets)
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    self.model.zero_grad()
                    output = self.model(inputs)
                    if isinstance(output, tuple):
                        output = output[0]
                    output = output.view(-1, output.size(-1))
                    targets = targets.view(-1)
            else:
                continue

            # Compute loss and gradients
            loss = criterion(output, targets)
            loss.backward()

            # Accumulate squared gradients (Fisher approximation)
            for name, param in self.model.named_parameters():
                if param.grad is not None and param.requires_grad:
                    fisher[name] += param.grad.data.pow(2)

            samples_processed += input_ids.size(0) if 'input_ids' in dir() else inputs.size(0)

        # Normalize by number of samples
        for name in fisher:
            fisher[name] /= max(samples_processed, 1)

        return fisher

    def consolidate_task(
        self,
        task_id: str,
        dataloader: DataLoader,
        criterion: Optional[nn.Module] = None,
        performance_metric: float = 0.0
    ):
        """
        Consolidate current model state for a completed task.

        This should be called after training on a task is complete.
        """
        print(f"Consolidating task: {task_id}")

        # Compute Fisher Information
        fisher = self.compute_fisher_information(dataloader, criterion)

        # Store optimal parameters
        optimal_params = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        # Create task memory
        task_memory = TaskMemory(
            task_id=task_id,
            fisher_information=fisher,
            optimal_params=optimal_params,
            num_samples=len(dataloader.dataset) if hasattr(dataloader, 'dataset') else 0,
            performance_metric=performance_metric
        )

        # Add to memories, respecting max_tasks
        self.task_memories.append(task_memory)
        if len(self.task_memories) > self.max_tasks:
            self.task_memories.pop(0)  # Remove oldest

        # Apply decay to older task importances
        for i, memory in enumerate(self.task_memories[:-1]):
            decay = self.gamma ** (len(self.task_memories) - i - 1)
            for name in memory.fisher_information:
                memory.fisher_information[name] *= decay

        print(f"Task {task_id} consolidated. Total tasks remembered: {len(self.task_memories)}")

    def compute_ewc_loss(self) -> torch.Tensor:
        """
        Compute EWC regularization loss.

        Loss = (λ/2) * Σ_tasks Σ_params F_i * (θ_i - θ*_i)²

        This penalizes changes to important parameters.
        """
        if not self.task_memories:
            return torch.tensor(0.0, device=self.device)

        ewc_loss = torch.tensor(0.0, device=self.device)

        for memory in self.task_memories:
            for name, param in self.model.named_parameters():
                if name in memory.fisher_information and param.requires_grad:
                    fisher = memory.fisher_information[name].to(self.device)
                    optimal = memory.optimal_params[name].to(self.device)

                    # EWC penalty: F * (θ - θ*)²
                    ewc_loss += (fisher * (param - optimal).pow(2)).sum()

        return (self.lambda_ewc / 2) * ewc_loss

    def get_parameter_importance(self) -> Dict[str, float]:
        """
        Get aggregated importance scores for each parameter.

        Higher = more important across all tasks = should change less.
        """
        if not self.task_memories:
            return {}

        importance = defaultdict(float)

        for memory in self.task_memories:
            for name, fisher in memory.fisher_information.items():
                importance[name] += fisher.sum().item()

        # Normalize
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}

        return dict(importance)

    def save(self, path: str):
        """Save EWC state to disk"""
        state = {
            'lambda_ewc': self.lambda_ewc,
            'gamma': self.gamma,
            'max_tasks': self.max_tasks,
            'task_memories': [
                {
                    'task_id': m.task_id,
                    'fisher_information': {k: v.cpu().numpy().tolist() for k, v in m.fisher_information.items()},
                    'optimal_params': {k: v.cpu().numpy().tolist() for k, v in m.optimal_params.items()},
                    'num_samples': m.num_samples,
                    'performance_metric': m.performance_metric
                }
                for m in self.task_memories
            ]
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(state, f)

    def load(self, path: str):
        """Load EWC state from disk"""
        with open(path, 'r') as f:
            state = json.load(f)

        self.lambda_ewc = state['lambda_ewc']
        self.gamma = state['gamma']
        self.max_tasks = state['max_tasks']

        self.task_memories = []
        for m in state['task_memories']:
            task_memory = TaskMemory(
                task_id=m['task_id'],
                fisher_information={
                    k: torch.tensor(v, device=self.device)
                    for k, v in m['fisher_information'].items()
                },
                optimal_params={
                    k: torch.tensor(v, device=self.device)
                    for k, v in m['optimal_params'].items()
                },
                num_samples=m['num_samples'],
                performance_metric=m['performance_metric']
            )
            self.task_memories.append(task_memory)


class EWCTrainer:
    """
    Trainer that integrates EWC with the neural language model.

    Enables true continuous learning without catastrophic forgetting.
    """

    def __init__(
        self,
        model: nn.Module,
        vocab,
        lambda_ewc: float = 400.0,
        learning_rate: float = 0.001,
        device: str = None
    ):
        self.model = model
        self.vocab = vocab
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding

        self.ewc = EWCModule(model, lambda_ewc=lambda_ewc)
        self.current_task_id = 0
        self.task_performance_history: List[Dict[str, float]] = []

    def train_on_task(
        self,
        task_data: List[Tuple[str, str, np.ndarray]],  # (input, response, substrate_embedding)
        task_id: Optional[str] = None,
        epochs: int = 10,
        batch_size: int = 8,
        consolidate: bool = True
    ) -> Dict[str, Any]:
        """
        Train on a new task with EWC protection for previous tasks.
        """
        task_id = task_id or f"task_{self.current_task_id}"
        self.current_task_id += 1

        # Prepare data
        dataloader = self._prepare_dataloader(task_data, batch_size)

        # Training loop
        self.model.train()
        losses = []
        ewc_losses = []

        for epoch in range(epochs):
            epoch_loss = 0
            epoch_ewc_loss = 0
            batches = 0

            for batch in dataloader:
                input_ids, target_ids, substrate_emb = [b.to(self.device) for b in batch]

                self.optimizer.zero_grad()

                # Forward pass
                logits, _ = self.model(input_ids, substrate_emb)
                logits = logits.view(-1, self.vocab.n_words)
                targets = target_ids.view(-1)

                # Task loss
                task_loss = self.criterion(logits, targets)

                # EWC regularization loss
                ewc_loss = self.ewc.compute_ewc_loss()

                # Total loss
                total_loss = task_loss + ewc_loss

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_loss += task_loss.item()
                epoch_ewc_loss += ewc_loss.item()
                batches += 1

            avg_loss = epoch_loss / max(batches, 1)
            avg_ewc = epoch_ewc_loss / max(batches, 1)
            losses.append(avg_loss)
            ewc_losses.append(avg_ewc)

        # Consolidate this task
        if consolidate:
            self.ewc.consolidate_task(
                task_id,
                dataloader,
                self.criterion,
                performance_metric=losses[-1] if losses else 0.0
            )

        # Record performance
        result = {
            'task_id': task_id,
            'final_loss': losses[-1] if losses else 0.0,
            'final_ewc_loss': ewc_losses[-1] if ewc_losses else 0.0,
            'loss_history': losses,
            'ewc_loss_history': ewc_losses
        }

        self.task_performance_history.append(result)

        return result

    def evaluate_on_task(
        self,
        task_data: List[Tuple[str, str, np.ndarray]],
        batch_size: int = 8
    ) -> float:
        """Evaluate model on a task and return loss"""
        dataloader = self._prepare_dataloader(task_data, batch_size)

        self.model.eval()
        total_loss = 0
        batches = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids, target_ids, substrate_emb = [b.to(self.device) for b in batch]

                logits, _ = self.model(input_ids, substrate_emb)
                logits = logits.view(-1, self.vocab.n_words)
                targets = target_ids.view(-1)

                loss = self.criterion(logits, targets)
                total_loss += loss.item()
                batches += 1

        return total_loss / max(batches, 1)

    def measure_forgetting(
        self,
        task_data_list: List[Tuple[str, List[Tuple[str, str, np.ndarray]]]]
    ) -> Dict[str, float]:
        """
        Measure catastrophic forgetting across all tasks.

        Returns loss on each task after training on all tasks.
        Lower is better. Large increases indicate forgetting.
        """
        results = {}

        for task_id, task_data in task_data_list:
            loss = self.evaluate_on_task(task_data)
            results[task_id] = loss

        return results

    def _prepare_dataloader(
        self,
        task_data: List[Tuple[str, str, np.ndarray]],
        batch_size: int
    ) -> DataLoader:
        """Prepare a DataLoader from task data"""
        input_ids_list = []
        target_ids_list = []
        substrate_embs = []

        for _, response, substrate_emb in task_data:
            # Encode response with SOS and EOS
            tokens = [self.vocab.word2idx.get('<SOS>', 2)]
            tokens += self.vocab.encode(response)
            tokens += [self.vocab.word2idx.get('<EOS>', 3)]
            tokens = tokens[:64]  # Max length

            # Create input (without last token) and target (without first token)
            input_tokens = tokens[:-1]
            target_tokens = tokens[1:]

            # Pad to same length
            max_len = 63
            input_padded = input_tokens + [0] * (max_len - len(input_tokens))
            target_padded = target_tokens + [0] * (max_len - len(target_tokens))

            input_ids_list.append(input_padded)
            target_ids_list.append(target_padded)
            substrate_embs.append(substrate_emb)

        # Create tensors
        input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        target_ids = torch.tensor(target_ids_list, dtype=torch.long)
        substrate_embs = torch.tensor(np.array(substrate_embs), dtype=torch.float32)

        dataset = TensorDataset(input_ids, target_ids, substrate_embs)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def get_forgetting_report(self) -> Dict[str, Any]:
        """Get a report on forgetting across training"""
        return {
            'total_tasks': len(self.task_performance_history),
            'ewc_lambda': self.ewc.lambda_ewc,
            'tasks_remembered': len(self.ewc.task_memories),
            'parameter_importance': self.ewc.get_parameter_importance(),
            'task_history': self.task_performance_history
        }

    def save(self, path: str):
        """Save trainer state"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'current_task_id': self.current_task_id,
            'task_performance_history': self.task_performance_history
        }, path)

        # Save EWC state
        self.ewc.save(path.replace('.pt', '_ewc.json'))

    def load(self, path: str):
        """Load trainer state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_task_id = checkpoint.get('current_task_id', 0)
        self.task_performance_history = checkpoint.get('task_performance_history', [])

        ewc_path = path.replace('.pt', '_ewc.json')
        if Path(ewc_path).exists():
            self.ewc.load(ewc_path)
