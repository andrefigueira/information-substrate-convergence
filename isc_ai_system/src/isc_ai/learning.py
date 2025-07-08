"""
Learning Engine for continuous improvement through interaction
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
import torch.nn.functional as F


class LearningEngine:
    """
    Manages the learning process for the self-modifying network.
    Implements reinforcement learning and self-supervised learning.
    """
    
    def __init__(self, network: nn.Module, config: Optional[Dict] = None):
        self.network = network
        self.config = config or self._default_config()
        
        # Optimizer for network parameters
        self.optimizer = optim.AdamW(
            self.network.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"]
        )
        
        # Learning components
        self.experience_buffer = deque(maxlen=self.config["buffer_size"])
        self.prediction_buffer = deque(maxlen=self.config["prediction_window"])
        self.feedback_history = deque(maxlen=100)
        
        # Metrics
        self.loss_history = deque(maxlen=100)
        self.prediction_accuracy = 0.0
        self.learning_progress = 0.0
        
    def _default_config(self) -> Dict:
        return {
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "buffer_size": 1000,
            "prediction_window": 5,
            "feedback_weight": 0.1,
            "consistency_weight": 0.3,
            "prediction_weight": 0.2,
            "batch_size": 16,
        }
    
    def learn_from_interaction(
        self, 
        user_input: str, 
        response: str, 
        internal_states: List[torch.Tensor]
    ):
        """
        Learn from a single interaction using multiple learning signals.
        """
        # Store experience
        experience = {
            "input": user_input,
            "response": response,
            "states": [s.detach() for s in internal_states] if internal_states else None,
            "timestamp": torch.tensor(len(self.experience_buffer)),
        }
        self.experience_buffer.append(experience)
        
        # Perform learning if we have enough experiences
        if len(self.experience_buffer) >= self.config["batch_size"]:
            losses = []
            
            # 1. Self-supervised prediction loss
            pred_loss = self._compute_prediction_loss()
            if pred_loss is not None:
                losses.append(pred_loss)
            
            # 2. Consistency loss (responses should be coherent)
            cons_loss = self._compute_consistency_loss()
            if cons_loss is not None:
                losses.append(cons_loss)
            
            # 3. Information maximization loss
            info_loss = self._compute_information_loss(internal_states)
            if info_loss is not None:
                losses.append(info_loss)
            
            # Combine losses and update
            if losses:
                total_loss = sum(losses)
                self._update_network(total_loss)
                
                # Update metrics
                self.loss_history.append(total_loss.item())
                self._update_learning_progress()
    
    def _compute_prediction_loss(self) -> Optional[torch.Tensor]:
        """
        Compute loss based on predicting future states or responses.
        """
        if len(self.experience_buffer) < self.config["prediction_window"] + 1:
            return None
        
        # Get recent experiences
        recent = list(self.experience_buffer)[-self.config["prediction_window"]-1:]
        
        # Try to predict the next state from previous ones
        if recent[0]["states"] and recent[-1]["states"]:
            # Use the network to predict future state
            past_state = recent[0]["states"][-1]  # Last layer of past interaction
            future_state = recent[-1]["states"][-1]  # Last layer of future interaction
            
            # Forward pass through network
            predicted, _ = self.network(past_state)
            
            # MSE loss between predicted and actual future
            loss = F.mse_loss(predicted, future_state)
            
            # Update prediction accuracy
            with torch.no_grad():
                similarity = F.cosine_similarity(predicted, future_state).mean()
                self.prediction_accuracy = 0.9 * self.prediction_accuracy + 0.1 * similarity.item()
            
            return loss * self.config["prediction_weight"]
        
        return None
    
    def _compute_consistency_loss(self) -> Optional[torch.Tensor]:
        """
        Ensure responses are consistent with previous responses.
        """
        if len(self.experience_buffer) < 2:
            return None
        
        # Get last two experiences
        exp1 = self.experience_buffer[-2]
        exp2 = self.experience_buffer[-1]
        
        if exp1["states"] and exp2["states"]:
            # States should maintain some similarity over time
            state1 = exp1["states"][-1]
            state2 = exp2["states"][-1]
            
            # Cosine similarity loss (want some consistency but not too much)
            similarity = F.cosine_similarity(state1, state2).mean()
            
            # Optimal similarity around 0.7 (not too similar, not too different)
            target_similarity = 0.7
            loss = (similarity - target_similarity) ** 2
            
            return loss * self.config["consistency_weight"]
        
        return None
    
    def _compute_information_loss(self, internal_states: List[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Encourage states that maximize information content.
        """
        if not internal_states:
            return None
        
        # Information loss: encourage diverse activations
        info_loss = 0.0
        
        for state in internal_states:
            # Encourage non-zero activations (avoid dead neurons)
            sparsity = torch.abs(state).mean()
            sparsity_loss = -torch.log(sparsity + 1e-8)
            
            # Encourage diversity in activations
            std = state.std()
            diversity_loss = -torch.log(std + 1e-8)
            
            info_loss += sparsity_loss + diversity_loss
        
        return info_loss / len(internal_states) * 0.1
    
    def apply_feedback(self, feedback_value: float):
        """
        Apply explicit feedback to recent experiences.
        """
        self.feedback_history.append(feedback_value)
        
        if len(self.experience_buffer) == 0:
            return
        
        # Get the most recent experience
        recent_exp = self.experience_buffer[-1]
        
        if recent_exp["states"]:
            # Create a feedback-based loss
            # Positive feedback -> reduce loss, negative feedback -> increase loss
            feedback_loss = -feedback_value * sum(
                state.abs().mean() for state in recent_exp["states"]
            ) / len(recent_exp["states"])
            
            # Update network based on feedback
            self._update_network(feedback_loss * self.config["feedback_weight"])
    
    def _update_network(self, loss: torch.Tensor):
        """
        Update network parameters based on computed loss.
        """
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
    
    def _update_learning_progress(self):
        """
        Calculate learning progress based on loss history.
        """
        if len(self.loss_history) < 10:
            self.learning_progress = 0.0
            return
        
        # Compare recent losses to older ones
        recent_losses = list(self.loss_history)[-10:]
        older_losses = list(self.loss_history)[-20:-10] if len(self.loss_history) >= 20 else list(self.loss_history)[:10]
        
        avg_recent = np.mean(recent_losses)
        avg_older = np.mean(older_losses)
        
        # Progress is positive if losses are decreasing
        if avg_older > 0:
            self.learning_progress = (avg_older - avg_recent) / avg_older
        else:
            self.learning_progress = 0.0
    
    def get_learning_metrics(self) -> Dict[str, float]:
        """
        Get current learning metrics.
        """
        avg_loss = np.mean(list(self.loss_history)) if self.loss_history else 0.0
        avg_feedback = np.mean(list(self.feedback_history)) if self.feedback_history else 0.0
        
        return {
            "average_loss": avg_loss,
            "prediction_accuracy": self.prediction_accuracy,
            "learning_progress": self.learning_progress,
            "average_feedback": avg_feedback,
            "experience_count": len(self.experience_buffer),
            "current_lr": self.optimizer.param_groups[0]['lr'],
        }
    
    def adapt_learning_rate(self):
        """
        Adapt learning rate based on progress.
        """
        metrics = self.get_learning_metrics()
        
        # Reduce learning rate if not making progress
        if metrics["learning_progress"] < 0.01 and metrics["experience_count"] > 50:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.9
        
        # Increase learning rate if making good progress
        elif metrics["learning_progress"] > 0.1:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = min(param_group['lr'] * 1.1, 0.01)
    
    def consolidate_learning(self):
        """
        Consolidate learning by rehearsing important experiences.
        """
        if len(self.experience_buffer) < 10:
            return
        
        # Select diverse experiences for rehearsal
        experiences = list(self.experience_buffer)
        
        # Simple rehearsal: re-process some old experiences
        sample_size = min(5, len(experiences) // 2)
        sample_indices = np.random.choice(len(experiences), sample_size, replace=False)
        
        for idx in sample_indices:
            exp = experiences[idx]
            if exp["states"]:
                # Forward pass through network
                for state in exp["states"]:
                    _, _ = self.network(state)
        
        # Adapt learning rate based on progress
        self.adapt_learning_rate()