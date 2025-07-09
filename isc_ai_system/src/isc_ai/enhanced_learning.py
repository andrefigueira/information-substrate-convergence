"""
Enhanced Learning Engine with phi-based optimization and caching
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
import torch.nn.functional as F
import time


class EnhancedLearningEngine:
    """
    Enhanced learning engine that optimizes for increasing phi values
    and implements efficient caching mechanisms.
    """
    
    def __init__(self, network: nn.Module, config: Optional[Dict] = None):
        self.network = network
        self.config = config or self._default_config()
        
        # Main optimizer for network parameters
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
        self.phi_history = deque(maxlen=100)
        self.prediction_accuracy = 0.0
        self.learning_progress = 0.0
        
        # Phi optimization
        self.target_phi = self.config["phi_target"]
        self.phi_growth_rate = self.config["phi_growth_rate"]
        self.last_phi = 0.0
        self.phi_momentum = 0.0
        
        # Loss component tracking
        self.loss_components = deque(maxlen=100)
        
    def _default_config(self) -> Dict:
        return {
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "buffer_size": 1000,
            "prediction_window": 5,
            "feedback_weight": 0.1,
            "consistency_weight": 0.15,
            "prediction_weight": 0.15,
            "information_weight": 0.2,
            "phi_weight": 0.4,  # Higher weight for phi optimization
            "phi_target": 2.0,
            "phi_growth_rate": 0.001,
            "phi_momentum_decay": 0.9,
            "batch_size": 16,
            "adaptive_phi_target": True,
            "min_phi": 0.5,
            "max_phi": 10.0,
        }
    
    def learn_from_interaction(
        self, 
        user_input: str, 
        response: str, 
        internal_states: List[torch.Tensor],
        phi_value: float
    ):
        """
        Learn from a single interaction with phi-aware optimization.
        """
        # Store experience with phi value
        experience = {
            "input": user_input,
            "response": response,
            "states": [s.detach() for s in internal_states] if internal_states else None,
            "phi": phi_value,
            "timestamp": torch.tensor(len(self.experience_buffer)),
        }
        self.experience_buffer.append(experience)
        
        # Track phi history
        self.phi_history.append(phi_value)
        
        # Update phi momentum for trend tracking
        if self.last_phi > 0:
            phi_change = phi_value - self.last_phi
            self.phi_momentum = (self.config["phi_momentum_decay"] * self.phi_momentum + 
                               (1 - self.config["phi_momentum_decay"]) * phi_change)
        self.last_phi = phi_value
        
        # Adaptive phi target based on progress
        if self.config["adaptive_phi_target"]:
            self._update_phi_target()
        
        # Perform learning if we have enough experiences
        if len(self.experience_buffer) >= self.config["batch_size"]:
            self._train_step(internal_states, phi_value)
    
    def _train_step(self, internal_states: List[torch.Tensor], current_phi: float):
        """
        Perform a training step with all loss components.
        """
        losses = {}
        
        # 1. Prediction loss
        pred_loss = self._compute_prediction_loss()
        if pred_loss is not None:
            losses["prediction"] = pred_loss
        
        # 2. Consistency loss
        cons_loss = self._compute_consistency_loss()
        if cons_loss is not None:
            losses["consistency"] = cons_loss
        
        # 3. Information loss
        info_loss = self._compute_information_loss(internal_states)
        if info_loss is not None:
            losses["information"] = info_loss
        
        # 4. Phi optimization loss (most important)
        phi_loss = self._compute_phi_loss(current_phi)
        if phi_loss is not None:
            losses["phi"] = phi_loss
        
        # Combine losses with weights
        if losses:
            total_loss = sum(losses.values())
            self._update_network(total_loss)
            
            # Track loss components
            loss_dict = {k: v.item() for k, v in losses.items()}
            loss_dict["total"] = total_loss.item()
            self.loss_components.append(loss_dict)
            
            # Update metrics
            self.loss_history.append(total_loss.item())
            self._update_learning_progress()
    
    def _compute_phi_loss(self, current_phi: float) -> Optional[torch.Tensor]:
        """
        Compute loss to optimize phi towards target value.
        """
        if current_phi is None:
            return None
        
        phi_tensor = torch.tensor(current_phi, requires_grad=True)
        target_tensor = torch.tensor(self.target_phi)
        
        # Base loss: distance from target
        base_loss = (phi_tensor - target_tensor) ** 2
        
        # Growth incentive: reward phi increases
        if len(self.phi_history) > 1:
            prev_phi = self.phi_history[-2]
            growth = current_phi - prev_phi
            growth_reward = -torch.tensor(growth) * 0.1  # Negative loss for positive growth
            
            # Momentum-based loss: maintain positive phi momentum
            momentum_loss = -self.phi_momentum * 0.05
            
            phi_loss = base_loss + growth_reward + momentum_loss
        else:
            phi_loss = base_loss
        
        # Scale by phi weight
        return phi_loss * self.config["phi_weight"]
    
    def _update_phi_target(self):
        """
        Adaptively update phi target based on recent performance.
        """
        if len(self.phi_history) < 20:
            return
        
        recent_phi = list(self.phi_history)[-10:]
        avg_recent = np.mean(recent_phi)
        
        # If consistently achieving target, increase it
        if avg_recent > self.target_phi * 0.95:
            self.target_phi = min(
                self.target_phi * (1 + self.phi_growth_rate),
                self.config["max_phi"]
            )
        # If far from target and decreasing, lower it temporarily
        elif avg_recent < self.target_phi * 0.5 and self.phi_momentum < 0:
            self.target_phi = max(
                self.target_phi * 0.95,
                self.config["min_phi"]
            )
    
    def _compute_prediction_loss(self) -> Optional[torch.Tensor]:
        """
        Compute loss based on predicting future states.
        """
        if len(self.experience_buffer) < self.config["prediction_window"] + 1:
            return None
        
        recent = list(self.experience_buffer)[-self.config["prediction_window"]-1:]
        
        if recent[0]["states"] and recent[-1]["states"]:
            past_state = recent[0]["states"][-1]
            future_state = recent[-1]["states"][-1]
            
            # Include phi in prediction
            past_phi = torch.tensor([recent[0]["phi"]])
            future_phi = torch.tensor([recent[-1]["phi"]])
            
            # Predict both state and phi evolution
            state_loss = 1.0 - F.cosine_similarity(past_state, future_state).mean()
            phi_loss = F.mse_loss(past_phi, future_phi)
            
            loss = (state_loss + phi_loss * 0.1) * self.config["prediction_weight"]
            
            # Update prediction accuracy
            with torch.no_grad():
                similarity = F.cosine_similarity(past_state, future_state).mean()
                self.prediction_accuracy = 0.9 * self.prediction_accuracy + 0.1 * similarity.item()
            
            return loss
        
        return None
    
    def _compute_consistency_loss(self) -> Optional[torch.Tensor]:
        """
        Ensure responses maintain consistency while allowing growth.
        """
        if len(self.experience_buffer) < 2:
            return None
        
        exp1 = self.experience_buffer[-2]
        exp2 = self.experience_buffer[-1]
        
        if exp1["states"] and exp2["states"]:
            state1 = exp1["states"][-1]
            state2 = exp2["states"][-1]
            
            similarity = F.cosine_similarity(state1, state2).mean()
            
            # Dynamic target based on phi change
            phi_change = abs(exp2["phi"] - exp1["phi"])
            # More phi change allows more state change
            target_similarity = 0.7 - min(phi_change * 0.1, 0.2)
            
            loss = (similarity - target_similarity) ** 2
            
            return loss * self.config["consistency_weight"]
        
        return None
    
    def _compute_information_loss(self, internal_states: List[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Encourage states that maximize information content and support high phi.
        """
        if not internal_states:
            return None
        
        info_loss = 0.0
        
        for i, state in enumerate(internal_states):
            # Encourage non-zero activations
            sparsity = torch.abs(state).mean()
            sparsity_loss = -torch.log(sparsity + 1e-8)
            
            # Encourage diversity
            std = state.std()
            diversity_loss = -torch.log(std + 1e-8)
            
            # Layer-specific scaling (deeper layers more important for phi)
            layer_weight = (i + 1) / len(internal_states)
            
            info_loss += (sparsity_loss + diversity_loss) * layer_weight
        
        return info_loss / len(internal_states) * self.config["information_weight"]
    
    def apply_feedback(self, feedback_value: float):
        """
        Apply feedback with phi-aware adjustments.
        """
        self.feedback_history.append(feedback_value)
        
        if len(self.experience_buffer) == 0:
            return
        
        recent_exp = self.experience_buffer[-1]
        recent_phi = recent_exp["phi"]
        
        # Adjust feedback based on phi performance
        if recent_phi > self.target_phi:
            # Bonus for high phi
            adjusted_feedback = feedback_value * 1.2
        elif self.phi_momentum > 0:
            # Bonus for improving phi
            adjusted_feedback = feedback_value * 1.1
        else:
            adjusted_feedback = feedback_value
        
        self.network.update_meta_weights(adjusted_feedback)
    
    def _update_network(self, loss: torch.Tensor):
        """
        Update network with phi-aware gradient scaling.
        """
        self.optimizer.zero_grad()
        loss.backward()
        
        # Scale gradients based on phi trend
        if self.phi_momentum > 0:
            # Amplify gradients when phi is improving
            for param in self.network.parameters():
                if param.grad is not None:
                    param.grad *= (1 + self.phi_momentum * 0.1)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
    
    def _update_learning_progress(self):
        """
        Calculate learning progress including phi improvement.
        """
        if len(self.loss_history) < 10 or len(self.phi_history) < 10:
            self.learning_progress = 0.0
            return
        
        # Loss-based progress
        recent_losses = list(self.loss_history)[-10:]
        older_losses = list(self.loss_history)[-20:-10] if len(self.loss_history) >= 20 else list(self.loss_history)[:10]
        
        avg_recent_loss = np.mean(recent_losses)
        avg_older_loss = np.mean(older_losses)
        
        loss_progress = 0.0
        if avg_older_loss > 0:
            loss_progress = (avg_older_loss - avg_recent_loss) / avg_older_loss
        
        # Phi-based progress
        recent_phi = list(self.phi_history)[-10:]
        older_phi = list(self.phi_history)[-20:-10] if len(self.phi_history) >= 20 else list(self.phi_history)[:10]
        
        avg_recent_phi = np.mean(recent_phi)
        avg_older_phi = np.mean(older_phi)
        
        phi_progress = 0.0
        if avg_older_phi > 0:
            phi_progress = (avg_recent_phi - avg_older_phi) / avg_older_phi
        
        # Combined progress (phi is more important)
        self.learning_progress = loss_progress * 0.3 + phi_progress * 0.7
    
    def get_learning_metrics(self) -> Dict[str, float]:
        """
        Get comprehensive learning metrics including phi statistics.
        """
        avg_loss = np.mean(list(self.loss_history)) if self.loss_history else 0.0
        avg_feedback = np.mean(list(self.feedback_history)) if self.feedback_history else 0.0
        avg_phi = np.mean(list(self.phi_history)) if self.phi_history else 0.0
        
        # Get recent loss components
        recent_components = {}
        if self.loss_components:
            recent = list(self.loss_components)[-10:]
            for component in ["prediction", "consistency", "information", "phi"]:
                values = [d.get(component, 0) for d in recent]
                recent_components[f"avg_{component}_loss"] = np.mean(values) if values else 0.0
        
        return {
            "average_loss": avg_loss,
            "prediction_accuracy": self.prediction_accuracy,
            "learning_progress": self.learning_progress,
            "average_feedback": avg_feedback,
            "experience_count": len(self.experience_buffer),
            "current_lr": self.optimizer.param_groups[0]['lr'],
            "average_phi": avg_phi,
            "current_phi": self.phi_history[-1] if self.phi_history else 0.0,
            "phi_momentum": self.phi_momentum,
            "phi_target": self.target_phi,
            **recent_components
        }
    
    def adapt_learning_rate(self):
        """
        Adapt learning rate based on phi and loss progress.
        """
        metrics = self.get_learning_metrics()
        
        # Different adaptation based on phi trend
        if self.phi_momentum < -0.01:
            # Phi decreasing - reduce learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.85
        elif self.phi_momentum > 0.01 and metrics["learning_progress"] > 0.05:
            # Both phi and overall progress positive - can increase
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = min(param_group['lr'] * 1.05, 0.01)
        elif metrics["learning_progress"] < 0.01 and metrics["experience_count"] > 50:
            # Stagnant - small reduction
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.95
    
    def consolidate_learning(self):
        """
        Consolidate learning focusing on high-phi experiences.
        """
        if len(self.experience_buffer) < 10:
            return
        
        experiences = list(self.experience_buffer)
        
        # Sort by phi value and select top experiences
        high_phi_experiences = sorted(experiences, key=lambda x: x["phi"], reverse=True)[:5]
        
        # Rehearse high-phi experiences
        for exp in high_phi_experiences:
            if exp["states"]:
                for state in exp["states"]:
                    _, _ = self.network(state)
        
        # Adapt learning rate
        self.adapt_learning_rate()