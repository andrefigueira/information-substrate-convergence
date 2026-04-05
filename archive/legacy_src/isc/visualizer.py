"""
Visualization module for creating ASCII representations of system state
"""

import numpy as np
from typing import List, Dict, Optional
import plotext as plt


class SystemVisualizer:
    """
    Creates ASCII visualizations of the ISC system's internal state.
    """
    
    def __init__(self):
        self.width = 80
        self.height = 20
    
    def plot_activation_pattern(self, activations: List[np.ndarray], title: str = "Neural Activations") -> str:
        """
        Create an ASCII plot of activation patterns.
        """
        if not activations:
            return "No activation data available."
        
        plt.clear_figure()
        
        # Plot each layer's average activation
        layer_means = []
        for i, activation in enumerate(activations):
            if isinstance(activation, np.ndarray):
                mean_activation = np.mean(activation)
                layer_means.append(mean_activation)
        
        if layer_means:
            plt.plot(layer_means, label="Mean Activation")
            plt.title(title)
            plt.xlabel("Layer")
            plt.ylabel("Activation")
            
            # Capture plot as string
            plt.show()
            return ""  # plotext prints directly
        
        return "No valid activation data."
    
    def plot_learning_progress(self, loss_history: List[float], title: str = "Learning Progress") -> str:
        """
        Plot learning progress over time.
        """
        if not loss_history:
            return "No learning history available."
        
        plt.clear_figure()
        plt.plot(loss_history)
        plt.title(title)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.show()
        
        return ""
    
    def visualize_information_flow(self, phi_values: List[float]) -> str:
        """
        Visualize information integration (Phi) over time.
        """
        if not phi_values:
            return "No Φ history available."
        
        plt.clear_figure()
        
        # Plot phi values
        plt.plot(phi_values, label="Φ")
        
        # Add threshold line
        if phi_values:
            avg_phi = np.mean(phi_values)
            plt.hline(avg_phi, color="red", label=f"Avg: {avg_phi:.3f}")
        
        plt.title("Information Integration (Φ) Over Time")
        plt.xlabel("Interaction")
        plt.ylabel("Φ Value")
        plt.show()
        
        return ""
    
    def create_concept_map(self, concepts: Dict[str, List[str]], max_concepts: int = 10) -> str:
        """
        Create a simple ASCII representation of concept relationships.
        """
        if not concepts:
            return "No concepts to display."
        
        # Limit to top concepts
        concept_items = list(concepts.items())[:max_concepts]
        
        output = []
        output.append("Concept Network:")
        output.append("=" * 50)
        
        for concept, connections in concept_items:
            if connections:
                # Main concept
                output.append(f"\n┌─ {concept.upper()}")
                
                # Connections
                for i, conn in enumerate(connections[:5]):
                    if i == len(connections[:5]) - 1:
                        output.append(f"└──> {conn}")
                    else:
                        output.append(f"├──> {conn}")
            else:
                output.append(f"\n○ {concept} (isolated)")
        
        return "\n".join(output)
    
    def create_state_summary(self, metrics: Dict) -> str:
        """
        Create a visual summary of system state.
        """
        phi = metrics.get('phi_value', 0.0)
        coherence = metrics.get('coherence_score', 0.0)
        interactions = metrics.get('total_interactions', 0)
        
        # Create bar representations
        phi_bar = self._create_bar(phi, max_val=2.0, width=20)
        coh_bar = self._create_bar(coherence, max_val=1.0, width=20)
        
        output = []
        output.append("System State Summary")
        output.append("=" * 50)
        output.append(f"Interactions: {interactions}")
        output.append(f"Φ (Phi):      {phi_bar} {phi:.3f}")
        output.append(f"Coherence:    {coh_bar} {coherence:.3f}")
        
        return "\n".join(output)
    
    def _create_bar(self, value: float, max_val: float = 1.0, width: int = 20) -> str:
        """
        Create a simple ASCII progress bar.
        """
        filled = int((value / max_val) * width)
        filled = min(filled, width)  # Cap at width
        
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}]"
    
    def plot_network_activity(self, layer_activities: Dict[str, List[float]]) -> str:
        """
        Plot activity patterns across network layers.
        """
        if not layer_activities:
            return "No network activity data."
        
        plt.clear_figure()
        
        # Plot each layer's activity
        for layer_name, activities in layer_activities.items():
            if activities:
                plt.plot(activities[-50:], label=layer_name)  # Last 50 values
        
        plt.title("Network Layer Activities")
        plt.xlabel("Time Step")
        plt.ylabel("Activity Level")
        plt.show()
        
        return ""
    
    def create_interaction_pattern(self, interactions: List[Dict]) -> str:
        """
        Visualize interaction patterns.
        """
        if not interactions:
            return "No interactions to visualize."
        
        output = []
        output.append("Interaction Pattern")
        output.append("=" * 50)
        
        for i, interaction in enumerate(interactions[-10:], 1):
            input_len = len(interaction.get('input', '').split())
            response_len = len(interaction.get('response', '').split())
            
            # Create simple visualization
            input_bar = "▸" * min(input_len // 5, 10)
            response_bar = "▹" * min(response_len // 5, 10)
            
            output.append(f"{i:2d}. IN:  {input_bar}")
            output.append(f"    OUT: {response_bar}")
        
        return "\n".join(output)