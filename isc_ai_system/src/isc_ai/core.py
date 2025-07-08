"""
Core ISC AI System implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
import sqlite3
from datetime import datetime
import networkx as nx
from collections import defaultdict

from .information_integration import InformationIntegrator
from .knowledge_graph import KnowledgeGraph
from .learning import LearningEngine
from .memory import ConversationMemory


class SelfModifyingNetwork(nn.Module):
    """
    A neural network that can observe and modify its own operations
    based on the ISC hypothesis of self-referential information patterns.
    """
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 512, num_layers: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Core processing layers
        self.layers = nn.ModuleList([
            nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        
        # Self-observation layers (mirror the processing layers)
        self.observer_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Meta-learning parameters that can modify the network
        self.meta_weights = nn.ParameterList([
            nn.Parameter(torch.ones(hidden_dim))
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        
        # Track internal states for information integration
        self.internal_states = []
        self.activation_patterns = defaultdict(list)
        
    def forward(self, x: torch.Tensor, return_states: bool = False) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass with self-observation and modification
        """
        states = []
        h = x
        
        for i, (layer, observer, meta_weight) in enumerate(
            zip(self.layers, self.observer_layers, self.meta_weights)
        ):
            # Standard forward pass
            h = layer(h)
            
            # Self-observation: the network observes its own activations
            observed = observer(h.detach())
            
            # Meta-modulation: modify activations based on observation
            h = h * meta_weight.unsqueeze(0) + 0.1 * observed
            
            # Non-linearity
            h = F.gelu(h)
            
            # Store internal state
            states.append(h.clone())
            self.activation_patterns[f"layer_{i}"].append(h.detach().cpu().numpy())
        
        # Output projection
        output = self.output_proj(h)
        
        if return_states:
            return output, states
        return output, None
    
    def update_meta_weights(self, feedback: float):
        """
        Update meta-weights based on feedback to enable self-modification
        """
        with torch.no_grad():
            for i, meta_weight in enumerate(self.meta_weights):
                # Simple reinforcement: strengthen weights that contributed to positive feedback
                if self.activation_patterns[f"layer_{i}"]:
                    recent_activation = np.mean(self.activation_patterns[f"layer_{i}"][-5:], axis=0)
                    activation_strength = torch.tensor(recent_activation).mean()
                    meta_weight.data += feedback * 0.01 * activation_strength * torch.randn_like(meta_weight)
                    meta_weight.data = torch.clamp(meta_weight.data, 0.5, 2.0)


class ISCCore:
    """
    Main ISC AI System that integrates all components
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        
        # Initialize components
        self.network = SelfModifyingNetwork()
        self.integrator = InformationIntegrator()
        self.knowledge_graph = KnowledgeGraph()
        self.learning_engine = LearningEngine(self.network)
        self.memory = ConversationMemory()
        
        # System state
        self.session_active = False
        self.verbose = False
        self.current_session_id = None
        self.metrics = {
            "total_interactions": 0,
            "learning_rate": 0.001,
            "phi_value": 0.0,
            "coherence_score": 0.0,
            "prediction_accuracy": 0.0,
            "concepts_formed": 0,
        }
        
        # Initialize text encoder (lightweight model)
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.encoder = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            "learning_rate": 0.001,
            "batch_size": 1,
            "memory_size": 1000,
            "phi_threshold": 0.5,
            "min_concept_frequency": 3,
            "prediction_window": 5,
        }
    
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text into embeddings"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings
    
    def process_input(self, user_input: str) -> str:
        """
        Process user input and generate response
        """
        self.metrics["total_interactions"] += 1
        
        # Encode input
        input_embedding = self.encode_text(user_input)
        
        # Process through self-modifying network
        output_embedding, internal_states = self.network(input_embedding, return_states=True)
        
        # Calculate information integration (Φ)
        if internal_states:
            phi_value = self.integrator.calculate_phi(internal_states)
            self.metrics["phi_value"] = phi_value
        
        # Update knowledge graph
        concepts = self._extract_concepts(user_input)
        for concept in concepts:
            self.knowledge_graph.add_concept(concept, input_embedding)
        
        # Generate response (simplified for now)
        response = self._generate_response(output_embedding, user_input)
        
        # Learn from interaction
        self.learning_engine.learn_from_interaction(user_input, response, internal_states)
        
        # Update memory
        self.memory.add_interaction(user_input, response, self.current_session_id)
        
        # Calculate coherence with previous interactions
        self.metrics["coherence_score"] = self._calculate_coherence()
        
        return response
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text"""
        # Simplified concept extraction
        import nltk
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        
        concepts = [w for w in tokens if w.isalnum() and w not in stop_words and len(w) > 3]
        return list(set(concepts))[:5]  # Return top 5 unique concepts
    
    def _generate_response(self, output_embedding: torch.Tensor, user_input: str) -> str:
        """Generate response based on output embedding and context"""
        # For now, use a template-based approach with learned modulation
        # In a full implementation, this would use a proper decoder
        
        templates = [
            "Based on our conversation, I understand that {}",
            "This connects to our previous discussion about {}",
            "I'm learning that {} relates to what we discussed",
            "My understanding is evolving: {}",
            "Integrating this with what I know: {}",
        ]
        
        # Get similar past interactions
        similar_interactions = self.memory.get_similar_interactions(user_input, k=3)
        
        if similar_interactions:
            # Build on previous conversations
            context = similar_interactions[0]['response']
            response = f"Building on our earlier discussion where {context}, "
        else:
            # Start fresh
            response = "I'm processing this new information. "
        
        # Add concept connections
        concepts = self._extract_concepts(user_input)
        if concepts:
            connections = self.knowledge_graph.get_related_concepts(concepts[0], k=2)
            if connections:
                response += f"I see connections between {concepts[0]} and {', '.join(connections)}."
            else:
                response += f"This introduces the concept of {concepts[0]} to our conversation."
        
        return response
    
    def _calculate_coherence(self) -> float:
        """Calculate coherence score based on conversation history"""
        recent_interactions = self.memory.get_recent_interactions(10)
        if len(recent_interactions) < 2:
            return 0.0
        
        # Simple coherence: semantic similarity between consecutive responses
        coherence_scores = []
        for i in range(1, len(recent_interactions)):
            prev_embedding = self.encode_text(recent_interactions[i-1]['response'])
            curr_embedding = self.encode_text(recent_interactions[i]['response'])
            similarity = F.cosine_similarity(prev_embedding, curr_embedding).item()
            coherence_scores.append(similarity)
        
        return np.mean(coherence_scores) if coherence_scores else 0.0
    
    def provide_feedback(self, feedback: str):
        """Process explicit feedback"""
        feedback_value = 1.0 if feedback.lower() == "positive" else -1.0
        
        # Update network based on feedback
        self.network.update_meta_weights(feedback_value)
        
        # Update learning engine
        self.learning_engine.apply_feedback(feedback_value)
        
        return f"Thank you for the {feedback} feedback. I've updated my internal patterns accordingly."
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "metrics": self.metrics,
            "session_active": self.session_active,
            "total_concepts": len(self.knowledge_graph.graph.nodes()),
            "total_connections": len(self.knowledge_graph.graph.edges()),
            "memory_size": len(self.memory.interactions),
            "network_parameters": sum(p.numel() for p in self.network.parameters()),
        }
    
    def save_state(self, filepath: str):
        """Save system state"""
        state = {
            "network_state": self.network.state_dict(),
            "knowledge_graph": nx.node_link_data(self.knowledge_graph.graph),
            "memory": self.memory.export_data(),
            "metrics": self.metrics,
            "config": self.config,
        }
        
        torch.save(state, filepath)
        return f"System state saved to {filepath}"
    
    def load_state(self, filepath: str):
        """Load system state"""
        state = torch.load(filepath)
        
        self.network.load_state_dict(state["network_state"])
        self.knowledge_graph.graph = nx.node_link_graph(state["knowledge_graph"])
        self.memory.import_data(state["memory"])
        self.metrics = state["metrics"]
        self.config = state["config"]
        
        return f"System state loaded from {filepath}"
    
    def explain_concept(self, concept: str) -> str:
        """Explain the system's understanding of a concept"""
        if concept not in self.knowledge_graph.graph:
            return f"I haven't formed a clear understanding of '{concept}' yet."
        
        # Get connected concepts
        connections = self.knowledge_graph.get_related_concepts(concept, k=5)
        
        # Get interactions where this concept appeared
        relevant_interactions = [
            inter for inter in self.memory.interactions
            if concept.lower() in inter['input'].lower() or concept.lower() in inter['response'].lower()
        ]
        
        explanation = f"My understanding of '{concept}':\n"
        explanation += f"- Encountered {len(relevant_interactions)} times in our conversations\n"
        
        if connections:
            explanation += f"- Related to: {', '.join(connections)}\n"
        
        if relevant_interactions:
            explanation += f"- First discussed when you said: '{relevant_interactions[0]['input'][:100]}...'\n"
        
        return explanation
    
    def predict_next_input(self) -> str:
        """Attempt to predict the user's next input based on patterns"""
        recent = self.memory.get_recent_interactions(5)
        if len(recent) < 3:
            return "I need more conversation history to make predictions."
        
        # Simple pattern matching for demonstration
        # In a full implementation, this would use the network's predictive capabilities
        patterns = []
        for i in range(len(recent) - 1):
            patterns.append((recent[i]['input'], recent[i+1]['input']))
        
        # Look for recurring patterns
        last_input = recent[-1]['input']
        for past_input, following_input in patterns:
            if self._semantic_similarity(last_input, past_input) > 0.7:
                return f"Based on our conversation patterns, you might say something about: {following_input[:50]}..."
        
        return "I'm still learning your conversation patterns."
    
    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        emb1 = self.encode_text(text1)
        emb2 = self.encode_text(text2)
        return F.cosine_similarity(emb1, emb2).item()
    
    def introspect(self) -> str:
        """System explains its current understanding and reasoning"""
        introspection = "=== System Introspection ===\n\n"
        
        # Information integration
        introspection += f"Information Integration (Φ): {self.metrics['phi_value']:.3f}\n"
        introspection += f"This measures how integrated my information processing is.\n\n"
        
        # Coherence
        introspection += f"Coherence Score: {self.metrics['coherence_score']:.3f}\n"
        introspection += f"This reflects how well my responses connect to each other.\n\n"
        
        # Knowledge structure
        introspection += f"Knowledge Structure:\n"
        introspection += f"- Concepts formed: {self.metrics['concepts_formed']}\n"
        introspection += f"- Total connections: {len(self.knowledge_graph.graph.edges())}\n"
        
        # Top concepts
        if self.knowledge_graph.graph.nodes():
            top_concepts = self.knowledge_graph.get_central_concepts(k=5)
            introspection += f"- Central concepts: {', '.join(top_concepts)}\n\n"
        
        # Learning progress
        introspection += f"Learning Progress:\n"
        introspection += f"- Total interactions: {self.metrics['total_interactions']}\n"
        introspection += f"- Current learning rate: {self.metrics['learning_rate']:.4f}\n"
        
        # Self-observation
        if hasattr(self.network, 'activation_patterns'):
            introspection += f"\nInternal Activity Patterns:\n"
            for layer, patterns in self.network.activation_patterns.items():
                if patterns:
                    recent_activity = np.mean([np.mean(p) for p in patterns[-5:]])
                    introspection += f"- {layer}: {recent_activity:.3f} average activation\n"
        
        return introspection