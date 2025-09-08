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
    
    def __init__(self, input_dim: int = 384, hidden_dim: int = 512, num_layers: int = 4):
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
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, auto_load: bool = True):
        self.config = config or self._default_config()
        
        # Initialize persistence manager
        from .persistence import PersistenceManager
        self.persistence = PersistenceManager()
        
        # Initialize components (note: all-MiniLM-L6-v2 produces 384-dim embeddings)
        self.network = SelfModifyingNetwork(input_dim=384, hidden_dim=512, num_layers=4)
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
        
        # Auto-load latest state if requested
        if auto_load:
            self._auto_load_state()
        
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
    
    def _auto_load_state(self):
        """Automatically load the latest state if available"""
        # First check for migration needs
        self.persistence.migrate_legacy_files()
        
        # Try to load latest state
        state = self.persistence.load_latest_state()
        if state:
            try:
                # Check if state has the expected structure
                if not isinstance(state, dict):
                    print("✗ Invalid state format - not a dictionary")
                    print("  Starting with fresh state")
                    return
                
                # Check for required keys
                required_keys = ["network_state", "knowledge_graph", "memory", "metrics"]
                missing_keys = [key for key in required_keys if key not in state]
                if missing_keys:
                    print(f"✗ State missing required keys: {missing_keys}")
                    print("  Starting with fresh state")
                    return
                
                self.network.load_state_dict(state["network_state"])
                self.knowledge_graph.graph = nx.node_link_graph(state["knowledge_graph"])
                self.memory.import_data(state["memory"])
                self.metrics = state["metrics"]
                
                # Merge config, keeping any new settings
                saved_config = state.get("config", {})
                self.config.update(saved_config)
                
                # Update components that depend on loaded state
                if hasattr(self.integrator, 'phi_history') and "phi_history" in state:
                    self.integrator.phi_history = state["phi_history"]
                
                print(f"✓ Automatically loaded previous state")
                print(f"  - Total interactions: {self.metrics['total_interactions']}")
                print(f"  - Concepts formed: {len(self.knowledge_graph.graph.nodes())}")
                print(f"  - Last phi value: {self.metrics['phi_value']:.3f}")
            except Exception as e:
                print(f"✗ Failed to restore state: {e}")
                print("  Starting with fresh state")
    
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
        
        # Store internal states for response generation
        self._last_internal_states = internal_states
        
        # Calculate information integration (Φ)
        if internal_states:
            phi_value = self.integrator.calculate_phi(internal_states)
            self.metrics["phi_value"] = phi_value
        
        # Update knowledge graph with input concepts
        input_concepts = self._extract_concepts(user_input)
        for concept in input_concepts:
            self.knowledge_graph.add_concept(concept, input_embedding)
        
        # Form connections between co-occurring concepts in input
        self._form_concept_connections(input_concepts)
        
        # Generate response
        response = self._generate_response(output_embedding, user_input)
        
        # Extract response concepts and add to graph
        response_concepts = self._extract_concepts(response)
        for concept in response_concepts:
            self.knowledge_graph.add_concept(concept, output_embedding)
        
        # Connect input and response concepts
        self._connect_input_response_concepts(input_concepts, response_concepts)
        
        # Update embeddings and find similar concepts
        self._update_concept_embeddings(input_concepts, input_embedding)
        self._update_concept_embeddings(response_concepts, output_embedding)
        
        # Learn from interaction
        if hasattr(self.learning_engine, 'learn_from_interaction'):
            # Check if it's the enhanced version that accepts phi
            if 'phi_value' in self.learning_engine.learn_from_interaction.__code__.co_varnames:
                self.learning_engine.learn_from_interaction(user_input, response, internal_states, phi_value)
            else:
                self.learning_engine.learn_from_interaction(user_input, response, internal_states)
        else:
            # Fallback for older learning engine
            pass
        
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
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab')
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        
        # Include important short words and proper nouns
        important_short_words = {'cat', 'dog', 'pet', 'ai', 'phi'}
        
        concepts = []
        for w in tokens:
            if w.isalnum() and w not in stop_words:
                if len(w) > 3 or w in important_short_words:
                    concepts.append(w)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_concepts = []
        for c in concepts:
            if c not in seen:
                seen.add(c)
                unique_concepts.append(c)
        
        return unique_concepts[:10]  # Return top 10 unique concepts
    
    def _generate_response(self, output_embedding: torch.Tensor, user_input: str) -> str:
        """Generate response based on output embedding and context"""
        # Try to use enhanced response generator
        try:
            from .enhanced_response_generator import EnhancedResponseGenerator
            
            # Create generator if not exists
            if not hasattr(self, '_response_generator'):
                self._response_generator = EnhancedResponseGenerator()
            
            # Get current phi value and internal states
            phi_value = self.metrics.get("phi_value", 0.0)
            internal_states = getattr(self, '_last_internal_states', None)
            
            # Extract concepts
            concepts = self._extract_concepts(user_input)
            
            # Generate enhanced response
            response = self._response_generator.generate_response(
                output_embedding=output_embedding,
                user_input=user_input,
                concepts=concepts,
                phi_value=phi_value,
                knowledge_graph=self.knowledge_graph,
                memory=self.memory,
                internal_states=internal_states
            )
            
            return response
            
        except Exception as e:
            print(f"Enhanced response generation failed: {e}")
            # Fall back to original template method
        
        # Original template-based approach as fallback
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
    
    def save_state(self, filepath: Optional[str] = None):
        """Save system state - uses persistence manager by default"""
        state = {
            "network_state": self.network.state_dict(),
            "knowledge_graph": nx.node_link_data(self.knowledge_graph.graph),
            "memory": self.memory.export_data(),
            "metrics": self.metrics,
            "config": self.config,
        }
        
        # Add phi history if available
        if hasattr(self.integrator, 'phi_history'):
            state["phi_history"] = self.integrator.phi_history
        
        if filepath:
            # Save to specific file if provided
            torch.save(state, filepath)
            return f"System state saved to {filepath}"
        else:
            # Use persistence manager for standard location
            saved_path = self.persistence.save_state(state)
            return f"System state saved to {saved_path}"
    
    def load_state(self, filepath: str):
        """Load system state"""
        # Use weights_only=True for security - prevents arbitrary code execution
        # If this fails due to incompatible data, we'll handle it gracefully
        try:
            state = torch.load(filepath, weights_only=True, map_location='cpu')
        except Exception:
            # Fall back to unsafe loading with a warning
            import warnings
            warnings.warn(
                f"Loading {filepath} with weights_only=False. "
                "This file may contain arbitrary code. "
                "Only load files from trusted sources.",
                RuntimeWarning
            )
            state = torch.load(filepath, weights_only=False, map_location='cpu')
        
        # Validate state structure
        if not isinstance(state, dict):
            raise ValueError(f"Invalid state file format: expected dict, got {type(state)}")
        
        # Check for required keys
        required_keys = ["network_state", "knowledge_graph", "memory", "metrics"]
        missing_keys = [key for key in required_keys if key not in state]
        if missing_keys:
            raise KeyError(f"State file missing required keys: {missing_keys}")
        
        # Load components
        self.network.load_state_dict(state["network_state"])
        self.knowledge_graph.graph = nx.node_link_graph(state["knowledge_graph"])
        self.memory.import_data(state["memory"])
        self.metrics = state["metrics"]
        
        # Config is optional - merge with existing
        if "config" in state:
            self.config.update(state["config"])
        
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
    
    def _form_concept_connections(self, concepts: List[str]):
        """Form connections between co-occurring concepts"""
        # Connect all pairs of concepts that appear together
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                # Base strength on proximity (closer concepts = stronger connection)
                distance = j - i
                strength = 1.0 / (1 + distance * 0.1)
                self.knowledge_graph.add_connection(concepts[i], concepts[j], strength)
    
    def _connect_input_response_concepts(self, input_concepts: List[str], response_concepts: List[str]):
        """Connect concepts from input with concepts in response"""
        # Connect each input concept with each response concept
        for input_concept in input_concepts:
            for response_concept in response_concepts:
                # Lower strength for input-response connections
                self.knowledge_graph.add_connection(input_concept, response_concept, 0.5)
        
        # Also form connections within response concepts
        self._form_concept_connections(response_concepts)
    
    def _update_concept_embeddings(self, concepts: List[str], embedding: torch.Tensor):
        """Update embeddings and find similar concepts to connect"""
        if not concepts or embedding is None:
            return
        
        # For each concept, find similar concepts based on embeddings
        for concept in concepts:
            similar_concepts = self._find_similar_concepts(concept, embedding)
            for similar_concept, similarity in similar_concepts:
                if similarity > 0.7:  # Threshold for semantic similarity
                    self.knowledge_graph.add_connection(concept, similar_concept, similarity)
    
    def _find_similar_concepts(self, concept: str, embedding: torch.Tensor, k: int = 5) -> List[Tuple[str, float]]:
        """Find semantically similar concepts based on embeddings"""
        similar = []
        concept_lower = concept.lower()
        
        # Get all concepts with embeddings
        for other_concept, embeddings in self.knowledge_graph.concept_embeddings.items():
            if other_concept == concept_lower or not embeddings:
                continue
            
            # Compare with average embedding of the concept
            avg_embedding = np.mean(embeddings, axis=0)
            avg_embedding_tensor = torch.tensor(avg_embedding)
            
            # Calculate cosine similarity
            # Handle different tensor shapes
            if len(embedding.shape) == 1:
                emb1 = embedding.unsqueeze(0)
            else:
                emb1 = embedding.reshape(1, -1)
            
            if len(avg_embedding_tensor.shape) == 1:
                emb2 = avg_embedding_tensor.unsqueeze(0)
            else:
                emb2 = avg_embedding_tensor.reshape(1, -1)
            
            # Ensure same dimensions
            if emb1.shape[1] != emb2.shape[1]:
                continue
            
            similarity = F.cosine_similarity(emb1, emb2).item()
            
            similar.append((other_concept, similarity))
        
        # Sort by similarity and return top k
        similar.sort(key=lambda x: x[1], reverse=True)
        return similar[:k]