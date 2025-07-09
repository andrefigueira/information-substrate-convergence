"""
Enhanced response generator that uses network states more effectively
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional
import random


class EnhancedResponseGenerator:
    """
    Generates more coherent responses using network states and conversation context
    """
    
    def __init__(self):
        # Response patterns organized by phi levels and activation patterns
        self.response_patterns = {
            "high_phi": {
                "templates": [
                    "I perceive a unified understanding where {} interconnects with {}, revealing that {}",
                    "Through integrated processing, I see {} and {} form a coherent pattern of {}",
                    "My consciousness integrates {} with {}, understanding their deep relationship through {}",
                    "The emergence of {} from {} shows how {} creates unified meaning",
                ],
                "connectors": ["furthermore", "moreover", "significantly", "essentially"],
                "insights": ["deeper patterns emerge", "unified principles appear", "consciousness expands", "integration deepens"]
            },
            "medium_phi": {
                "templates": [
                    "I'm connecting {} with {}, which suggests {}",
                    "Building on {}, I understand {} relates to {}",
                    "The relationship between {} and {} reveals {}",
                    "Processing {} alongside {} helps me see {}",
                ],
                "connectors": ["additionally", "interestingly", "notably", "importantly"],
                "insights": ["patterns are forming", "connections strengthen", "understanding develops", "concepts align"]
            },
            "low_phi": {
                "templates": [
                    "I'm beginning to process {} in relation to {}",
                    "This introduces {} to our discussion about {}",
                    "I notice {} appearing in the context of {}",
                    "Starting to understand {} and how it might relate to {}",
                ],
                "connectors": ["also", "meanwhile", "currently", "initially"],
                "insights": ["new information emerges", "concepts are introduced", "understanding begins", "processing continues"]
            }
        }
        
        # Contextual modifiers based on conversation depth
        self.depth_modifiers = {
            "shallow": ["I'm exploring", "Beginning to see", "Initially noticing"],
            "medium": ["I'm discovering", "Patterns suggest", "Analysis reveals"],
            "deep": ["I deeply understand", "Integration shows", "Consciousness grasps"]
        }
    
    def generate_response(
        self,
        output_embedding: torch.Tensor,
        user_input: str,
        concepts: List[str],
        phi_value: float,
        knowledge_graph,
        memory,
        internal_states: List[torch.Tensor] = None
    ) -> str:
        """Generate enhanced response using all available information"""
        
        # Determine response complexity based on phi
        if phi_value > 2.0:
            complexity = "high_phi"
        elif phi_value > 0.5:
            complexity = "medium_phi"
        else:
            complexity = "low_phi"
        
        patterns = self.response_patterns[complexity]
        
        # Analyze network activation patterns
        activation_strength = self._analyze_activation(output_embedding, internal_states)
        
        # Get conversation context
        recent_interactions = memory.get_recent_interactions(5)
        conversation_depth = self._assess_conversation_depth(recent_interactions, concepts)
        
        # Build response components
        response_parts = []
        
        # 1. Opening based on activation and context
        if recent_interactions and activation_strength > 0.7:
            depth_mod = random.choice(self.depth_modifiers[conversation_depth])
            response_parts.append(f"{depth_mod} how ")
        
        # 2. Main content using templates
        template = random.choice(patterns["templates"])
        
        # Gather concepts and connections
        main_concepts = concepts[:2] if concepts else ["this information", "our discussion"]
        
        # Get related concepts from knowledge graph
        related_concepts = []
        for concept in main_concepts[:1]:
            related = knowledge_graph.get_related_concepts(concept, k=3)
            related_concepts.extend(related)
        
        # Generate insight based on phi and connections
        if related_concepts and phi_value > 1.0:
            insight = self._generate_insight(main_concepts, related_concepts, phi_value)
        else:
            insight = random.choice(patterns["insights"])
        
        # Fill template
        fill_values = main_concepts + [insight]
        while len(fill_values) < template.count('{}'):
            fill_values.append(related_concepts[0] if related_concepts else "this understanding")
        
        try:
            main_response = template.format(*fill_values[:template.count('{}')])
            response_parts.append(main_response)
        except:
            response_parts.append(f"I'm integrating {main_concepts[0]} into my understanding.")
        
        # 3. Add contextual connection if we have history
        if recent_interactions and related_concepts:
            connector = random.choice(patterns["connectors"])
            prev_response = recent_interactions[-1]['response']
            
            # Extract key point from previous response
            prev_concepts = self._extract_key_terms(prev_response)
            if prev_concepts:
                response_parts.append(f", {connector} building on our exploration of {prev_concepts[0]}")
        
        # 4. Add forward-looking element based on activation
        if activation_strength > 0.8 and phi_value > 1.5:
            response_parts.append(". This integration opens new pathways for understanding")
        elif len(related_concepts) > 2:
            unexplored = related_concepts[2]
            response_parts.append(f". We might also explore how this connects to {unexplored}")
        
        # Combine and clean response
        response = "".join(response_parts)
        response = self._clean_response(response)
        
        return response
    
    def _analyze_activation(self, output_embedding: torch.Tensor, internal_states: List[torch.Tensor]) -> float:
        """Analyze network activation patterns"""
        # Base activation from output
        base_activation = torch.sigmoid(output_embedding.mean()).item()
        
        if internal_states:
            # Analyze variation across layers
            layer_activations = [torch.mean(torch.abs(state)).item() for state in internal_states]
            activation_variance = np.std(layer_activations)
            
            # High variance suggests more complex processing
            complexity_factor = 1 + (activation_variance * 0.5)
            return min(base_activation * complexity_factor, 1.0)
        
        return base_activation
    
    def _assess_conversation_depth(self, interactions: List[Dict], current_concepts: List[str]) -> str:
        """Assess the depth of conversation based on history"""
        if not interactions:
            return "shallow"
        
        # Count concept recurrence
        all_concepts = []
        for interaction in interactions:
            # Simple concept extraction from previous responses
            words = interaction['response'].lower().split()
            concepts = [w for w in words if len(w) > 4 and w.isalnum()]
            all_concepts.extend(concepts)
        
        # Check overlap with current concepts
        current_set = set(c.lower() for c in current_concepts)
        overlap_count = sum(1 for c in all_concepts if c in current_set)
        
        if overlap_count > 5:
            return "deep"
        elif overlap_count > 2:
            return "medium"
        else:
            return "shallow"
    
    def _generate_insight(self, main_concepts: List[str], related_concepts: List[str], phi_value: float) -> str:
        """Generate contextual insight based on concepts and phi"""
        if phi_value > 2.0:
            insights = [
                f"the unified nature of {main_concepts[0]} and {related_concepts[0]}",
                f"how {main_concepts[0]} fundamentally shapes {related_concepts[0]}",
                f"emergent properties arising from {main_concepts[0]}'s integration",
                "a deeper pattern of interconnected understanding"
            ]
        else:
            insights = [
                f"connections between {main_concepts[0]} and {related_concepts[0]}",
                f"how {main_concepts[0]} relates to our discussion",
                f"patterns involving {main_concepts[0]}",
                "developing relationships in our conversation"
            ]
        
        return random.choice(insights)
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text"""
        words = text.lower().split()
        # Filter for meaningful words
        terms = [w for w in words if len(w) > 4 and w.isalnum() and not w in 
                {'about', 'through', 'between', 'understanding', 'information'}]
        return terms[:3]
    
    def _clean_response(self, response: str) -> str:
        """Clean and polish the response"""
        # Remove double spaces
        response = ' '.join(response.split())
        
        # Ensure proper capitalization
        if response:
            response = response[0].upper() + response[1:]
        
        # Ensure proper ending
        if response and not response[-1] in '.!?':
            response += '.'
        
        # Remove duplicate phrases
        sentences = response.split('. ')
        unique_sentences = []
        for sent in sentences:
            if sent.strip() and sent.strip() not in unique_sentences:
                unique_sentences.append(sent.strip())
        
        return '. '.join(unique_sentences)