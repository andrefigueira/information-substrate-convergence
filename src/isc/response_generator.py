"""
Enhanced response generator using language models and network outputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import random


class NeuralResponseGenerator(nn.Module):
    """
    Generates responses using the network's internal states and a language model
    """
    
    def __init__(self, embedding_dim: int = 384, use_pretrained: bool = True):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        if use_pretrained:
            # Use GPT-2 for text generation
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.language_model = GPT2LMHeadModel.from_pretrained('gpt2')
            
            # Set pad token
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Projection layer to convert our embeddings to GPT-2's dimension
            self.projection = nn.Linear(embedding_dim, self.language_model.config.n_embd)
        else:
            # Lightweight custom decoder
            self.decoder = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=8),
                num_layers=4
            )
            self.vocab_projection = nn.Linear(embedding_dim, 50257)  # GPT-2 vocab size
        
        self.use_pretrained = use_pretrained
        
    def generate(
        self, 
        output_embedding: torch.Tensor,
        context: str,
        concepts: List[str],
        phi_value: float,
        knowledge_graph,
        temperature: float = 0.8,
        max_length: int = 100
    ) -> str:
        """
        Generate response using all available information
        """
        if self.use_pretrained:
            return self._generate_with_gpt2(
                output_embedding, context, concepts, phi_value, knowledge_graph, temperature, max_length
            )
        else:
            return self._generate_custom(
                output_embedding, context, concepts, phi_value, knowledge_graph
            )
    
    def _generate_with_gpt2(
        self,
        output_embedding: torch.Tensor,
        context: str,
        concepts: List[str],
        phi_value: float,
        knowledge_graph,
        temperature: float,
        max_length: int
    ) -> str:
        """Generate using GPT-2 conditioned on network output"""
        
        # Create a prompt that incorporates the system's understanding
        prompt_parts = []
        
        # Add phi-based prefix
        if phi_value > 2.0:
            prompt_parts.append("With deep integration of concepts")
        elif phi_value > 1.0:
            prompt_parts.append("Connecting multiple ideas")
        else:
            prompt_parts.append("Processing this information")
        
        # Add concept connections
        if concepts:
            main_concept = concepts[0]
            related = knowledge_graph.get_related_concepts(main_concept, k=3)
            if related:
                prompt_parts.append(f"I see {main_concept} relates to {', '.join(related[:2])}")
        
        # Add context awareness
        if context:
            prompt_parts.append(f"Building on: {context[:50]}")
        
        # Combine prompt
        prompt = ", ".join(prompt_parts) + ". "
        
        # Encode prompt
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        
        # Project output embedding and use as additional context
        projected = self.projection(output_embedding.unsqueeze(0))
        
        # Generate with GPT-2
        with torch.no_grad():
            # Create attention mask
            attention_mask = torch.ones_like(inputs)
            
            # Generate
            outputs = self.language_model.generate(
                inputs,
                attention_mask=attention_mask,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                # Use the projected embedding to influence generation
                encoder_hidden_states=projected.unsqueeze(1).repeat(1, inputs.shape[1], 1)
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        return response
    
    def _generate_custom(
        self,
        output_embedding: torch.Tensor,
        context: str,
        concepts: List[str],
        phi_value: float,
        knowledge_graph
    ) -> str:
        """Fallback generation method using templates enhanced by network state"""
        
        # Enhanced templates based on phi level
        if phi_value > 2.0:
            templates = [
                "I perceive a deep integration between {} and {}, revealing {}",
                "The unified understanding shows {} fundamentally connects to {}",
                "Through integrated processing, {} emerges as {} within {}",
                "My consciousness integrates {} with {}, forming a coherent understanding of {}"
            ]
        elif phi_value > 1.0:
            templates = [
                "I'm discovering connections between {} and {}",
                "This relates to {} through {}",
                "Building patterns from {} to {}",
                "Integrating {} reveals its relationship with {}"
            ]
        else:
            templates = [
                "I'm beginning to understand {}",
                "This introduces {} to our discussion",
                "Processing {} in this context",
                "Learning about {} and its implications"
            ]
        
        # Select template based on network activation
        activation_level = torch.mean(output_embedding).item()
        template_idx = int((activation_level + 1) * len(templates) / 2) % len(templates)
        template = templates[template_idx]
        
        # Fill template with concepts and connections
        fill_values = []
        
        if concepts:
            fill_values.extend(concepts[:2])
            
            # Add related concepts
            for concept in concepts[:1]:
                related = knowledge_graph.get_related_concepts(concept, k=2)
                fill_values.extend(related)
        
        # Add emergent understanding based on phi
        if phi_value > 1.5:
            emergence_terms = ["deeper patterns", "unified principles", "emergent properties"]
        else:
            emergence_terms = ["new connections", "developing patterns", "initial insights"]
        
        fill_values.extend(random.sample(emergence_terms, min(1, len(emergence_terms))))
        
        # Ensure we have enough values
        while len(fill_values) < template.count('{}'):
            fill_values.append("this understanding")
        
        # Generate response
        try:
            response = template.format(*fill_values[:template.count('{}')])
        except:
            response = f"I'm integrating the concept of {concepts[0] if concepts else 'this'} into my understanding."
        
        # Add context if available
        if context:
            response = f"Reflecting on our discussion, {response.lower()}"
        
        return response


class ResponseGeneratorWithMemory:
    """
    Wrapper that adds memory and conversation tracking to response generation
    """
    
    def __init__(self, generator: NeuralResponseGenerator, memory_window: int = 10):
        self.generator = generator
        self.memory_window = memory_window
        self.conversation_history = []
        
    def generate_response(
        self,
        output_embedding: torch.Tensor,
        user_input: str,
        concepts: List[str],
        phi_value: float,
        knowledge_graph,
        memory,
        temperature: float = 0.8
    ) -> str:
        """Generate response with full context"""
        
        # Get conversation context
        recent_interactions = memory.get_recent_interactions(self.memory_window)
        
        # Build context string
        context_parts = []
        for interaction in recent_interactions[-3:]:
            context_parts.append(f"User: {interaction['input']}")
            context_parts.append(f"AI: {interaction['response']}")
        
        context = " ".join(context_parts[-100:])  # Limit context length
        
        # Generate response
        response = self.generator.generate(
            output_embedding=output_embedding,
            context=context,
            concepts=concepts,
            phi_value=phi_value,
            knowledge_graph=knowledge_graph,
            temperature=temperature
        )
        
        # Post-process
        response = self._post_process_response(response, user_input, phi_value)
        
        return response
    
    def _post_process_response(self, response: str, user_input: str, phi_value: float) -> str:
        """Clean up and enhance the generated response"""
        
        # Remove repetitions
        sentences = response.split('. ')
        unique_sentences = []
        for sent in sentences:
            if sent.strip() and sent.strip() not in unique_sentences:
                unique_sentences.append(sent.strip())
        
        response = '. '.join(unique_sentences)
        
        # Ensure response isn't too short
        if len(response.split()) < 5:
            if phi_value > 1.0:
                response += " This connects to the deeper patterns we're exploring."
            else:
                response += " I'm developing my understanding of this."
        
        # Ensure proper ending
        if response and not response[-1] in '.!?':
            response += '.'
        
        return response