#!/usr/bin/env python3
"""
Consciousness-Driven Token Generation for ISC
Implements autoregressive generation with phi-based scoring and self-observation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import math

@dataclass
class GenerationConfig:
    """Configuration for consciousness-driven generation"""
    max_length: int = 100
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    beam_size: int = 5
    repetition_penalty: float = 1.2
    phi_weight: float = 0.3  # Weight for phi-based scoring
    observation_depth: int = 3  # Layers of self-observation
    
class ObserverLayer(nn.Module):
    """Self-observation layer that monitors token generation quality"""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Observer attention mechanism
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Quality assessment layers
        self.coherence_assessor = nn.Linear(hidden_dim, 1)
        self.relevance_assessor = nn.Linear(hidden_dim, 1)
        self.phi_assessor = nn.Linear(hidden_dim, 1)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Observe and assess the quality of hidden states
        
        Returns:
            Dictionary with quality scores and refined states
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Self-attention for observation
        Q = self.query_proj(hidden_states)
        K = self.key_proj(hidden_states)
        V = self.value_proj(hidden_states)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.hidden_dim)
        
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        observed_states = torch.matmul(attn_weights, V)
        
        # Assess quality metrics
        coherence = torch.sigmoid(self.coherence_assessor(observed_states))
        relevance = torch.sigmoid(self.relevance_assessor(observed_states))
        phi_score = torch.sigmoid(self.phi_assessor(observed_states))
        
        return {
            'observed_states': observed_states,
            'coherence': coherence,
            'relevance': relevance,
            'phi_score': phi_score,
            'attention_weights': attn_weights
        }

class ConsciousnessLMHead(nn.Module):
    """Enhanced language model head with consciousness-driven generation"""
    def __init__(self, hidden_dim: int, vocab_size: int, embedding_dim: int = 768):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Token embeddings for input
        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.position_embeddings = nn.Embedding(512, embedding_dim)  # Max seq length
        
        # Projection layers
        self.concept_projection = nn.Linear(hidden_dim, embedding_dim)
        
        # Transformer-style generation layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embedding_dim) for _ in range(4)
        ])
        
        # Observer layers for self-monitoring
        self.observers = nn.ModuleList([
            ObserverLayer(embedding_dim) for _ in range(3)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(embedding_dim, vocab_size)
        
        # Phi computation module
        self.phi_computer = PhiComputer(embedding_dim)
        
    def forward(self, concept_vector: torch.Tensor, 
                input_ids: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with consciousness-driven generation
        
        Args:
            concept_vector: ISC concept vector
            input_ids: Previously generated token IDs for autoregressive generation
            past_key_values: Cached key-value pairs for efficiency
            
        Returns:
            Dictionary with logits and consciousness metrics
        """
        # Ensure concept_vector has batch dimension
        if concept_vector.dim() == 1:
            concept_vector = concept_vector.unsqueeze(0)
            
        batch_size = concept_vector.size(0)
        
        # Project concept vector
        concept_embed = self.concept_projection(concept_vector)
        
        if input_ids is None:
            # Start of generation - use concept embedding
            hidden_states = concept_embed.unsqueeze(1)  # Add sequence dimension
            seq_len = 1
        else:
            # Continue generation - combine concept with token embeddings
            seq_len = input_ids.size(1)
            token_embeds = self.token_embeddings(input_ids)
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
            position_embeds = self.position_embeddings(position_ids)
            
            # Combine embeddings
            hidden_states = token_embeds + position_embeds
            
            # Add concept influence to first position
            hidden_states[:, 0] = hidden_states[:, 0] + concept_embed
            
        # Apply transformer blocks with self-observation
        consciousness_metrics = []
        
        for i, (transformer, observer) in enumerate(zip(self.transformer_blocks, self.observers[:len(self.transformer_blocks)])):
            # Transform hidden states
            hidden_states = transformer(hidden_states)
            
            # Observe and assess quality
            obs_results = observer(hidden_states)
            consciousness_metrics.append(obs_results)
            
            # Refine states based on observation
            hidden_states = hidden_states + 0.1 * obs_results['observed_states']
            
        # Final output projection
        logits = self.output_projection(hidden_states)
        
        # Compute phi for the generated sequence
        phi_value = self.phi_computer(hidden_states)
        
        return {
            'logits': logits,
            'hidden_states': hidden_states,
            'phi_value': phi_value,
            'consciousness_metrics': consciousness_metrics
        }
    
    def generate(self, concept_vector: torch.Tensor, 
                 tokenizer, 
                 config: GenerationConfig,
                 context_vectors: Optional[torch.Tensor] = None) -> Tuple[List[int], float]:
        """
        Generate tokens using consciousness-driven autoregressive generation
        
        Returns:
            Tuple of (generated token IDs, average phi score)
        """
        # Ensure concept_vector has batch dimension
        if concept_vector.dim() == 1:
            concept_vector = concept_vector.unsqueeze(0)
            
        device = concept_vector.device
        batch_size = concept_vector.size(0)
        
        # Initialize with BOS token if available
        if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
            input_ids = torch.full((batch_size, 1), tokenizer.bos_token_id, device=device)
        else:
            # Start with the most likely token from concept
            with torch.no_grad():
                initial_output = self.forward(concept_vector)
                logits = initial_output['logits']
                # Get the token with highest probability from the first position
                if logits.dim() == 3:  # [batch, seq, vocab]
                    input_ids = logits[:, 0, :].argmax(dim=-1).unsqueeze(1)
                else:  # [batch, vocab]
                    input_ids = logits.argmax(dim=-1).unsqueeze(1)
        
        # Track phi scores
        phi_scores = []
        
        # Autoregressive generation loop
        for step in range(config.max_length):
            with torch.no_grad():
                # Get model outputs
                outputs = self.forward(concept_vector, input_ids)
                logits = outputs['logits'][:, -1, :]  # Get last position
                phi_scores.append(outputs['phi_value'].item())
                
                # Apply repetition penalty
                if config.repetition_penalty != 1.0:
                    for token_id in input_ids[0].tolist():
                        logits[:, token_id] /= config.repetition_penalty
                
                # Apply temperature
                if config.temperature != 1.0:
                    logits = logits / config.temperature
                
                # Top-k filtering
                if config.top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, config.top_k)[0][..., -1, None]
                    logits[indices_to_remove] = -float('Inf')
                
                # Top-p (nucleus) filtering
                if config.top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > config.top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float('Inf')
                
                # Sample from distribution
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                # Check for EOS token
                if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
                    if next_token.item() == tokenizer.eos_token_id:
                        break
        
        # Return generated tokens and average phi score
        generated_tokens = input_ids[0].tolist()
        avg_phi = sum(phi_scores) / len(phi_scores) if phi_scores else 0.0
        
        return generated_tokens, avg_phi

class TransformerBlock(nn.Module):
    """Single transformer block for sequence processing"""
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x

class PhiComputer(nn.Module):
    """Compute phi (integrated information) for hidden states"""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.integration_assessor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute phi value for hidden states"""
        # Pool across sequence dimension
        pooled = hidden_states.mean(dim=1)
        
        # Assess integration
        phi = self.integration_assessor(pooled)
        
        return phi.squeeze(-1)

def beam_search_generate(model: ConsciousnessLMHead, 
                        concept_vector: torch.Tensor,
                        tokenizer,
                        config: GenerationConfig) -> Tuple[List[int], float]:
    """
    Beam search with consciousness-based scoring
    
    Returns best sequence and its phi score
    """
    # Ensure concept_vector has batch dimension
    if concept_vector.dim() == 1:
        concept_vector = concept_vector.unsqueeze(0)
        
    device = concept_vector.device
    beam_size = config.beam_size
    
    # Initialize beams
    beams = [([], 0.0, concept_vector)]  # (tokens, score, hidden_state)
    
    for step in range(config.max_length):
        candidates = []
        
        for tokens, score, hidden in beams:
            # Convert tokens to tensor
            if tokens:
                input_ids = torch.tensor([tokens], device=device)
            else:
                input_ids = None
                
            # Get predictions
            with torch.no_grad():
                outputs = model.forward(concept_vector, input_ids)
                logits = outputs['logits'][:, -1, :]
                phi_value = outputs['phi_value'].item()
                
            # Get top-k tokens
            topk_probs, topk_ids = torch.topk(F.softmax(logits, dim=-1), k=min(config.top_k, logits.size(-1)))
            
            # Add candidates
            for i in range(topk_ids.size(-1)):
                token = topk_ids[0, i].item()
                token_prob = topk_probs[0, i].item()
                
                # Calculate consciousness-aware score
                new_score = score + math.log(token_prob) + config.phi_weight * phi_value
                
                candidates.append((tokens + [token], new_score, hidden))
                
        # Select top beams
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:beam_size]
        
        # Check for EOS in all beams
        if all(hasattr(tokenizer, 'eos_token_id') and 
               tokenizer.eos_token_id in beam[0] for beam in beams):
            break
    
    # Return best beam
    best_tokens, best_score, _ = beams[0]
    return best_tokens, best_score / len(best_tokens) if best_tokens else 0.0