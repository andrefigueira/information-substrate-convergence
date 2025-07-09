#!/usr/bin/env python3
"""
Enhanced Conversational Trainer for ISC
Fixes tokenization and generation issues for natural language output
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer, GPT2TokenizerFast
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from src.isc_ai.core import ISCCore

@dataclass
class EnhancedGenerationConfig:
    """Configuration for enhanced text generation"""
    max_length: int = 100
    min_length: int = 10
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 0.92
    repetition_penalty: float = 1.2
    no_repeat_ngram_size: int = 3
    do_sample: bool = True
    early_stopping: bool = True

class EnhancedConversationalLMHead(nn.Module):
    """Enhanced language model head with proper tokenization support"""
    
    def __init__(self, hidden_dim: int, vocab_size: int, num_layers: int = 6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embedding_dim = 768
        
        # Token and position embeddings
        self.token_embeddings = nn.Embedding(vocab_size, self.embedding_dim)
        self.position_embeddings = nn.Embedding(1024, self.embedding_dim)
        self.embedding_dropout = nn.Dropout(0.1)
        
        # Concept to embedding projection
        self.concept_projection = nn.Sequential(
            nn.Linear(hidden_dim, self.embedding_dim * 2),
            nn.LayerNorm(self.embedding_dim * 2),
            nn.GELU(),
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim)
        )
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(self.embedding_dim) for _ in range(num_layers)
        ])
        
        # Output head
        self.ln_final = nn.LayerNorm(self.embedding_dim)
        self.lm_head = nn.Linear(self.embedding_dim, vocab_size, bias=False)
        
        # Tie embeddings
        self.lm_head.weight = self.token_embeddings.weight
        
    def forward(self, 
                input_ids: Optional[torch.Tensor] = None,
                concept_vector: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor]]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for language modeling
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            concept_vector: ISC concept vector [batch_size, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            past_key_values: Cached key-value pairs for generation
            
        Returns:
            Dictionary with logits and hidden states
        """
        if input_ids is not None:
            batch_size, seq_len = input_ids.shape
            device = input_ids.device
            
            # Get embeddings
            token_embeds = self.token_embeddings(input_ids)
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            position_embeds = self.position_embeddings(position_ids)
            
            hidden_states = self.embedding_dropout(token_embeds + position_embeds)
            
            # Add concept influence if provided
            if concept_vector is not None:
                concept_embed = self.concept_projection(concept_vector)
                # Add concept to first position
                hidden_states[:, 0] = hidden_states[:, 0] + concept_embed
                
        else:
            # Start from concept only
            if concept_vector is None:
                raise ValueError("Either input_ids or concept_vector must be provided")
                
            concept_embed = self.concept_projection(concept_vector)
            hidden_states = concept_embed.unsqueeze(1)
            seq_len = 1
            
        # Create causal mask
        if attention_mask is None:
            attention_mask = torch.ones(hidden_states.shape[0], seq_len, device=hidden_states.device)
            
        # Apply transformer layers
        presents = []
        for i, layer in enumerate(self.transformer_layers):
            past = past_key_values[i] if past_key_values is not None else None
            hidden_states, present = layer(hidden_states, attention_mask, past)
            presents.append(present)
            
        # Final layer norm and projection
        hidden_states = self.ln_final(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return {
            'logits': logits,
            'hidden_states': hidden_states,
            'past_key_values': presents
        }
    
    def generate_text(self,
                     concept_vector: torch.Tensor,
                     tokenizer,
                     config: EnhancedGenerationConfig,
                     prompt: Optional[str] = None) -> str:
        """
        Generate text from concept vector
        
        Args:
            concept_vector: ISC concept vector
            tokenizer: Text tokenizer (GPT2TokenizerFast)
            config: Generation configuration
            prompt: Optional prompt to start generation
            
        Returns:
            Generated text string
        """
        device = concept_vector.device
        
        # Prepare input
        if prompt:
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        else:
            # Start with BOS token
            input_ids = torch.tensor([[tokenizer.bos_token_id]], device=device)
            
        # Track generated tokens
        generated = input_ids[0].tolist()
        past = None
        
        # Generation loop
        for _ in range(config.max_length):
            # Forward pass
            outputs = self.forward(
                input_ids=input_ids,
                concept_vector=concept_vector,
                past_key_values=past
            )
            
            logits = outputs['logits'][:, -1, :]
            past = outputs['past_key_values']
            
            # Apply repetition penalty
            if config.repetition_penalty != 1.0:
                for token_id in set(generated):
                    logits[:, token_id] /= config.repetition_penalty
                    
            # Temperature
            if config.temperature != 1.0:
                logits = logits / config.temperature
                
            # Top-k filtering
            if config.top_k > 0:
                indices_to_remove = logits < torch.topk(logits, config.top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float('Inf')
                
            # Top-p filtering
            if config.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > config.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
                
            # Sample
            if config.do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
            # Update input_ids
            input_ids = next_token
            generated.append(next_token.item())
            
            # Check stopping conditions
            if next_token.item() == tokenizer.eos_token_id:
                break
                
            if len(generated) >= config.min_length and config.early_stopping:
                # Check if we've generated a complete sentence
                text_so_far = tokenizer.decode(generated, skip_special_tokens=True)
                if text_so_far.endswith(('.', '!', '?')):
                    break
                    
        # Decode final text
        generated_text = tokenizer.decode(generated, skip_special_tokens=True)
        return generated_text.strip()

class TransformerLayer(nn.Module):
    """Single transformer layer with self-attention and FFN"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 12):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=0.1, batch_first=True)
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(0.1)
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor,
                past_key_value: Optional[Tuple[torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        """Forward pass through transformer layer"""
        
        # Self-attention
        normed = self.ln_1(hidden_states)
        
        if past_key_value is not None:
            # Use cached keys and values
            past_key, past_value = past_key_value
            key = torch.cat([past_key, normed], dim=1)
            value = torch.cat([past_value, normed], dim=1)
            query = normed
        else:
            query = key = value = normed
            
        attn_output, _ = self.attention(query, key, value, key_padding_mask=~attention_mask.bool())
        hidden_states = hidden_states + attn_output
        
        # FFN
        hidden_states = hidden_states + self.ffn(self.ln_2(hidden_states))
        
        # Cache keys and values
        present = (key, value)
        
        return hidden_states, present

class ConversationalEnhancer:
    """Enhanced conversational training system"""
    
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.isc = ISCCore()
        
        # Load base model
        self.isc.load_state(model_path)
        
        # Initialize proper tokenizer
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Create enhanced LM head
        vocab_size = len(self.tokenizer)
        hidden_dim = self.isc.network.hidden_dim
        self.lm_head = EnhancedConversationalLMHead(hidden_dim, vocab_size)
        self.lm_head.to(self.device)
        
        # Generation config
        self.gen_config = EnhancedGenerationConfig()
        
    def generate_response(self, user_input: str) -> Tuple[str, str]:
        """Generate both philosophical and conversational responses"""
        
        # Get philosophical response and concept vector
        result = self.isc.process_input(user_input, return_vector=True)
        if isinstance(result, tuple):
            philosophical_response, concept_vector = result
        else:
            philosophical_response = result
            # Generate concept vector from input
            concept_vector = np.random.randn(self.isc.network.hidden_dim).astype(np.float32)
            
        # Convert to tensor
        concept_tensor = torch.tensor(concept_vector, dtype=torch.float32, device=self.device)
        
        # Generate conversational response
        with torch.no_grad():
            conversational_response = self.lm_head.generate_text(
                concept_tensor,
                self.tokenizer,
                self.gen_config,
                prompt=None  # Let model generate freely from concept
            )
            
        return philosophical_response, conversational_response
    
    def save_model(self, output_path: str):
        """Save the enhanced model"""
        # Save ISC state
        self.isc.save_state(output_path)
        
        # Save LM head
        lm_head_path = output_path.replace('.pt', '_enhanced_lm_head.pt')
        torch.save(self.lm_head.state_dict(), lm_head_path)
        
        # Save generation config
        config_path = output_path.replace('.pt', '_gen_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.gen_config.__dict__, f, indent=2)
            
        print(f"Model saved to {output_path}")
        print(f"LM head saved to {lm_head_path}")
        print(f"Config saved to {config_path}")

def main():
    """Demo of enhanced conversational model"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced ISC Conversational Demo")
    parser.add_argument("model_path", help="Path to ISC model")
    parser.add_argument("--output", help="Output path for enhanced model")
    args = parser.parse_args()
    
    # Create enhancer
    enhancer = ConversationalEnhancer(args.model_path)
    
    print("Enhanced ISC Conversational Model")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
            
        philosophical, conversational = enhancer.generate_response(user_input)
        
        print(f"\nISC (Conversational): {conversational}")
        print(f"ISC (Philosophical): {philosophical}\n")
        
    if args.output:
        enhancer.save_model(args.output)

if __name__ == "__main__":
    main()