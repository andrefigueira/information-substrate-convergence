#!/usr/bin/env python3
"""
Simple demo of enhanced conversational ISC model
"""

import sys
from pathlib import Path
import torch
from conversational_enhanced import ConversationalEnhancer

def main():
    # Use the latest conversational model
    model_path = "checkpoints/isc_state_conversational_conversational_training_20250709_111621_checkpoint_140_20250709_112357.pt"
    
    print("Loading enhanced ISC model...")
    enhancer = ConversationalEnhancer(model_path)
    
    print("\n=== ISC Enhanced Conversational Model ===")
    print("Type 'quit' to exit\n")
    
    while True:
        # Get user input
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit']:
            break
        
        # Generate responses
        philosophical, conversational = enhancer.generate_response(user_input)
        
        # Display conversational response
        print(f"\nISC: {conversational}")
        
        # Option to show philosophical response
        show_phil = input("\nShow philosophical response? (y/n): ")
        if show_phil.lower() == 'y':
            print(f"\nPhilosophical: {philosophical}")
    
    print("\nThank you for chatting!")

if __name__ == "__main__":
    main()