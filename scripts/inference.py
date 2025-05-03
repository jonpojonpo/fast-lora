#!/usr/bin/env python3
"""
Performs inference using the trained and converted GGUF model.
"""

import os
import argparse
from llama_cpp import Llama

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run inference with GGUF model')
    parser.add_argument('--model_path', type=str, default='models/gguf/model.gguf', help='Path to the GGUF model')
    parser.add_argument('--prompt', type=str, help='Prompt to use for inference')
    parser.add_argument('--temp', type=float, default=0.7, help='Temperature for sampling')
    parser.add_argument('--max_tokens', type=int, default=512, help='Maximum number of tokens to generate')
    parser.add_argument('--top_p', type=float, default=0.95, help='Top-p sampling value')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    return parser.parse_args()

def format_prompt(user_prompt, model_path):
    """Format the prompt based on model type."""
    if "qwen" in model_path.lower():
        # Qwen chat format
        return f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
    else:
        # Default Llama format
        return f"<s>[INST] {user_prompt} [/INST] "

def load_model(model_path):
    """Load the GGUF model."""
    try:
        model = Llama(
            model_path=model_path,
            n_ctx=4096,  # context window size
            n_batch=512  # batch size for prompt processing
        )
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def generate_response(model, prompt, temp, max_tokens, top_p, model_path):
    """Generate a response from the model."""
    try:
        formatted_prompt = format_prompt(prompt, model_path)
        
        # Set appropriate stop tokens based on model type
        stop_tokens = ["</s>", "[INST]"]
        if "qwen" in model_path.lower():
            stop_tokens = ["<|im_end|>", "<|im_start|>"]
        
        response = model.create_completion(
            formatted_prompt,
            max_tokens=max_tokens,
            temperature=temp,
            top_p=top_p,
            stop=stop_tokens,
            echo=False
        )
        
        return response['choices'][0]['text'].strip()
    except Exception as e:
        print(f"Error generating response: {e}")
        return None

def interactive_mode(model, temp, max_tokens, top_p, model_path):
    """Run the model in interactive mode."""
    print("\nEntering interactive mode. Type 'exit' to quit.\n")
    print("-------------------------------------------")
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ['exit', 'quit', 'q']:
            break
        
        print("\nGenerating response...")
        response = generate_response(model, user_input, temp, max_tokens, top_p, model_path)
        
        if response:
            print(f"\nModel: {response}")
        else:
            print("\nModel failed to generate a response.")
    
    print("\nExiting interactive mode.")

def main():
    args = parse_arguments()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return
    
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path)
    
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    print("Model loaded successfully.")
    
    if args.interactive:
        interactive_mode(model, args.temp, args.max_tokens, args.top_p, args.model_path)
    elif args.prompt:
        print("Generating response...")
        response = generate_response(model, args.prompt, args.temp, args.max_tokens, args.top_p, args.model_path)
        
        if response:
            print(f"\nResponse: {response}")
        else:
            print("Failed to generate a response.")
    else:
        print("Error: Either provide a prompt or use interactive mode.")
        print("Use --prompt "Your prompt here" or --interactive")

if __name__ == "__main__":
    main()
