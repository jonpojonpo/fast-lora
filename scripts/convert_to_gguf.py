#!/usr/bin/env python3
"""
Converts trained LoRA model to GGUF format for inference.
"""

import os
import argparse
import subprocess
import tempfile
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser(description='Convert LoRA model to GGUF format')
    parser.add_argument('--lora_model_path', type=str, default='models/lora', help='Path to the trained LoRA model')
    parser.add_argument('--base_model', type=str, default='meta-llama/Llama-2-7b-hf', help='Base model that was fine-tuned')
    parser.add_argument('--output_dir', type=str, default='models/gguf', help='Directory to save the GGUF model')
    parser.add_argument('--output_name', type=str, default='model', help='Name of the output GGUF file (without extension)')
    parser.add_argument('--quantization', type=str, default='q4_k_m', 
                        choices=['q4_0', 'q4_1', 'q5_0', 'q5_1', 'q8_0', 'q4_k_m'], 
                        help='Quantization method')
    return parser.parse_args()

def merge_lora_with_base_model(base_model, lora_model_path, temp_dir):
    """Merge LoRA weights with the base model."""
    try:
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoModelForCausalLM
        
        print(f"Loading LoRA model from {lora_model_path}...")
        model = AutoPeftModelForCausalLM.from_pretrained(
            lora_model_path,
            device_map="auto",
            torch_dtype="auto"
        )
        
        print("Merging LoRA weights with base model...")
        merged_model = model.merge_and_unload()
        
        print(f"Saving merged model to {temp_dir}...")
        merged_model.save_pretrained(temp_dir)
        merged_model.config.save_pretrained(temp_dir)
        
        # Also save the tokenizer
        model.tokenizer.save_pretrained(temp_dir)
        
        return True
    except Exception as e:
        print(f"Error merging models: {e}")
        return False

def convert_to_gguf(temp_dir, output_dir, output_name, quantization):
    """Convert the merged model to GGUF format using llama.cpp tools."""
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f"{output_name}.gguf")
        
        # Run the conversion script (assuming llama.cpp tools are installed)
        cmd = [
            "python", "-m", "llama_cpp.model_converter",
            "--outfile", output_path,
            "--outtype", "f16",
            "--quantize", quantization,
            temp_dir
        ]
        
        print(f"Running conversion command: {' '.join(cmd)}")
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print(process.stdout)
        
        if os.path.exists(output_path):
            print(f"Successfully created GGUF model at {output_path}")
            return True
        else:
            print(f"Conversion failed: GGUF file not found at {output_path}")
            print(f"Error: {process.stderr}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"Conversion failed with error: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False
    except Exception as e:
        print(f"Error during conversion: {e}")
        return False

def main():
    args = parse_arguments()
    
    # Create a temporary directory for the merged model
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Created temporary directory: {temp_dir}")
        
        # Step 1: Merge LoRA with base model
        success = merge_lora_with_base_model(
            args.base_model,
            args.lora_model_path,
            temp_dir
        )
        
        if not success:
            print("Failed to merge LoRA with base model. Aborting.")
            return
        
        # Step 2: Convert merged model to GGUF format
        success = convert_to_gguf(
            temp_dir,
            args.output_dir,
            args.output_name,
            args.quantization
        )
        
        if success:
            print(f"Conversion complete! GGUF model saved to {os.path.join(args.output_dir, args.output_name + '.gguf')}")
        else:
            print("Conversion failed.")

if __name__ == "__main__":
    main()
