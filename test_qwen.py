#!/usr/bin/env python3
"""
Test script for Fast-LoRA with Qwen3-0.6B model.
"""

import os
import argparse
import subprocess
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser(description="Test Fast-LoRA with Qwen3-0.6B")
    parser.add_argument('--mode', type=str, choices=['dataset', 'train', 'convert', 'inference', 'all'], 
                        default='all', help='Operational mode')
    parser.add_argument('--input_dir', type=str, default='input', help='Directory with input text files')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory for output files')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode for inference')
    return parser.parse_args()

def run_command(command, description):
    """Run a command and display output."""
    print(f"\n{'-'*50}")
    print(f"{description}...")
    print(f"{'-'*50}\n")
    
    try:
        process = subprocess.run(command, check=True)
        print(f"\n{description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nError executing {description}: {e}")
        return False

def create_dataset(args):
    """Create a synthetic dataset from input text files using a default LLM."""
    command = [
        "python", "scripts/create_dataset.py",
        "--input_dir", args.input_dir,
        "--output_dir", args.output_dir
    ]
    
    return run_command(command, "Dataset creation")

def train_qwen_lora(args):
    """Train a LoRA model on Qwen3-0.6B."""
    command = [
        "python", "scripts/train_lora.py",
        "--dataset_path", f"{args.output_dir}/synthetic_dataset.jsonl",
        "--output_dir", "models/lora/qwen-0.6b",
        "--base_model", "Qwen/Qwen3-0.6B",
        "--lora_r", "8",
        "--lora_alpha", "16",
        "--batch_size", "4",
        "--num_epochs", "3"
    ]
    
    return run_command(command, "Qwen3-0.6B LoRA training")

def convert_to_gguf(args):
    """Convert LoRA model to GGUF format."""
    command = [
        "python", "scripts/convert_to_gguf.py",
        "--lora_model_path", "models/lora/qwen-0.6b",
        "--base_model", "Qwen/Qwen3-0.6B",
        "--output_dir", "models/gguf",
        "--output_name", "qwen-0.6b",
        "--quantization", "q4_k_m"
    ]
    
    # Run the conversion
    success = run_command(command, "GGUF conversion")
    
    # Verify that the output file exists even if the command reports success
    output_file = os.path.join("models/gguf", "qwen-0.6b.gguf")
    if success and not os.path.exists(output_file):
        print(f"Warning: Conversion reported success but output file {output_file} was not found.")
        return False
    
    return success

def run_inference(args):
    """Run inference with the GGUF model."""
    command = [
        "python", "scripts/inference.py",
        "--model_path", "models/gguf/qwen-0.6b.gguf"
    ]
    
    if args.interactive:
        command.append("--interactive")
    else:
        prompt = "Explain how LoRA fine-tuning works in simple terms."
        command.extend(["--prompt", prompt])
    
    return run_command(command, "Model inference")

def main():
    args = parse_arguments()
    
    print("\n" + "="*50)
    print("       Testing Fast-LoRA with Qwen3-0.6B       ")
    print("="*50 + "\n")
    
    # Check if there are input files
    if args.mode in ['dataset', 'all']:
        input_files = list(Path(args.input_dir).glob('*.txt'))
        if not input_files:
            print(f"Warning: No .txt files found in {args.input_dir} directory.")
            print("Please add text files before running dataset creation.")
            return
    
    # Execute operations based on mode
    if args.mode == 'dataset' or args.mode == 'all':
        if not create_dataset(args):
            print("Dataset creation failed. Stopping workflow.")
            return
    
    if args.mode == 'train' or args.mode == 'all':
        if not train_qwen_lora(args):
            print("Qwen3-0.6B LoRA training failed. Stopping workflow.")
            return
    
    if args.mode == 'convert' or args.mode == 'all':
        if not convert_to_gguf(args):
            print("GGUF conversion failed. Stopping workflow.")
            return
    
    if args.mode == 'inference' or args.mode == 'all':
        if not run_inference(args):
            print("Inference failed.")
            return
    
    print("\n" + "="*50)
    print("     Qwen3-0.6B Testing Complete     ")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()