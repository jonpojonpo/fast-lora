#!/usr/bin/env python3
"""
Fast Lora - Main script to orchestrate the workflow.
"""

import os
import argparse
import subprocess
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser(description="Fast Lora - Turn text into a LoRA and GGUF model")
    
    # Main operational mode
    parser.add_argument('--mode', type=str, choices=['dataset', 'train', 'convert', 'inference', 'all'], 
                        default='all', help='Operational mode')
    
    # Dataset generation parameters
    parser.add_argument('--input_dir', type=str, default='input', help='Directory with input text files')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory for output files')
    parser.add_argument('--llm_model', type=str, default='gpt-3.5-turbo', help='LLM model for synthetic data generation')
    
    # Training parameters
    parser.add_argument('--dataset_path', type=str, default='output/synthetic_dataset.jsonl', help='Path to the dataset file')
    parser.add_argument('--base_model', type=str, default='meta-llama/Llama-2-7b-hf', help='Base model to fine-tune')
    parser.add_argument('--lora_r', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha')
    parser.add_argument('--batch_size', type=int, default=4, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    
    # Conversion parameters
    parser.add_argument('--lora_model_path', type=str, default='models/lora', help='Path to the trained LoRA model')
    parser.add_argument('--gguf_dir', type=str, default='models/gguf', help='Directory to save the GGUF model')
    parser.add_argument('--model_name', type=str, default='model', help='Name of the output model file')
    parser.add_argument('--quantization', type=str, default='q4_k_m', help='Quantization method')
    
    # Inference parameters
    parser.add_argument('--gguf_model_path', type=str, default='models/gguf/model.gguf', help='Path to the GGUF model')
    parser.add_argument('--prompt', type=str, help='Prompt for inference')
    parser.add_argument('--interactive', action='store_true', help='Interactive inference mode')
    
    # HF Hub parameters
    parser.add_argument('--upload_to_hub', action='store_true', help='Upload models to HuggingFace Hub')
    parser.add_argument('--hf_repo_name', type=str, help='HuggingFace repository name (username/repo)')
    parser.add_argument('--hf_token', type=str, help='HuggingFace API token')
    
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
    """Create a synthetic dataset from input text files."""
    command = [
        "python", "scripts/create_dataset.py",
        "--input_dir", args.input_dir,
        "--output_dir", args.output_dir,
        "--llm_model", args.llm_model
    ]
    
    return run_command(command, "Dataset creation")

def train_lora(args):
    """Train a LoRA model."""
    command = [
        "python", "scripts/train_lora.py",
        "--dataset_path", args.dataset_path,
        "--output_dir", args.lora_model_path,
        "--base_model", args.base_model,
        "--lora_r", str(args.lora_r),
        "--lora_alpha", str(args.lora_alpha),
        "--batch_size", str(args.batch_size),
        "--num_epochs", str(args.epochs)
    ]
    
    return run_command(command, "LoRA training")

def convert_to_gguf(args):
    """Convert LoRA model to GGUF format."""
    command = [
        "python", "scripts/convert_to_gguf.py",
        "--lora_model_path", args.lora_model_path,
        "--base_model", args.base_model,
        "--output_dir", args.gguf_dir,
        "--output_name", args.model_name,
        "--quantization", args.quantization
    ]
    
    return run_command(command, "GGUF conversion")

def run_inference(args):
    """Run inference with the GGUF model."""
    command = [
        "python", "scripts/inference.py",
        "--model_path", args.gguf_model_path
    ]
    
    if args.interactive:
        command.append("--interactive")
    elif args.prompt:
        command.extend(["--prompt", args.prompt])
    else:
        print("Warning: No prompt provided and interactive mode not enabled.")
        print("Adding --interactive flag automatically.")
        command.append("--interactive")
    
    return run_command(command, "Model inference")

def upload_models(args):
    """Upload models to Hugging Face Hub."""
    # Upload LoRA model
    lora_command = [
        "python", "scripts/upload_to_hub.py",
        "--model_path", args.lora_model_path,
        "--model_type", "lora",
        "--repo_name", f"{args.hf_repo_name}-lora",
        "--token", args.hf_token,
        "--commit_message", "Upload LoRA model"
    ]
    
    lora_success = run_command(lora_command, "LoRA model upload")
    
    # Upload GGUF model
    gguf_command = [
        "python", "scripts/upload_to_hub.py",
        "--model_path", os.path.join(args.gguf_dir, f"{args.model_name}.gguf"),
        "--model_type", "gguf",
        "--repo_name", f"{args.hf_repo_name}-gguf",
        "--token", args.hf_token,
        "--commit_message", "Upload GGUF model"
    ]
    
    gguf_success = run_command(gguf_command, "GGUF model upload")
    
    return lora_success and gguf_success

def ensure_directories_exist():
    """Ensure all required directories exist."""
    dirs = ['input', 'output', 'models', 'models/lora', 'models/gguf']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Directory '{dir_path}' is ready.")

def main():
    args = parse_arguments()
    
    print("\n" + "="*50)
    print("             Fast Lora Workflow               ")
    print("="*50 + "\n")
    
    # Ensure directories exist
    ensure_directories_exist()
    
    # Check if there are input files
    if args.mode in ['dataset', 'all']:
        input_files = list(Path(args.input_dir).glob('*.txt'))
        if not input_files:
            print(f"Warning: No .txt files found in {args.input_dir} directory.")
            print("Please add text files before running dataset creation.")
            return
    
    # Execute the requested operations
    if args.mode == 'dataset' or args.mode == 'all':
        if not create_dataset(args):
            print("Dataset creation failed. Stopping workflow.")
            return
    
    if args.mode == 'train' or args.mode == 'all':
        if not train_lora(args):
            print("LoRA training failed. Stopping workflow.")
            return
    
    if args.mode == 'convert' or args.mode == 'all':
        if not convert_to_gguf(args):
            print("GGUF conversion failed. Stopping workflow.")
            return
    
    if args.mode == 'inference' or args.mode == 'all':
        if not run_inference(args):
            print("Inference failed.")
            return
    
    # Upload to HF Hub if requested
    if args.upload_to_hub:
        if not args.hf_repo_name or not args.hf_token:
            print("Error: HuggingFace repository name and token are required for uploading.")
            print("Please provide --hf_repo_name and --hf_token parameters.")
            return
        
        if not upload_models(args):
            print("Model upload failed.")
            return
    
    print("\n" + "="*50)
    print("          Fast Lora Workflow Complete          ")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
