#!/usr/bin/env python3
"""
Uploads models to Hugging Face Hub.
"""

import os
import argparse
from huggingface_hub import HfApi, create_repo, upload_file, upload_folder

def parse_arguments():
    parser = argparse.ArgumentParser(description='Upload models to Hugging Face Hub')
    parser.add_argument('--model_path', type=str, help='Path to the model to upload')
    parser.add_argument('--model_type', type=str, choices=['lora', 'gguf'], help='Type of model')
    parser.add_argument('--repo_name', type=str, help='Name of the repository (username/repo)')
    parser.add_argument('--token', type=str, help='HuggingFace API token')
    parser.add_argument('--commit_message', type=str, default='Upload model', help='Commit message')
    return parser.parse_args()

def validate_inputs(args):
    """Validate the command line arguments."""
    if not os.path.exists(args.model_path):
        print(f"Error: Model path {args.model_path} does not exist.")
        return False
    
    if not args.token:
        print("Error: HuggingFace token is required.")
        return False
    
    if not args.repo_name:
        print("Error: Repository name is required.")
        return False
    
    return True

def upload_to_hub(model_path, model_type, repo_name, token, commit_message):
    """Upload the model to the Hugging Face Hub."""
    try:
        # Initialize the Hugging Face API
        api = HfApi(token=token)
        
        # Create or get the repository
        repo_url = create_repo(
            repo_name,
            token=token,
            exist_ok=True,
            repo_type="model",
        )
        
        print(f"Repository URL: {repo_url}")
        
        # Upload the model based on type
        if model_type == 'gguf' and os.path.isfile(model_path):
            # For GGUF, which is typically a single file
            print(f"Uploading GGUF file: {model_path}")
            upload_file(
                path_or_fileobj=model_path,
                path_in_repo=os.path.basename(model_path),
                repo_id=repo_name,
                token=token,
                commit_message=commit_message,
            )
        elif model_type == 'lora' and os.path.isdir(model_path):
            # For LoRA, which is a directory with model files
            print(f"Uploading LoRA model directory: {model_path}")
            upload_folder(
                folder_path=model_path,
                repo_id=repo_name,
                token=token,
                commit_message=commit_message,
            )
        else:
            print(f"Error: Unsupported model type or path format. {model_type=}, {model_path=}")
            return False
        
        print(f"Successfully uploaded model to {repo_name}")
        return True
    
    except Exception as e:
        print(f"Error uploading to Hugging Face Hub: {e}")
        return False

def main():
    args = parse_arguments()
    
    # Validate inputs
    if not validate_inputs(args):
        return
    
    # Upload to HF Hub
    success = upload_to_hub(
        args.model_path,
        args.model_type,
        args.repo_name,
        args.token,
        args.commit_message
    )
    
    if success:
        print("Upload completed successfully.")
    else:
        print("Upload failed.")

if __name__ == "__main__":
    main()
