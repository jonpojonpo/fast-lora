#!/usr/bin/env python3
"""
Utility functions for data processing and formatting.
"""

import os
import json
import glob
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import random

def read_text_files(directory: str, extension: str = "txt") -> List[Dict[str, str]]:
    """Read all text files from a directory."""
    files = []
    pattern = os.path.join(directory, f"*.{extension}")
    
    for file_path in glob.glob(pattern):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                files.append({
                    "filename": os.path.basename(file_path),
                    "content": content
                })
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    return files

def save_jsonl(data: List[Dict[str, Any]], output_path: str) -> None:
    """Save data as JSONL (one JSON object per line)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))
    return data

def split_dataset(data: List[Dict[str, Any]], train_ratio: float = 0.8, 
                 val_ratio: float = 0.1, test_ratio: float = 0.1, 
                 seed: int = 42) -> Dict[str, List[Dict[str, Any]]]:
    """Split a dataset into train, validation, and test sets."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    # Shuffle data with a fixed seed for reproducibility
    random.seed(seed)
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    # Calculate split points
    n_samples = len(shuffled_data)
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)
    
    # Split the data
    train_data = shuffled_data[:train_end]
    val_data = shuffled_data[train_end:val_end]
    test_data = shuffled_data[val_end:]
    
    return {
        "train": train_data,
        "validation": val_data,
        "test": test_data
    }

def format_for_training(data: List[Dict[str, Any]], format_type: str = "qa", 
                       instruction_template: Optional[str] = None) -> List[Dict[str, str]]:
    """Format data for different training formats."""
    formatted_data = []
    
    if format_type == "qa":
        # Format for question-answering
        for item in data:
            if "question" in item and "answer" in item:
                if instruction_template:
                    text = instruction_template.replace("{question}", item["question"]).replace("{answer}", item["answer"])
                    formatted_data.append({"text": text})
                else:
                    formatted_data.append({
                        "instruction": item["question"],
                        "response": item["answer"]
                    })
    
    elif format_type == "completion":
        # Format for text completion
        for item in data:
            if "prompt" in item and "completion" in item:
                formatted_data.append({
                    "prompt": item["prompt"],
                    "completion": item["completion"]
                })
    
    elif format_type == "alpaca":
        # Format for Alpaca-style dataset
        for item in data:
            if "instruction" in item and "response" in item:
                formatted_item = {
                    "instruction": item["instruction"],
                    "response": item["response"]
                }
                if "input" in item:
                    formatted_item["input"] = item["input"]
                formatted_data.append(formatted_item)
    
    return formatted_data

def convert_to_dataset_format(data: List[Dict[str, Any]], format_type: str = "huggingface") -> Any:
    """Convert processed data to various dataset formats."""
    if format_type == "huggingface":
        # Convert to format suitable for HuggingFace datasets
        try:
            from datasets import Dataset
            return Dataset.from_dict({k: [item.get(k, "") for item in data] for k in data[0].keys()})
        except ImportError:
            print("Warning: HuggingFace datasets library not found. Returning raw data.")
            return data
    
    elif format_type == "pandas":
        # Convert to pandas DataFrame
        return pd.DataFrame(data)
    
    else:
        # Return as-is for other formats
        return data

def main():
    # Example usage
    print("This is a utility module and not meant to be run directly.")
    print("Import and use these functions in your scripts instead.")

if __name__ == "__main__":
    main()
