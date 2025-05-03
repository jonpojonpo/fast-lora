#!/usr/bin/env python3
"""
Transforms text files from input directory into a synthetic dataset for training.
"""

import os
import json
import glob
import argparse
from pathlib import Path

# You might need additional imports for LLM integration
# e.g., openai, transformers, etc.

def parse_arguments():
    parser = argparse.ArgumentParser(description='Create a synthetic dataset from input text files')
    parser.add_argument('--input_dir', type=str, default='input', help='Directory containing input text files')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save the synthetic dataset')
    parser.add_argument('--llm_model', type=str, default='gpt-3.5-turbo', help='LLM model to use for generation')
    return parser.parse_args()

def read_input_files(input_dir):
    """Read all text files from the input directory."""
    data = []
    for file_path in glob.glob(os.path.join(input_dir, '*.txt')):
        with open(file_path, 'r', encoding='utf-8') as f:
            data.append({
                'filename': os.path.basename(file_path),
                'content': f.read()
            })
    return data

def generate_synthetic_examples(input_data, model_name):
    """Use an LLM to transform input text into synthetic training examples."""
    # This is a placeholder for actual LLM integration
    # You would use an API like OpenAI or a local model here
    
    synthetic_examples = []
    
    for item in input_data:
        # For testing purposes, we'll create realistic synthetic data based on the content
        if "lora" in item['content'].lower():
            examples = [
                {
                    "question": "What is LoRA and how does it work?", 
                    "answer": "LoRA (Low-Rank Adaptation) is a fine-tuning technique that reduces trainable parameters by adding low-rank matrices to existing weights instead of modifying them directly. This preserves the original capabilities of the model while adding new knowledge or specializations."
                },
                {
                    "question": "What is the GGUF format used for?", 
                    "answer": "GGUF (GPT-Generated Unified Format) is a format for storing large language models that is optimized for inference, especially on consumer hardware. It supports various quantization methods to reduce the model size while maintaining reasonable performance."
                },
                {
                    "question": "What are the main steps in the Fast-LoRA workflow?", 
                    "answer": "The Fast-LoRA workflow consists of: 1) Starting with text files containing specialized knowledge, 2) Transforming text into a question-answer dataset, 3) Fine-tuning a pre-trained model with LoRA, 4) Converting the model to GGUF for efficient inference, and 5) Deploying or sharing the resulting model."
                },
                {
                    "question": "Why is LoRA more efficient than traditional fine-tuning?", 
                    "answer": "LoRA is more efficient than traditional fine-tuning because it significantly reduces the number of trainable parameters by adding low-rank matrices to existing weights instead of modifying all model weights directly. This approach requires less computational resources and memory while still allowing effective adaptation to new domains or tasks."
                },
                {
                    "question": "What type of tone and style should a model trained on this content adopt?", 
                    "answer": "The model should adopt a formal and educational tone, with a focus on clear explanations of technical concepts. It should avoid ambiguity, provide specific details when discussing processes or techniques, and maintain clarity and precision while remaining helpful and informative to users with varying levels of technical knowledge."
                }
            ]
        else:
            # Generic examples for other content
            examples = [
                {"question": f"What is the main topic of {item['filename']}?", 
                 "answer": f"The main topic appears to be related to {item['filename'].split('.')[0]}."},
                {"question": "Could you summarize this content?", 
                 "answer": "The content covers several important concepts and provides detailed explanations of the subject matter."},
                {"question": "What is the most important concept mentioned?", 
                 "answer": "While several concepts are discussed, the core focus seems to be on the fundamental principles and their applications."},
                {"question": "How would you apply this knowledge in practice?", 
                 "answer": "This knowledge can be applied by following the outlined steps and integrating the concepts into relevant workflows or systems."},
                {"question": "What are the benefits of this approach?", 
                 "answer": "The benefits include improved efficiency, better performance, and more effective results compared to traditional methods."}
            ]
        
        synthetic_examples.extend(examples)
    
    return synthetic_examples

def save_dataset(examples, output_dir):
    """Save the synthetic examples as a dataset."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as jsonl (one JSON object per line)
    output_path = os.path.join(output_dir, 'synthetic_dataset.jsonl')
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"Saved {len(examples)} examples to {output_path}")

def main():
    args = parse_arguments()
    
    print(f"Reading input files from {args.input_dir}...")
    input_data = read_input_files(args.input_dir)
    
    if not input_data:
        print(f"No text files found in {args.input_dir}. Please add .txt files to continue.")
        return
    
    print(f"Found {len(input_data)} text files. Generating synthetic examples...")
    synthetic_examples = generate_synthetic_examples(input_data, args.llm_model)
    
    print(f"Saving dataset to {args.output_dir}...")
    save_dataset(synthetic_examples, args.output_dir)
    
    print("Done!")

if __name__ == "__main__":
    main()
