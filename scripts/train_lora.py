#!/usr/bin/env python3
"""
Trains a LoRA model based on the synthetic dataset.
"""

import os
import argparse
import torch
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a LoRA model')
    parser.add_argument('--dataset_path', type=str, default='output/synthetic_dataset.jsonl', help='Path to the dataset file')
    parser.add_argument('--output_dir', type=str, default='models/lora', help='Directory to save the model')
    parser.add_argument('--base_model', type=str, default='meta-llama/Llama-2-7b-hf', help='Base model to fine-tune')
    parser.add_argument('--lora_r', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout')
    parser.add_argument('--batch_size', type=int, default=4, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    return parser.parse_args()

def prepare_dataset(dataset_path, tokenizer, max_length):
    """Prepare the dataset for training."""
    dataset = load_dataset('json', data_files=dataset_path, split='train')
    
    def preprocess_function(examples):
        # Format as instruction-response pairs
        texts = []
        for question, answer in zip(examples['question'], examples['answer']):
            # Check if we're using Qwen model
            if "qwen" in tokenizer.name_or_path.lower():
                # Qwen chat format
                texts.append(f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>")
            else:
                # Default Llama format
                texts.append(f"<s>[INST] {question} [/INST] {answer} </s>")
        
        return tokenizer(texts, truncation=True, max_length=max_length, padding='max_length')
    
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=['question', 'answer'],
    )
    
    return tokenized_dataset

def configure_lora(model, lora_r, lora_alpha, lora_dropout):
    """Configure and apply LoRA to the model."""
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    return model

def train_model(model, tokenizer, dataset, output_dir, batch_size, learning_rate, num_epochs):
    """Train the LoRA model."""
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        gradient_accumulation_steps=4,
        fp16=True,
        report_to="none",
    )
    
    # Initialize data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train the model
    trainer.train()
    
    # Save the final model
    model.save_pretrained(output_dir)
    
    return model

def main():
    args = parse_arguments()
    
    print(f"Loading base model {args.base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    print("Preparing dataset...")
    dataset = prepare_dataset(args.dataset_path, tokenizer, args.max_length)
    
    print("Configuring LoRA...")
    model = configure_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout)
    
    print("Starting training...")
    train_model(
        model, 
        tokenizer,
        dataset, 
        args.output_dir, 
        args.batch_size, 
        args.learning_rate, 
        args.num_epochs
    )
    
    print(f"Training complete! Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
