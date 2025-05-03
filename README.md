# Fast Lora

Fast Lora is an opinionated workflow to transform text files into LoRA fine-tuned models and export them to quantized .gguf format for efficient inference.

## Overview

The workflow consists of these main steps:
- Start with text files in the `./input` directory
- Use LLMs to transform them into a synthetic dataset for training
- Train your LoRA model on a base LLM
- Convert to .gguf format for efficient inference
- Optionally upload to Hugging Face Hub
- Run inference with your custom model

## Setup

### Requirements

- Python 3.8+
- PyTorch
- Transformers library
- PEFT library
- Hugging Face Hub account (for model uploading, optional)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/fast-lora.git
cd fast-lora
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

1. Add your text files to the `input` directory
2. Run the full workflow:
```bash
python fast_lora.py
```

### Step-by-Step Usage

#### 1. Creating a Synthetic Dataset

Add `.txt` files to the `input` directory, then run:

```bash
python fast_lora.py --mode dataset --llm_model gpt-3.5-turbo
```

For OpenAI models, you'll need to set your API key:
```bash
export OPENAI_API_KEY=your_api_key_here
```

#### 2. Training a LoRA Model

Once your dataset is created:

```bash
python fast_lora.py --mode train --base_model meta-llama/Llama-2-7b-hf
```

Training parameters can be customized:
```bash
python fast_lora.py --mode train \
  --base_model meta-llama/Llama-2-7b-hf \
  --lora_r 16 \
  --lora_alpha 32 \
  --batch_size 8 \
  --epochs 5
```

#### 3. Converting to GGUF Format

After training your LoRA model:

```bash
python fast_lora.py --mode convert
```

You can specify the quantization method:
```bash
python fast_lora.py --mode convert --quantization q4_0
```

Available quantization methods:
- `q4_0`: 4-bit quantization (smaller size, lower quality)
- `q4_1`: 4-bit quantization with improved method
- `q5_0`: 5-bit quantization
- `q5_1`: 5-bit quantization with improved method
- `q8_0`: 8-bit quantization (larger size, higher quality)
- `q4_k_m`: 4-bit quantization with k-quants (recommended)

#### 4. Running Inference

Test your model with:

```bash
python fast_lora.py --mode inference --interactive
```

Or specify a prompt:
```bash
python fast_lora.py --mode inference --prompt "What is the purpose of LoRA fine-tuning?"
```

#### 5. Uploading to Hugging Face Hub (Optional)

To share your models:

```bash
python fast_lora.py --upload_to_hub --hf_repo_name yourusername/your-model-name --hf_token your_hf_token
```

This will create two repositories on Hugging Face Hub:
- `yourusername/your-model-name-lora`: containing the LoRA model
- `yourusername/your-model-name-gguf`: containing the GGUF model

## Advanced Configuration

### Using a Different Base Model

You can specify any model from Hugging Face that supports LoRA fine-tuning:

```bash
python fast_lora.py --base_model mistralai/Mistral-7B-v0.1
```

### Custom LLM for Dataset Creation

You can use different LLMs for creating the synthetic dataset:

```bash
python fast_lora.py --mode dataset --llm_model claude-2
```

For using Claude, set:
```bash
export ANTHROPIC_API_KEY=your_api_key_here
```

### Full Parameter List

Run the following to see all available parameters:

```bash
python fast_lora.py --help
```

## Directory Structure

- `input/`: Place your text files here
- `output/`: Contains the generated synthetic dataset
- `models/lora/`: Contains the trained LoRA model
- `models/gguf/`: Contains the converted GGUF model
- `scripts/`: Contains individual scripts for each step
  - `create_dataset.py`: Generates synthetic dataset 
  - `train_lora.py`: Trains LoRA model
  - `convert_to_gguf.py`: Converts to GGUF format
  - `inference.py`: Runs inference
  - `upload_to_hub.py`: Uploads to Hugging Face Hub
  - `utils/`: Helper functions and utilities

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face for their transformers and PEFT libraries
- The llama.cpp project for GGUF format and quantization methods