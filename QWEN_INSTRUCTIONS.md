# Using Fast-LoRA with Qwen3-0.6B

This document provides instructions on how to use Fast-LoRA specifically with the Qwen3-0.6B model.

## Prerequisites

Before starting, ensure you have:
- A machine with sufficient GPU memory (at least 8GB recommended)
- Dependencies installed from `requirements.txt`
- Hugging Face account and token (for downloading the model)

## Usage Instructions

### 1. Create a Synthetic Dataset

```bash
# Create a dataset from your text files in the input directory
python fast_lora.py --mode dataset
```

### 2. Train Qwen3-0.6B with LoRA

```bash
# Train Qwen3-0.6B with LoRA
python fast_lora.py --mode train \
  --base_model Qwen/Qwen3-0.6B \
  --lora_r 8 \
  --lora_alpha 16 \
  --batch_size 4 \
  --epochs 3
```

### 3. Convert to GGUF Format

```bash
# Convert to GGUF format for efficient inference
python fast_lora.py --mode convert \
  --base_model Qwen/Qwen3-0.6B \
  --lora_model_path models/lora \
  --output_name qwen-0.6b \
  --quantization q4_k_m
```

### 4. Run Inference

```bash
# Run inference with interactive mode
python fast_lora.py --mode inference \
  --gguf_model_path models/gguf/qwen-0.6b.gguf \
  --interactive
```

## Important Notes for Qwen3-0.6B

1. **Chat Format**: 
   - The code has been updated to support Qwen3's specific chat format:
   ```
   <|im_start|>user
   Your question here
   <|im_end|>
   <|im_start|>assistant
   Model's response
   <|im_end|>
   ```

2. **Memory Requirements**:
   - Even though Qwen3-0.6B is relatively small, you'll need adequate GPU memory for training
   - Consider reducing batch size if you encounter memory issues

3. **Quantization**:
   - The recommended quantization for Qwen3-0.6B is q4_k_m
   - For better quality but larger file size, try q8_0

4. **Models Directory Structure**:
   - The trained LoRA model will be saved to `models/lora/`
   - The converted GGUF model will be saved to `models/gguf/qwen-0.6b.gguf`

5. **GGUF Conversion**:
   - The script attempts to use `llama_cpp.model_converter` for conversion
   - If this module is not available, a fallback method is used to create a test GGUF file
   - For production use, you would need one of these options:
     - Install the full llama.cpp toolchain
     - Use the ct2-transformers-converter tool (pip install ct2-transformers)
     - Use the direct exllama2 conversion tools

## Troubleshooting

- If you encounter CUDA out-of-memory errors, try:
  - Reducing batch size (`--batch_size 2` or even `1`)
  - Using gradient accumulation (adjust in `train_lora.py`)
  - Training with 8-bit precision

- If the model generation is poor quality:
  - Try increasing LoRA rank (`--lora_r 16` or higher)
  - Increase training epochs (`--epochs 5` or more)
  - Improve your synthetic dataset with more diverse examples

## Using the Test Script

For convenience, you can use the test script that's specifically configured for Qwen3-0.6B:

```bash
# Run the full workflow
python test_qwen.py

# Or run specific steps
python test_qwen.py --mode dataset
python test_qwen.py --mode train
python test_qwen.py --mode convert
python test_qwen.py --mode inference --interactive
```

This script uses the optimal parameters for Qwen3-0.6B already set.