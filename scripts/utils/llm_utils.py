#!/usr/bin/env python3
"""
Utility functions for working with LLMs.
"""

import os
import json
import requests
from typing import List, Dict, Any, Optional

# Placeholder for actual API implementations
# You would need to implement these based on the LLM providers you want to use

def generate_with_openai(prompt: str, model: str = "gpt-3.5-turbo", temperature: float = 0.7) -> str:
    """Generate text using OpenAI API."""
    try:
        import openai
        
        # Check for API key
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        openai.api_key = api_key
        
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
        )
        
        return response.choices[0].message.content
    except ImportError:
        return "Error: OpenAI package not installed. Run 'pip install openai' to use this feature."
    except Exception as e:
        return f"Error with OpenAI API: {str(e)}"

def generate_with_anthropic(prompt: str, model: str = "claude-2", temperature: float = 0.7) -> str:
    """Generate text using Anthropic API."""
    try:
        # Check for API key
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
        
        headers = {
            "x-api-key": api_key,
            "content-type": "application/json"
        }
        
        data = {
            "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
            "model": model,
            "temperature": temperature,
            "max_tokens_to_sample": 1000
        }
        
        response = requests.post(
            "https://api.anthropic.com/v1/complete",
            headers=headers,
            data=json.dumps(data)
        )
        
        response.raise_for_status()
        return response.json().get("completion", "")
    except Exception as e:
        return f"Error with Anthropic API: {str(e)}"

def generate_with_local_model(prompt: str, model_path: str, temperature: float = 0.7) -> str:
    """Generate text using a local model with llama.cpp."""
    try:
        from llama_cpp import Llama
        
        # Check if model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Load the model
        llm = Llama(model_path=model_path, n_ctx=2048)
        
        # Generate response
        response = llm.create_completion(
            prompt,
            max_tokens=1000,
            temperature=temperature,
            stop=["\n\n"]
        )
        
        return response["choices"][0]["text"]
    except ImportError:
        return "Error: llama-cpp-python package not installed. Run 'pip install llama-cpp-python' to use this feature."
    except Exception as e:
        return f"Error with local model: {str(e)}"

def select_llm_provider(provider: str) -> callable:
    """Select the appropriate LLM provider function."""
    providers = {
        "openai": generate_with_openai,
        "anthropic": generate_with_anthropic,
        "local": generate_with_local_model,
    }
    
    return providers.get(provider.lower(), generate_with_openai)

def create_synthetic_dataset(input_texts: List[str], output_format: str = "qa", 
                           provider: str = "openai", model: str = "gpt-3.5-turbo") -> List[Dict[str, Any]]:
    """Create a synthetic dataset from input texts using an LLM."""
    dataset = []
    llm_function = select_llm_provider(provider)
    
    for text in input_texts:
        # Create a prompt based on the desired output format
        if output_format == "qa":
            prompt = f"""Based on the following text, generate 5 diverse question-answer pairs 
            that capture the style, knowledge, and tone of the original text. Format each pair as 
            a JSON object with 'question' and 'answer' fields.
            
            Text: {text}
            """
        elif output_format == "completion":
            prompt = f"""Based on the following text, generate 5 diverse text completion examples 
            that capture the style, knowledge, and tone of the original text. Format each example as 
            a JSON object with 'prompt' and 'completion' fields.
            
            Text: {text}
            """
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        # Generate examples using the selected LLM
        response = llm_function(prompt, model=model)
        
        # Parse the response and add to dataset
        try:
            # Try to parse as JSON list first
            examples = json.loads(response)
            if isinstance(examples, list):
                dataset.extend(examples)
            else:
                # If it's a single object, wrap it in a list
                dataset.append(examples)
        except json.JSONDecodeError:
            # If not valid JSON, try to extract JSON objects from the text
            import re
            json_objects = re.findall(r'\{[^\{\}]*\}', response)
            for json_obj in json_objects:
                try:
                    example = json.loads(json_obj)
                    dataset.append(example)
                except json.JSONDecodeError:
                    pass
    
    return dataset

def main():
    # Example usage
    print("This is a utility module and not meant to be run directly.")
    print("Import and use these functions in your scripts instead.")

if __name__ == "__main__":
    main()
