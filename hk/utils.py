import torch
import os
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from typing import List, Optional

def clear_gpu_memory():
    """Clear GPU memory to free up resources"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()

def load_base_model_and_tokenizer(model_id: str, device: str = "cuda:0"):
    """
    Load base model and tokenizer
    
    Args:
        model_id: Hugging Face model ID
        device: Device to load the model on
    
    Returns:
        model, tokenizer
    """
    print(f"Loading base model and tokenizer: {model_id}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set padding side to left for better generation
    tokenizer.padding_side = 'left'
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map={"": device} if device.startswith("cuda") else None,
    )
    
    # Move model to device if not already done
    if not device.startswith("cuda"):
        model = model.to(device)
    
    return model, tokenizer

def get_lora_model(model, lora_config: dict):
    """
    Apply LoRA configuration to model
    
    Args:
        model: Base model
        lora_config: LoRA configuration parameters
    
    Returns:
        model with LoRA applied
    """
    lora_config_obj = LoraConfig(
        r=lora_config["r"],
        lora_alpha=lora_config["alpha"],
        target_modules=lora_config["target_modules"],
        lora_dropout=lora_config["dropout"],
        bias=lora_config["bias"],
        task_type=TaskType.CAUSAL_LM
    )
    
    try:
        # Apply LoRA
        model = get_peft_model(model, lora_config_obj)
        print(f"Applied LoRA with target modules: {lora_config['target_modules']}")
    except Exception as e:
        print(f"Error applying LoRA: {str(e)}")
        raise
    
    return model

def load_lora_model(base_model_id: str, adapter_path: str, device: str = "cuda:0"):
    """
    Load a LoRA adapter and apply it to a base model
    
    Args:
        base_model_id: Hugging Face model ID for base model
        adapter_path: Path to LoRA adapter
        device: Device to load the model on
    
    Returns:
        model, tokenizer
    """
    print(f"Loading base model: {base_model_id}")
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map={"": device} if device.startswith("cuda") else None,
    )
    
    # Move model to device if not already done
    if not device.startswith("cuda"):
        base_model = base_model.to(device)
    
    # Load tokenizer
    try:
        # Try to load tokenizer from adapter path first
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        print(f"Loaded tokenizer from adapter path: {adapter_path}")
    except Exception:
        # Fall back to original model tokenizer
        print(f"Loading tokenizer from base model: {base_model_id}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set padding side to left for better generation
    tokenizer.padding_side = 'left'
    
    # Load and apply LoRA adapter
    try:
        model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            is_trainable=False  # Set to False for inference
        )
        print(f"Successfully loaded LoRA adapter from {adapter_path}")
    except Exception as e:
        print(f"Error loading LoRA adapter: {str(e)}")
        raise
    
    # Set model to evaluation mode
    model.eval()
    
    return model, tokenizer
