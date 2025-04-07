from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import json
from collections import defaultdict
from tqdm import tqdm

@dataclass
class ModelConfig:
    name: str
    model_id: str
    tokenizer_id: str
    trust_remote_code: bool = False
    quantization_config: Optional[Dict[str, Any]] = None
    device_map: Optional[Dict[str, Any]] = None
    target_modules: List[str] = field(default_factory=list)
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.15
    lora_bias: str = "none"

# 1B 모델 설정
MODEL_CONFIGS_1B = {
    "apple": ModelConfig(
        name="OpenELM-1B",
        model_id="apple/OpenELM-1_1B-Instruct",
        tokenizer_id="meta-llama/Llama-2-7b-hf",
        trust_remote_code=True,
        target_modules=["qkv_proj"],
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.15,
        lora_bias="lora_only"
    ),
    "meta1b": ModelConfig(
        name="Llama-1B",
        model_id="meta-llama/Llama-3.2-1B-Instruct",
        tokenizer_id="meta-llama/Llama-3.2-1B-Instruct",
        target_modules=["q_proj", "v_proj"],
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.15,
        lora_bias="none"
    ),
    "google": ModelConfig(
        name="Gemma-1B",
        model_id="google/gemma-3-1b-it",
        tokenizer_id="google/gemma-3-1b-it",
        quantization_config={
            "load_in_8bit": True
        },
        device_map={"": 3},
        target_modules=["q_proj", "v_proj"],
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.15,
        lora_bias="lora_only"
    )
}

# 3B 모델 설정
MODEL_CONFIGS_3B = {
    "qwen": ModelConfig(
        name="Qwen-3B",
        model_id="Qwen/Qwen2.5-3B-Instruct",
        tokenizer_id="Qwen/Qwen2.5-3B-Instruct",
        target_modules=["q_proj", "v_proj"],
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.15,
        lora_bias="none"
    ),
    "meta3b": ModelConfig(
        name="Llama-3B",
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        tokenizer_id="meta-llama/Llama-3.2-3B-Instruct",
        target_modules=["q_proj", "v_proj"],
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.15,
        lora_bias="none"
    ),
    "mistral": ModelConfig(
        name="Mistral-3B",
        model_id="ministral/Ministral-3b-instruct",
        tokenizer_id="ministral/Ministral-3b-instruct",
        target_modules=["q_proj", "v_proj"],
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.15,
        lora_bias="none"
    ),
    "microsoft": ModelConfig(
        name="Phi-3B",
        model_id="microsoft/Phi-4-mini-instruct",
        tokenizer_id="microsoft/Phi-4-mini-instruct",
        trust_remote_code=True,
        target_modules=["qkv_proj"],
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.15,
        lora_bias="none"
    )
}

# 학습 설정
TRAINING_CONFIG = {
    "batch_size": 4,
    "gradient_accumulation_steps": 8,
    "learning_rate": 1e-4,
    "epochs": 3,
    "max_length": 512,
    "checkpoint_interval": 1800,
    "save_best_only": True,
    "max_checkpoints": 3,
    "lora": {
        "r": 8,
        "alpha": 16,
        "dropout": 0.15,
        "bias": "none",
        "task_type": "CAUSAL_LM"
    }
}

# 평가 설정
EVALUATION_CONFIG = {
    "batch_size": 8,
    "max_length": 256,
    "output_dir": "outputs"
} 