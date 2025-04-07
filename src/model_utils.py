# intent-reconstruction/src/model_utils.py
import logging
import torch
import os
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from configs.model_configs import ModelConfig
from typing import List


def _ensure_list(value) -> List:
    """값이 리스트가 아니면 리스트로 변환합니다."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple) or isinstance(value, set):
        return list(value)
    return [value]


def load_model_and_tokenizer(model_config: ModelConfig, device="cuda:3"):
    """모델과 토크나이저를 로드하는 함수"""
    logging.info(f"Loading {model_config.name} model and tokenizer")
    
    # CUDA 디바이스 설정
    if isinstance(device, str):
        device = torch.device(device)
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.tokenizer_id,
        trust_remote_code=model_config.trust_remote_code
    )
    
    # 토크나이저 설정
    tokenizer.padding_side = 'left'  # 왼쪽 패딩으로 설정
    tokenizer.pad_token = tokenizer.eos_token  # 패딩 토큰 설정
    
    # target_modules를 항상 리스트로 보장
    target_modules = _ensure_list(model_config.target_modules)
    logging.info(f"Target modules for {model_config.name}: {target_modules}")
    
    # 모델별 로딩 및 LoRA 적용 전략
    if "Gemma" in model_config.name:
        # Gemma 모델 (양자화 모델)의 경우 특별 처리
        model_kwargs = {
            "trust_remote_code": model_config.trust_remote_code,
            "device_map": {"": device},  
            "attn_implementation": "eager"
        }
        
        if model_config.quantization_config:
            quantization_config = BitsAndBytesConfig(**model_config.quantization_config)
            model_kwargs["quantization_config"] = quantization_config
        
        # 모델 로드 (이동 없음)
        model = AutoModelForCausalLM.from_pretrained(
            model_config.model_id,
            **model_kwargs
        )
        
        # Gemma 모델용 LoRA 설정
        lora_config = LoraConfig(
            r=model_config.lora_r,
            lora_alpha=model_config.lora_alpha,
            target_modules=target_modules,
            lora_dropout=model_config.lora_dropout,
            bias=model_config.lora_bias,
            task_type=TaskType.CAUSAL_LM,
            modules_to_save=None  # Gemma 모델은 modules_to_save 없음
        )
        
    else:
        # 다른 모델들의 경우
        if "OpenELM" in model_config.name:
            # OpenELM 모델 특수 처리
            model_kwargs = {
                "trust_remote_code": model_config.trust_remote_code,
                "torch_dtype": torch.float32,  
                "device_map": None
            }
            
            # 모델 로드 후 명시적으로 디바이스로 이동
            model = AutoModelForCausalLM.from_pretrained(
                model_config.model_id,
                **model_kwargs
            ).to(device)  
            
        else:
            # 일반 모델 (Llama 등)
            model_kwargs = {
                "trust_remote_code": model_config.trust_remote_code,
                "torch_dtype": torch.float16,
                "device_map": None
            }
            
            # 모델 로드
            model = AutoModelForCausalLM.from_pretrained(
                model_config.model_id,
                **model_kwargs
            ).to(device)
        
        # 일반 모델용 LoRA 설정
        lora_config = LoraConfig(
            r=model_config.lora_r,
            lora_alpha=model_config.lora_alpha,
            target_modules=target_modules,
            lora_dropout=model_config.lora_dropout,
            bias=model_config.lora_bias,
            task_type=TaskType.CAUSAL_LM,
            modules_to_save=None  # 모든 모델에 대해 modules_to_save 명시적 설정
        )
    
    try:
        # LoRA 적용
        model = get_peft_model(model, lora_config)
        
        # 학습 모드 설정
        model.train()
        # 그라디언트 초기화 - 모든 파라미터에 대해 개별적으로 처리
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    param.grad = None
        
        # LoRA 어댑터 활성화 상태 확인
        if hasattr(model, 'active_adapter'):
            logging.info(f"Active adapter: {model.active_adapter}")
        
        # 학습 가능한 파라미터 수 확인
        param_count = sum(p.numel() for p in model.parameters())
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Model parameters: Total={param_count}, Trainable={trainable_count}")
        
        # LoRA 어댑터 상태 확인
        if hasattr(model, 'peft_config'):
            adapter_names = list(model.peft_config.keys()) if model.peft_config else []
            logging.info(f"LoRA adapter names: {adapter_names}")
            
            # 어댑터 설정 확인
            for name, config in model.peft_config.items():
                logging.info(f"Adapter {name} config: r={config.r}, alpha={config.lora_alpha}, "
                           f"target_modules={config.target_modules}")
        
        logging.info(f"Applied LoRA to {model_config.name} with target modules: {target_modules}")
    except Exception as e:
        logging.error(f"Error applying LoRA to {model_config.name}: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        raise
    
    return model, tokenizer


def load_trained_model_and_tokenizer(model_config, checkpoint_path, device="cuda:3"):
    """학습된 모델과 토크나이저를 로드하는 함수"""
    logging.info(f"Loading trained {model_config.name} from {checkpoint_path}")
    
    # CUDA 디바이스 설정
    if isinstance(device, str):
        device = torch.device(device)
    
    # 토크나이저 로드
    try:
        # 먼저 체크포인트에서 토크나이저 로드 시도
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        logging.info(f"Loaded tokenizer from checkpoint: {checkpoint_path}")
    except Exception as e:
        # 체크포인트에서 로드 실패 시 원본 모델 ID에서 로드
        logging.warning(f"Failed to load tokenizer from checkpoint: {str(e)}")
        logging.info(f"Loading tokenizer from original model ID: {model_config.tokenizer_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_config.tokenizer_id, trust_remote_code=model_config.trust_remote_code)
    
    # 토크나이저 설정 - 명시적으로 설정
    tokenizer.padding_side = 'left'  # 왼쪽 패딩으로 설정
    tokenizer.pad_token = tokenizer.eos_token  # 패딩 토큰 설정
    
    # 설정 확인 로깅
    logging.info(f"Tokenizer settings: padding_side={tokenizer.padding_side}, pad_token={tokenizer.pad_token}")
    
    # 모델 로드 방식 결정
    try:
        # 체크포인트에 adapter_config.json이 있는지 확인
        adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            logging.info(f"Found adapter_config.json. Loading as LoRA adapter: {adapter_config_path}")
            
            # 원본 기본 모델 로드
            logging.info(f"Loading base model: {model_config.model_id}")
            base_model = AutoModelForCausalLM.from_pretrained(
                model_config.model_id,  # 원본 모델 ID 사용
                trust_remote_code=model_config.trust_remote_code,
                device_map={"": device} if "Gemma" in model_config.name else None,
                torch_dtype=torch.float32 if "OpenELM" in model_config.name else torch.float16,
            )
            
            # Gemma 외 모델은 디바이스로 명시적 이동
            if "Gemma" not in model_config.name and device is not None:
                base_model = base_model.to(device)
                logging.info(f"Moved base model to device: {device}")
            
            logging.info(f"Base model loaded successfully. Model type: {type(base_model).__name__}")
            
            # LoRA 모델 로드 전 파라미터 상태 로깅
            param_count = sum(p.numel() for p in base_model.parameters())
            trainable_count = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
            logging.info(f"Base model parameters: Total={param_count}, Trainable={trainable_count}")
            
            # LoRA 어댑터 로드 및 적용
            try:
                import peft
                logging.info(f"PEFT version: {peft.__version__}")
                
                # 로딩 전 어댑터 상태 확인
                adapter_names = []
                if hasattr(base_model, 'peft_config'):
                    adapter_names = list(base_model.peft_config.keys()) if base_model.peft_config else []
                    logging.info(f"Pre-loading adapter names: {adapter_names}")
                
                # 명시적으로 PeftModel 클래스를 사용하여 로드
                model = PeftModel.from_pretrained(
                    base_model, 
                    checkpoint_path,
                    device_map={"": device} if "Gemma" in model_config.name else None,
                    is_trainable=False  # 평가 모드용
                )
                
                # 로딩 후 어댑터 상태 확인
                if hasattr(model, 'peft_config'):
                    adapter_names = list(model.peft_config.keys()) if model.peft_config else []
                    logging.info(f"Post-loading adapter names: {adapter_names}")
                
                # 어댑터 활성화 확인
                if hasattr(model, 'active_adapter'):
                    logging.info(f"Active adapter: {model.active_adapter}")
                
                # 로딩 후 파라미터 상태 확인
                param_count_after = sum(p.numel() for p in model.parameters())
                trainable_count_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
                logging.info(f"LoRA model parameters: Total={param_count_after}, Trainable={trainable_count_after}")
                
                # LoRA 어댑터 설정 확인
                if hasattr(model, 'peft_config'):
                    for name, config in model.peft_config.items():
                        logging.info(f"Adapter {name} config: r={config.r}, alpha={config.lora_alpha}, "
                                   f"target_modules={config.target_modules}")
                
                logging.info(f"Successfully loaded LoRA adapter from {checkpoint_path}")
            except Exception as e:
                logging.error(f"Error loading LoRA adapter: {str(e)}")
                import traceback
                logging.error(traceback.format_exc())
                
                # 기본 모델 사용 (LoRA 로드 실패 시)
                model = base_model
                logging.warning(f"Using base model without LoRA adapter due to loading error")
            
        else:
            # 일반 모델로 로드 (LoRA가 아닌 경우)
            logging.info(f"No adapter config found. Loading as standard model: {checkpoint_path}")
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                trust_remote_code=model_config.trust_remote_code,
                device_map={"": device} if "Gemma" in model_config.name else None,
                torch_dtype=torch.float32 if "OpenELM" in model_config.name else torch.float16,
            )
            
            if "Gemma" not in model_config.name and device is not None:
                model = model.to(device)
                logging.info(f"Moved model to device: {device}")
                
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        raise
    
    return model, tokenizer

def save_model(model, tokenizer, output_dir):
    """모델과 토크나이저를 저장하는 함수"""
    try:
        # 저장 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 모델 저장
        logging.info(f"Saving model to {output_dir}")
        model.save_pretrained(output_dir)
        
        # 토크나이저 저장
        logging.info(f"Saving tokenizer to {output_dir}")
        tokenizer.save_pretrained(output_dir)
        
        logging.info(f"Model and tokenizer successfully saved to {output_dir}")
        return True
    except Exception as e:
        logging.error(f"Error saving model: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return False

def clear_gpu_memory():
    """GPU 메모리 정리"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()
        logging.info("GPU memory cleared")
