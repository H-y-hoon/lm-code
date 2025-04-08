# intent-reconstruction/src/model_utils_v2.py
import logging
import torch
import os
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from configs.model_configs import ModelConfig
from typing import List, Dict, Optional, Union, Tuple

# 로깅 설정
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

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
    """모델과 토크나이저를 로드하는 함수 (학습용)"""
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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # 패딩 토큰 설정
    
    # target_modules를 항상 리스트로 보장
    target_modules = _ensure_list(model_config.target_modules)
    logging.info(f"Target modules for {model_config.name}: {target_modules}")
    
    # 모델별 로딩 및 LoRA 적용 전략
    is_gemma = "Gemma" in model_config.name
    is_openelm = "OpenELM" in model_config.name
    
    # 모델별 로드 설정
    model_kwargs = {
        "trust_remote_code": model_config.trust_remote_code,
    }
    
    # Gemma 모델 특수 처리
    if is_gemma:
        model_kwargs["device_map"] = {"": device}
        model_kwargs["attn_implementation"] = "eager"
        # Gemma 모델은 8비트로 로드 필요
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model_kwargs["quantization_config"] = quantization_config
    else:
        # 다른 모델들은 모델별 특성에 맞게 처리
        model_kwargs["torch_dtype"] = torch.float32 if is_openelm else torch.float16
    
    # 모델 로드
    try:
        if is_gemma:
            # Gemma는 device_map으로 자동 배치되므로 별도 이동 없음
            model = AutoModelForCausalLM.from_pretrained(
                model_config.model_id,
                **model_kwargs
            )
        else:
            # 다른 모델들은 명시적으로 디바이스 이동
            model = AutoModelForCausalLM.from_pretrained(
                model_config.model_id,
                **model_kwargs
            ).to(device)
        
        # LoRA 설정
        lora_config = LoraConfig(
            r=model_config.lora_r,
            lora_alpha=model_config.lora_alpha,
            target_modules=target_modules,
            lora_dropout=model_config.lora_dropout,
            bias=model_config.lora_bias,
            task_type=TaskType.CAUSAL_LM,
            modules_to_save=None
        )
        
        # LoRA 적용
        model = get_peft_model(model, lora_config)
        
        # 학습 모드 설정
        model.train()
        
        # 학습 가능한 파라미터 확인
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        logging.info(f"모델 파라미터: 총={total_params}, 학습 가능={trainable_params} ({trainable_params/total_params*100:.2f}%)")
        
        # LoRA 어댑터 상태 확인
        if hasattr(model, 'active_adapter'):
            logging.info(f"활성 어댑터: {model.active_adapter}")
            
        return model, tokenizer
        
    except Exception as e:
        logging.error(f"모델 로드 중 오류 발생: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        raise

def load_trained_model_and_tokenizer(model_config: ModelConfig, checkpoint_path: str, device="cuda:3"):
    """학습된 모델과 토크나이저를 로드하는 함수 (평가용)"""
    logging.info(f"Loading trained {model_config.name} from {checkpoint_path}")
    
    # CUDA 디바이스 설정
    if isinstance(device, str):
        device = torch.device(device)
    
    # 토크나이저 로드
    try:
        # 체크포인트에서 토크나이저 로드 시도
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        logging.info(f"Loaded tokenizer from checkpoint: {checkpoint_path}")
    except Exception as e:
        # 체크포인트에서 로드 실패 시 원본 모델 ID에서 로드
        logging.warning(f"Failed to load tokenizer from checkpoint: {str(e)}")
        logging.info(f"Loading tokenizer from original model ID: {model_config.tokenizer_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_config.tokenizer_id, trust_remote_code=model_config.trust_remote_code)
    
    # 토크나이저 설정
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logging.info(f"Tokenizer settings: padding_side={tokenizer.padding_side}, pad_token={tokenizer.pad_token}")
    
    # 모델 로드
    try:
        # 모델별 특성 확인
        is_gemma = "Gemma" in model_config.name
        is_openelm = "OpenELM" in model_config.name
        
        # 모델 로드 설정
        model_kwargs = {
            "trust_remote_code": model_config.trust_remote_code,
        }
        
        # Gemma 모델 특수 처리
        if is_gemma:
            model_kwargs["device_map"] = {"": device}
            model_kwargs["attn_implementation"] = "eager"
            # Gemma 모델은 8비트로 로드 필요
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            model_kwargs["quantization_config"] = quantization_config
        else:
            # 다른 모델들은 모델별 특성에 맞게 처리
            model_kwargs["torch_dtype"] = torch.float32 if is_openelm else torch.float16
        
        # 모델 로드
        if is_gemma:
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                **model_kwargs
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                **model_kwargs
            ).to(device)
        
        # 평가 모드 설정
        model.eval()
                
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        raise
    
    return model, tokenizer

def save_model(model, tokenizer, output_dir: str) -> bool:
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

class EarlyStopping:
    """Early stopping 클래스"""
    
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if score < self.best_score - self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False
