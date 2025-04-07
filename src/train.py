# intent-reconstruction/src/train.py
import os
import torch
import logging
import time
import gc
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from src.data_utils import IntentDataset, collate_fn
from src.model_utils import load_model_and_tokenizer, save_model, clear_gpu_memory
from configs.model_configs import MODEL_CONFIGS_1B, MODEL_CONFIGS_3B, TRAINING_CONFIG
import sys
from typing import Dict, List, Optional

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

# 데이터셋 경로 설정 - 절대/상대 경로 모두 지원하도록 함수화
def get_dataset_path(dataset_name):
    """데이터셋 이름에 해당하는 경로 반환"""
    # 기본 디렉토리 경로 (상대 경로와 절대 경로 모두 처리)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 데이터셋 매핑 - 다양한 경로 패턴 시도
    paths_to_try = [
        # 방법 1: 상대 경로
        os.path.join(base_dir, "data", "BlendX" if "Blend" in dataset_name else "MixX", f"{dataset_name}.json"),
        # 방법 2: 절대 경로
        os.path.join("/home/user/workspace/hyunku/experiment/lm/code/data", 
                     "BlendX" if "Blend" in dataset_name else "MixX", 
                     f"{dataset_name}.json"),
        # 방법 3: 단순 코드 디렉토리 상대 경로
        os.path.join("data", "BlendX" if "Blend" in dataset_name else "MixX", f"{dataset_name}.json"),
    ]
    
    # 모든 경로 시도
    for path in paths_to_try:
        if os.path.exists(path):
            logging.info(f"데이터셋 파일 찾음: {path}")
            return path
    
    # 모든 경로를 시도했지만 파일을 찾지 못한 경우
    raise FileNotFoundError(f"데이터셋 파일을 찾을 수 없습니다: {dataset_name}. 시도한 경로: {paths_to_try}")

def get_collate_fn(pad_token_id):
    """배치 데이터 처리 함수"""
    def collate_fn(batch):
        # None 항목 필터링
        batch = [item for item in batch if item is not None]
        if not batch:
            return {}
        
        # 모든 텐서가 이미 동일한 크기로 패딩되어 있으므로 단순히 스택
        # 텐서 차원을 명시적으로 지정하여 스택
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        
        # 레이블이 있는 경우에만 스택
        if 'labels' in batch[0]:
            labels = torch.stack([item['labels'] for item in batch])
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    
    return collate_fn

# 체크포인트 저장 함수
def save_checkpoint(model, tokenizer, checkpoint_dir, epoch, step=None):
    """모델 체크포인트 저장"""
    if step == "best":
        checkpoint_path = os.path.join(checkpoint_dir, "best_model")
    else:
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch}_step{step}")
    
    os.makedirs(checkpoint_path, exist_ok=True)
    model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)
    logging.info(f"Saved checkpoint to {checkpoint_path}")

def train_model(model_name, model_size, dataset_name=None):
    """
    모델 학습 함수
    
    Args:
        model_name (str): 모델 이름 (예: "Llama-1B")
        model_size (str): 모델 크기 ("1b" 또는 "3b")
        dataset_name (str, optional): 특정 데이터셋 이름. None이면 모든 데이터셋에 대해 학습
    """
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # 모델 설정 선택
    model_configs = MODEL_CONFIGS_1B if model_size == "1b" else MODEL_CONFIGS_3B
    model_config = None
    
    # 모델 이름으로 설정 찾기
    for config in model_configs.values():
        if config.name == model_name:
            model_config = config
            break
    
    if model_config is None:
        raise ValueError(f"모델 설정을 찾을 수 없습니다: {model_name}")
    
    # 데이터셋 선택
    datasets_to_train = []
    if dataset_name:
        # 데이터셋 경로 가져오기
        try:
            data_path = get_dataset_path(dataset_name)
            datasets_to_train = [(dataset_name, data_path)]
        except (ValueError, FileNotFoundError) as e:
            logging.error(str(e))
            return
    else:
        # 모든 데이터셋에 대해 학습
        for name in ["BlendATIS", "BlendSNIPS", "BlendCLINC150", "BlendBanking77", 
                     "MixATIS", "MixSNIPS", "MixCLINC150", "MixBanking77"]:
            try:
                data_path = get_dataset_path(name)
                datasets_to_train.append((name, data_path))
            except (ValueError, FileNotFoundError) as e:
                logging.warning(f"데이터셋 {name} 건너뜀: {str(e)}")
                continue
    
    if not datasets_to_train:
        logging.error("학습할 수 있는 데이터셋이 없습니다.")
        return
    
    # 모델 및 토크나이저 로드
    model, tokenizer = load_model_and_tokenizer(model_config, device)
    
    # 학습 설정
    batch_size = TRAINING_CONFIG["batch_size"]
    gradient_accumulation_steps = TRAINING_CONFIG["gradient_accumulation_steps"]
    learning_rate = TRAINING_CONFIG["learning_rate"]
    epochs = TRAINING_CONFIG["epochs"]
    max_length = TRAINING_CONFIG["max_length"]
    checkpoint_interval = TRAINING_CONFIG["checkpoint_interval"]
    save_best_only = TRAINING_CONFIG["save_best_only"]
    max_checkpoints = TRAINING_CONFIG["max_checkpoints"]
    
    # 체크포인트 디렉토리 설정
    checkpoint_dir = os.path.join("checkpoints", f"{model_name}_{model_size}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 학습 루프
    for dataset_name, data_path in datasets_to_train:
        logging.info(f"데이터셋 {dataset_name} 학습 시작")
        
        # 데이터셋 로드
        train_dataset = IntentDataset(data_path, tokenizer, max_length=max_length, split="train")
        val_dataset = IntentDataset(data_path, tokenizer, max_length=max_length, split="val")
        
        # 데이터로더 생성
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=get_collate_fn(tokenizer.pad_token_id)
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=get_collate_fn(tokenizer.pad_token_id)
        )
        
        # 옵티마이저 설정
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        
        # 학습 루프
        best_val_loss = float('inf')
        last_checkpoint_time = time.time()
        checkpoint_count = 0
        
        for epoch in range(epochs):
            # 학습 모드
            model.train()
            total_train_loss = 0
            train_steps = 0
            
            # 진행 상황 표시
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            
            for batch_idx, batch in enumerate(progress_bar):
                # 배치 데이터를 GPU로 이동
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # 순전파
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / gradient_accumulation_steps
                
                # 역전파
                loss.backward()
                
                # 그라디언트 누적
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                # 손실 누적
                total_train_loss += loss.item() * gradient_accumulation_steps
                train_steps += 1
                
                # 진행 상황 업데이트
                progress_bar.set_postfix({"loss": loss.item() * gradient_accumulation_steps})
            
            # 평균 학습 손실 계산
            avg_train_loss = total_train_loss / train_steps
            logging.info(f"Epoch {epoch+1}/{epochs} - Average training loss: {avg_train_loss:.4f}")
            
            # 검증 모드
            model.eval()
            total_val_loss = 0
            val_steps = 0
            
            # 진행 상황 표시
            progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            
            with torch.no_grad():
                for batch in progress_bar:
                    # 배치 데이터를 GPU로 이동
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    # 순전파
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    
                    # 손실 누적
                    total_val_loss += loss.item()
                    val_steps += 1
                    
                    # 진행 상황 업데이트
                    progress_bar.set_postfix({"loss": loss.item()})
            
            # 평균 검증 손실 계산
            avg_val_loss = total_val_loss / val_steps
            logging.info(f"Epoch {epoch+1}/{epochs} - Average validation loss: {avg_val_loss:.4f}")
            
            # 체크포인트 저장 (시간 기반)
            current_time = time.time()
            if current_time - last_checkpoint_time >= checkpoint_interval:
                save_checkpoint(model, tokenizer, checkpoint_dir, epoch+1, batch_idx)
                last_checkpoint_time = current_time
                checkpoint_count += 1
                
                # 최대 체크포인트 수 제한
                if checkpoint_count >= max_checkpoints:
                    logging.info(f"최대 체크포인트 수({max_checkpoints})에 도달했습니다.")
            
            # 최고 성능 모델 저장
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_checkpoint(model, tokenizer, checkpoint_dir, epoch+1, "best")
                logging.info(f"새로운 최고 성능 모델 저장 (검증 손실: {best_val_loss:.4f})")
        
        # 데이터셋 학습 완료
        logging.info(f"데이터셋 {dataset_name} 학습 완료")
    
    # 학습 완료
    logging.info(f"모델 {model_name} 학습 완료")
    
    # 메모리 정리
    del model
    del tokenizer
    clear_gpu_memory()
    gc.collect()

def train(model_size="1b", model_name=None, dataset_name=None):
    """
    모델 학습 함수
    
    Args:
        model_size (str): 모델 크기 ("1b" 또는 "3b")
        model_name (str, optional): 특정 모델 이름. None이면 모든 모델에 대해 학습
        dataset_name (str, optional): 특정 데이터셋 이름. None이면 모든 데이터셋에 대해 학습
    """
    # 모델 설정 선택
    model_configs = MODEL_CONFIGS_1B if model_size == "1b" else MODEL_CONFIGS_3B
    
    # 모델 이름이 지정된 경우
    if model_name:
        # 모델 이름으로 설정 찾기
        model_config = None
        for config in model_configs.values():
            if config.name == model_name:
                model_config = config
                break
        
        if model_config is None:
            logging.error(f"모델 설정을 찾을 수 없습니다: {model_name}")
            return
        
        # 단일 모델 학습
        train_model(model_name, model_size, dataset_name)
    else:
        # 모든 모델에 대해 학습
        for config in model_configs.values():
            train_model(config.name, model_size, dataset_name)

if __name__ == "__main__":
    # 명령행 인자 파싱
    if len(sys.argv) > 1:
        model_size = sys.argv[1]
        model_name = sys.argv[2] if len(sys.argv) > 2 else None
        dataset_name = sys.argv[3] if len(sys.argv) > 3 else None
        
        train(model_size, model_name, dataset_name)
    else:
        # 기본값으로 학습
        train()
