# intent-reconstruction/src/train_v2.py
import os
import json
import torch
import logging
import argparse
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Dict, Optional, Union, Tuple
import re

# 로컬 모듈 임포트
from src.data_utils import (
    IntentDataset, 
    collate_fn, 
    extract_sentence_from_prompt, 
    normalize_intent_output,
    extract_intent_set
)
from src.model_utils import (
    load_model_and_tokenizer, 
    save_model, 
    clear_gpu_memory,
    EarlyStopping
)
from src.evaluate import evaluate_model
from configs.model_configs import MODEL_CONFIGS_1B, MODEL_CONFIGS_3B, TRAINING_CONFIG

# 로깅 설정
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def get_dataset_path(dataset_name: str) -> str:
    """Get dataset path"""
    # 기본 디렉토리 경로
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 데이터셋 매핑
    paths_to_try = [
        # 상대 경로
        os.path.join(base_dir, "data", "BlendX" if "Blend" in dataset_name else "MixX", f"{dataset_name}.json"),
        # 코드 디렉토리 상대 경로
        os.path.join("data", "BlendX" if "Blend" in dataset_name else "MixX", f"{dataset_name}.json"),
    ]
    
    # 모든 경로 시도
    for path in paths_to_try:
        if os.path.exists(path):
            logging.info(f"Dataset file found: {path}")
            return path
    
    # 파일을 찾지 못한 경우
    raise FileNotFoundError(f"Dataset file not found: {dataset_name}")

def train_model(
    model_config,
    dataset_name: str,
    output_dir: str,
    device: Union[str, torch.device] = "cuda:3",
    batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 1e-4,
    epochs: int = 3,
    checkpoint_interval: int = 1800,  # 체크포인트 저장 간격 (초)
    save_best_only: bool = True,
    max_checkpoints: int = 3,
    eval_during_training: bool = True
) -> Dict:
    """Train a single model"""
    # 시작 시간 기록
    start_time = datetime.now()
    
    try:
        # 데이터셋 경로 가져오기
        data_path = get_dataset_path(dataset_name)
        
        # 모델 및 토크나이저 로드
        logging.info(f"Loading model {model_config.name} (dataset: {dataset_name})")
        model, tokenizer = load_model_and_tokenizer(model_config, device)
        
        # 데이터셋 로드
        train_dataset = IntentDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            max_length=TRAINING_CONFIG["max_length"],
            split="train"
        )
        
        val_dataset = IntentDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            max_length=TRAINING_CONFIG["max_length"],
            split="val"
        )
        
        # 데이터셋 정보 로깅
        logging.info(f"Training dataset size: {len(train_dataset)} samples")
        logging.info(f"Validation dataset size: {len(val_dataset)} samples")
        
        # 데이터 로더 생성
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id)
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id)
        )
        
        # 최적화 도구 설정
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 학습률 스케줄러 설정
        total_steps = len(train_loader) * epochs // gradient_accumulation_steps
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=1e-6
        )
        
        # Early stopping 초기화
        early_stopping = EarlyStopping(patience=5, min_delta=0.01)
        
        # 체크포인트 관리
        checkpoints = []
        best_val_loss = float('inf')
        
        # 모델 저장 경로 수정 - 모델명/데이터셋명 형식으로 저장
        model_base_dir = os.path.join(output_dir, model_config.name)
        os.makedirs(model_base_dir, exist_ok=True)
        
        model_output_dir = os.path.join(model_base_dir, dataset_name)
        os.makedirs(model_output_dir, exist_ok=True)
        
        # 로그 파일 설정
        log_file = os.path.join(model_output_dir, "training_log.txt")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)
        
        # 학습 상태 저장 경로
        training_state_path = os.path.join(model_output_dir, "training_state.json")
        
        # 학습 시작 로깅
        logging.info(f"Starting training for model {model_config.name} - dataset: {dataset_name}")
        logging.info(f"Training settings: batch_size={batch_size}, grad_accum={gradient_accumulation_steps}, lr={learning_rate}, epochs={epochs}")
        
        # 학습 루프
        global_step = 0
        step_loss = 0.0
        last_checkpoint_time = datetime.now()
        
        train_losses = []
        val_losses = []
        
        try:
            for epoch in range(epochs):
                # 에포크 시작 로깅
                logging.info(f"Starting epoch {epoch+1}/{epochs}")
                
                # 학습 모드 설정
                model.train()
                
                # 배치 루프
                epoch_loss = 0.0
                optimizer.zero_grad()
                
                for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
                    try:
                        # 배치 데이터 추출
                        input_ids = batch["input_ids"].to(device)
                        attention_mask = batch["attention_mask"].to(device)
                        labels = batch["labels"].to(device)
                        
                        # 모델 출력 계산
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        
                        # 손실 계산
                        loss = outputs.loss / gradient_accumulation_steps
                        loss.backward()
                        
                        # 누적 손실 업데이트
                        step_loss += loss.item() * gradient_accumulation_steps
                        epoch_loss += loss.item() * gradient_accumulation_steps
                        
                        # 경사 축적 및 최적화
                        if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                            # 경사 클리핑
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            
                            # 최적화 단계
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad()
                            
                            # 체크포인트 저장
                            time_since_last_checkpoint = (datetime.now() - last_checkpoint_time).total_seconds()
                            if time_since_last_checkpoint >= checkpoint_interval:
                                checkpoint_path = os.path.join(model_output_dir, f"checkpoint-{global_step}")
                                save_model(model, tokenizer, checkpoint_path)
                                checkpoints.append(checkpoint_path)
                                last_checkpoint_time = datetime.now()
                                
                                # 체크포인트 수 제한
                                if len(checkpoints) > max_checkpoints:
                                    old_checkpoint = checkpoints.pop(0)
                                    if os.path.exists(old_checkpoint) and old_checkpoint != os.path.join(model_output_dir, "best_model"):
                                        import shutil
                                        shutil.rmtree(old_checkpoint, ignore_errors=True)
                                        logging.info(f"Removed old checkpoint: {old_checkpoint}")
                            
                            # 상태 초기화
                            step_loss = 0.0
                            global_step += 1
                    
                    except Exception as e:
                        logging.error(f"Error processing batch: {str(e)}")
                        continue
                
                # 에포크 손실 계산
                avg_train_loss = epoch_loss / len(train_loader)
                logging.info(f"Epoch {epoch+1}/{epochs} training loss: {avg_train_loss:.5f}")
                
                # 검증 모드로 전환
                model.eval()
                val_loss = 0.0
                
                # 검증 루프
                with torch.no_grad():
                    for val_batch in tqdm(val_loader, desc="Validating"):
                        try:
                            # 배치 데이터 추출
                            input_ids = val_batch["input_ids"].to(device)
                            attention_mask = val_batch["attention_mask"].to(device)
                            labels = val_batch["labels"].to(device)
                            
                            # 모델 출력 계산
                            outputs = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels
                            )
                            
                            # 손실 누적
                            val_loss += outputs.loss.item()
                        
                        except Exception as e:
                            logging.error(f"Error processing validation batch: {str(e)}")
                            continue
                
                # 평균 검증 손실 계산
                if len(val_loader) > 0:
                    avg_val_loss = val_loss / len(val_loader)
                    logging.info(f"Epoch {epoch+1}/{epochs} validation loss: {avg_val_loss:.5f}")
                    
                    # 최고 모델 저장
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        best_model_path = os.path.join(model_output_dir, "best_model")
                        save_model(model, tokenizer, best_model_path)
                        logging.info(f"New best model saved (loss: {best_val_loss:.5f})")
                        
                        # 체크포인트 목록에 추가
                        if best_model_path not in checkpoints:
                            checkpoints.append(best_model_path)
                    
                    # Early stopping 확인
                    if early_stopping(avg_val_loss):
                        logging.info(f"Early stopping activated. No improvement in validation loss for {early_stopping.patience} epochs.")
                        break
                else:
                    logging.warning("Validation loader is empty! Skipping validation...")
                    # 이 경우 검증 없이 최신 모델을 저장
                    best_model_path = os.path.join(model_output_dir, "best_model")
                    save_model(model, tokenizer, best_model_path)
                    logging.info(f"Saved model without validation")
                
                # 학습 상태 저장
                training_state = {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "best_val_loss": best_val_loss,
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "checkpoints": checkpoints,
                    "training_time": str(datetime.now() - start_time)
                }
                
                with open(training_state_path, "w", encoding="utf-8") as f:
                    json.dump(training_state, f, indent=2, ensure_ascii=False)
            
            # 학습 완료 후 최종 모델 저장
            final_model_path = os.path.join(model_output_dir, "final_model")
            save_model(model, tokenizer, final_model_path)
            logging.info(f"Final model saved: {final_model_path}")
            
            # 학습 완료 메트릭 계산
            training_time = datetime.now() - start_time
            logging.info(f"Training completed. Total time: {training_time}")
            
            # 평가 실행 (학습 중 평가 옵션이 켜진 경우)
            if eval_during_training:
                logging.info("Skipping automatic evaluation. Please run evaluation separately.")
                # 자동 평가 부분 주석 처리
                # eval_output_dir = os.path.join("outputs")
                # os.makedirs(eval_output_dir, exist_ok=True)
                
                # logging.info(f"Evaluating best model... (dataset: {dataset_name})")
                
                # # 메모리 정리
                # del model
                # clear_gpu_memory()
                
                # # 최고 모델 평가
                # eval_results = evaluate_model(
                #     model_config,
                #     dataset_name,
                #     eval_output_dir,
                #     device,
                #     batch_size=8
                # )
                
                # # 평가 결과 로깅
                # logging.info(f"Evaluation results: accuracy={eval_results.get('accuracy', 0):.4f}")
            
            # 결과 반환
            return {
                "model_name": model_config.name,
                "dataset_name": dataset_name,
                "best_val_loss": best_val_loss,
                "training_time": str(training_time),
                "epochs_completed": epoch + 1,
                "model_path": best_model_path
            }
        
        except KeyboardInterrupt:
            logging.info("Training manually interrupted. Saving current state...")
            interrupt_model_path = os.path.join(model_output_dir, "interrupted_model")
            save_model(model, tokenizer, interrupt_model_path)
            logging.info(f"Interrupted model saved: {interrupt_model_path}")
            
            return {
                "model_name": model_config.name,
                "dataset_name": dataset_name,
                "status": "interrupted",
                "best_val_loss": best_val_loss,
                "training_time": str(datetime.now() - start_time),
                "epochs_completed": epoch + 1,
                "model_path": interrupt_model_path
            }
    
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        
        return {
            "model_name": model_config.name,
            "dataset_name": dataset_name,
            "status": "error",
            "error": str(e),
            "training_time": str(datetime.now() - start_time)
        }
    
    finally:
        # 모든 경우에 메모리 정리
        try:
            clear_gpu_memory()
        except:
            pass

def train_models(
    model_configs: Dict,
    dataset_names: List[str],
    output_dir: str,
    model_size: str = None,
    model_name: str = None,
    dataset_name: str = None,
    device: str = "cuda:3",
    batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 1e-4,
    epochs: int = 3,
    checkpoint_interval: int = 1800,
    save_best_only: bool = True,
    max_checkpoints: int = 3,
    eval_during_training: bool = True
) -> Dict:
    """Train multiple models"""
    # 학습 시작 시간
    start_time = datetime.now()
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 학습할 모델 선택
    models_to_train = {}
    if model_name:
        for model_id, config in model_configs.items():
            if config.name == model_name:
                models_to_train[model_id] = config
    else:
        models_to_train = model_configs
    
    if not models_to_train:
        logging.error(f"No models found to train: {model_name}")
        return {}
    
    # 학습할 데이터셋 선택
    datasets_to_train = [dataset_name] if dataset_name else dataset_names
    
    # 결과 저장용 딕셔너리
    training_results = {}
    
    # 각 모델과 데이터셋 조합에 대해 학습
    for model_id, model_config in models_to_train.items():
        model_results = {}
        for ds_name in datasets_to_train:
            try:
                # 학습 상태 로깅
                logging.info(f"Starting training for model {model_config.name}, dataset {ds_name}")
                
                # 모델 학습
                result = train_model(
                    model_config,
                    ds_name,
                    output_dir,
                    device,
                    batch_size=batch_size,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    learning_rate=learning_rate,
                    epochs=epochs,
                    checkpoint_interval=checkpoint_interval,
                    save_best_only=save_best_only,
                    max_checkpoints=max_checkpoints,
                    eval_during_training=eval_during_training
                )
                
                # 결과 저장
                model_results[ds_name] = result
                
                # 다음 학습 전에 메모리 정리
                clear_gpu_memory()
                
            except Exception as e:
                logging.error(f"Error training model {model_config.name}, dataset {ds_name}: {str(e)}")
                model_results[ds_name] = {"status": "error", "error": str(e)}
                continue
        
        # 모델별 결과 저장
        if model_results:
            training_results[model_config.name] = model_results
    
    # 총 소요 시간 계산
    total_time = datetime.now() - start_time
    
    return training_results

def train(model_size=None, model_name=None, dataset_name=None):
    """Run model training"""
    if model_size not in ["1b", "3b"]:
        raise ValueError("model_size must be '1b' or '3b'")
    
    # 장치 설정
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # 출력 디렉토리 설정
    output_dir = os.path.join("models")
    os.makedirs(output_dir, exist_ok=True)
    
    # 지원하는 모든 데이터셋 목록
    all_datasets = [
        "BlendATIS", "BlendSNIPS", "BlendCLINC150", "BlendBanking77",
        "MixATIS", "MixSNIPS", "MixCLINC150", "MixBanking77"
    ]
    
    # 모델 설정 선택
    model_configs = MODEL_CONFIGS_1B if model_size == "1b" else MODEL_CONFIGS_3B
    
    # 훈련 설정 가져오기
    training_params = {
        "batch_size": TRAINING_CONFIG["batch_size"],
        "gradient_accumulation_steps": TRAINING_CONFIG["gradient_accumulation_steps"],
        "learning_rate": TRAINING_CONFIG["learning_rate"],
        "epochs": TRAINING_CONFIG["epochs"],
        "checkpoint_interval": TRAINING_CONFIG["checkpoint_interval"],
        "save_best_only": TRAINING_CONFIG["save_best_only"],
        "max_checkpoints": TRAINING_CONFIG["max_checkpoints"],
    }
    
    # 훈련 실행
    results = train_models(
        model_configs, 
        all_datasets, 
        output_dir, 
        model_size=model_size,
        model_name=model_name,
        dataset_name=dataset_name,
        device=device,
        **training_params
    )
    
    return results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Intent Detection Model Training")
    parser.add_argument("--model-size", choices=["1b", "3b"], required=True, help="Model size (1b or 3b)")
    parser.add_argument("--model", type=str, help="Specific model to train (e.g., Llama-1B)")
    parser.add_argument("--dataset", type=str, help="Specific dataset to train on (e.g., MixATIS)")
    parser.add_argument("--gpu", type=int, default=3, help="GPU ID to use (default: 3)")
    parser.add_argument("--batch-size", type=int, help="Training batch size (default: from config file)")
    parser.add_argument("--epochs", type=int, help="Number of epochs (default: from config file)")
    parser.add_argument("--lr", type=float, help="Learning rate (default: from config file)")
    
    args = parser.parse_args()
    
    # 학습 실행
    train(model_size=args.model_size, model_name=args.model, dataset_name=args.dataset)

if __name__ == "__main__":
    main()