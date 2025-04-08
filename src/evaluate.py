import os
import json
import torch
import logging
import pandas as pd
import argparse
import re
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Dict, Optional, Union, Set, Tuple

# 로컬 모듈 임포트
from src.data_utils import (
    IntentDataset, 
    collate_fn, 
    extract_sentence_from_prompt, 
    normalize_intent_output,
    extract_intent_set
)
from src.model_utils import (
    load_trained_model_and_tokenizer,
    clear_gpu_memory
)
from configs.model_configs import MODEL_CONFIGS_1B, MODEL_CONFIGS_3B, EVALUATION_CONFIG

# 로깅 설정
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)

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

def generate_model_output(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    device: Union[str, torch.device] = "cuda:3"
) -> str:
    """Generate model output"""
    try:
        # 입력을 GPU로 이동
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # 생성 파라미터 설정
        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": 50,
            "min_new_tokens": 1,
            "do_sample": True,
            "num_beams": 2,
            "temperature": 0.1,
            "top_p": 0.95,
            "repetition_penalty": 1.5,
            "no_repeat_ngram_size": 2,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": True,
            "early_stopping": True
        }
        
        # 모델 생성
        with torch.no_grad():
            outputs = model.generate(**gen_kwargs)
        
        # 생성된 텍스트 디코딩
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 디버깅을 위한 로깅
        input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        new_tokens = len(outputs[0]) - len(input_ids[0])
        logging.debug(f"Input text: {input_text[:100]}...")
        logging.debug(f"Generated text: {generated_text[:100]}...")
        logging.debug(f"New tokens generated: {new_tokens}")
        
        # 입력 부분을 제외한 새로 생성된 부분만 추출
        result = generated_text[len(input_text):].strip()
        
        return result
        
    except Exception as e:
        logging.error(f"Error generating model output: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return ""

def process_model_output(generated_text: str) -> str:
    """Process generated output"""
    if not generated_text or not isinstance(generated_text, str):
        return ""
    
    # 원본 출력 로깅 (디버깅용)
    logging.debug(f"Raw model output: {generated_text[:100]}...")
    
    # 괄호 형식 찾기
    pattern = r"\([^)]+\)"
    matches = re.findall(pattern, generated_text)
    if matches:
        # 괄호 형식이 여러 개 있는 경우 첫 번째 것만 사용
        return matches[0]
    
    # 괄호 없이 atis_ 형식이 있으면 괄호로 감싸기
    atis_pattern = r"atis_\w+"
    atis_matches = re.findall(atis_pattern, generated_text)
    if atis_matches:
        return f"({', '.join(atis_matches)})"
    
    # 마지막 시도: "인텐트는 ~ 입니다" 패턴 찾기
    intent_pattern = r"인텐트는\s+([^.]+)입니다"
    intent_match = re.search(intent_pattern, generated_text, re.IGNORECASE)
    if intent_match:
        intent_text = intent_match.group(1).strip()
        if "," in intent_text:
            return f"({intent_text})"
        else:
            return f"({intent_text})"
    
    return generated_text.strip()

def evaluate_model(
    model_config,
    dataset_name: str,
    output_dir: str,
    device: Union[str, torch.device] = "cuda:3",
    batch_size: int = 4
) -> Dict:
    """Evaluate a single model"""
    # 모델 ID 설정 - 실제 Hugging Face 모델 ID 사용
    model_id = model_config.model_id
    
    # 데이터셋 경로
    data_path = get_dataset_path(dataset_name)
    
    try:
        # 모델 및 토크나이저 로드
        logging.info(f"Loading model {model_config.name} (dataset: {dataset_name})")
        model, tokenizer = load_trained_model_and_tokenizer(model_config, model_id, device)
        
        # 모델을 평가 모드로 설정
        model.eval()
        
        # 데이터셋 로드
        test_dataset = IntentDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            max_length=EVALUATION_CONFIG["max_length"],
            split="test"
        )
        
        # 데이터 로더 생성
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id)
        )
        
        # 결과 저장용 변수
        all_predictions = []
        all_ground_truths = []
        all_sentences = []
        
        # 평가 루프
        logging.info(f"Evaluating model {model_config.name} on dataset {dataset_name}...")
        
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            if not batch:
                continue
                
            # 배치 데이터를 GPU로 이동
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            
            # 원본 문장 및 의도 추출
            raw_intents = batch.get("raw_intents", [])
            
            # 배치 내 각 항목에 대해 추론
            for i in range(len(input_ids)):
                # 모델 출력 생성
                generated_text = generate_model_output(
                    model, 
                    tokenizer, 
                    input_ids[i:i+1], 
                    attention_mask[i:i+1], 
                    device
                )
                
                # 원본 문장 추출
                sentence = tokenizer.decode(input_ids[i], skip_special_tokens=True)
                extracted_sentence = extract_sentence_from_prompt(sentence)
                
                # 생성된 텍스트에서 의도 추출
                predicted_intent = process_model_output(generated_text)
                
                # 정답 의도 가져오기
                if i < len(raw_intents):
                    true_intent = f"({', '.join(raw_intents[i])})"
                else:
                    true_intent = "(unknown)"
                
                # 디버깅을 위한 로깅 (매 20번째 샘플마다)
                if batch_idx == 0 and i < 5:  # 첫 배치의 처음 5개 샘플만 로깅
                    logging.info(f"샘플 #{i+1}:")
                    logging.info(f"  원본 프롬프트: {sentence[:200]}...")
                    logging.info(f"  추출된 문장: {extracted_sentence}")
                    logging.info(f"  모델 출력 (원본): {generated_text[:200]}...")
                    logging.info(f"  가공된 모델 출력: {predicted_intent}")
                    logging.info(f"  실제 인텐트: {true_intent}")
                    logging.info("-" * 50)
                
                # 결과 저장
                all_predictions.append(predicted_intent)
                all_ground_truths.append(true_intent)
                all_sentences.append(extracted_sentence)
                
            # 배치마다 출력되는 로그 제거
        
        # 정확도 계산
        correct = sum(1 for p, g in zip(all_predictions, all_ground_truths) if p == g)
        total = len(all_predictions)
        accuracy = correct / total if total > 0 else 0
        
        # 의도 개수별 통계
        intent_counts = {}
        for gt in all_ground_truths:
            intents = extract_intent_set(gt)
            count = len(intents)
            if count not in intent_counts:
                intent_counts[count] = {"total": 0, "correct": 0}
            intent_counts[count]["total"] += 1
        
        # 의도 개수별 정확도 계산
        for i, (pred, gt) in enumerate(zip(all_predictions, all_ground_truths)):
            pred_intents = extract_intent_set(pred)
            gt_intents = extract_intent_set(gt)
            
            # 정확히 일치하는 경우
            if pred_intents == gt_intents:
                count = len(gt_intents)
                if count in intent_counts:
                    intent_counts[count]["correct"] += 1
        
        # 결과 로깅 추가
        logging.info(f"=== 평가 결과: {model_config.name} - {dataset_name} ===")
        logging.info(f"종합 정확도: {accuracy:.4f} ({correct}/{total})")
        
        # 인텐트 개수별 정확도 로깅
        logging.info("인텐트 개수별 정확도:")
        for count, stats in sorted(intent_counts.items()):
            count_accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            logging.info(f"  {count}개 인텐트: {count_accuracy:.4f} ({stats['correct']}/{stats['total']})")
        
        # 결과 저장
        results = {
            "model": model_config.name,
            "dataset": dataset_name,
            "accuracy": accuracy,
            "total_samples": total,
            "correct_predictions": correct,
            "intent_counts": intent_counts
        }
        
        # 결과를 CSV 파일로 저장
        output_path = os.path.join(output_dir, model_config.name, dataset_name)
        os.makedirs(output_path, exist_ok=True)
        
        # 예측 결과 저장
        df = pd.DataFrame({
            "sentence": all_sentences,
            "prediction": all_predictions,
            "ground_truth": all_ground_truths,
            "correct": [p == g for p, g in zip(all_predictions, all_ground_truths)]
        })
        
        csv_path = os.path.join(output_path, "results.csv")
        df.to_csv(csv_path, index=False)
        logging.info(f"Results saved to {csv_path}")
        
        return results
        
    except Exception as e:
        logging.error(f"Error evaluating model: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return {}

def evaluate_models(
    model_names: Optional[List[str]] = None,
    dataset_names: Optional[List[str]] = None,
    output_dir: str = "outputs",
    device: Union[str, torch.device] = "cuda:3"
) -> Dict:
    """Evaluate multiple models on specified datasets"""
    # 모델 및 데이터셋 목록 설정
    if model_names is None:
        model_names = []
        for config in [MODEL_CONFIGS_1B, MODEL_CONFIGS_3B]:
            model_names.extend([config[model_id].name for model_id in config])
    
    if dataset_names is None:
        dataset_names = [
            "BlendATIS", "BlendSNIPS", "BlendCLINC150", "BlendBanking77",
            "MixATIS", "MixSNIPS", "MixCLINC150", "MixBanking77"
        ]
    
    # 결과 저장
    all_results = {}
    
    # 각 모델 및 데이터셋에 대해 평가 수행
    for model_name in model_names:
        model_results = {}
        
        # 모델 설정 찾기
        model_config = None
        for config in [MODEL_CONFIGS_1B, MODEL_CONFIGS_3B]:
            for model_id, cfg in config.items():
                if cfg.name == model_name:
                    model_config = cfg
                    break
            if model_config:
                break
        
        if not model_config:
            logging.warning(f"Model configuration not found for {model_name}")
            continue
        
        for dataset_name in dataset_names:
            try:
                logging.info(f"Evaluating {model_name} on {dataset_name}")
                result = evaluate_model(model_config, dataset_name, output_dir, device)
                model_results[dataset_name] = result
            except Exception as e:
                logging.error(f"Error evaluating {model_name} on {dataset_name}: {str(e)}")
                model_results[dataset_name] = {}
        
        all_results[model_name] = model_results
    
    # 전체 평가 결과 요약 로깅 추가
    logging.info("\n=== 전체 평가 결과 요약 ===")
    for model_name, model_results in all_results.items():
        logging.info(f"\n{model_name}:")
        for dataset_name, result in model_results.items():
            if result and "accuracy" in result:
                logging.info(f"  {dataset_name}: {result['accuracy']:.4f}")
            else:
                logging.info(f"  {dataset_name}: 평가 실패")
    
    return all_results

def evaluate(
    model_size: str = None,
    model_name: str = None,
    dataset_name: str = None,
    device: str = "cuda:3",
    output_dir: str = "outputs"
) -> Dict:
    """Run model evaluation"""
    if model_size not in ["1b", "3b"]:
        raise ValueError("model_size must be '1b' or '3b'")
    
    # 장치 설정
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # 출력 디렉토리 설정
    os.makedirs(output_dir, exist_ok=True)
    
    # 지원하는 모든 데이터셋 목록
    all_datasets = [
        "BlendATIS", "BlendSNIPS", "BlendCLINC150", "BlendBanking77",
        "MixATIS", "MixSNIPS", "MixCLINC150", "MixBanking77"
    ]
    
    # 모델 설정 선택
    model_configs = MODEL_CONFIGS_1B if model_size == "1b" else MODEL_CONFIGS_3B
    
    # 모델 및 데이터셋 목록 설정
    model_names = [model_name] if model_name else None
    dataset_names = [dataset_name] if dataset_name else all_datasets
    
    # 평가 실행
    results = evaluate_models(
        model_names=model_names,
        dataset_names=dataset_names,
        output_dir=output_dir,
        device=device
    )
    
    return results

def main():
    """Main function for evaluation"""
    parser = argparse.ArgumentParser(description="Evaluate models on intent datasets")
    parser.add_argument("--model-size", choices=["1b", "3b"], required=True, help="Model size (1b or 3b)")
    parser.add_argument("--model-name", type=str, help="Specific model to evaluate")
    parser.add_argument("--dataset-name", type=str, help="Specific dataset to evaluate on")
    parser.add_argument("--device", type=str, default="cuda:3", help="Device to use for evaluation")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to save results")
    
    args = parser.parse_args()
    
    # 평가 실행
    results = evaluate(
        model_size=args.model_size,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        device=args.device,
        output_dir=args.output_dir
    )
    
    logging.info("Evaluation completed")

if __name__ == "__main__":
    main()