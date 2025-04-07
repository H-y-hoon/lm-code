# intent-reconstruction/src/evaluate.py
import os
import json
import torch
import logging
import pandas as pd
import re
from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data_utils import IntentDataset, collate_fn
from src.model_utils import load_trained_model_and_tokenizer, clear_gpu_memory
from configs.model_configs import MODEL_CONFIGS_1B, MODEL_CONFIGS_3B, EVALUATION_CONFIG
import sys
import gc
from typing import Optional
import argparse

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

# 데이터셋 경로 설정 함수 추가
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

def normalize_intent_output(text):
    """의도 출력을 정규화하는 함수"""
    # 빈 텍스트 처리
    if not text or text.strip() == "":
        logging.warning(f"빈 텍스트가 입력되었습니다.")
        return "(atis_flight)"  # 기본 의도 반환
        
    # 불필요한 텍스트 제거
    text = re.sub(r'\[INST\].*?Sentence:.*?Intent:', '', text, flags=re.DOTALL)
    text = re.sub(r'\[/INST\]', '', text)
    text = re.sub(r'Answer:.*?(\(|$)', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'Note that.*?(\(|$)', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'Explanation:.*?(\(|$)', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'Intents:.*?(\(|$)', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\n.*?(\(|$)', r'\1', text, flags=re.DOTALL)
    
    # 디버깅을 위한 로깅 추가
    logging.debug(f"Before normalize text: {text}")
    
    # 다양한 형식의 의도 추출
    intents = []
    
    # 1. 괄호 안의 콤마로 구분된 의도 형식 처리 (atis_flight, atis_airfare)
    parenthesis_match = re.search(r'\((.*?)\)', text)
    if parenthesis_match:
        # 괄호 안의 내용에서 쉼표로 구분된 의도 추출
        content = parenthesis_match.group(1)
        intents = [intent.strip() for intent in content.split(',')]
        logging.debug(f"괄호에서 추출한 의도: {intents}")
    
    # 2. "#" 구분자로 연결된 의도 형식 처리 (atis_flight#atis_airfare)
    elif "#" in text:
        hash_intents = re.findall(r'atis_[a-z_#]+', text)
        for hash_intent in hash_intents:
            if "#" in hash_intent:
                intents.extend([i.strip() for i in hash_intent.split("#")])
            else:
                intents.append(hash_intent.strip())
    
    # 3. 개별 atis_ 접두어가 있는 의도 추출
    elif not intents:
        atis_matches = re.findall(r'atis_\w+', text)
        if atis_matches:
            intents = atis_matches
    
    # 4. 의도가 추출되지 않은 경우 키워드 매핑 시도
    if not intents:
        intent_keywords = {
            "flight": "atis_flight",
            "time": "atis_flight_time", 
            "fare": "atis_airfare",
            "aircraft": "atis_aircraft",
            "ground service": "atis_ground_service",
            "ground transportation": "atis_ground_service",
            "airport": "atis_airport",
            "airline": "atis_airline",
            "distance": "atis_distance",
            "abbreviation": "atis_abbreviation",
            "ground fare": "atis_ground_fare",
            "quantity": "atis_quantity",
            "city": "atis_city",
            "flight number": "atis_flight_no",
            "flight no": "atis_flight_no",
            "capacity": "atis_capacity",
            "seat": "atis_capacity",
            "meal": "atis_meal",
            "restriction": "atis_restriction",
            "cheapest": "atis_cheapest",
            "least expensive": "atis_cheapest"
        }
        
        # 텍스트 내 키워드 찾기
        text_lower = text.lower()
        for keyword, intent in intent_keywords.items():
            if keyword in text_lower:
                intents.append(intent)
    
    # 유효한 의도만 포함 및 중복 제거
    valid_intents = []
    seen = set()
    
    # 유효한 의도 목록
    valid_intent_types = [
        "atis_flight", "atis_flight_time", "atis_airfare", "atis_aircraft",
        "atis_ground_service", "atis_airport", "atis_airline", "atis_distance",
        "atis_abbreviation", "atis_ground_fare", "atis_quantity", "atis_city",
        "atis_flight_no", "atis_capacity", "atis_meal", "atis_restriction", "atis_cheapest"
    ]
    
    for intent in intents:
        intent = intent.strip()
        
        # atis_ 접두어가 없는 경우 추가
        if not intent.startswith('atis_') and intent in ["flight", "airfare", "airport", "airline", 
                                                    "flight_time", "aircraft", "ground_service"]:
            intent = f"atis_{intent}"
        
        # 비정상적인 의도 형식 수정 (예: atis_intent_atiti_flights -> atis_flight)
        if intent.startswith('atis_intent_') or 'atiti' in intent or 'atcis' in intent:
            if 'flight' in intent.lower():
                intent = "atis_flight"
            elif 'air' in intent.lower():
                intent = "atis_airfare"
        
        # 유효한 의도만 포함
        if intent in valid_intent_types and intent not in seen:
            valid_intents.append(intent)
            seen.add(intent)
    
   # 결과가 없으면 문맥에 따른 기본값 추론
    if not valid_intents:
        logging.warning(f"의도를 추출할 수 없습니다. 원본 텍스트: {text}")
        valid_intents.append("atis_flight")  # 가장 일반적인 의도를 기본값으로 사용
    
    # 정규화된 형식으로 반환
    return f"({', '.join(valid_intents)})"

def count_intents(intent_str):
    """의도 레이블 문자열에서 의도 개수를 계산"""
    # 괄호 제거 후 쉼표로 분리
    intents = intent_str.strip('()').split(',')
    return len([i.strip() for i in intents if i.strip()])

def generate_model_output(model, tokenizer, input_ids, attention_mask, device):
    """모델 출력 생성 함수"""
    try:
        # 입력을 GPU로 이동
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # 생성 파라미터 설정
        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": 128,           # 더 많은 새 토큰 생성
            "min_new_tokens": 10,            # 최소 새 토큰 수 설정
            "do_sample": True,               # 샘플링 활성화
            "num_beams": 5,                  # 빔 서치 사용
            "temperature": 0.7,              # 적절한 온도 설정
            "top_p": 0.9,                    # nucleus sampling
            "repetition_penalty": 1.2,       # 반복 패널티
            "no_repeat_ngram_size": 3,       # n-gram 반복 방지
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": True,
            "early_stopping": True           # 조기 종료 활성화
        }
        
        # 모델 생성
        with torch.no_grad():
            outputs = model.generate(**gen_kwargs)
        
        # 생성된 텍스트 디코딩
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 디버깅을 위한 로깅
        input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        new_tokens = len(outputs[0]) - len(input_ids[0])
        logging.info(f"입력 텍스트: {input_text[:100]}...")
        logging.info(f"생성된 텍스트: {generated_text[:100]}...")
        logging.info(f"새로 생성된 토큰 수: {new_tokens}")
        
        return generated_text
        
    except Exception as e:
        logging.error(f"모델 출력 생성 중 오류 발생: {str(e)}")
        return None

def process_model_output(generated_text, input_text, tokenizer, sample_idx):
    """모델 출력을 처리하는 함수"""
    try:
        # 디버깅 로그
        if sample_idx < 5:  # 처음 5개 샘플만 상세 로깅
            logging.info(f"샘플 {sample_idx} 생성된 전체 텍스트: {generated_text[:100]}...")
            logging.info(f"샘플 {sample_idx} 입력 텍스트: {input_text[:100]}...")
        
        # 새로운 마커에 따른 결과 추출
        predicted_text = ""
        
        # 방법 1: "The intents are:" 이후 텍스트 추출
        if "The intents are:" in generated_text:
            parts = generated_text.split("The intents are:", 1)
            if len(parts) > 1:
                predicted_text = parts[1].strip()
                logging.info(f"마커 'The intents are:' 이후 텍스트 추출: {predicted_text[:30]}...")
        
        # 방법 2: "Intents:" 이후 텍스트 추출
        if not predicted_text and "Intents:" in generated_text:
            parts = generated_text.split("Intents:", 1)
            if len(parts) > 1:
                predicted_text = parts[1].strip()
                logging.info(f"마커 'Intents:' 이후 텍스트 추출: {predicted_text[:30]}...")
        
        # 방법 3: 입력 이후 텍스트 추출 (백업)
        if not predicted_text and input_text in generated_text and len(generated_text) > len(input_text):
            predicted_text = generated_text[len(input_text):].strip()
            logging.info(f"입력 이후 텍스트 추출: {predicted_text[:30]}...")
            
        # 방법 4: 입력 마지막 줄 이후 텍스트 추출 (백업)
        if not predicted_text and "\n\n" in generated_text:
            last_lines = generated_text.split("\n\n")
            if len(last_lines) > 1:
                predicted_text = last_lines[-1].strip()
                logging.info(f"마지막 줄 텍스트 추출: {predicted_text[:30]}...")
                
        # 방법 5: 입력/출력 동일 시 처리 (마지막 방어선)
        if not predicted_text or input_text.strip() == generated_text.strip():
            logging.warning(f"샘플 {sample_idx}: 모델이 의미 있는 출력을 생성하지 못했습니다.")
            # 기본 의도 추가 (강제 응답)
            predicted_text = "(atis_flight)"
            logging.info(f"기본 의도로 대체: {predicted_text}")
        
        # 괄호 유무 확인 및 포맷 수정
        if not "(" in predicted_text and not ")" in predicted_text:
            # 괄호가 없는 경우 유효한 의도 텍스트만 있으면 괄호로 감싸기
            atis_matches = re.findall(r'atis_\w+', predicted_text)
            if atis_matches:
                predicted_text = f"({', '.join(atis_matches)})"
                logging.info(f"괄호 추가: {predicted_text}")
        
        # 의도 정규화
        predicted_text = normalize_intent_output(predicted_text)
        
        return predicted_text
    except Exception as e:
        logging.error(f"출력 처리 중 오류 발생 (샘플 {sample_idx}): {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return "(atis_flight)"  # 오류 발생 시 기본 의도 반환

def evaluate_model(model_config, device, test_path, output_dir, dataset_name):
    """단일 모델 평가 함수"""
    try:
        # 최적 체크포인트 경로 찾기
        model_base_dir = os.path.join("outputs", f"{model_config.name}_{dataset_name}")
        best_checkpoint_dir = os.path.join(model_base_dir, "best_model")
        
        # best_model 디렉토리가 없으면 다른 체크포인트 찾기
        if not os.path.exists(best_checkpoint_dir):
            checkpoints = [item for item in os.listdir(model_base_dir) 
                          if item.startswith("checkpoint_epoch") and 
                          os.path.isdir(os.path.join(model_base_dir, item))]
            
            if not checkpoints:
                logging.error(f"No checkpoints found for model {model_config.name} in {model_base_dir}")
                return
                
            # 가장 최근 체크포인트 선택
            best_checkpoint_dir = os.path.join(model_base_dir, checkpoints[-1])
        
        # 학습된 모델과 토크나이저 로드
        logging.info(f"Loading trained model from {best_checkpoint_dir}")
        model, tokenizer = load_trained_model_and_tokenizer(model_config, best_checkpoint_dir, device)
        
        # 모델을 평가 모드로 설정
        model.eval()
        
        # 테스트 데이터셋 로드
        test_dataset = IntentDataset(test_path, tokenizer, max_length=EVALUATION_CONFIG["max_length"], split="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=EVALUATION_CONFIG["batch_size"],
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id)
        )
        
        # 결과 저장을 위한 변수 초기화
        all_predictions = []
        all_labels = []
        
        # 평가 루프
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if not batch:  # 빈 배치 처리
                    continue
                    
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                # 모델 출력 생성
                generated_texts = []
                input_texts = []
                for i in range(len(input_ids)):
                    # 입력 텍스트 디코딩
                    input_text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
                    input_texts.append(input_text)
                    
                    # 모델 출력 생성
                    text = generate_model_output(model, tokenizer, input_ids[i:i+1], attention_mask[i:i+1], device)
                    if text is None:
                        logging.warning(f"배치 {batch_idx}, 샘플 {i}에서 생성 실패")
                        continue
                    generated_texts.append(text)
                
                # 생성된 텍스트에서 의도 추출
                for i, (text, input_text) in enumerate(zip(generated_texts, input_texts)):
                    intent = process_model_output(text, input_text, tokenizer, batch_idx * test_loader.batch_size + i)
                    if intent:
                        all_predictions.append(intent)
                    else:
                        all_predictions.append("unknown")
                
                # 레이블 처리
                for label in labels:
                    label_text = tokenizer.decode(label, skip_special_tokens=True)
                    all_labels.append(label_text)
                
                if batch_idx % 10 == 0:
                    logging.info(f"배치 {batch_idx} 처리 완료")
        
        # 정확도 계산
        correct = sum(1 for p, l in zip(all_predictions, all_labels) if p == l)
        accuracy = correct / len(all_predictions) if all_predictions else 0
        
        logging.info(f"모델 {model_config.name}의 {dataset_name} 데이터셋 평가 결과:")
        logging.info(f"전체 샘플 수: {len(all_predictions)}")
        logging.info(f"정확한 예측 수: {correct}")
        logging.info(f"정확도: {accuracy:.4f}")
        
        # 결과 저장
        results = {
            "accuracy": accuracy,
            "total_samples": len(all_predictions),
            "correct_samples": correct
        }
        
        # CSV 파일로 결과 저장
        results_df = pd.DataFrame({
            "prediction": all_predictions,
            "label": all_labels,
            "correct": [p == l for p, l in zip(all_predictions, all_labels)]
        })
        csv_path = os.path.join(output_dir, f"{model_config.name}_{dataset_name}_results.csv")
        results_df.to_csv(csv_path, index=False)
        logging.info(f"결과가 CSV 파일로 저장되었습니다: {csv_path}")
        
        return results
        
    except Exception as e:
        logging.error(f"모델 평가 중 오류 발생: {str(e)}")
        return {"accuracy": 0.0, "total_samples": 0, "correct_samples": 0}

def evaluate_models_sequentially(model_configs: dict, device: str, dataset_names: list, output_dir: str, model_name: Optional[str] = None, dataset_name: Optional[str] = None):
    """모델들을 순차적으로 평가"""
    # 모델 설정 필터링
    models_to_evaluate = {}
    if model_name:
        for model_id, config in model_configs.items():
            if config.name == model_name:
                models_to_evaluate[model_id] = config
    else:
        models_to_evaluate = model_configs
    
    if not models_to_evaluate:
        logging.error(f"No matching models found to evaluate")
        return
    
    # 데이터셋 필터링
    datasets_to_evaluate = [dataset_name] if dataset_name else dataset_names
    
    # 각 모델과 데이터셋 조합에 대해 평가
    results = {}
    for model_id, model_config in models_to_evaluate.items():
        model_results = {}
        for ds_name in datasets_to_evaluate:
            try:
                test_path = get_dataset_path(ds_name)
                result = evaluate_model(model_config, device, test_path, output_dir, ds_name)
                if result:
                    model_results[ds_name] = result
            except Exception as e:
                logging.error(f"Error evaluating {model_config.name} on {ds_name}: {str(e)}")
                continue
        
        if model_results:
            results[model_config.name] = model_results
    
    # 종합 결과 저장
    if results:
        summary_path = os.path.join(output_dir, "evaluation_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logging.info(f"Evaluation summary saved to {summary_path}")
    
    return results

def evaluate(model_name=None, dataset_name=None):
    """모델 평가 실행 함수"""
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # 출력 디렉토리 설정
    output_dir = os.path.join("outputs", "evaluation")
    os.makedirs(output_dir, exist_ok=True)
    
    # 지원하는 모든 데이터셋 목록
    all_datasets = [
        "BlendATIS", "BlendSNIPS", "BlendCLINC150", "BlendBanking77",
        "MixATIS", "MixSNIPS", "MixCLINC150", "MixBanking77"
    ]
    
    # 모델 설정 - 1B와 3B 모두 평가
    all_model_configs = {**MODEL_CONFIGS_1B, **MODEL_CONFIGS_3B}
    
    # 평가 실행
    results = evaluate_models_sequentially(
        all_model_configs, 
        device, 
        all_datasets, 
        output_dir, 
        model_name,
        dataset_name
    )
    
    # 결과 요약 출력
    if results:
        logging.info("\n===== EVALUATION SUMMARY =====")
        for model, model_results in results.items():
            logging.info(f"\n===== 모델: {model} =====")
            for ds, metrics in model_results.items():
                correct = metrics.get('correct_samples', 0)
                total = metrics.get('total_samples', 0)
                accuracy = metrics['accuracy']
                intent_stats = metrics.get('intent_stats', {})
                
                logging.info(f"\n데이터셋: {ds}")
                logging.info(f"전체 정확도: {accuracy:.4f} ({correct}/{total})")
                
                # 의도 개수별 정확도 테이블 형식으로 출력
                logging.info("\n< 의도 개수별 정확도 >")
                logging.info("----------------------------------")
                logging.info("의도 개수 | 정확도 | 정답/전체")
                logging.info("----------------------------------")
                
                intent_accuracy = metrics.get('intent_accuracy', {})
                
                # 정렬된 의도 개수별 출력
                for count in sorted([int(c) for c in intent_accuracy.keys() if c and c.isdigit()], key=int):
                    count_str = str(count)
                    acc = intent_accuracy[count_str]
                    stats = intent_stats.get(count_str, {'correct': 0, 'total': 0})
                    correct_count = stats.get('correct', 0)
                    total_count = stats.get('total', 0)
                    logging.info(f"{count:^9} | {acc:.4f} | {correct_count}/{total_count}")
                
                logging.info("----------------------------------\n")
        
    return results
                
if __name__ == "__main__":
    # 명령행 인자 파서 설정
    parser = argparse.ArgumentParser(description="의도 분류 모델 평가")
    parser.add_argument("--model", type=str, help="평가할 모델 이름 (예: Llama-1B)")
    parser.add_argument("--dataset", type=str, help="평가할 데이터셋 이름 (예: BlendATIS)")
    parser.add_argument("--gpu", type=int, default=3, help="사용할 GPU ID (기본값: 3)")
    
    args = parser.parse_args()
    
    # GPU 설정
    device_id = args.gpu if torch.cuda.is_available() else "cpu"
    if isinstance(device_id, int):
        device = f"cuda:{device_id}"
        logging.info(f"Using GPU: {device}")
    else:
        device = device_id
        logging.info(f"Using device: {device}")
    
    # 모델과 데이터셋 로그
    if args.model:
        logging.info(f"평가할 모델: {args.model}")
    else:
        logging.info("모든 모델 평가")
    
    if args.dataset:
        logging.info(f"평가할 데이터셋: {args.dataset}")
    else:
        logging.info("모든 데이터셋 평가")
    
    # 평가 실행
    try:
        results = evaluate(model_name=args.model, dataset_name=args.dataset)
        logging.info("평가 완료")
    except Exception as e:
        logging.error(f"평가 중 오류 발생: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())