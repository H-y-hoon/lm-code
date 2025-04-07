# intent-reconstruction/src/data_utils.py
import json
import logging
import torch
from torch.utils.data import Dataset
import os
import ast

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def collate_fn(batch, pad_token_id):
    """배치 데이터를 패딩하여 동일한 크기로 만드는 함수"""
    # None 항목 필터링
    batch = [item for item in batch if item is not None]
    if not batch:
        return {}
    
    try:
        # 텐서 크기 확인 및 로깅
        input_sizes = [item['input_ids'].size() for item in batch]
        attention_sizes = [item['attention_mask'].size() for item in batch]
        
        if len(set(input_sizes)) > 1:
            logging.warning(f"입력 텐서 크기가 일치하지 않습니다: {input_sizes}")
        
        if len(set(attention_sizes)) > 1:
            logging.warning(f"어텐션 마스크 텐서 크기가 일치하지 않습니다: {attention_sizes}")
        
        # 모든 텐서 스택 시도
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        
        # 레이블이 있는 경우에만 스택
        if 'labels' in batch[0]:
            label_sizes = [item['labels'].size() for item in batch]
            if len(set(label_sizes)) > 1:
                logging.warning(f"레이블 텐서 크기가 일치하지 않습니다: {label_sizes}")
            
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
    except Exception as e:
        # 오류 발생 시 상세 정보 로깅
        logging.error(f"배치 처리 중 오류 발생: {str(e)}")
        
        # 배치 내 각 항목의 상세 정보 출력
        for i, item in enumerate(batch):
            if item is not None:
                logging.error(f"배치 항목 {i}: input_ids 크기={item['input_ids'].size()}, "
                            f"attention_mask 크기={item['attention_mask'].size()}, "
                            f"keys={item.keys()}")
        
        # 비상 대책: 배치 크기가 1인 배치 반환
        if len(batch) > 0:
            return {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v 
                   for k, v in batch[0].items()}
        return {}

class IntentDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=256, split="train"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # split 이름 매핑 (검증 데이터 처리)
        self.split = "dev" if split == "val" else split
        self.is_test = split == "test"  # test 모드 여부 확인
        
        # 데이터 경로 확인
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_path}")
        
        # 데이터 로드
        try:
            self.data = []
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line.strip())
                    if item["split"] == self.split:  # self.split 사용
                        self.data.append(item)
            
            if len(self.data) == 0:
                raise ValueError(f"데이터가 비어 있습니다: {data_path} (split: {self.split})")
                
            logging.info(f"데이터 {len(self.data)}개 로드 완료: {data_path} (split: {self.split})")
            
            # 데이터 유효성 검사
            self._validate_data()
            
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON 파싱 오류: {str(e)}")
        
        # 학습/검증용 프롬프트 템플릿 정의
        self.train_prompt_template = (
            '''
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            "Task: Identify ALL intent types from the following sentence. IMPORTANT: You MUST identify ALL applicable intents, not just one.\n\n"
            "Instruction: For the given sentence, identify ALL applicable intent types and output them in a comma-separated list within parentheses. "
            "If multiple intents are present, include all of them. "
            "Do not include any additional text or explanation.\n"
            "Available Intent types:\n"
            "- atis_flight: Flight information (most common request)\n"
            "- atis_flight_time: Flight arrival/departure time\n"
            "- atis_airfare: Flight ticket fare\n"
            "- atis_aircraft: Questions about aircraft type\n"
            "- atis_ground_service: Ground service, such as transportation to the airport\n"
            "- atis_airport: Questions about the airport\n"
            "- atis_airline: Information about the airline\n"
            "- atis_distance: Distance information (e.g. distance between cities)\n"
            "- atis_abbreviation: Abbreviation meaning (e.g. airport code)\n"
            "- atis_ground_fare: Ground transportation fare\n"
            "- atis_quantity: Quantity information (e.g. number of flights)\n"
            "- atis_city: City information\n"
            "- atis_flight_no: Flight number\n"
            "- atis_capacity: Capacity information, such as number of seats on the aircraft\n"
            "- atis_meal: Questions about meals on board\n"
            "- atis_restriction: Restrictions on flights, etc.\n"
            "- atis_cheapest: Request for the cheapest flight<|eot_id|>\n"

            "<|start_header_id|>user<|end_header_id|>\n"
            "Sentence: {sentence}<|eot_id|>\n"

            "<|start_header_id|>assistant<|end_header_id|>\n"
            "Intent: {intent}<|eot_id|>"
            '''
        )

        
        # 테스트용 프롬프트 템플릿
        self.test_prompt_template = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            "Task: Identify ALL intents from the following sentence. IMPORTANT: You MUST identify ALL applicable intents, not just one.\n"
            "Instruction: For the given sentence, identify ALL applicable intent types and output them in a comma-separated list within parentheses. "
            "If multiple intents are present, include all of them. "
            "Do not include any additional text, explanation, or SQL queries. "
            "The output should be in the format: (intent_type1, intent_type2, ...)\n"
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            "Sentence: {sentence}\n"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            "Intent: "
        )

        
        # 샘플 데이터로 테스트
        try:
            first_item = self.data[0]
            test_item = self.__getitem__(0)
            logging.info(f"데이터셋 초기화 성공. 첫 번째 항목 형식: {test_item.keys()}")
        except Exception as e:
            logging.error(f"샘플 데이터 처리 중 오류: {str(e)}")
            raise
    
    def _validate_data(self):
        """데이터셋 유효성 검사"""
        valid_count = 0
        invalid_count = 0
        
        for i, item in enumerate(self.data):
            try:
                # 필수 필드 확인
                if "utterance" not in item or "intent" not in item:
                    logging.warning(f"필수 필드 누락 (인덱스: {i}): {item}")
                    invalid_count += 1
                    continue
                
                # 문장 확인
                if not item["utterance"] or not item["utterance"].strip():
                    logging.warning(f"빈 문장 (인덱스: {i})")
                    invalid_count += 1
                    continue
                
                # 의도 레이블 확인
                intents = self._parse_intents(item["intent"])
                if not intents:
                    logging.warning(f"유효하지 않은 의도 레이블 (인덱스: {i}): {item['intent']}")
                    invalid_count += 1
                    continue
                
                valid_count += 1
            except Exception as e:
                logging.error(f"데이터 검증 중 오류 (인덱스: {i}): {str(e)}")
                invalid_count += 1
        
        logging.info(f"데이터 검증 완료: 유효={valid_count}, 유효하지 않음={invalid_count}, 총={len(self.data)}")
    
    def _parse_intents(self, intent_str):
        """의도 레이블 문자열을 파싱하는 함수"""
        try:
            # 문자열이 이미 리스트인 경우
            if isinstance(intent_str, list):
                return intent_str
            
            # 문자열이 튜플 형태인 경우 (예: "(atis_flight, atis_airfare)")
            if isinstance(intent_str, str) and intent_str.startswith("(") and intent_str.endswith(")"):
                # 괄호 제거 후 쉼표로 분리
                intent_str = intent_str.strip("()")
                intents = [i.strip() for i in intent_str.split(",")]
                return intents
            
            # 문자열이 단일 의도인 경우
            if isinstance(intent_str, str):
                return [intent_str]
            
            # 그 외의 경우
            return []
        except Exception as e:
            logging.error(f"의도 레이블 파싱 중 오류: {str(e)}, 레이블: {intent_str}")
            return []
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        """데이터셋에서 항목을 가져오는 메서드"""
        try:
            item = self.data[idx]
            sentence = item["utterance"]
            
            # 빈 문장 처리
            if not sentence or not sentence.strip():
                logging.warning(f"빈 문장 발견 (인덱스: {idx})")
                return None
            
            # 의도 레이블 처리
            intents = self._parse_intents(item["intent"])
            
            # 의도가 없는 경우 처리
            if not intents:
                logging.warning(f"의도 레이블 없음 (인덱스: {idx})")
                return None
            
            # 프롬프트 템플릿 선택
            if self.is_test:
                prompt = self.test_prompt_template.format(sentence=sentence)
            else:
                prompt = self.train_prompt_template.format(sentence=sentence, intent=f"({', '.join(intents)})")
    
            # 전체 텍스트 토크나이즈
            try:
                tokenized = self.tokenizer(
                    prompt,
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",  # 항상 max_length로 패딩
                    return_tensors=None
                )
                
                # 입력 텐서 생성 - 1차원 텐서로 변환
                input_ids = torch.tensor(tokenized["input_ids"], dtype=torch.long)
                attention_mask = torch.tensor(tokenized["attention_mask"], dtype=torch.long)
                
                # 레이블 토큰화 (모든 모드에서 동일하게 처리)
                label_text = f"({', '.join(intents)})"
                label_tokenized = self.tokenizer(
                    label_text,
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",  # 항상 max_length로 패딩
                    return_tensors=None
                )
                
                # 레이블 텐서 생성 - 1차원 텐서로 변환
                labels = torch.tensor(label_tokenized["input_ids"], dtype=torch.long)
                
                # 결과 반환 - 모든 텐서가 동일한 크기(max_length)를 가지도록 보장
                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                    "intents": intents  # 원본 의도 레이블도 함께 반환
                }
            except Exception as e:
                logging.error(f"토크나이즈 처리 중 오류 발생 (인덱스: {idx}, 문장: {sentence[:30]}...): {str(e)}")
                return None
                
        except Exception as e:
            logging.error(f"데이터 처리 중 오류 발생 (인덱스: {idx}): {str(e)}")
            # 스택 트레이스 추가
            import traceback
            logging.error(traceback.format_exc())
            return None
