# intent-reconstruction/src/data_utils_v2.py
import json
import logging
import torch
from torch.utils.data import Dataset
import os
from typing import List, Dict, Union, Optional, Tuple, Set
import re

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

class IntentDataset(Dataset):
    """Intent classification dataset class"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 512,
        split: str = "train"
    ):
        """
        Args:
            data_path: Path to data file
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
            split: Dataset split (train, dev, test)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = "dev" if split == "val" else split
        self.is_test = split == "test"
        
        # Load data
        self.data = self._load_data(data_path)
        
        # Define prompt templates
        self.train_prompt_template = (
            """
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            Task: Identify ALL intents from the following sentence. IMPORTANT: You MUST identify ALL applicable intents, not just one.

            Instruction: For the given sentence, identify ALL applicable intent types and output them in a comma-separated list within parentheses.
            If multiple intents are present, include all of them.
            Do not include any additional text, explanation, or SQL queries.
            The output should be in the format: (intent_type1, intent_type2, ...)

            Available Intent types:
            - atis_flight: Flight information (most common request)
            - atis_flight_time: Flight arrival/departure time
            - atis_airfare: Flight ticket fare
            - atis_aircraft: Questions about aircraft type
            - atis_ground_service: Ground service, such as transportation to the airport
            - atis_airport: Questions about the airport
            - atis_airline: Information about the airline
            - atis_distance: Distance information (e.g. distance between cities)
            - atis_abbreviation: Abbreviation meaning (e.g. airport code)
            - atis_ground_fare: Ground transportation fare
            - atis_quantity: Quantity information (e.g. number of flights)
            - atis_city: City information
            - atis_flight_no: Flight number
            - atis_capacity: Capacity information, such as number of seats on the aircraft
            - atis_meal: Questions about meals on board
            - atis_restriction: Restrictions on flights, etc.
            - atis_cheapest: Request for the cheapest flight<|eot_id|>

            <|start_header_id|>user<|end_header_id|>
            Sentence: {utterance}<|eot_id|>

            <|start_header_id|>assistant<|end_header_id|>
            Intent: {intent}<|eot_id|>
            """
        )
        
        # Test prompt template
        self.test_prompt_template = (
            """
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            Task: Identify ALL intents from the following sentence. IMPORTANT: You MUST identify ALL applicable intents, not just one.

            Instruction: For the given sentence, identify ALL applicable intent types and output them in a comma-separated list within parentheses.
            If multiple intents are present, include all of them.
            Do not include any additional text, explanation, or SQL queries.
            The output should be in the format: (intent_type1, intent_type2, ...)

            Available Intent types:
            - atis_flight: Flight information (most common request)
            - atis_flight_time: Flight arrival/departure time
            - atis_airfare: Flight ticket fare
            - atis_aircraft: Questions about aircraft type
            - atis_ground_service: Ground service, such as transportation to the airport
            - atis_airport: Questions about the airport
            - atis_airline: Information about the airline
            - atis_distance: Distance information (e.g. distance between cities)
            - atis_abbreviation: Abbreviation meaning (e.g. airport code)
            - atis_ground_fare: Ground transportation fare
            - atis_quantity: Quantity information (e.g. number of flights)
            - atis_city: City information
            - atis_flight_no: Flight number
            - atis_capacity: Capacity information, such as number of seats on the aircraft
            - atis_meal: Questions about meals on board
            - atis_restriction: Restrictions on flights, etc.
            - atis_cheapest: Request for the cheapest flight<|eot_id|>

            <|start_header_id|>user<|end_header_id|>
            Sentence: {utterance}<|eot_id|>

            <|start_header_id|>assistant<|end_header_id|>
            Intent: 
            """
        )
        
        # Log initialization
        logging.info(f"Loaded {len(self.data)} samples from {data_path} (split: {self.split})")
        
    def _load_data(self, data_path: str) -> List[Dict]:
        """데이터 파일을 로드하는 함수"""
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # split에 따라 필터링
            filtered_data = []
            for item in data:
                if item["split"] == self.split:
                    # Standardize intent format
                    item["intent"] = self._standardize_intents(item["intent"])
                    if item["intent"]:  # Only add if there are intents
                        filtered_data.append(item)
            
            return filtered_data
            
        except Exception as e:
            logging.error(f"Error loading data from {data_path}: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return []
    
    def _standardize_intents(self, intent_data) -> List[str]:
        """Convert various intent formats to standard list format"""
        if isinstance(intent_data, list):
            return intent_data
            
        if isinstance(intent_data, str):
            # String in list format (e.g. "['intent1', 'intent2']")
            if intent_data.startswith('[') and intent_data.endswith(']'):
                try:
                    return eval(intent_data)
                except:
                    pass
                    
            # String in parentheses format (e.g. "(intent1, intent2)")
            if intent_data.startswith('(') and intent_data.endswith(')'):
                return [i.strip() for i in intent_data.strip('()').split(',') if i.strip()]
                
            # Single intent
            return [intent_data]
            
        return []
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            item = self.data[idx]
            sentence = item["utterance"].strip()
            intents = item["intent"]
            
            # 디버깅용 로깅 추가
            if idx < 5 and self.is_test:
                logging.info(f"데이터 샘플 {idx}:")
                logging.info(f"  원본 문장: {sentence}")
                logging.info(f"  인텐트: {intents}")
            
            # Format prompt based on split
            if self.is_test:
                # For test, only include system and user parts
                prompt = self.test_prompt_template.format(utterance=sentence)
                
                # 디버깅용 로깅 추가
                if idx < 5:
                    logging.info(f"  생성된 프롬프트: {prompt[:200]}...")
            else:
                # For train/val, include system, user, and assistant parts
                intent_str = f"({', '.join(intents)})"
                prompt = self.train_prompt_template.format(utterance=sentence, intent=intent_str)
            
            # Tokenize
            tokenized = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors=None
            )
            
            # Convert to tensors
            input_ids = torch.tensor(tokenized["input_ids"], dtype=torch.long)
            attention_mask = torch.tensor(tokenized["attention_mask"], dtype=torch.long)
            
            # Prepare result
            result = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "raw_intents": intents,  # Original intent list
            }
            
            # For train/val, create labels
            if not self.is_test:
                # Prepare intent label format
                intent_label = f"({', '.join(intents)})"
                
                # Tokenize label
                label_tokenized = self.tokenizer(
                    intent_label,
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                    return_tensors=None
                )
                
                # Create label tensor
                labels = torch.tensor(label_tokenized["input_ids"], dtype=torch.long)
                result["labels"] = labels
                result["label_text"] = intent_label
            
            return result
            
        except Exception as e:
            logging.error(f"Error processing sample {idx}: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return None

def collate_fn(batch, pad_token_id):
    """Batch data processing function"""
    # Filter None items
    batch = [item for item in batch if item is not None]
    if not batch:
        return {}
    
    try:
        # Get common keys
        keys = batch[0].keys()
        
        result = {}
        for key in keys:
            if key in ['input_ids', 'attention_mask', 'labels']:
                # Stack padded tensors
                try:
                    result[key] = torch.stack([item[key] for item in batch])
                except:
                    # Handle variable length tensors if needed
                    max_length = max(item[key].size(0) for item in batch)
                    padded_tensors = []
                    
                    for item in batch:
                        tensor = item[key]
                        if tensor.size(0) < max_length:
                            padding = torch.full((max_length - tensor.size(0),), 
                                              pad_token_id if key != 'labels' else -100,
                                              dtype=tensor.dtype)
                            tensor = torch.cat([tensor, padding])
                        padded_tensors.append(tensor)
                    
                    result[key] = torch.stack(padded_tensors)
            elif key == 'raw_intents' or key == 'label_text':
                # Collect as list
                result[key] = [item[key] for item in batch]
        
        return result
        
    except Exception as e:
        logging.error(f"Error in batch processing: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        
        # Fallback: return first item as batch of 1
        if len(batch) > 0:
            return {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else [v] 
                   for k, v in batch[0].items()}
        return {}

def extract_sentence_from_prompt(prompt: str) -> str:
    """프롬프트에서 입력 문장을 추출하는 함수
    
    Args:
        prompt (str): 전체 프롬프트 텍스트
        
    Returns:
        str: 추출된 문장 또는 "unknown"
    """
    if not prompt or not isinstance(prompt, str):
        logging.warning("Invalid prompt input")
        return "unknown"
    
    try:
        # 프롬프트에서 입력 문장 추출 시도 (완전히 새로운 방식)
        # 1. 명시적인 'Sentence:' 패턴 찾기
        sentence_pattern = r"Sentence:\s*([^\n<]+)"
        sentence_match = re.search(sentence_pattern, prompt, re.IGNORECASE)
        
        if sentence_match:
            sentence = sentence_match.group(1).strip()
            if sentence and sentence != "Sentence:":
                if not sentence.startswith("IMPORTANT:"):
                    return sentence
        
        # 실패한 경우 전체 프롬프트 출력
        logging.warning(f"Failed to extract sentence. Full prompt: {prompt[:500]}...")
        return "unknown"
    except Exception as e:
        logging.error(f"Error in extract_sentence_from_prompt: {e}")
        return "unknown"

def normalize_intent_output(text: str) -> str:
    """모델 출력을 그대로 반환하는 함수 (정규화 없음)
    
    Args:
        text (str): 모델 출력 텍스트
        
    Returns:
        str: 원본 텍스트
    """
    if not text or not isinstance(text, str):
        return ""
    
    return text

def extract_intent_set(intent_str: str) -> Set[str]:
    """인텐트 문자열에서 인텐트 집합을 추출하는 함수
    
    Args:
        intent_str (str): 인텐트 문자열 (예: "(intent1, intent2)")
        
    Returns:
        Set[str]: 인텐트 집합
    """
    if not intent_str or not isinstance(intent_str, str):
        return set()
    
    # 괄호 제거 후 쉼표로 분리
    clean_str = intent_str.strip('() ').strip()
    if not clean_str:
        return set()
    
    # 쉼표로 구분된 의도들을 집합으로 변환
    intents = {intent.strip() for intent in clean_str.split(',') if intent.strip()}
    return intents
