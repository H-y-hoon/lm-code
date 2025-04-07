import json
import torch
from torch.utils.data import Dataset, DataLoader
import os
from typing import Dict, List, Optional, Union, Any
from transformers import AutoTokenizer

# batch 내 데이터의 길이가 다르기에, Padding 수행
def collate_fn(batch, pad_token_id):
    batch = [item for item in batch if item is not None]
    if not batch:
        return {}

    max_length = max(len(item["input_ids"]) for item in batch)
    
    input_ids = []
    attention_mask = []
    labels = []
    
    for item in batch:
        curr_len = len(item["input_ids"])
        padding_len = max_length - curr_len # calculate padding length
        
        # Pad input IDs (left padding)
        input_ids.append(
            torch.cat([
                torch.full((padding_len,), pad_token_id, dtype=torch.long),
                item["input_ids"]
            ])
        )
        
        # Pad attention mask (left padding)
        attention_mask.append(
            torch.cat([
                torch.zeros(padding_len, dtype=torch.long),
                item["attention_mask"]
            ])
        )
        
        # Pad labels if present (left padding with -100 to ignore in loss)
        if "labels" in item:
            labels.append(
                torch.cat([
                    torch.full((padding_len,), -100, dtype=torch.long),
                    item["labels"]
                ])
            )
    
    batch_dict = {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask)
    }
    
    if labels:
        batch_dict["labels"] = torch.stack(labels)
    
    return batch_dict

class IntentDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 256, split: str = "train"):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.split = "dev" if split == "val" else split
        self.is_test = split == "test"
    
        self.data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                if item["split"] == self.split:
                    self.data.append(item)
        
        self.train_prompt_template = ( # you can check the prompt template using the following command: print(tokenizer.get_chat_template)
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
            "- atis_cheapest: Request for the cheapest flight\n"
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            "Sentence: {sentence}\n"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            "Intent: {intent}<|eot_id|>"
        )

        
        self.test_prompt_template = (
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
            "- atis_cheapest: Request for the cheapest flight\n"
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            "Sentence: {sentence}\n"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            "Intent: "
        )
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]

        sentence = item["utterance"].strip()
        intent_label = item["intent"].strip() if not self.is_test else ""
            
        if not self.is_test:
            intent_list = eval(intent_label) if isinstance(intent_label, str) else intent_label
            if isinstance(intent_list, list):
                intent_label = f"({', '.join(intent_list)})"
            else:
                intent_label = f"({intent_label})"

        if not intent_label or intent_label == "()":
            intent_label = "(unknown)"
            
        # Apply prompt templates
        if self.is_test:
            prompt = self.test_prompt_template.format(sentence=sentence)
        else:
            prompt = self.train_prompt_template.format(sentence=sentence, intent=intent_label)

        # Tokenize prompt
        tokenized = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors=None)

        input_ids = torch.tensor(tokenized["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(tokenized["attention_mask"], dtype=torch.long)
            
        if self.is_test: # Test mode: don't return labels
            return {"input_ids": input_ids,
                    "attention_mask": attention_mask}
        else: # Training/validation mode: include labels
            prompt_without_intent = self.train_prompt_template.format(sentence=sentence, intent="")
            tokenized_prompt = self.tokenizer(
                    prompt_without_intent,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors=None)
            prompt_length = len(tokenized_prompt["input_ids"])

            labels = input_ids.clone()
            labels[:prompt_length] = -100 # Set prompt part to -100 to ignore in loss calculation
                
            return {"input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels}

def create_dataloaders(data_path: str, tokenizer, batch_size: int, max_length: int = 256):
    dataloaders = {}
    for split in ["train", "val", "test"]:
        dataset = IntentDataset(
                data_path=data_path,
                tokenizer=tokenizer,
                max_length=max_length,
                split=split)
        dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == "train"),  # Shuffle only training data
                collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id))
        dataloaders[split] = dataloader
    return dataloaders

if __name__ == "__main__":    
    data_path = "/home/user/workspace/hyunku/experiment/lm/code/data/BlendX/BlendATIS.json"
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    if tokenizer.pad_token is None: # padding token이 없으면 eos_token을 padding token으로 설정
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left' # 왼쪽 패딩 사용

    dataloaders = create_dataloaders(
        data_path=data_path,
        tokenizer=tokenizer,
        batch_size=4,
        max_length=256
    )

    for split, dataloader in dataloaders.items(): # split : train, val, test
        batch = next(iter(dataloader))
        print(list(batch.keys()))
        print(batch['input_ids'].shape)
        print(batch['attention_mask'].shape)
        print(batch['labels'].shape)
        exit()