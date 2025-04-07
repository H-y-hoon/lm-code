import os
import json
import torch
import argparse
import re
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Union

# Import custom modules
from dataloader import IntentDataset, collate_fn
from utils import load_lora_model, clear_gpu_memory

def normalize_intent_output(text):
    """Normalize model output to extract intent labels"""
    # Remove instruction and system prompts
    text = re.sub(r'\[INST\].*?Sentence:.*?Intent:', '', text, flags=re.DOTALL)
    text = re.sub(r'\[/INST\]', '', text)
    text = re.sub(r'Answer:.*?(\(|$)', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'Note that.*?(\(|$)', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'Explanation:.*?(\(|$)', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'Intents:.*?(\(|$)', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\n.*?(\(|$)', r'\1', text, flags=re.DOTALL)
    
    # Extract intents using different patterns
    intents = []
    
    # 1. Extract from parentheses with comma separation (atis_flight, atis_airfare)
    parenthesis_match = re.search(r'\((.*?)\)', text)
    if parenthesis_match:
        content = parenthesis_match.group(1)
        intents = [intent.strip() for intent in content.split(',')]
    
    # 2. Extract from hash-separated format (atis_flight#atis_airfare)
    elif "#" in text:
        hash_intents = re.findall(r'atis_[a-z_#]+', text)
        for hash_intent in hash_intents:
            if "#" in hash_intent:
                intents.extend([i.strip() for i in hash_intent.split("#")])
            else:
                intents.append(hash_intent.strip())
    
    # 3. Extract individual atis_ prefixed intents
    elif not intents:
        atis_matches = re.findall(r'atis_\w+', text)
        if atis_matches:
            intents = atis_matches
    
    # 4. Try keyword mapping if no intents found
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
        
        text_lower = text.lower()
        for keyword, intent in intent_keywords.items():
            if keyword in text_lower:
                intents.append(intent)
    
    # Normalize and deduplicate
    valid_intents = []
    seen = set()
    
    # Valid intent types
    valid_intent_types = [
        "atis_flight", "atis_flight_time", "atis_airfare", "atis_aircraft",
        "atis_ground_service", "atis_airport", "atis_airline", "atis_distance",
        "atis_abbreviation", "atis_ground_fare", "atis_quantity", "atis_city",
        "atis_flight_no", "atis_capacity", "atis_meal", "atis_restriction", "atis_cheapest"
    ]
    
    for intent in intents:
        intent = intent.strip()
        
        # Add atis_ prefix if missing
        if not intent.startswith('atis_') and intent in ["flight", "airfare", "airport", "airline", 
                                                    "flight_time", "aircraft", "ground_service"]:
            intent = f"atis_{intent}"
        
        # Fix malformed intents
        if intent.startswith('atis_intent_') or 'atiti' in intent or 'atcis' in intent:
            if 'flight' in intent.lower():
                intent = "atis_flight"
            elif 'air' in intent.lower():
                intent = "atis_airfare"
        
        # Only include valid intents
        if intent in valid_intent_types and intent not in seen:
            valid_intents.append(intent)
            seen.add(intent)
    
    # Fall back to default intent if none found
    if not valid_intents:
        print(f"No valid intents extracted from: {text}")
        valid_intents.append("atis_flight")  # Default intent
    
    # Return normalized format
    return f"({', '.join(valid_intents)})"

def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Inference for intent recognition")
    
    # Model arguments
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-1B-Instruct", 
                      help="Base model ID from Hugging Face")
    parser.add_argument("--ckpt_dir", type=str, required=True, 
                      help="Path to saved LoRA adapter directory")
    parser.add_argument("--device", type=str, default="cuda:0", 
                      help="Device to use for inference")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, required=True, 
                      help="Path to test dataset file")
    parser.add_argument("--max_length", type=int, default=256, 
                      help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=4, 
                      help="Batch size for inference")
    
    # Output arguments
    parser.add_argument("--output_file", type=str, default="results.json", 
                      help="Path to save results")
    parser.add_argument("--interactive", action="store_true", 
                      help="Enable interactive mode for testing single inputs")
    
    # Generation arguments
    parser.add_argument("--max_new_tokens", type=int, default=100, 
                      help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.3, 
                      help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.9, 
                      help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=int, default=50, 
                      help="Top-k sampling parameter")
    
    return parser.parse_args()

def evaluate_model(args):
    """Run model evaluation on test dataset"""
    # Set device
    device = args.device
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        device = "cpu"
    
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    model, tokenizer = load_lora_model(args.model_id, args.ckpt_dir, device)
    
    # Create test dataset
    test_dataset = IntentDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        split="test"
    )
    
    # Create test dataloader
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id)
    )
    
    # Extract ground truth data
    ground_truth_data = []
    for item in test_dataset.data:
        try:
            intents = eval(item["intent"]) if isinstance(item["intent"], str) else item["intent"]
            if isinstance(intents, list):
                ground_truth_data.append({
                    "sentence": item["utterance"],
                    "intents": intents
                })
            else:
                ground_truth_data.append({
                    "sentence": item["utterance"],
                    "intents": [intents]
                })
        except Exception as e:
            print(f"Error extracting ground truth: {str(e)}")
            ground_truth_data.append({
                "sentence": item["utterance"],
                "intents": []
            })
    
    # Initialize metrics and results
    results = []
    correct = 0
    total = 0
    
    # Generate predictions
    print(f"Starting evaluation on {len(test_dataset)} test samples")
    
    for batch_idx, batch in enumerate(test_dataloader):
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        # Generate predictions
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            num_return_sequences=1,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
            use_cache=True
        )
        
        # Process each sample in batch
        for sample_idx in range(len(outputs)):
            # Calculate global index
            global_idx = batch_idx * args.batch_size + sample_idx
            if global_idx >= len(ground_truth_data):
                break
            
            # Decode output
            output_ids = outputs[sample_idx]
            input_text = tokenizer.decode(input_ids[sample_idx], skip_special_tokens=True)
            generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
            
            # Extract prediction text
            if input_text in generated_text:
                predicted_text = generated_text[len(input_text):].strip()
            else:
                predicted_text = generated_text.strip()
            
            # Handle empty predictions
            if not predicted_text:
                predicted_text = "empty generation"
                print(f"Empty prediction for sample {global_idx}")
            
            # Extract and normalize intent
            predicted_intents = normalize_intent_output(predicted_text)
            predicted_intents_list = predicted_intents.strip('()').split(', ')
            predicted_intents_list = [intent.strip() for intent in predicted_intents_list if intent.strip()]
            
            # Get ground truth
            ground_truth_intents = ground_truth_data[global_idx]["intents"]
            
            # Check if prediction is correct
            is_correct = set(predicted_intents_list) == set(ground_truth_intents)
            if is_correct:
                correct += 1
            total += 1
            
            # Print progress
            if (global_idx + 1) % 10 == 0:
                print(f"Processed {global_idx + 1}/{len(ground_truth_data)} samples")
            
            # Store result
            results.append({
                "sample_id": global_idx,
                "sentence": ground_truth_data[global_idx]["sentence"],
                "ground_truth": ground_truth_intents,
                "prediction": predicted_intents_list,
                "is_correct": is_correct,
                "predicted_text": predicted_text
            })
    
    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0
    print(f"Overall accuracy: {accuracy:.4f} ({correct}/{total})")
    
    # Save results
    output_data = {
        "model": args.model_id,
        "adapter": args.ckpt_dir,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results
    }
    
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {args.output_file}")
    
    return accuracy, results

def interactive_mode(args):
    """Run model in interactive mode for single inputs"""
    # Set device
    device = args.device
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        device = "cpu"
    
    # Load model and tokenizer
    model, tokenizer = load_lora_model(args.model_id, args.ckpt_dir, device)
    
    print("\n==== Interactive Intent Recognition Mode ====")
    print("Enter a sentence to predict its intent.")
    print("Type 'exit' or 'quit' to end the session.\n")
    
    # Prompt template
    prompt_template = (
        "[INST] <<SYS>>\n"
        "Task: Identify ALL intents from the following sentence. IMPORTANT: You MUST identify ALL applicable intents, not just one.\n"
        "Instruction: For the given sentence, identify ALL applicable intent types and output them in a comma-separated list within parentheses. "
        "If multiple intents are present, include all of them. "
        "Do not include any additional text, explanation, or SQL queries. "
        "The output should be in the format: (intent_type1, intent_type2, ...)\n\n"
        "<</SYS>>\n"
        "Sentence: {}\n"
        "Intent:\n"
        "[/INST]"
    )
    
    while True:
        # Get user input
        user_input = input("\nEnter a sentence (or 'exit' to quit): ")
        
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting interactive mode.")
            break
        
        if not user_input.strip():
            continue
        
        # Create prompt
        prompt = prompt_template.format(user_input)
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate prediction
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2,
                use_cache=True
            )
        
        # Decode and process output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract prediction
        input_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        if input_text in generated_text:
            predicted_text = generated_text[len(input_text):].strip()
        else:
            predicted_text = generated_text.strip()
        
        # Normalize intent
        predicted_intents = normalize_intent_output(predicted_text)
        
        print(f"Raw model output: {predicted_text}")
        print(f"Normalized intent: {predicted_intents}")

def main():
    args = get_args()
    
    # Clear GPU memory before starting
    clear_gpu_memory()
    
    if args.interactive:
        interactive_mode(args)
    else:
        evaluate_model(args)
    
    # Clear GPU memory after finishing
    clear_gpu_memory()

if __name__ == "__main__":
    main()
