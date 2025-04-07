import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.optim import AdamW
import sys
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from dataloader import create_dataloaders
from utils import load_base_model_and_tokenizer, get_lora_model, clear_gpu_memory
from typing import List, Dict, Tuple
import numpy as np

class ValueNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

def get_args():
    parser = argparse.ArgumentParser()
    
    # Model arguments
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:3")
    
    # LoRA configuration
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_bias", type=str, default="none", choices=["none", "all", "lora_only"])
    parser.add_argument("--target_modules", type=str, nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"])
    
    # Data arguments
    parser.add_argument("--data_path", type=str, default="/home/user/workspace/hyunku/experiment/lm/code/data/BlendX/BlendATIS.json")
    parser.add_argument("--max_length", type=int, default=256)
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--value_learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--ppo_epochs", type=int, default=3)
    parser.add_argument("--target_bert_score", type=float, default=0.8)
    parser.add_argument("--max_iterations", type=int, default=100)
    
    # PPO specific arguments
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lambda_", type=float, default=0.95)
    parser.add_argument("--clip_epsilon", type=float, default=0.2)
    parser.add_argument("--value_loss_coef", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="/home/user/workspace/hyunku/experiment/lm/code/hk/ckpt")
    parser.add_argument("--save_ckpt_name", type=str, default="lora_model")
    
    return parser.parse_args()

class PPOTrainer:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            print("CUDA not available, switching to CPU")
            self.device = "cpu"
        
        print(f"Using device: {self.device}")
        
        # Load base model and tokenizer
        self.model, self.tokenizer = load_base_model_and_tokenizer(args.model_id, self.device)
        
        # Apply LoRA
        lora_config = {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "bias": args.lora_bias,
            "target_modules": args.target_modules
        }
        self.model = get_lora_model(self.model, lora_config)
        
        # Initialize value network
        self.value_network = ValueNetwork(input_dim=768).to(self.device)  # SentenceBERT embedding dimension
        
        # Load sentence transformer for BERT score
        self.sentence_transformer = SentenceTransformer('sentence-transformers/all-mpnet-base-v2').to(self.device)
        
        # Create dataloaders
        self.dataloaders = create_dataloaders(
            args.data_path, 
            self.tokenizer, 
            args.batch_size, 
            args.max_length)
        
        # Initialize optimizers
        self.policy_optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        self.value_optimizer = AdamW(
            self.value_network.parameters(),
            lr=args.value_learning_rate,
            weight_decay=args.weight_decay
        )
        
        # Initialize scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.policy_optimizer,
            max_lr=args.learning_rate,
            epochs=args.epochs,
            steps_per_epoch=len(self.dataloaders["train"]),
            pct_start=args.warmup_ratio,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=10000.0
        )
    
    def calculate_bert_score(self, original_sentences: List[str], reconstructed_sentences: List[str]) -> float:
        """Calculate BERT score between original and reconstructed sentences"""
        with torch.no_grad():
            original_embeddings = self.sentence_transformer.encode(original_sentences, convert_to_tensor=True)
            reconstructed_embeddings = self.sentence_transformer.encode(reconstructed_sentences, convert_to_tensor=True)
            
            # Calculate cosine similarity
            scores = torch.nn.functional.cosine_similarity(original_embeddings, reconstructed_embeddings)
            return scores.mean().item()
    
    def get_state_embedding(self, sentences: List[str]) -> torch.Tensor:
        """Get state embedding using SentenceBERT"""
        with torch.no_grad():
            embeddings = self.sentence_transformer.encode(sentences, convert_to_tensor=True)
            return embeddings
    
    def generate_reconstructed_sentence(self, intents: str) -> str:
        """Generate sentence from predicted intents"""
        prompt = f"Generate a sentence that contains the following intents: {intents}"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def compute_gae(self, rewards: List[float], values: List[float], dones: List[bool]) -> List[float]:
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.args.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.args.gamma * self.args.lambda_ * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        return advantages
    
    def train(self):
        print(f"Starting PPO training for {self.args.epochs} epochs")
        
        for epoch in range(self.args.epochs):
            self.model.train()
            self.value_network.train()
            total_policy_loss = 0.0
            total_value_loss = 0.0
            iteration = 0
            
            while iteration < self.args.max_iterations:
                # Collect trajectories
                states = []
                actions = []
                rewards = []
                values = []
                dones = []
                old_log_probs = []
                
                for batch in self.dataloaders["train"]:
                    # Move batch to device
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    
                    # Get state embedding
                    original_sentences = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
                    state_embeddings = self.get_state_embedding(original_sentences)
                    
                    # Get value estimates
                    with torch.no_grad():
                        value_estimates = self.value_network(state_embeddings).squeeze(-1).tolist()
                    
                    # Generate predictions
                    with torch.no_grad():
                        outputs = self.model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=100,
                            num_return_sequences=1,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            return_dict_in_generate=True,
                            output_scores=True
                        )
                    
                    # Get log probabilities
                    log_probs = torch.log_softmax(outputs.scores[0], dim=-1)
                    action_log_probs = log_probs.gather(1, outputs.sequences[:, 1:2]).squeeze(-1)
                    
                    # Decode predictions
                    predicted_intents = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs.sequences]
                    
                    # Generate reconstructed sentences
                    reconstructed_sentences = [self.generate_reconstructed_sentence(intents) for intents in predicted_intents]
                    
                    # Calculate rewards
                    bert_scores = [self.calculate_bert_score([orig], [rec]) for orig, rec in zip(original_sentences, reconstructed_sentences)]
                    
                    # Store trajectory data
                    states.extend(state_embeddings)
                    actions.extend(outputs.sequences)
                    rewards.extend(bert_scores)
                    values.extend(value_estimates)
                    dones.extend([False] * len(bert_scores))
                    old_log_probs.extend(action_log_probs)
                    
                    # Check if target BERT score is reached
                    if np.mean(bert_scores) >= self.args.target_bert_score:
                        print(f"Target BERT score reached: {np.mean(bert_scores):.4f}")
                        self.save_model()
                        return
                
                # Convert lists to tensors
                states = torch.stack(states)
                actions = torch.stack(actions)
                rewards = torch.tensor(rewards, device=self.device)
                values = torch.tensor(values, device=self.device)
                dones = torch.tensor(dones, device=self.device)
                old_log_probs = torch.stack(old_log_probs)
                
                # Compute advantages
                advantages = torch.tensor(self.compute_gae(rewards.tolist(), values.tolist(), dones.tolist()), device=self.device)
                returns = advantages + values
                
                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # PPO training loop
                for ppo_epoch in range(self.args.ppo_epochs):
                    # Update value network
                    self.value_optimizer.zero_grad()
                    value_preds = self.value_network(states).squeeze(-1)
                    value_loss = F.mse_loss(value_preds, returns)
                    value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), self.args.max_grad_norm)
                    self.value_optimizer.step()
                    
                    # Update policy network
                    self.policy_optimizer.zero_grad()
                    
                    # Get new log probabilities
                    with torch.no_grad():
                        outputs = self.model.generate(
                            input_ids=actions,
                            attention_mask=torch.ones_like(actions),
                            max_new_tokens=100,
                            num_return_sequences=1,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            return_dict_in_generate=True,
                            output_scores=True
                        )
                    
                    new_log_probs = torch.log_softmax(outputs.scores[0], dim=-1)
                    new_action_log_probs = new_log_probs.gather(1, actions[:, 1:2]).squeeze(-1)
                    
                    # Calculate policy loss
                    ratio = torch.exp(new_action_log_probs - old_log_probs)
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.args.clip_epsilon, 1.0 + self.args.clip_epsilon) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Calculate entropy bonus
                    entropy = -(new_log_probs * torch.exp(new_log_probs)).sum(-1).mean()
                    
                    # Total loss
                    loss = policy_loss + self.args.value_loss_coef * value_loss - self.args.entropy_coef * entropy
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    self.policy_optimizer.step()
                    self.scheduler.step()
                    
                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                
                iteration += 1
                
                # Print progress
                print(f'Epoch: {epoch+1}/{self.args.epochs}, '
                      f'Iteration: {iteration}/{self.args.max_iterations}, '
                      f'Policy Loss: {total_policy_loss/self.args.ppo_epochs:.4f}, '
                      f'Value Loss: {total_value_loss/self.args.ppo_epochs:.4f}, '
                      f'Average Reward: {torch.mean(rewards):.4f}')
            
            print(f'[Epoch] : {epoch+1}/{self.args.epochs}, '
                  f'[Average Policy Loss] : {total_policy_loss/iteration:.4f}, '
                  f'[Average Value Loss] : {total_value_loss/iteration:.4f}')
        
        self.save_model()
        print("Training completed")
    
    def save_model(self):
        save_dir = os.path.join(self.args.output_dir, self.args.save_ckpt_name)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        torch.save(self.value_network.state_dict(), os.path.join(save_dir, "value_network.pt"))
        print(f"Successfully saved model to {save_dir}")

if __name__ == "__main__":
    PPOTrainer(get_args()).train()
