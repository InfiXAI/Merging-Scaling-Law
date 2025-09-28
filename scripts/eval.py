from torch.utils.data import Dataset, DataLoader
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import os
import glob
import pickle
import hashlib
import json


def build_prompt(problem: str, solution: str, tokenizer=None) -> str:
    if solution:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": problem.strip()},
            {"role": "assistant", "content": solution.strip()},
        ]
        messages = tokenizer.apply_chat_template(messages, tokenize=False)
    else:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": problem.strip()},
        ]
        messages = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return messages


def get_cache_key(file_path: str, tokenizer_name: str, max_length: int) -> str:
    """生成缓存键，基于文件内容、tokenizer和参数"""
    with open(file_path, 'r', encoding='utf-8') as f:
        file_content = f.read()
    
    cache_data = {
        'file_content_hash': hashlib.md5(file_content.encode()).hexdigest(),
        'tokenizer_name': tokenizer_name,
        'max_length': max_length
    }
    
    cache_str = json.dumps(cache_data, sort_keys=True)
    return hashlib.md5(cache_str.encode()).hexdigest()


class EvalDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512, cache_dir="./dataset_cache", use_cache=True):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)

        self.classification = None
        self.set_classification(self.dataset[0])
        
        self.cached_data = None

    def __len__(self):
        return len(self.dataset)

    def set_classification(self, row):
        if "classification" in row:
            self.classification = row["classification"].lower()
        else:
            self.classification = "code"
        print(f"loaded {self.classification} data!")

    def preprocess_data(self, file_path=None):
        if not self.use_cache:
            return self._tokenize_all_data()
            
        if file_path:
            cache_key = get_cache_key(file_path, self.tokenizer.name_or_path, self.max_length)
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            
            if os.path.exists(cache_file):
                print(f"Loading cached data from {cache_file}")
                try:
                    with open(cache_file, 'rb') as f:
                        self.cached_data = pickle.load(f)
                    print(f"Successfully loaded {len(self.cached_data)} cached samples")
                    return self.cached_data
                except Exception as e:
                    print(f"Failed to load cache: {e}, will regenerate...")
            
            print("Tokenizing data (this may take a while)...")
            self.cached_data = self._tokenize_all_data()
            
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(self.cached_data, f)
                print(f"Cached data saved to {cache_file}")
            except Exception as e:
                print(f"Failed to save cache: {e}")
        else:
            self.cached_data = self._tokenize_all_data()
            
        return self.cached_data

    def _tokenize_all_data(self):
        cached_data = []
        print("Tokenizing dataset...")
        
        for idx in tqdm(range(len(self.dataset)), desc="Tokenizing"):
            item = self.dataset[idx]

            if "instruction" in item:
                query = item["instruction"]
                solution = item["output"]
            elif "problem" in item:
                query = item["problem"]
                solution = item["solution"]
            else:
                raise ValueError("Invalid dataset format.")
                
            messages = build_prompt(query, solution, self.tokenizer)

            encoded = self.tokenizer(
                messages,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            )

            input_ids = encoded["input_ids"].squeeze(0)

            prefix_text = build_prompt(query, "", self.tokenizer)
            prefix = self.tokenizer(
                prefix_text,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.max_length
            )

            prefix_ids = prefix["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)

            labels = input_ids.clone()
            labels[:len(prefix_ids)] = -100
            labels[attention_mask == 0] = -100

            cached_data.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })
            
        return cached_data

    def __getitem__(self, idx):
        if self.cached_data is None:
            return self._get_item_original(idx)
        
        return self.cached_data[idx]

    def _get_item_original(self, idx):
        item = self.dataset[idx]

        if "instruction" in item:
            query = item["instruction"]
            solution = item["output"]
        elif "problem" in item:
            query = item["problem"]
            solution = item["solution"]
        else:
            raise ValueError("Invalid dataset format.")
            
        messages = build_prompt(query, solution, self.tokenizer)

        encoded = self.tokenizer(
            messages,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )

        input_ids = encoded["input_ids"].squeeze(0)

        prefix_text = build_prompt(query, "", self.tokenizer)
        prefix = self.tokenizer(
            prefix_text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_length
        )

        prefix_ids = prefix["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        labels[:len(prefix_ids)] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/path/to/pretrained-model")
    parser.add_argument("--tokenizer", type=str, default="/path/to/tokenizer")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--dataset", type=str, default="./data")
    parser.add_argument("--file", type=str, default="")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output", type=str, default="./output")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--no_cache", action="store_true")
    args = parser.parse_args()
    
    device_count = torch.cuda.device_count()
    print(f"Found {device_count} GPUs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    if args.file:
        files = [args.file]
    else:
        files = glob.glob(f"{args.dataset}/*.json")

    results = []
    total_loss_overall = 0
    total_token_overall = 0
    
    for file in files:
        print(f"\nProcessing file: {file}")
        dataset = json.load(open(file))
        test_dataset = EvalDataset(
            dataset, 
            tokenizer, 
            max_length=args.max_length,
            cache_dir=args.cache_dir,
            use_cache=not args.no_cache
        )
        
        test_dataset.preprocess_data(file_path=file)
        
        dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                active_tokens = (labels != -100).sum().item()
                total_loss += loss.item() * active_tokens
                total_tokens += active_tokens

        avg_ce_loss = total_loss / total_tokens
        problem_type = os.path.basename(file).replace(".json", "")
        results.append(
            {
                "problem": problem_type,
                "CE Loss": avg_ce_loss,
                "class": test_dataset.classification
            }
        )
        total_loss_overall += total_loss
        total_token_overall += total_tokens

    overall_loss = total_loss_overall / total_token_overall

    domain = 'all'
    model_name = os.path.basename(args.model)
    output_dir = os.path.join(args.output, model_name, domain)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "results.csv")
    
    df = pd.DataFrame(results)
    new_row = pd.DataFrame([
        {
            "problem": "Avg.",
            "CE Loss": df["CE Loss"].mean(),
            "class": "average"
        },
        {
            "problem": "Overall",
            "CE Loss": overall_loss,
            "class": "overall"
        },
    ])
    df = pd.concat([df, new_row], ignore_index=True)
    print(df)
    df.to_csv(output_file, index=False)