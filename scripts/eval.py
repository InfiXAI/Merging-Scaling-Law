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
import swanlab
import requests
import time
from urllib.parse import urlparse

def extract_model_name(model_path: str) -> str:
    """
    Extract model name from HuggingFace URL or local path

    Args:
        model_path: Model path (can be HuggingFace URL, repo name, or local path)

    Returns:
        Extracted model name

    Examples:
        "https://huggingface.co/microsoft/DialoGPT-medium" -> "microsoft/DialoGPT-medium"
        "microsoft/DialoGPT-medium" -> "microsoft/DialoGPT-medium"
        "/path/to/local/model" -> "model"
    """
    # Remove trailing slashes
    model_path = model_path.rstrip('/')

    # Check if it's a URL
    if model_path.startswith('http://') or model_path.startswith('https://'):
        parsed = urlparse(model_path)
        # Extract path after domain, remove leading slash
        path_parts = parsed.path.lstrip('/').split('/')
        # For huggingface.co URLs, typically format is: /org/model or /org/model/tree/branch
        if len(path_parts) >= 2:
            # Return org/model format
            return f"{path_parts[0]}/{path_parts[1]}"
        else:
            # Fallback to last part
            return path_parts[-1] if path_parts else model_path

    # Check if it's already in org/model format (HuggingFace style)
    if '/' in model_path and not model_path.startswith('/'):
        # Likely already in format like "microsoft/DialoGPT-medium"
        return model_path

    # Otherwise treat as local path and return basename
    return os.path.basename(model_path)

def send_callback(callback_url: str, task_id: str, model_id: str, benchmark_id: str,
                  status: str, score: float, evaluator_scores: dict = None,
                  error_message: str = None, signature: str = None, api_key: str = None,
                  max_retries: int = 3):
    """
    Send evaluation callback to specified URL

    Args:
        callback_url: The callback API endpoint URL
        task_id: Unique task identifier
        model_id: Model identifier
        benchmark_id: Benchmark identifier
        status: Evaluation status ("succeed" or "failed")
        score: Overall evaluation score
        evaluator_scores: Dictionary of individual evaluator scores
        error_message: Error message if status is "failed"
        signature: Signature parameter (usually experiment_name)
        api_key: API key for authentication
        max_retries: Maximum number of retry attempts
    """
    if evaluator_scores is None:
        evaluator_scores = {}
    
    # Prepare the payload
    payload = {
        "task_id": task_id,
        "model_id": model_id,
        "benchmark_id": benchmark_id,
        "status": status,
        "score": score,
        "evaluator_scores": evaluator_scores,
        "error_message": error_message
    }
    
    # Prepare headers
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
        'X-API-Key': api_key or ''  # Use provided API key or default
    }
    
    # Add signature parameter to URL if provided
    url = callback_url
    if signature:
        separator = '&' if '?' in url else '?'
        url = f"{url}{separator}signature={signature}"
    
    # Attempt to send the callback with retries
    for attempt in range(max_retries):
        try:
            print(f"Sending callback to {url} (attempt {attempt + 1}/{max_retries})")
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            print(f"Payload: {payload}")
            if response.status_code == 200:
                print("Callback sent successfully!")
                return True
            else:
                print(f"Callback failed with status code: {response.status_code}")
                print(f"Response: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"Error sending callback (attempt {attempt + 1}): {e}")
            
        if attempt < max_retries - 1:
            print(f"Retrying in 5 seconds...")
            time.sleep(5)
    
    print(f"Failed to send callback after {max_retries} attempts")
    return False


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
    parser.add_argument("--experiment_name", type=str, default="eval_experiment",
                       help="Name of the experiment for logging")
    parser.add_argument("--use_swanlab", action="store_true",
                       help="Enable SwanLab logging")
    parser.add_argument("--swanlab_mode", type=str, default="local",
                       choices=["local", "cloud", "disabled"],
                       help="SwanLab mode: 'local' for local logging, 'cloud' for cloud logging, 'disabled' to disable")

    # Callback-related arguments
    parser.add_argument("--callback_url", type=str, default="",
                       help="Callback URL for sending evaluation results")
    parser.add_argument("--task_id", type=str, default="",
                       help="Task ID for callback")
    parser.add_argument("--model_id", type=str, default="",
                       help="Model ID for callback")
    parser.add_argument("--benchmark_id", type=str, default="",
                       help="Benchmark ID for callback")
    parser.add_argument("--api_key", type=str, default="",
                       help="API key for callback authentication")
    args = parser.parse_args()
    
    device_count = torch.cuda.device_count()
    print(f"Found {device_count} GPUs")

    # Device selection: prefer CUDA, fallback to CPU (avoid MPS due to compatibility issues)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    # Initialize callback parameters
    callback_enabled = bool(args.callback_url)
    if callback_enabled:
        print(f"Callback enabled: {args.callback_url}")
        # Use experiment_name as signature if not provided explicitly
        signature = args.experiment_name
        
        # Generate default IDs if not provided
        task_id = args.task_id or f"eval_task_{int(time.time())}"
        model_id = args.model_id or extract_model_name(args.model)
        benchmark_id = args.benchmark_id or os.path.basename(args.dataset)
        
        print(f"Callback parameters - Task ID: {task_id}, Model ID: {model_id}, Benchmark ID: {benchmark_id}")
    
    try:
        # Normalize model and tokenizer paths (extract from HuggingFace URLs if needed)
        model_path = extract_model_name(args.model)
        tokenizer_path = extract_model_name(args.tokenizer)

        print(f"Loading model from: {model_path}")
        print(f"Loading tokenizer from: {tokenizer_path}")

        # Initialize SwanLab if enabled
        if args.use_swanlab and args.swanlab_mode != "disabled":
            print(f"Initializing SwanLab in '{args.swanlab_mode}' mode...")
            swanlab.init(
                project="model-evaluation",
                experiment_name=args.experiment_name,
                config={
                    "model_path": model_path,
                    "tokenizer_path": tokenizer_path,
                    "max_length": args.max_length,
                    "batch_size": args.batch_size,
                    "dataset_path": args.dataset,
                    "device_count": device_count,
                },
                mode=args.swanlab_mode
            )

        # Load model with appropriate device map
        # Use device_map="auto" only with CUDA, otherwise load to CPU to avoid MPS issues
        if torch.cuda.is_available():
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
        else:
            # On CPU or MPS, use float32 and explicit device placement
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32
            )
            model = model.to(device)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Set pad_token to eos_token: {tokenizer.eos_token}")

        # Adjust max_length based on model's max position embeddings
        model_max_length = getattr(model.config, 'n_positions', None) or \
                          getattr(model.config, 'max_position_embeddings', None) or \
                          args.max_length

        if args.max_length > model_max_length:
            print(f"Warning: Requested max_length ({args.max_length}) exceeds model's max ({model_max_length})")
            print(f"Adjusting max_length to {model_max_length}")
            args.max_length = model_max_length

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
            batch_losses = []

            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    active_tokens = (labels != -100).sum().item()
                    
                    batch_loss = loss.item()
                    batch_losses.append(batch_loss)
                    
                    total_loss += batch_loss * active_tokens
                    total_tokens += active_tokens
                    
                    # Log batch-level metrics to SwanLab
                    if args.use_swanlab:
                        problem_type = os.path.basename(file).replace(".json", "")
                        swanlab.log({
                            f"{problem_type}/batch_ce_loss": batch_loss,
                            f"{problem_type}/batch_tokens": active_tokens,
                            f"{problem_type}/batch_idx": batch_idx
                        })

            avg_ce_loss = total_loss / total_tokens
            problem_type = os.path.basename(file).replace(".json", "")
            
            # Log file-level metrics
            print(f"Problem type: {problem_type}, CE Loss: {avg_ce_loss:.4f}, Total tokens: {total_tokens}")
            if args.use_swanlab:
                swanlab.log({
                    f"{problem_type}/avg_ce_loss": avg_ce_loss,
                    f"{problem_type}/total_tokens": total_tokens,
                    f"{problem_type}/total_samples": len(dataset)
                })
            
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
        # Use normalized model path for output directory (replace / with _ for filesystem compatibility)
        model_name = model_path.replace('/', '_')
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
        
        # Log final overall metrics to SwanLab
        if args.use_swanlab:
            avg_ce_loss = df["CE Loss"].mean()
            swanlab.log({
                "overall/average_ce_loss": avg_ce_loss,
                "overall/overall_ce_loss": overall_loss,
                "overall/total_tokens": total_token_overall,
                "overall/num_problems": len(results) - 2,  # Exclude Avg. and Overall rows
            })
            
            # Log per-problem results as a summary table
            problem_results = {}
            for _, row in df.iterrows():
                if row["problem"] not in ["Avg.", "Overall"]:
                    problem_results[f"ce_loss/{row['problem']}"] = row["CE Loss"]
            
            swanlab.log(problem_results)
            
            print(f"SwanLab experiment '{args.experiment_name}' completed successfully!")
            swanlab.finish()
        
        # Send callback if enabled
        if callback_enabled:
            try:
                # Prepare evaluator scores from individual problem results
                evaluator_scores = {}
                for _, row in df.iterrows():
                    if row["problem"] not in ["Avg.", "Overall"]:
                        evaluator_scores[row["problem"]] = float(row["CE Loss"])
                
                # Send success callback
                send_callback(
                    callback_url=args.callback_url,
                    task_id=task_id,
                    model_id=model_id,
                    benchmark_id=benchmark_id,
                    status="success",
                    score=float(overall_loss),  # Use overall loss as main score
                    evaluator_scores=evaluator_scores,
                    signature=signature,
                    api_key=args.api_key
                )
            except Exception as callback_error:
                print(f"Error sending success callback: {callback_error}")
                
    except Exception as e:
        print(f"Evaluation failed with error: {e}")
        
        # Send failure callback if enabled
        if callback_enabled:
            try:
                send_callback(
                    callback_url=args.callback_url,
                    task_id=task_id,
                    model_id=model_id,
                    benchmark_id=benchmark_id,
                    status="failed",
                    score=0.0,
                    error_message=str(e),
                    signature=signature,
                    api_key=args.api_key
                )
            except Exception as callback_error:
                print(f"Error sending failure callback: {callback_error}")
        
        # Re-raise the original exception
        raise e