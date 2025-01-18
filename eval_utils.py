# derived from reference
import os
import json
import requests
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "hellaswag")

hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

enc = AutoTokenizer.from_pretrained("microsoft/phi-1_5")

def download_file(url, dest_path):
    """Downloads a file from a URL to a destination path."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def write_evalfile(filename, datas):
    """Writes evaluation data to a binary file."""
    with open(filename, "wb") as f:
        torch.save(datas, f)

def download(split):
    """Downloads HellaSwag dataset to DATA_CACHE_DIR."""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = hellaswags[split]
    data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)
    else:
        print(f"{data_filename} already exists, skipping download...")

def render_example(example):
    """
    Given the example as a dictionary, render it as three torch tensors:
    - tokens (the tokens of context + completion, of size 4xN, as there are always 4 candidates)
    - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods)
    - label (the index of the correct completion, which we hope has the highest likelihood)
    """
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    # data needed to reproduce this eval on the C size
    data = {
        "label": label,
        "ctx_tokens": None,
        "ending_tokens": [],
    }

    # gather up all the tokens
    ctx_tokens = enc.encode(ctx, add_special_tokens=False)
    data["ctx_tokens"] = ctx_tokens
    tok_rows = []
    mask_rows = []
    for end in endings:
        end_tokens = enc.encode(end) # note: prepending " " because GPT-2 tokenizer, removed for phi tokenizer
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_tokens))
        data["ending_tokens"].append(end_tokens)

    # have to be careful during the collation because the number of tokens in each row can differ
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label


def iterate_examples(split):
    """Iterates through examples in the HellaSwag dataset."""
    download(split)
    with open(os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl"), "r") as f:
        for line in f:
            example = json.loads(line)
            yield example

# In essence, we need to compute metrics here
class HellaSwagDataset(torch.utils.data.Dataset):
    """Dataset for HellaSwag evaluation that handles multiple choice format"""
    def __init__(self, split="val"):
        self.sources = load_dataset("Rowan/hellaswag", split="validation")
        self.size = len(self.sources)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        example = self.sources[idx]
        data, tokens, mask, label = render_example(example)
        label = {
            "tokens": tokens,
            "mask": mask,
            "label": label
        } # we need this to be able to calculate metrics properly
        return {
            "tokens": tokens,  # Shape: (4, seq_len) - one row per choice
            "label": label,   # Dict containing tokens, mask and label
            "idx": idx        # For distributed training tracking
        }

class IterableHellaSwagDataset(torch.utils.data.IterableDataset):
    """Dataset for HellaSwag evaluation that handles multiple choice format"""
    def __init__(self, split="val"):
        self.source = iterate_examples(split=split)

    def __iter__(self, idx):
        example = iter(self.examples)
        data, tokens, mask, label = render_example(example)
        label = {
            "tokens": tokens,
            "mask": mask,
            "label": label
        } # we need this to be able to calculate metrics properly
        return {
            "tokens": tokens,  # Shape: (4, seq_len) - one row per choice
            "label": label,   # Dict containing tokens, mask and label
            "idx": idx        # For distributed training tracking
        }


class DataCollatorForHellaSwag:
    """
    Data collator that handles batching multiple choice examples for HellaSwag evaluation.
    Each example has 4 choices that need to be processed together.
    """
    def __init__(self, pad_token_id=enc.pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        batch_size = len(examples)
        
        # Find max sequence length across all examples and all choices
        max_len = max(example["tokens"].shape[1] for example in examples)
        
        # Initialize tensors - preserving the 4 choices dimension
        tokens = torch.full(
            (batch_size, 4, max_len), 
            -100, 
            dtype=torch.long
        )
        
        label_mask = torch.zeros(
            (batch_size, 4, max_len),
            dtype=torch.long
        )
        label_indices = torch.zeros(batch_size, dtype=torch.long)
        indices = torch.zeros(batch_size, dtype=torch.long)
        
        # Fill tensors while keeping the 4 choices structure
        for i, example in enumerate(examples):
            seq_len = example["tokens"].shape[1]
            tokens[i, :, :seq_len] = example["tokens"]
            # Fill label dict tensors
            label_mask[i, :, :seq_len] = example["label"]["mask"]
            label_indices[i] = example["label"]["label"]
            indices[i] = example["idx"]
        
        # Reshape tokens to flatten the choices dimension into batch dimension
        tokens = tokens.view(batch_size * 4, max_len)
        label_mask = label_mask.view(batch_size * 4, max_len)
            
        return {
            "input_ids": tokens,           # Shape: (batch_size * 4, seq_len)
            "mask": label_mask,             # Shape: (batch_size * 4, seq_len)
            "labels": label_indices ,     # Shape: (batch_size,)
            "idx": indices                 # Shape: (batch_size,)
        }

class HellaswagMetrics:
    def __init__(self):
        self.correct = 0
        self.correct_norm = 0
        self.total = 0

    def __call__(self, logits, tokens, labels, mask, compute_last=False):

        # Calculate losses
        shift_logits = (logits[..., :-1, :]).contiguous()
        shift_tokens = (tokens[..., 1:]).contiguous()
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_shift_tokens = shift_tokens.view(-1)
        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
        shift_losses = shift_losses.view(tokens.size(0), -1)

        # Get masked losses
        shift_mask = (mask[..., 1:]).contiguous()
        masked_shift_losses = shift_losses * shift_mask
        sum_loss = masked_shift_losses.sum(dim=1) 
        avg_loss = sum_loss / shift_mask.sum(dim=1)

        # Reshape losses from (batch_size * 4, seq_len) back to (batch_size, 4)
        batch_size = tokens.size(0) // 4
        sum_loss = sum_loss.view(batch_size, 4)
        avg_loss = avg_loss.view(batch_size, 4)

        # Get predictions
        preds = sum_loss.argmin(dim=-1)
        preds_norm = avg_loss.argmin(dim=-1)

        # Accumulate stats
        self.correct += (preds == labels).sum().item()
        self.correct_norm += (preds_norm == labels).sum().item()
        self.total += len(preds)

        if compute_last:
            return self.compute_metrics()
        return None

    def compute_metrics(self):
        accuracy = self.correct / self.total
        accuracy_norm = self.correct_norm / self.total

        metrics = {
            "hellaswag_acc": accuracy,
            "hellaswag_acc_norm": accuracy_norm,
        }

        # Reset accumulators
        self.correct = 0
        self.correct_norm = 0
        self.total = 0

        return metrics




