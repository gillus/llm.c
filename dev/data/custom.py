import os
import argparse
import multiprocessing as mp
import pandas as pd
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import requests
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from data_common import write_datafile, download_file, write_evalfile
os.environ["TOKENIZERS_PARALLELISM"] = "false"


HEADERS_INFO = {
    "gpt-2": {"magic": 0x67607432, "version": 1, "token_dtype": np.uint16},
    "llama-3": {"magic": 0x6c6c6d61, "version": 1, "token_dtype": np.uint32},
    "custom": {"magic": 0x63757374, "version": 1, "token_dtype": np.uint32},  # example values
}

parser = argparse.ArgumentParser(description="Preprocess the custom dataset")
parser.add_argument("--dataset_path", type=str, required=True,
                    help="Path to the parquet file of the dataset (with a 'text' column)")
parser.add_argument("-s", "--shard_size", type=int, default=10**8,
                    help="Size of each data shard (in tokens)")
parser.add_argument("-v", "--vocab_size", type=int, default=30000,
                    help="Size of the vocabulary for the tokenizer")
args = parser.parse_args()

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "simpleenglishwiki_cache")
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

dataset = load_dataset("parquet", data_files=args.dataset_path, split="train")
name = f"{args.dataset_path}_custom"

print("Training custom tokenizer on dataset...")

# Initialize a BPE tokenizer
custom_tokenizer_model = Tokenizer(models.BPE(unk_token="<unk>"))
custom_tokenizer_model.pre_tokenizer = pre_tokenizers.ByteLevel()

# Set up a trainer with your desired vocabulary size and special tokens.
trainer = trainers.BpeTrainer(
    vocab_size=args.vocab_size,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
)

custom_tokenizer_model.train_from_iterator(dataset["text"], trainer=trainer)

custom_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=custom_tokenizer_model,
    bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>",
    mask_token="<mask>"
)

print("Tokenizer training completed.")
TOKENIZER_SAVE_PATH = os.path.join(DATA_CACHE_DIR, "custom_tokenizer")

os.makedirs(TOKENIZER_SAVE_PATH, exist_ok=True)
custom_tokenizer.save_pretrained(TOKENIZER_SAVE_PATH)

print(f"Tokenizer saved at {TOKENIZER_SAVE_PATH}")

def tokenize_custom(doc):
    """
    Tokenizes a single document using the custom tokenizer.
    An end-of-document token (here we use the eos token id) is added as a delimiter.
    """
    # Encode without adding any extra special tokens automatically.
    tokens = custom_tokenizer.encode(doc["text"], add_special_tokens=False)
    # Prepend an end-of-document token as delimiter (adjust if you prefer appending instead)
    tokens = [custom_tokenizer.eos_token_id] + tokens
    tokens_np = np.array(tokens)
    # Ensure tokens fit within uint32 range
    assert (0 <= tokens_np).all() and (tokens_np < 2**32).all(), "Token values out of uint32 range"
    tokens_np_uint = tokens_np.astype(np.uint32)
    return tokens_np_uint

token_dtype = np.uint32
nprocs = max(1, os.cpu_count() - 2) 
with mp.Pool(nprocs) as pool:
    shard_index = 0
    all_tokens_np = np.empty((args.shard_size,), dtype=token_dtype)
    token_count = 0
    progress_bar = None
    tokenize = tokenize_custom

    for tokens in pool.imap(tokenize, dataset, chunksize=16):
        if token_count + len(tokens) < args.shard_size:
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            if progress_bar is None:
                progress_bar = tqdm(total=args.shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"{name}_{split}_{shard_index:06d}.bin")
            remainder = args.shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np.tolist(), "custom")
            shard_index += 1
            progress_bar = None
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens) - remainder

    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"{name}_{split}_{shard_index:06d}.bin")
        write_datafile(filename, (all_tokens_np[:token_count]).tolist(), "custom")
