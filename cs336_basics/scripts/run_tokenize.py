import numpy as np
from tqdm import tqdm

from cs336_basics.tokenizer import Tokenizer

data_path = dict(train="data/TinyStoriesV2-GPT4-train.txt", val="data/TinyStoriesV2-GPT4-valid.txt")
vocab_filepath = "outputs/tinystories_vocab.json"
merges_filepath = "outputs/tinystories_merges.txt"
special_tokens = ["<|endoftext|>"]
dataset_name = "tinystories"

tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)

for split in ["train", "val"]:
    with open(data_path[split]) as f:
        text = f.read()
    encoded = tokenizer.encode(text)

    # save the ids
    total_chunks = 1024
    chunk_size = len(encoded) // total_chunks
    arr = np.memmap(f"data/tinystories/{split}.bin", dtype=np.uint16, mode="w+", shape=(len(encoded),))
    idx = 0
    for batch_idx in tqdm(range(total_chunks), desc=f"Writing {split}.bin"):
        batch = encoded[idx : idx + chunk_size]
        arr[idx : idx + chunk_size] = batch
        idx += chunk_size
arr.flush()
