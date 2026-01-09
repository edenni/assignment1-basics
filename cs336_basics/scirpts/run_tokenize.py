import numpy as np
from tqdm import tqdm

from cs336_basics.tokenizer import Tokenizer

data_path = dict(train="data/TinyStoriesV2-GPT4-train.txt", val="data/TinyStoriesV2-GPT4-valid.txt")
vocab_filepath = "outputs/tinystories_vocab.json"
merges_filepath = "outputs/tinystories_merges.txt"
special_tokens = ["<|endoftext|>"]


tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)

for split in ["train", "val"]:
    with open(data_path[split]) as f:
        text = f.read()
    encoded = tokenizer.encode(text)

    # save the ids
    total_batches = 1024
    batch_size = len(encoded) // total_batches
    arr = np.memmap(f"data/dataset/tinystories_{split}.bin", dtype=np.uint16, mode="w+", shape=(len(encoded),))
    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f"Writing {split}.bin"):
        batch = encoded[idx : idx + batch_size]
        arr[idx : idx + batch_size] = batch
        idx += batch_size
arr.flush()
