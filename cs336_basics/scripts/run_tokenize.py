import os
from pathlib import Path

import numpy as np

from cs336_basics.rust_tokenizer import RustTokenizer as Tokenizer

data_path = dict(train="data/owt_train.txt", val="data/owt_valid.txt")
vocab_filepath = "outputs/owt_vocab.json"
merges_filepath = "outputs/owt_merges.txt"
special_tokens = ["<|endoftext|>"]
output_dir = "data/owt"

Path(output_dir).mkdir(parents=True, exist_ok=True)

tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)

num_threads = int(os.environ.get("BPE_THREADS", os.cpu_count() or 1))

for split in ["train", "val"]:
    out_path = f"{output_dir}/{split}.bin"
    ids = tokenizer.encode_file(
        data_path[split], min_heap=True, num_threads=num_threads
    )
    arr = np.memmap(out_path, dtype=np.uint16, mode="w+", shape=(len(ids),))
    arr[:] = ids
    arr.flush()
    del arr
    print(f"{split}: wrote {len(ids):,} tokens to {out_path}")
