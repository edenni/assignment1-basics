import cProfile

import wandb

from cs336_basics.train_bpe import train_bpe
from cs336_basics.utils import save_voacb_and_merge

input_path = "data/TinyStoriesV2-GPT4-train.txt"
output_vocab_path = "outputs/tinystories_vocab.json"
output_merge_path = "outputs/tinystories_merges.txt"

wandb_name = "cs336_basics"
wandb_run_name = "train_bpe_tinystories"
wandb_logging = False

vocab_size = 10000
special_tokens = ["<|endoftext|>"]
num_processes = 24

config_keys = [k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys}

if wandb_logging:
    wandb.init(project=wandb_name, name=wandb_run_name, config=config)

pr = cProfile.Profile()
pr.enable()
vocab, merges = train_bpe(input_path, vocab_size, special_tokens, num_processes=num_processes)
pr.disable()
pr.print_stats(sort="time")
save_voacb_and_merge(vocab, merges, output_vocab_path, output_merge_path)
