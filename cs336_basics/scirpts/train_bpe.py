import cProfile

import wandb

from cs336_basics.train_bpe import train_bpe

text_source = "data/TinyStoriesV2-GPT4-train.txt"
output_vocab_path = "data/out/tinystories_vocab.json"
output_merge_path = "data/out/tinystories_merges.txt"

wandb_name = "cs336_basics"
wandb_run_name = "train_bpe_tinystories"
wandb_logging = False

special_tokens = ["<|endoftext|>"]
vocab_size = 10000
num_workers = 20

config_keys = [k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys}

if wandb_logging:
    wandb.init(project=wandb_name, name=wandb_run_name, config=config)

pr = cProfile.Profile()
pr.enable()
vocab, merges = train_bpe(text_source, vocab_size, special_tokens, progress_bar=True, num_workers=num_workers)
pr.disable()

pr.print_stats(sort="time")
