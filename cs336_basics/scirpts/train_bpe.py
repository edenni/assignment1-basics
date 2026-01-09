import cProfile

from cs336_basics.train_bpe import train_bpe
from cs336_basics.utils import save_vocab_and_merges

input_path = "data/TinyStoriesV2-GPT4-train.txt"
output_vocab_path = "outputs/tinystories_vocab.plk"
output_merge_path = "outputs/tinystories_merges.plk"

vocab_size = 10000
special_tokens = ["<|endoftext|>"]
num_processes = 24

pr = cProfile.Profile()
pr.enable()
vocab, merges = train_bpe(input_path, vocab_size, special_tokens, num_processes=num_processes)
pr.disable()
pr.print_stats(sort="time")
save_vocab_and_merges(vocab, merges, output_vocab_path, output_merge_path)
