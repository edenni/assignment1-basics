from cs336_basics.train_bpe import train_bpe
from cs336_basics.utils import save_vocab_and_merges

input_path = "data/owt_train.txt"
output_vocab_path = "outputs/owt_vocab.json"
output_merge_path = "outputs/owt_merges.txt"

vocab_size = 32000
special_tokens = ["<|endoftext|>"]
num_processes = 24

vocab, merges = train_bpe(input_path, vocab_size, special_tokens, num_processes=num_processes)
save_vocab_and_merges(vocab, merges, output_vocab_path, output_merge_path)
