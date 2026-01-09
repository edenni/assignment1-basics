from collections.abc import Iterable, Iterator

import regex as re
from tqdm import tqdm

from cs336_basics.pretokenization import simple_splitter
from cs336_basics.train_bpe import train_bpe
from cs336_basics.utils import load_vocab_and_merges


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.cache = {}

        if self.special_tokens:
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
            for token in self.special_tokens:
                if (token_bytes := token.encode("utf-8")) not in self.inv_vocab:
                    idx = len(self.vocab)
                    self.vocab[idx] = token_bytes
                    self.inv_vocab[token_bytes] = idx

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        vocab, merges = load_vocab_and_merges(vocab_filepath, merges_filepath)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        if self.special_tokens is not None:
            special_pattern = "(" + "|".join(re.escape(k) for k in self.special_tokens) + ")"
            chunks = re.split(special_pattern, text)
        else:
            chunks = [text]
        ids = []
        for chunk in tqdm(chunks, desc="Encoding"):
            ids += self._encode_chunk(chunk)
        return ids

    def _encode_chunk(self, text: str) -> list[int]:
        if self.special_tokens and text in self.special_tokens:
            return [self.inv_vocab[text.encode("utf-8")]]

        pre_tokens = list(simple_splitter(text))
        pre_tokens = [tuple(bytes([b]) for b in t.encode("utf8")) for t in pre_tokens]

        merged_tokens = []
        for pre_token in pre_tokens:
            if pre_token in self.cache:
                merged_tokens.append(self.cache[pre_token])
                continue

            merged = pre_token
            for token_pair in self.merges:
                new_merged = []
                i = 0
                while i < len(merged):
                    if i + 1 < len(merged) and tuple(merged[i : i + 2]) == token_pair:
                        new_merged.append(token_pair[0] + token_pair[1])
                        i += 2
                    else:
                        new_merged.append(merged[i])
                        i += 1
                # if merged != new_merged:
                #     print(merged, new_merged)
                merged = new_merged
            merged_tokens.append(merged)
            self.cache[pre_token] = merged

        encoded = []
        for tokens in merged_tokens:
            for token in tokens:
                encoded.append(self.inv_vocab[token])
        return encoded

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for line in iterable:
            ids = self.encode(line)
            yield from ids

    def decode(self, ids: list[int]) -> str:
        tokens = [self.vocab[i] for i in ids]
        tokens = b"".join(tokens)
        return tokens.decode("utf-8", errors="replace")


if __name__ == "__main__":
    input_path = "./data/TinyStoriesV2-GPT4-valid.txt"
    vocab_size = 5000
    special_tokens = ["<|endoftext|>"]

    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    tokenizer = Tokenizer(vocab, merges, special_tokens)
    inp = tokenizer.encode("hello world<|endoftext|>")
    print(inp)
    print(tokenizer.decode(inp))
