from collections.abc import Iterable, Iterator

import bpe_rs

from cs336_basics.train_bpe import train_bpe
from cs336_basics.utils import load_vocab_and_merges


class RustTokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self._rs = bpe_rs.Tokenizer(vocab, merges, special_tokens)
        self.special_tokens = special_tokens

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "RustTokenizer":
        vocab, merges = load_vocab_and_merges(vocab_filepath, merges_filepath)
        return cls(vocab, merges, special_tokens)

    @classmethod
    def train_from_file(
        cls,
        input_path: str,
        vocab_size: int,
        special_tokens: list[str],
        num_processes: int = 1,
    ) -> "RustTokenizer":
        vocab, merges = train_bpe(
            input_path, vocab_size, special_tokens, num_processes, use_rust=True
        )
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str, min_heap: bool = False) -> list[int]:
        return self._rs.encode(text, min_heap)

    def encode_file(
        self, input_path: str, min_heap: bool = False, num_threads: int = 0
    ):
        """Parallel-encode an entire file, returning a numpy uint16 array.

        Boundary-splits on the first special token so the result equals
        single-threaded ``encode(open(input_path).read())`` byte-for-byte
        (provided all ids fit in uint16).
        """
        return self._rs.encode_file(str(input_path), min_heap, num_threads)

    def encode_iterable(
        self, iterable: Iterable[str], min_heap: bool = False
    ) -> Iterator[int]:
        for line in iterable:
            yield from self._rs.encode(line, min_heap)

    def decode(self, ids: list[int]) -> str:
        return self._rs.decode(ids)

    @property
    def vocab(self) -> dict[int, bytes]:
        return self._rs.vocab

    @property
    def merges(self) -> list[tuple[bytes, bytes]]:
        return self._rs.merges

    @property
    def vocab_size(self) -> int:
        return self._rs.vocab_size


if __name__ == "__main__":
    tok = RustTokenizer.train_from_file(
        "./data/TinyStoriesV2-GPT4-valid.txt", 5000, ["<|endoftext|>"]
    )
    ids = tok.encode("hello world<|endoftext|>")
    print(ids)
    print(tok.decode(ids))
