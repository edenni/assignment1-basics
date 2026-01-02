import logging
import os
from collections import Counter, defaultdict
from collections.abc import Callable
from typing import BinaryIO

import regex as re


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT = re.compile(PAT)


def simple_splitter(chunk: str):
    for match in PAT.finditer(chunk):
        yield match.group()


def pretokenize(
    input_path: str,
    num_processes: int = 4,
    special_token: str = "<|endoftext|>",
    splitter: Callable = simple_splitter,
) -> dict[str, int]:
    """Pre-tokenize the corpus, split by special token and return the counts of pre-tokens."""
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes * 4, special_token.encode("utf8"))

    if num_processes > 1:
        raise NotImplementedError("Multi-processing is not yet supported for pre-tokenization.")
    else:
        counter = defaultdict(int)
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            with open(input_path, "rb") as f:
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")

            # Run pre-tokenization on your chunk and store the counts for each pre-token
            mini_chunks = chunk.split(special_token)
            for mini_chunk in mini_chunks:
                for pretoken in splitter(mini_chunk):
                    counter[pretoken] += 1

    return counter


special_tokens = ["<|endoftext|>"]
special_token_bytes = [x.encode() for x in special_tokens]

vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
for i, token_bytes in enumerate(special_token_bytes):
    vocab[i + 256] = bytes(token_bytes)

pretoken_counts = pretokenize("./data/TinyStoriesV2-GPT4-valid.txt", 1)
pretoken_counts: dict[tuple[bytes], int] = {tuple(k.encode("utf8")): v for k, v in pretoken_counts.items()}

token_counts = defaultdict(int)
for pretoken, count in pretoken_counts.items():
    for i in range(len(pretoken) - 1):
        token_counts[pretoken[i : i + 2]] += count


def get_most(counts: Counter[tuple[bytes], int]):
    """Break ties in pair frequency by preferring the lexicographically greater pair"""
    ties = []
    max_count = 0
    for tokens, count in counts.items():
        if count < max_count:
            continue
        elif count > max_count:
            max_count = count
            ties = [tokens]
        elif count == max_count:
            ties.append(tokens)
    return max(ties), max_count


tokens_to_merge, _ = get_most(token_counts)
new_id = len(vocab)
left, right = tokens_to_merge


def decode(tokens):
    chars = []
    for t in tokens:
        try:
            if t > 255:
                raise
            t = chr(t)
        except:
            pass
        chars.append(t)
    return chars


for tokens, count in pretoken_counts.copy().items():
    new_tokens = []
    updated = False
    i = 0
    while i < len(tokens) - 1:
        if tokens[i] == left and tokens[i + 1] == right:
            new_tokens.append(new_id)
            updated = True
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    if updated:
        pretoken_counts[tuple(new_tokens)] = pretoken_counts.pop(tokens)
        print(f"{decode(tokens)} -> {decode(new_tokens)}")
