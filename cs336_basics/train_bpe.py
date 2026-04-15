from collections import defaultdict

from tqdm import tqdm

import bpe_rs
from cs336_basics.pretokenization import pretokenize


def get_most(counts: dict[tuple[bytes], int]) -> tuple[tuple[bytes, bytes], int]:
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


def merge(
    pre_token_counts: dict[tuple[bytes], int],
    token_pair_counts: dict[tuple[bytes, bytes], int],
    tokens_to_merge: tuple[bytes, bytes],
) -> dict[tuple[bytes], int]:

    left, right = tokens_to_merge
    merged_token = left + right

    for tokens, count in list(pre_token_counts.items()):
        len_tokens = len(tokens)
        old_len = len_tokens - 1
        new_tokens = [None] * len_tokens
        i = j = 0
        updated = False

        while i < len_tokens:
            if i + 1 < len_tokens and tokens[i] == left and tokens[i + 1] == right:
                new_tokens[j] = merged_token
                i += 2
                updated = True
            else:
                new_tokens[j] = tokens[i]
                i += 1
            j += 1

        if not updated:
            continue

        new_tokens = tuple(new_tokens[:j])
        for k in range(old_len):
            token_pair_counts[(tokens[k], tokens[k + 1])] -= count
        for k in range(len(new_tokens) - 1):
            token_pair_counts[(new_tokens[k], new_tokens[k + 1])] += count
        pre_token_counts[new_tokens] = pre_token_counts.pop(tokens)

    token_pair_counts.pop(tokens_to_merge, None)

    return pre_token_counts


def _one_step(
    vocab: dict[int, bytes],
    pre_token_counts: dict[tuple[bytes], int],
    token_pair_counts: dict[tuple[bytes, bytes], int] = None,
):
    # init token pair counts
    if token_pair_counts is None:
        token_pair_counts = defaultdict(int)
        for pre_token, count in pre_token_counts.items():
            for i in range(len(pre_token) - 1):
                token_pair_counts[pre_token[i : i + 2]] += count

    tokens_to_merge, _ = get_most(token_pair_counts)

    # merge
    pre_token_counts = merge(pre_token_counts, token_pair_counts, tokens_to_merge)
    vocab[len(vocab)] = tokens_to_merge[0] + tokens_to_merge[1]
    return vocab, pre_token_counts, token_pair_counts, tokens_to_merge


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str], num_processes: int = 1, use_rust = True
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # init vocab
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for i, token in enumerate(special_tokens):
        vocab[i + 256] = token.encode("utf-8")

    if use_rust:
        # pre-tokenization (Rust)
        pre_token_counts: dict[bytes, int] = bpe_rs.pretokenize_file(
            str(input_path), special_tokens, num_processes
        )

        # expand each pretoken bytes into list of 1-byte symbols
        words = [
            ([bytes([b]) for b in k], v) for k, v in pre_token_counts.items() if k
        ]

        num_merges = vocab_size - len(vocab)
        merges: list[tuple[bytes, bytes]] = bpe_rs.train_merges(words, num_merges)
        for a, b in merges:
            vocab[len(vocab)] = a + b
    else:
        # pre-tokenization
        pre_token_counts = pretokenize(input_path, num_processes, special_tokens)
        pre_token_counts: dict[tuple[bytes], int] = {
            tuple(bytes([b]) for b in k.encode("utf8")): v for k, v in pre_token_counts.items()
        }

        # merge
        merges: list[tuple[bytes, bytes]] = []
        token_pair_counts = None
        for i in tqdm(range(vocab_size - len(vocab)), desc="Train BPE"):
            vocab, pre_token_counts, token_pair_counts, merged = _one_step(vocab, pre_token_counts, token_pair_counts)
            merges.append(merged)
    return vocab, merges


if __name__ == "__main__":
    vocab, merges = train_bpe(
        "./data/TinyStoriesV2-GPT4-valid.txt", 2000, ["<|endoftext|>"]
    )
