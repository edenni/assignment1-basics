from collections import defaultdict

from tqdm import tqdm

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
    """Merge the token pair with max count."""
    left, right = tokens_to_merge
    merged_token = left + right

    for tokens, count in pre_token_counts.copy().items():
        new_tokens = []
        updated = False
        i = 0

        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == left and tokens[i + 1] == right:
                new_tokens.append(merged_token)
                updated = True

                # update pair counts
                token_pair_counts[tokens_to_merge] -= count
                if i > 0:
                    token_pair_counts[(tokens[i - 1], left)] -= count
                    token_pair_counts[(tokens[i - 1], merged_token)] += count
                if i + 2 < len(tokens):
                    token_pair_counts[(right, tokens[i + 2])] -= count
                    token_pair_counts[(merged_token, tokens[i + 2])] += count
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        if updated:
            pre_token_counts[tuple(new_tokens)] = pre_token_counts.pop(tokens)
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
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # init vocab
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for i, token in enumerate(special_tokens):
        vocab[i + 256] = token.encode("utf-8")

    # pre-tokenization
    pre_token_counts = pretokenize(input_path, 1)
    pre_token_counts: dict[tuple[bytes], int] = {
        tuple(bytes([b]) for b in k.encode("utf8")): v for k, v in pre_token_counts.items()
    }

    # merge
    merges: list[tuple[bytes, bytes]] = []
    token_pair_counts = None
    for i in tqdm(range(vocab_size - len(vocab))):
        vocab, pre_token_counts, token_pair_counts, merged = _one_step(vocab, pre_token_counts, token_pair_counts)
        merges.append(merged)
    return vocab, merges


if __name__ == "__main__":
    train_bpe("./data/TinyStoriesV2-GPT4-valid.txt", 258, ["<|endoftext|>"])
