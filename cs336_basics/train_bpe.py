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


def merge(pretoken_counts: dict[tuple[bytes], int], tokens_to_merge: tuple[bytes, bytes]) -> dict[tuple[bytes], int]:
    """Merge the token pair with max count."""
    left, right = tokens_to_merge
    merged_token = left + right

    for tokens, count in pretoken_counts.copy().items():
        new_tokens = []
        updated = False
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == left and tokens[i + 1] == right:
                new_tokens.append(merged_token)
                updated = True
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        if updated:
            pretoken_counts[tuple(new_tokens)] = pretoken_counts.pop(tokens)
    return pretoken_counts


def one_step(pretoken_counts: dict[tuple[bytes], int], vocab: dict[int, bytes]):
    # count byte pairs
    token_pair_counts = defaultdict(int)
    for pretoken, count in pretoken_counts.items():
        for i in range(len(pretoken) - 1):
            token_pair_counts[pretoken[i : i + 2]] += count
    tokens_to_merge, _ = get_most(token_pair_counts)

    # merge
    pretoken_counts = merge(pretoken_counts, tokens_to_merge)
    vocab[len(vocab)] = tokens_to_merge[0] + tokens_to_merge[1]
    return pretoken_counts, vocab, tokens_to_merge


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # init vocab
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for i, token in enumerate(special_tokens):
        vocab[i + 256] = token.encode("utf-8")

    # pre-tokenization
    pretoken_counts = pretokenize(input_path, 1)
    pretoken_counts: dict[tuple[bytes], int] = {
        tuple(bytes([b]) for b in k.encode("utf8")): v for k, v in pretoken_counts.items()
    }

    # merge
    merges = []
    for i in tqdm(range(vocab_size - len(vocab))):
        pretoken_counts, vocab, merged = one_step(pretoken_counts, vocab)
        merges.append(merged)
    return vocab, merges


if __name__ == "__main__":
    train_bpe("../data/TinyStoriesV2-GPT4-valid.txt", 258, ["<|endoftext|>"])
