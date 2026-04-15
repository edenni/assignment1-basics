import bpe_rs


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str], num_processes: int = 1
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # init vocab
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for i, token in enumerate(special_tokens):
        vocab[i + 256] = token.encode("utf-8")

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
    return vocab, merges


if __name__ == "__main__":
    vocab, merges = train_bpe(
        "./data/TinyStoriesV2-GPT4-valid.txt", 2000, ["<|endoftext|>"]
    )
