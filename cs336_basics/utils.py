import json
import os
import pickle
from functools import lru_cache

import torch


@lru_cache
def gpt2_bytes_to_unicode():
    """
    https://github.com/openai/gpt-2/blob/master/src/encoder.py#L9

    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def load_vocab_and_merges(vocab_path: str | os.PathLike, merges_path: str | os.PathLike):
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(vocab_path) as vocab_f:
        gpt2_vocab = json.load(vocab_f)
    gpt2_bpe_merges = []
    with open(merges_path) as f:
        for line in f:
            cleaned_line = line.rstrip()
            if cleaned_line and len(cleaned_line.split(" ")) == 2:
                gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
    # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
    # just return the original bytes, so we don't force students to use
    # any particular encoding scheme.
    vocab = {
        gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
    }

    merges = [
        (
            bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
            bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
        )
        for merge_token_1, merge_token_2 in gpt2_bpe_merges
    ]
    return vocab, merges


def save_vocab_and_merges(
    vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], vocab_path: str, merges_path: str
):
    byte_to_unicode = gpt2_bytes_to_unicode()

    # Reverse the mapping from unicode characters to bytes
    unicode_to_byte = {v: k for k, v in byte_to_unicode.items()}

    # Convert the byte tokens in the vocab back to string tokens using the unicode mapping
    reversed_vocab = {"".join([byte_to_unicode[b] for b in bytes_token]): k for k, bytes_token in vocab.items()}

    # Convert the byte sequences in merges back to string tokens
    reversed_merges = [
        " ".join(["".join([byte_to_unicode[b] for b in merge[0]]), "".join([byte_to_unicode[b] for b in merge[1]])])
        for merge in merges
    ]

    # Save the vocab dictionary as a JSON file
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(reversed_vocab, f, ensure_ascii=False)

    # Save the merges list to a file
    with open(merges_path, "w", encoding="utf-8") as f:
        for merge in reversed_merges:
            f.write(merge + "\n")


# def save_vocab_and_merges(vocab, merges, vocab_path, merges_path):
#     with open(vocab_path, "wb") as f:
#         pickle.dump(vocab, f)
#     with open(merges_path, "wb") as f:
#         pickle.dump(merges, f)


# def load_vocab_and_merges(vocab_path, merges_path):
#     with open(vocab_path, "rb") as f:
#         vocab = pickle.load(f)
#     with open(merges_path, "rb") as f:
#         merges = pickle.load(f)
#     return vocab, merges


def save_checkpoint(model, optimizer, iteration, out):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )


def load_checkpoint(src, model, optimizer):
    state_dict = torch.load(src)
    model.load_state_dict(state_dict["model"])
    optimizer.load_state_dict(state_dict["optimizer"])
    return state_dict["iteration"]
