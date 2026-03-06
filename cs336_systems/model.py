import math

import torch.cuda.nvtx as nvtx

from cs336_basics.model import dropout_fn, softmax


@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(q, k, v, mask=None, dropout=0.0):
    with nvtx.range("computing attention scores"):
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    if mask is not None:
        att.masked_fill_(~mask, float("-inf"))
    with nvtx.range("softmax"):
        att = softmax(att)
    if dropout:
        att = dropout_fn(att, dropout)
    with nvtx.range("final matmul"):
        out = att @ v
    return out
