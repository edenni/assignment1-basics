import math

import torch
import torch.nn as nn
from einops import rearrange, repeat


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        self.reset_parameters()

    def forward(self, x):
        return x @ self.W.T

    def reset_parameters(self):
        std = math.sqrt(2 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.W, std=std, a=-3 * std, b=3 * std)


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embed = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        self.reset_parameters()

    def forward(self, x):
        return torch.index_select(self.embed, 0, x.reshape(-1)).view(*x.size(), -1)

    def reset_parameters(self):
        std = math.sqrt(2 / (self.num_embeddings + self.embedding_dim))
        nn.init.trunc_normal_(self.embed, std=std, a=-3 * std, b=3 * std)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    @torch.autocast("cuda", enabled=False)
    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        reverse_rms = torch.rsqrt((x * x).mean(-1) + self.eps).unsqueeze(-1)
        out = x * reverse_rms * self.gain
        return out.type(in_dtype)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        self.w2 = nn.Parameter(torch.empty(d_model, d_ff, device=device, dtype=dtype))
        self.w3 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        self.swish = Swish()
        self.reset_parameters()

    def forward(self, x):
        y1 = self.swish(x @ self.w1.T)
        y2 = x @ self.w3.T
        y = y1 * y2
        return y @ self.w2.T

    def reset_parameters(self):
        std = math.sqrt(2 / (self.d_model + self.d_ff))
        for w in (self.w1, self.w2, self.w3):
            nn.init.trunc_normal_(w, std=std, a=-3 * std, b=3 * std)


def rotate_pair(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


# https://kexue.fm/archives/8265
# https://github.com/lucidrains/rotary-embedding-torch/blob/main/rotary_embedding_torch/rotary_embedding_torch.py
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        theta = 1.0 / (theta ** (torch.arange(0, d_k, 2) / d_k))
        i = torch.arange(max_seq_len)
        theta = torch.outer(i, theta)
        self.register_buffer("theta", theta, persistent=False)

    @torch.autocast("cuda", enabled=False)
    def forward(self, x, token_positions):
        in_dtype = x.dtype
        if token_positions is not None:
            theta = self.theta[token_positions]  # seq_len d_k // 2
        else:
            theta = self.theta[: x.size(-2)]
        theta = repeat(theta, "... n -> ... (n r)", r=2)
        x = x * theta.cos() + rotate_pair(x) * theta.sin()
        return x.type(in_dtype)


def softmax(x, dim=-1):
    o = x - x.max(dim=dim, keepdim=True)[0]
    return o.exp() / o.exp().sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(q, k, v, mask=None):
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    if mask is not None:
        att.masked_fill_(~mask, float("-inf"))
    return softmax(att) @ v


if __name__ == "__main__":
    d = 64
    max_seq_len = 128
    theta_base = 10

    x = torch.arange(24).reshape(3, 8)

    print(softmax(x).sum(dim=1))
