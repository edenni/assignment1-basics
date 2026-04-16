import math

import torch
import torch.nn as nn
from einops import rearrange, repeat
from tqdm import tqdm

from cs336_systems.fa2_torch import TritonFlashAttnFunc


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        self.reset_parameters()

    def forward(self, x):
        return x @ self.weight.T

    def reset_parameters(self):
        std = math.sqrt(2 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weight, std=std, a=-3 * std, b=3 * std)


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


def dropout_fn(x, p: float):
    mask = torch.bernoulli(torch.ones_like(x) * (1 - p))
    return x * mask / (1 - p)


class Dropout(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        assert 0 <= p < 1
        self.p = p

    def forward(self, x):
        if self.training:
            x = dropout_fn(x, self.p)
        return x


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
    def forward(self, x, token_positions=None):
        in_dtype = x.dtype
        if token_positions is not None:
            theta = self.theta[token_positions]  # seq_len d_k // 2
        else:
            theta = self.theta[: x.size(-2)]
        theta = repeat(theta, "... n -> ... (n r)", r=2)
        x = x * theta.cos() + rotate_pair(x) * theta.sin()
        return x.type(in_dtype)


def softmax(x, dim=-1, t=1.0):
    o = x - x.max(dim=dim, keepdim=True)[0]
    assert t > 0, "temperature must be greater than 0"
    o /= t
    return o.exp() / o.exp().sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(q, k, v, mask=None, dropout=0.0):
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    if mask is not None:
        att.masked_fill_(~mask, float("-inf"))
    att = softmax(att)
    if dropout:
        att = dropout_fn(att, dropout)
    return att @ v


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads=None, theta=10000, max_seq_len=8192, dropout=0.1, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        assert d_model % num_heads == 0
        assert num_heads % self.num_kv_heads == 0
        self.d_head = d_model // num_heads
        self.num_groups = num_heads // self.num_kv_heads  # Q heads per KV head

        self.q_proj = Linear(d_model, num_heads * self.d_head, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, self.num_kv_heads * self.d_head, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, self.num_kv_heads * self.d_head, device=device, dtype=dtype)
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        if theta > 0:
            self.rope = RotaryPositionalEmbedding(theta, self.d_head, max_seq_len, device=device)
        else:
            self.rope = None
        self.dropout = dropout

    def forward(self, x, token_positions=None, kv_cache=None):
        B, L, D = x.size()
        q = self.q_proj(x).view(B, L, self.num_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.d_head).transpose(1, 2)

        if self.rope:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # append to KV cache if provided
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)
        new_kv_cache = (k, v)

        # expand KV heads to match Q heads: (B, num_kv_heads, S, d) -> (B, num_heads, S, d)
        S = k.size(2)
        k_exp, v_exp = k, v
        if self.num_groups > 1:
            k_exp = k.unsqueeze(2).expand(-1, -1, self.num_groups, -1, -1).reshape(B, self.num_heads, S, self.d_head)
            v_exp = v.unsqueeze(2).expand(-1, -1, self.num_groups, -1, -1).reshape(B, self.num_heads, S, self.d_head)

        if kv_cache is not None:
            # single-token decode: use simple matmul attention (flash attention overhead not worth it for L=1)
            att = (q @ k_exp.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
            y = torch.softmax(att, dim=-1) @ v_exp
        else:
            y = TritonFlashAttnFunc.apply(q, k_exp, v_exp, True)

        y = y.transpose(1, 2).contiguous().view(B, L, D)
        o = self.o_proj(y)
        return o, new_kv_cache


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        num_kv_heads: int | None = None,
        dropout: float = 0.1,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads, num_kv_heads, theta, max_seq_len, dropout, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.do1 = Dropout(dropout)
        self.do2 = Dropout(dropout)

    def forward(self, x, token_positions=None, kv_cache=None):
        y = self.ln1(x)
        y, new_kv_cache = self.attn(y, token_positions, kv_cache)
        y = self.do1(y)
        x = x + y
        y = self.ln2(x)
        y = self.ffn(y)
        y = self.do2(y)
        x = x + y
        return x, new_kv_cache


class Transformer(nn.Module):
    def __init__(
        self,
        num_layers: int,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float = 10_000,
        num_kv_heads: int | None = None,
        dropout: float = 0.1,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, context_length, theta, num_kv_heads, dropout, device=device, dtype=dtype)
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

        # Pre-layer scaler
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(num_layers, device=device))
        self.x0_lambdas = nn.Parameter(torch.ones(num_layers, device=device))
        for i in range(num_layers):
            self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(num_layers - 1, 1))
        # Decaying x0 init: earlier layers get more input embedding blending
        for i in range(num_layers):
            self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(num_layers - 1, 1))

    def _forward(self, x, token_positions=None, kv_caches=None):
        x = self.token_embeddings(x)
        new_kv_caches = []
        x0 = x
        for i, layer in enumerate(self.layers):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = layer(x, token_positions, cache)
            new_kv_caches.append(new_cache)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x, new_kv_caches

    def forward(self, x, token_positions=None, kv_caches=None):
        return self._forward(x, token_positions, kv_caches)[0]

    def _sample_token(self, logits, top_p, t):
        if t == 0:
            return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        probs = torch.softmax(logits[:, -1, :] / t, dim=-1)
        if top_p < 1.0:
            sorted_values, sorted_idx = probs.sort(-1, descending=True)
            mask = sorted_values.cumsum(-1) <= top_p
            mask[:, 0] = True
            orig_mask = mask.gather(-1, sorted_idx.argsort(-1))
            probs.masked_fill_(~orig_mask, 0.0)
            probs /= probs.sum(-1, keepdim=True)
        return torch.multinomial(probs, 1)

    @torch.inference_mode()
    def generate(
        self,
        prompt,
        eos_token_id: int,
        top_p: float = 1.0,
        t: float = 1.0,
        max_steps: int = 32,
    ):
        input_seq = prompt

        # prefill: process entire prompt, build KV cache
        logits, kv_caches = self._forward(input_seq)
        out = self._sample_token(logits, top_p, t)
        input_seq = torch.cat([input_seq, out], dim=-1)

        # decode: one token at a time with KV cache
        for _ in tqdm(range(max_steps - 1)):
            if (out[-1:] == eos_token_id).all(dim=-1).item():
                break
            seq_len = input_seq.size(1) - 1
            pos = torch.tensor([[seq_len]], device=input_seq.device)
            logits, kv_caches = self._forward(out, token_positions=pos, kv_caches=kv_caches)
            out = self._sample_token(logits, top_p, t)
            input_seq = torch.cat([input_seq, out], dim=-1)
        return input_seq


def cross_entropy(inputs, targets):
    o = inputs - inputs.max(dim=-1, keepdim=True)[0]
    log_softmax = o - torch.logsumexp(o, dim=-1, keepdim=True)
    target_loss = log_softmax.gather(dim=-1, index=targets.unsqueeze(-1))
    return -target_loss.mean()


if __name__ == "__main__":
    d = 64
    max_seq_len = 128
    theta_base = 10

    x = torch.arange(24).reshape(3, 8)

    print(softmax(x).sum(dim=1))
