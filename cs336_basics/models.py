import math

import torch
import torch.nn as nn


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

    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        reverse_rms = torch.rsqrt((x * x).mean(-1) + self.eps).unsqueeze(-1)
        out = x * reverse_rms * self.gain
        return out.to(in_dtype)


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
        nn.init.trunc_normal_(self.w1, std=std, a=-3 * std, b=3 * std)
        nn.init.trunc_normal_(self.w2, std=std, a=-3 * std, b=3 * std)
        nn.init.trunc_normal_(self.w3, std=std, a=-3 * std, b=3 * std)


if __name__ == "__main__":
    model = SwiGLU(16, 64)
    x = torch.randn(10, 16)
    print(model(x).shape)
