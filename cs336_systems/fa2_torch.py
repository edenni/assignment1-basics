import timeit

import torch
import triton
import triton.language as tl
from einops import rearrange


class FlashAttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B, H, N, D = Q.shape
        scale = D**-0.5
        B_q = 64
        B_kv = 64

        # placeholder
        O = torch.empty_like(Q)
        L = torch.empty((B, H, N))

        for b in range(B):
            for h in range(H):
                q = Q[b, h]
                k = K[b, h]
                v = V[b, h]

                for i in range(0, N, B_q):
                    q_tile = q[i : i + B_q]  # T * D
                    m_i = torch.full((q_tile.shape[0],), float("-inf"), device=q.device)  # T
                    l_i = torch.zeros((q_tile.shape[0],), device=q.device)
                    o_i = torch.zeros_like(q_tile)

                    for j in range(0, N, B_kv):
                        k_tile = k[j : j + B_kv]  # T * D
                        v_tile = v[j : j + B_kv]  # T * D

                        scores = torch.matmul(q_tile, k_tile.T) * scale  # T * T
                        m_ij = scores.max(dim=-1).values  # T
                        m_new = torch.maximum(m_i, m_ij)

                        exp_scores = torch.exp(scores - m_new[:, None])
                        l_i = torch.exp(m_i - m_new) * l_i + exp_scores.sum(dim=-1)
                        o_i = torch.exp(m_i - m_new)[:, None] * o_i + exp_scores @ v_tile
                        m_i = m_new
                    o_i /= l_i[:, None]
                    O[b, h, i : i + B_q] = o_i
                    L[b, h, i : i + B_q] = l_i
        ctx.save_for_backward(Q, K, V, O, L)
        return O

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Backward pass for FlashAttention is not implemented yet.")


x = torch.randn(4, 64, 16, 768, device="cuda", requires_grad=True)
output = FlashAttentionFunc.apply(x, x, x)
actual = torch.nn.functional.scaled_dot_product_attention(x, x, x)
torch.testing.assert_close(output, actual, atol=1e-5, rtol=1e-5)

custom_time = timeit.timeit(
    "FlashAttentionFunc.apply(x, x, x)",
    globals=globals(),
    number=1000,
)
torch_time = timeit.timeit(
    "torch.nn.functional.scaled_dot_product_attention(x, x, x)",
    globals=globals(),
    number=1000,
)
print(f"Custom FlashAttention time: {custom_time:.4f} seconds")
print(f"PyTorch FlashAttention time: {torch_time:.4f} seconds")
