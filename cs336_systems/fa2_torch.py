import torch
import triton
import triton.language as tl


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


@triton.jit
def flash_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    N_QUERIES,
    N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    q_tile = tl.load(Q_block_ptr, boundary_check=(0, 1))  # [Q_TILE_SIZE, D]
    o_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)  # [Q_TILE_SIZE, D]
    l_i = tl.zeros((q_tile.shape[0],), dtype=tl.float32)  # [Q_TILE_SIZE]
    m_i = tl.full((q_tile.shape[0],), float("-inf"), dtype=tl.float32)  # [Q_TILE_SIZE]

    for _ in range(0, N_KEYS, K_TILE_SIZE):
        k_tile = tl.load(K_block_ptr, boundary_check=(0, 1))  # [K_TILE_SIZE, D]
        v_tile = tl.load(V_block_ptr, boundary_check=(0, 1))  # [K_TILE_SIZE, D]

        scores = tl.dot(q_tile, tl.trans(k_tile), allow_tf32=False) * scale  # [Q_TILE_SIZE, K_TILE_SIZE]
        m_ij = tl.maximum(m_i, scores.max(axis=1))  # [Q_TILE_SIZE]

        exp_scores = tl.exp(scores - m_ij[:, None])  # [Q_TILE_SIZE, K_TILE_SIZE]
        l_i = tl.exp(m_i - m_ij) * l_i + tl.sum(exp_scores, axis=1)  # [Q_TILE_SIZE]
        o_i = tl.exp(m_i - m_ij)[:, None] * o_i + tl.dot(exp_scores, v_tile, allow_tf32=False)  # [Q_TILE_SIZE, D]
        m_i = m_ij

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    o_i /= l_i[:, None]  # Store the output and the normalizer

    tl.store(O_block_ptr, o_i, boundary_check=(0, 1))
    tl.store(L_block_ptr, m_i + tl.math.log(l_i), boundary_check=(0,))


def _rowsum(x, y):
    return torch.sum(x * y, dim=-1)

def _flash_bwd_kernel_torch(q, k, v, o, do, l):
    # q, k, v, o, do: [B, L, D]
    # l: [B, L]
    d = q.shape[-1]
    scale = d ** -0.5
    D = _rowsum(o, do)  # [B, L]
    s = q @ k.mT * scale # [B, L, L]
    p = torch.exp(s - l.unsqueeze(-1)) # [B, L, L]
    dv = p.mT @ do
    dp = do @ v.mT
    ds = p * (dp - D.unsqueeze(-1))  # [B, L, L]
    dq = ds @ k * scale
    dk = ds.mT @ q * scale
    return dq, dk, dv


class TritonFlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        assert Q.is_cuda and Q.is_contiguous()
        bs, Nq, d = Q.shape
        Nk = K.shape[1]
        scale = 1 / (d**0.5)

        ctx.Q_TILE_SIZE = 16
        ctx.K_TILE_SIZE = 16

        out_final = torch.empty(bs, Nq, d, device=Q.device, dtype=torch.float32)
        l_final = torch.empty(bs, Nq, device=Q.device, dtype=torch.float32)

        flash_fwd_kernel[(triton.cdiv(Nq, ctx.Q_TILE_SIZE), bs)](
            Q,
            K,
            V,
            out_final,
            l_final,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            out_final.stride(0),
            out_final.stride(1),
            out_final.stride(2),
            l_final.stride(0),
            l_final.stride(1),
            N_QUERIES=Nq,
            N_KEYS=Nk,
            scale=scale,
            D=d,
            Q_TILE_SIZE=ctx.Q_TILE_SIZE,
            K_TILE_SIZE=ctx.K_TILE_SIZE,
        )
        # ctx.is_causal = is_causal
        ctx.save_for_backward(Q, K, V, out_final, l_final)
        return out_final

    @staticmethod
    def backward(ctx, do):
        Q, K, V, O, L = ctx.saved_tensors
        dq, dk, dv = _flash_bwd_kernel_torch(Q, K, V, O, do, L)
        return dq, dk, dv, None

if __name__ == "__main__":
    x = torch.randn(4, 32, 128, device="cuda", dtype=torch.float32, requires_grad=True)
    output = TritonFlashAttnFunc.apply(x, x, x)
    actual = torch.nn.functional.scaled_dot_product_attention(x, x, x)
    torch.testing.assert_close(output, actual, atol=1e-5, rtol=1e-5)

    loss = output.sum()
    loss.backward()
    x_grad = x.grad.clone()
    x.grad.zero_()
    loss = actual.sum()
    loss.backward()
    torch.testing.assert_close(x.grad, x_grad, atol=1e-5, rtol=1e-5)

    # output = FlashAttentionFunc.apply(x, x, x)
    # actual = torch.nn.functional.scaled_dot_product_attention(x, x, x)
    # torch.testing.assert_close(output, actual, atol=1e-5, rtol=1e-5)

    # custom_time = timeit.timeit(
    #     "FlashAttentionFunc.apply(x, x, x)",
    #     globals=globals(),
    #     number=100,
    # )
    # torch_time = timeit.timeit(
    #     "torch.nn.functional.scaled_dot_product_attention(x, x, x)",
    #     globals=globals(),
    #     number=100,
    # )
    # print(f"Custom FlashAttention time: {custom_time:.4f} seconds")
    # print(f"PyTorch FlashAttention time: {torch_time:.4f} seconds")
