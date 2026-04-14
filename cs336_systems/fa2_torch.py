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
    stride_qh,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lh,
    stride_lq,
    N_QUERIES,
    N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    NH: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    bh = tl.program_id(1)
    batch_index = bh // NH
    head_index = bh % NH

    # Offset each pointer with the corresponding batch and head index
    # multiplied with the respective strides for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb + head_index * stride_qh,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb + head_index * stride_kh,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb + head_index * stride_vh,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob + head_index * stride_oh,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb + head_index * stride_lh,
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
    # q, k, v, o, do: [B, H, L, D]
    # l: [B, H, L]
    d = q.shape[-1]
    scale = d ** -0.5
    D = _rowsum(o, do)  # [B, H, L]
    s = q @ k.mT * scale # [B, H, L, L]
    p = torch.exp(s - l.unsqueeze(-1)) # [B, H, L, L]
    dv = p.mT @ do
    dp = do @ v.mT
    ds = p * (dp - D.unsqueeze(-1))  # [B, H, L, L]
    dq = ds @ k * scale
    dk = ds.mT @ q * scale
    return dq, dk, dv


@triton.jit
def flash_bwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    dO_ptr,
    L_ptr,
    D_ptr,
    dQ_ptr,
    dK_ptr,
    dV_ptr,
    stride_qb,
    stride_qh,
    stride_qn,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_on,
    stride_od,
    stride_dob,
    stride_doh,
    stride_don,
    stride_dod,
    stride_lb,
    stride_lh,
    stride_ln,
    stride_db,
    stride_dh,
    stride_dn,
    stride_dqb,
    stride_dqh,
    stride_dqn,
    stride_dqd,
    stride_dkb,
    stride_dkh,
    stride_dkn,
    stride_dkd,
    stride_dvb,
    stride_dvh,
    stride_dvn,
    stride_dvd,
    N,
    scale,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NH: tl.constexpr,
):
    pid_n = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // NH
    h = bh % NH

    # Load K(j), V(j)
    k_block_ptr = tl.make_block_ptr(
        K_ptr + b * stride_kb + h * stride_kh,
        shape=(N, D),
        strides=(stride_kn, stride_kd),
        offsets=(pid_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, D),
        order=(1, 0),
    )
    v_block_ptr = tl.make_block_ptr(
        V_ptr + b * stride_vb + h * stride_vh,
        shape=(N, D),
        strides=(stride_vn, stride_vd),
        offsets=(pid_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, D),
        order=(1, 0),
    )
    k = tl.load(k_block_ptr, boundary_check=(0, 1))
    v = tl.load(v_block_ptr, boundary_check=(0, 1))

    dk = tl.zeros((BLOCK_N, D), dtype=tl.float32)
    dv = tl.zeros((BLOCK_N, D), dtype=tl.float32)

    # Pointers for Q, dO, L, D tiles (will be advanced in loop)
    q_block_ptr = tl.make_block_ptr(
        Q_ptr + b * stride_qb + h * stride_qh,
        shape=(N, D),
        strides=(stride_qn, stride_qd),
        offsets=(0, 0),
        block_shape=(BLOCK_M, D),
        order=(1, 0),
    )
    do_block_ptr = tl.make_block_ptr(
        dO_ptr + b * stride_dob + h * stride_doh,
        shape=(N, D),
        strides=(stride_don, stride_dod),
        offsets=(0, 0),
        block_shape=(BLOCK_M, D),
        order=(1, 0),
    )
    l_block_ptr = tl.make_block_ptr(
        L_ptr + b * stride_lb + h * stride_lh,
        shape=(N,),
        strides=(stride_ln,),
        offsets=(0,),
        block_shape=(BLOCK_M,),
        order=(0,),
    )
    D_block_ptr = tl.make_block_ptr(
        D_ptr + b * stride_db + h * stride_dh,
        shape=(N,),
        strides=(stride_dn,),
        offsets=(0,),
        block_shape=(BLOCK_M,),
        order=(0,),
    )

    for i in range(0, N, BLOCK_M):
        q = tl.load(q_block_ptr, boundary_check=(0, 1))
        do = tl.load(do_block_ptr, boundary_check=(0, 1))
        l_tile = tl.load(l_block_ptr, boundary_check=(0,))
        D_tile = tl.load(D_block_ptr, boundary_check=(0,))

        # S_i^(j) and P_i^(j)
        qk = tl.dot(q, tl.trans(k), allow_tf32=False) * scale
        p = tl.exp(qk - l_tile[:, None])

        # dV(j) += P^T @ dO
        dv += tl.dot(tl.trans(p), do, allow_tf32=False)

        # dP and dS
        dp = tl.dot(do, tl.trans(v), allow_tf32=False)
        ds = p * (dp - D_tile[:, None]) * scale

        # Atomic add dQ(i)
        offs_m = i + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)
        dQ_ptrs = dQ_ptr + b * stride_dqb + h * stride_dqh + offs_m[:, None] * stride_dqn + offs_d[None, :] * stride_dqd
        dq = tl.dot(ds, k, allow_tf32=False)
        mask = (offs_m[:, None] < N) & (offs_d[None, :] < D)
        tl.atomic_add(dQ_ptrs, dq, mask=mask)

        # dK(j) += dS^T @ Q
        dk += tl.dot(tl.trans(ds), q, allow_tf32=False)

        q_block_ptr = q_block_ptr.advance((BLOCK_M, 0))
        do_block_ptr = do_block_ptr.advance((BLOCK_M, 0))
        l_block_ptr = l_block_ptr.advance((BLOCK_M,))
        D_block_ptr = D_block_ptr.advance((BLOCK_M,))

    # Store dK(j), dV(j)
    dk_block_ptr = tl.make_block_ptr(
        dK_ptr + b * stride_dkb + h * stride_dkh,
        shape=(N, D),
        strides=(stride_dkn, stride_dkd),
        offsets=(pid_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, D),
        order=(1, 0),
    )
    dv_block_ptr = tl.make_block_ptr(
        dV_ptr + b * stride_dvb + h * stride_dvh,
        shape=(N, D),
        strides=(stride_dvn, stride_dvd),
        offsets=(pid_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, D),
        order=(1, 0),
    )
    tl.store(dk_block_ptr, dk, boundary_check=(0, 1))
    tl.store(dv_block_ptr, dv, boundary_check=(0, 1))


class TritonFlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        assert Q.is_cuda and Q.is_contiguous()
        bs, H, Nq, d = Q.shape
        Nk = K.shape[2]
        scale = 1 / (d**0.5)

        ctx.Q_TILE_SIZE = 16
        ctx.K_TILE_SIZE = 16

        out_final = torch.empty(bs, H, Nq, d, device=Q.device, dtype=torch.float32)
        l_final = torch.empty(bs, H, Nq, device=Q.device, dtype=torch.float32)

        flash_fwd_kernel[(triton.cdiv(Nq, ctx.Q_TILE_SIZE), bs * H)](
            Q,
            K,
            V,
            out_final,
            l_final,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            Q.stride(3),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            K.stride(3),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            V.stride(3),
            out_final.stride(0),
            out_final.stride(1),
            out_final.stride(2),
            out_final.stride(3),
            l_final.stride(0),
            l_final.stride(1),
            l_final.stride(2),
            N_QUERIES=Nq,
            N_KEYS=Nk,
            scale=scale,
            D=d,
            Q_TILE_SIZE=ctx.Q_TILE_SIZE,
            K_TILE_SIZE=ctx.K_TILE_SIZE,
            NH=H,
        )
        # ctx.is_causal = is_causal
        ctx.save_for_backward(Q, K, V, out_final, l_final)
        return out_final

    @staticmethod
    def backward(ctx, do):
        Q, K, V, O, L = ctx.saved_tensors
        bs, H, N, d = Q.shape
        scale = 1 / (d**0.5)

        BLOCK_M = 16
        BLOCK_N = 16

        D = _rowsum(O, do)

        dQ = torch.zeros_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)

        flash_bwd_kernel[(triton.cdiv(N, BLOCK_N), bs * H)](
            Q, K, V, O, do, L, D, dQ, dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            D.stride(0), D.stride(1), D.stride(2),
            dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
            dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
            dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
            N=N, scale=scale,
            D=d, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, NH=H,
        )

        return dQ, dK, dV, None

def test_timing_flash_forward_backward():
    bs = 4
    n_heads = 16
    d_head = 64
    sequence_length = 8192
    q, k, v = torch.randn(
        3, bs, n_heads, sequence_length, d_head, device='cuda', dtype=torch.bfloat16, requires_grad=True
    )
    flash = torch.compile(TritonFlashAttnFunc.apply)


    def flash_forward_backward():
        o = flash(q, k, v, True)
        loss = o.sum()
        loss.backward()

    results = triton.testing.do_bench(flash_forward_backward, rep=100, warmup=100)
    print(results)


if __name__ == "__main__":
    x = torch.randn(4, 8, 32, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
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
    # test_timing_flash_forward_backward()

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
