import torch
import math

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

class MyFlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        B, N, d = q.shape
        Br = 32
        Bc = 32

        O = torch.zeros_like(q)
        l = torch.zeros((B, N, 1), device=q.device, dtype=q.dtype)
        m = torch.full((B, N, 1), -torch.inf, device=q.device, dtype=q.dtype)

        for i in range(0, N, Br):
            qi = q[:, i:i+Br, :]
            br = qi.shape[1]

            for j in range(0, N, Bc):
                if is_causal and j > i + br - 1:
                    break

                kj = k[:, j:j+Bc, :]
                vj = v[:, j:j+Bc, :]
                bc = kj.shape[1]

                S_ij = torch.matmul(qi, kj.transpose(-2, -1)) / math.sqrt(d)

                if is_causal:
                    row_idx = torch.arange(i, i + br, device=q.device).unsqueeze(1)
                    col_idx = torch.arange(j, j + bc, device=q.device).unsqueeze(0)
                    S_ij = S_ij.masked_fill(col_idx > row_idx, float('-inf'))

                m_ij = torch.max(S_ij, dim=-1, keepdim=True)[0]
                m_new = torch.maximum(m[:, i:i+Br, :], m_ij)

                P_ij = torch.exp(S_ij - m_new)

                # 关键：用 where 处理 m_i == -inf 的初始情况
                correction = torch.where(
                    m[:, i:i+Br, :] == -torch.inf,
                    torch.zeros_like(m[:, i:i+Br, :]),
                    torch.exp(m[:, i:i+Br, :] - m_new)
                )

                l[:, i:i+Br, :] = l[:, i:i+Br, :] * correction + P_ij.sum(dim=-1, keepdim=True)
                O[:, i:i+Br, :] = O[:, i:i+Br, :] * correction + torch.matmul(P_ij, vj)
                m[:, i:i+Br, :] = m_new

        O_final = O / l
        # L shape 必须是 (B, N)，测试会按这个 shape 查找
        L = (m + torch.log(l)).squeeze(-1)

        ctx.save_for_backward(q, k, v, O_final, L)
        ctx.is_causal = is_causal

        return O_final

    @staticmethod
    def backward(ctx, dO):
        q, k, v, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        B, N, d = q.shape
        Br = 32
        Bc = 32

        dQ = torch.zeros_like(q)
        dK = torch.zeros_like(k)
        dV = torch.zeros_like(v)

        D = (dO * O).sum(dim=-1, keepdim=True)  # (B, N, 1)

        for i in range(0, N, Br):
            qi = q[:, i:i+Br, :]
            doi = dO[:, i:i+Br, :]
            di = D[:, i:i+Br, :]
            li = L[:, i:i+Br]  # (B, br)
            br = qi.shape[1]

            for j in range(0, N, Bc):
                if is_causal and j > i + br - 1:
                    break

                kj = k[:, j:j+Bc, :]
                vj = v[:, j:j+Bc, :]
                bc = kj.shape[1]

                S_ij = torch.matmul(qi, kj.transpose(-2, -1)) / math.sqrt(d)

                if is_causal:
                    row_idx = torch.arange(i, i + br, device=q.device).unsqueeze(1)
                    col_idx = torch.arange(j, j + bc, device=q.device).unsqueeze(0)
                    S_ij = S_ij.masked_fill(col_idx > row_idx, float('-inf'))

                P_ij = torch.exp(S_ij - li.unsqueeze(-1))  # (B, br, bc)

                dV[:, j:j+Bc, :] += torch.matmul(P_ij.transpose(-2, -1), doi)

                dP_ij = torch.matmul(doi, vj.transpose(-2, -1))

                dS_ij = P_ij * (dP_ij - di) / math.sqrt(d)

                if is_causal:
                    dS_ij = dS_ij.masked_fill(col_idx > row_idx, 0.0)

                dQ[:, i:i+Br, :] += torch.matmul(dS_ij, kj)
                dK[:, j:j+Bc, :] += torch.matmul(dS_ij.transpose(-2, -1), qi)

        return dQ, dK, dV, None


if triton is not None:

    @triton.jit
    def _flash_attention_forward_kernel(
        q,
        k,
        v,
        o,
        lse,
        stride_qb: tl.constexpr,
        stride_qq: tl.constexpr,
        stride_qd: tl.constexpr,
        stride_kb: tl.constexpr,
        stride_kk: tl.constexpr,
        stride_kd: tl.constexpr,
        stride_vb: tl.constexpr,
        stride_vk: tl.constexpr,
        stride_vd: tl.constexpr,
        stride_ob: tl.constexpr,
        stride_oq: tl.constexpr,
        stride_od: tl.constexpr,
        stride_lb: tl.constexpr,
        stride_lq: tl.constexpr,
        n_queries: tl.constexpr,
        n_keys: tl.constexpr,
        head_dim: tl.constexpr,
        scale: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        is_causal: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        query_block_idx = tl.program_id(1)

        offs_m = query_block_idx * block_m + tl.arange(0, block_m)
        offs_n = tl.arange(0, block_n)
        offs_d = tl.arange(0, head_dim)

        q_block = tl.load(
            q + batch_idx * stride_qb + offs_m[:, None] * stride_qq + offs_d[None, :] * stride_qd,
            mask=offs_m[:, None] < n_queries,
            other=0.0,
        )

        m_i = tl.full([block_m], -float("inf"), tl.float32)
        l_i = tl.zeros([block_m], tl.float32)
        acc = tl.zeros([block_m, head_dim], tl.float32)

        for start_n in range(0, n_keys, block_n):
            key_offsets = start_n + offs_n
            k_block = tl.load(
                k + batch_idx * stride_kb + key_offsets[:, None] * stride_kk + offs_d[None, :] * stride_kd,
                mask=key_offsets[:, None] < n_keys,
                other=0.0,
            )
            v_block = tl.load(
                v + batch_idx * stride_vb + key_offsets[:, None] * stride_vk + offs_d[None, :] * stride_vd,
                mask=key_offsets[:, None] < n_keys,
                other=0.0,
            )

            scores = tl.dot(q_block, tl.trans(k_block)) * scale
            mask = key_offsets[None, :] < n_keys
            if is_causal:
                mask = mask & (key_offsets[None, :] <= offs_m[:, None])
            scores = tl.where(mask, scores, -float("inf"))

            m_ij = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            p = tl.exp(scores - m_new[:, None])
            alpha = tl.exp(m_i - m_new)

            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p, v_block)
            m_i = m_new

        acc = acc / l_i[:, None]
        lse_block = m_i + tl.log(l_i)

        tl.store(
            o + batch_idx * stride_ob + offs_m[:, None] * stride_oq + offs_d[None, :] * stride_od,
            acc,
            mask=offs_m[:, None] < n_queries,
        )
        tl.store(
            lse + batch_idx * stride_lb + offs_m * stride_lq,
            lse_block,
            mask=offs_m < n_queries,
        )


    @triton.jit
    def _flash_attention_backward_kernel(
        q,
        k,
        v,
        o,
        lse,
        do,
        dq,
        dk,
        dv,
        stride_qb: tl.constexpr,
        stride_qq: tl.constexpr,
        stride_qd: tl.constexpr,
        stride_kb: tl.constexpr,
        stride_kk: tl.constexpr,
        stride_kd: tl.constexpr,
        stride_vb: tl.constexpr,
        stride_vk: tl.constexpr,
        stride_vd: tl.constexpr,
        stride_ob: tl.constexpr,
        stride_oq: tl.constexpr,
        stride_od: tl.constexpr,
        stride_lb: tl.constexpr,
        stride_lq: tl.constexpr,
        stride_dob: tl.constexpr,
        stride_doq: tl.constexpr,
        stride_dod: tl.constexpr,
        stride_dqb: tl.constexpr,
        stride_dqq: tl.constexpr,
        stride_dqd: tl.constexpr,
        stride_dkb: tl.constexpr,
        stride_dkk: tl.constexpr,
        stride_dkd: tl.constexpr,
        stride_dvb: tl.constexpr,
        stride_dvk: tl.constexpr,
        stride_dvd: tl.constexpr,
        n_queries: tl.constexpr,
        n_keys: tl.constexpr,
        head_dim: tl.constexpr,
        scale: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        is_causal: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        query_block_idx = tl.program_id(1)

        offs_m = query_block_idx * block_m + tl.arange(0, block_m)
        offs_n = tl.arange(0, block_n)
        offs_d = tl.arange(0, head_dim)

        q_block = tl.load(
            q + batch_idx * stride_qb + offs_m[:, None] * stride_qq + offs_d[None, :] * stride_qd,
            mask=offs_m[:, None] < n_queries,
            other=0.0,
        )
        do_block = tl.load(
            do + batch_idx * stride_dob + offs_m[:, None] * stride_doq + offs_d[None, :] * stride_dod,
            mask=offs_m[:, None] < n_queries,
            other=0.0,
        )
        o_block = tl.load(
            o + batch_idx * stride_ob + offs_m[:, None] * stride_oq + offs_d[None, :] * stride_od,
            mask=offs_m[:, None] < n_queries,
            other=0.0,
        )
        lse_block = tl.load(
            lse + batch_idx * stride_lb + offs_m * stride_lq,
            mask=offs_m < n_queries,
            other=0.0,
        )

        d_i = tl.sum(do_block * o_block, axis=1)
        dq_acc = tl.zeros([block_m, head_dim], tl.float32)

        for start_n in range(0, n_keys, block_n):
            key_offsets = start_n + offs_n
            k_block = tl.load(
                k + batch_idx * stride_kb + key_offsets[:, None] * stride_kk + offs_d[None, :] * stride_kd,
                mask=key_offsets[:, None] < n_keys,
                other=0.0,
            )
            v_block = tl.load(
                v + batch_idx * stride_vb + key_offsets[:, None] * stride_vk + offs_d[None, :] * stride_vd,
                mask=key_offsets[:, None] < n_keys,
                other=0.0,
            )

            scores = tl.dot(q_block, tl.trans(k_block)) * scale
            mask = (offs_m[:, None] < n_queries) & (key_offsets[None, :] < n_keys)
            if is_causal:
                mask = mask & (key_offsets[None, :] <= offs_m[:, None])

            p = tl.exp(tl.where(mask, scores - lse_block[:, None], -float("inf")))
            dp = tl.dot(do_block, tl.trans(v_block))
            ds = p * (dp - d_i[:, None]) * scale
            ds = tl.where(mask, ds, 0.0)

            dq_acc += tl.dot(ds, k_block)
            dk_update = tl.dot(tl.trans(ds), q_block)
            dv_update = tl.dot(tl.trans(p), do_block)

            tl.atomic_add(
                dk + batch_idx * stride_dkb + key_offsets[:, None] * stride_dkk + offs_d[None, :] * stride_dkd,
                dk_update,
                mask=key_offsets[:, None] < n_keys,
                sem="relaxed",
            )
            tl.atomic_add(
                dv + batch_idx * stride_dvb + key_offsets[:, None] * stride_dvk + offs_d[None, :] * stride_dvd,
                dv_update,
                mask=key_offsets[:, None] < n_keys,
                sem="relaxed",
            )

        tl.store(
            dq + batch_idx * stride_dqb + offs_m[:, None] * stride_dqq + offs_d[None, :] * stride_dqd,
            dq_acc,
            mask=offs_m[:, None] < n_queries,
        )


class MyFlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        if triton is None or not q.is_cuda:
            return MyFlashAttentionPytorch.forward(ctx, q, k, v, is_causal)

        B, Nq, d = q.shape
        _, Nk, _ = k.shape
        block_m = 32
        block_n = 32

        O = torch.empty_like(q)
        L = torch.empty((B, Nq), device=q.device, dtype=torch.float32)

        grid = (B, triton.cdiv(Nq, block_m))
        _flash_attention_forward_kernel[grid](
            q,
            k,
            v,
            O,
            L,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            O.stride(0),
            O.stride(1),
            O.stride(2),
            L.stride(0),
            L.stride(1),
            Nq,
            Nk,
            d,
            1.0 / math.sqrt(d),
            block_m,
            block_n,
            is_causal,
        )

        ctx.save_for_backward(q, k, v, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        q, k, v, O, L = ctx.saved_tensors
        if triton is None or not dO.is_cuda:
            return MyFlashAttentionPytorch.backward(ctx, dO)

        B, Nq, d = q.shape
        _, Nk, _ = k.shape
        block_m = 32
        block_n = 32

        dQ = torch.empty_like(q)
        dK = torch.zeros_like(k)
        dV = torch.zeros_like(v)

        grid = (B, triton.cdiv(Nq, block_m))
        _flash_attention_backward_kernel[grid](
            q,
            k,
            v,
            O,
            L,
            dO,
            dQ,
            dK,
            dV,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            O.stride(0),
            O.stride(1),
            O.stride(2),
            L.stride(0),
            L.stride(1),
            dO.stride(0),
            dO.stride(1),
            dO.stride(2),
            dQ.stride(0),
            dQ.stride(1),
            dQ.stride(2),
            dK.stride(0),
            dK.stride(1),
            dK.stride(2),
            dV.stride(0),
            dV.stride(1),
            dV.stride(2),
            Nq,
            Nk,
            d,
            1.0 / math.sqrt(d),
            block_m,
            block_n,
            ctx.is_causal,
        )

        return dQ, dK, dV, None
