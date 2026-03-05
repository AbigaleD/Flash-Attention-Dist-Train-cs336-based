import torch
import math

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