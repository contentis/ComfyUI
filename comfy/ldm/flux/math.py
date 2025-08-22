import torch
from einops import rearrange
from torch import Tensor

from comfy.ldm.modules.attention import optimized_attention
import comfy.model_management
from comfy.ops import compile_decorator

def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, mask=None) -> Tensor:
    q_shape = q.shape
    k_shape = k.shape

    if pe is not None:
        q = q.reshape(*q.shape[:-1], -1, 1, 2)
        k = k.reshape(*k.shape[:-1], -1, 1, 2)
        q, k = pe_emb(pe[..., 0], pe[..., 1], q[..., 0], q[..., 1],  k[..., 0],  k[..., 1])
        q = q.reshape(*q_shape)
        k = k.reshape(*k_shape)
    heads = q.shape[1]
    x = optimized_attention(q, k, v, heads, skip_reshape=True, mask=mask)
    return x

def pe_emb(pe0: Tensor, pe1: Tensor, q0: Tensor, q1: Tensor, k0: Tensor, k1: Tensor):
    # Knowing the dtype ahead of time allows for better fusion when scripting.
    @compile_decorator
    def _pe_emb_dyn(pe0: Tensor, pe1: Tensor, q0: Tensor, q1: Tensor, k0: Tensor, k1: Tensor):
        pe_dtype = pe0.dtype
        qk_dtype = q0.dtype
        q = (pe0 * q0.to(dtype=pe_dtype) + pe1 * q1.to(dtype=pe_dtype)).to(dtype=qk_dtype)
        k = (pe0 * k0.to(dtype=pe_dtype) + pe1 * k1.to(dtype=pe_dtype)).to(dtype=qk_dtype)
        return q, k
    if pe0.dtype == torch.float32 and q0.dtype == torch.bfloat16:
        return _pe_emb(pe0, pe1, q0, q1, k0, k1)
    return _pe_emb_dyn(pe0, pe1, q0, q1, k0, k1)

@compile_decorator
def _pe_emb(pe0: Tensor, pe1: Tensor, q0: Tensor, q1: Tensor, k0: Tensor, k1: Tensor):
    q = (pe0 * q0.to(dtype=torch.float32) + pe1 * q1.to(dtype=torch.float32)).to(dtype=torch.bfloat16)
    k = (pe0 * k0.to(dtype=torch.float32) + pe1 * k1.to(dtype=torch.float32)).to(dtype=torch.bfloat16)
    return q, k

def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    if comfy.model_management.is_device_mps(pos.device) or comfy.model_management.is_intel_xpu() or comfy.model_management.is_directml_enabled():
        device = torch.device("cpu")
    else:
        device = pos.device

    scale = torch.linspace(0, (dim - 2) / dim, steps=dim//2, dtype=torch.float64, device=device)
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos.to(dtype=torch.float32, device=device), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.to(dtype=torch.float32, device=pos.device)

@compile_decorator
def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor):
    q_shape = list(xq.size())
    k_shape = list(xk.size())
    pairs_q = q_shape[-1] // 2
    pairs_k = k_shape[-1] // 2

    freqs = freqs_cis  # shape (..., P, 2, 2) where P is pairs
    xq_vec = xq.to(freqs.dtype).reshape(q_shape[:-1] + [pairs_q, 2])  # (..., P, 2)
    xk_vec = xk.to(freqs.dtype).reshape(k_shape[:-1] + [pairs_k, 2])  # (..., P, 2)

    xq_out = torch.matmul(freqs, xq_vec.unsqueeze(-1)).squeeze(-1)  # (..., P, 2)
    xk_out = torch.matmul(freqs, xk_vec.unsqueeze(-1)).squeeze(-1)  # (..., P, 2)

    xq_out = xq_out.reshape(q_shape).to(dtype=xq.dtype)
    xk_out = xk_out.reshape(k_shape).to(dtype=xk.dtype)
    return xq_out, xk_out

