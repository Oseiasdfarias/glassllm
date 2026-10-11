"""Independent CPU PyTorch equations used only as numerical test oracles."""
import numpy as np
import torch
import torch.nn.functional as F


def rmsnorm(x, gamma, eps):
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps) * gamma


def rope_angles(d_k, max_len, base):
    # Derive frequencies here: never import the production angle builder.
    exponents = torch.arange(0, d_k, 2, dtype=torch.float64) / d_k
    frequencies = torch.pow(torch.tensor(base, dtype=torch.float64), -exponents)
    angles = torch.arange(max_len, dtype=torch.float64)[:, None] * frequencies[None, :]
    return angles.cos(), angles.sin()


def rope(x, cos, sin):
    even, odd = x[..., ::2], x[..., 1::2]
    pairs = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return pairs.flatten(-2)


def to_torch(module):
    return {name: torch.tensor(p.data, dtype=torch.float64, requires_grad=True)
            for name, p in module.named_parameters().items()}


def attention(params, x, cfg, start=0):
    batch, length, _ = x.shape
    width = cfg.d_model // cfg.n_heads
    q = (x @ params["W_Q"]).reshape(batch, length, cfg.n_heads, width).transpose(1, 2)
    k = (x @ params["W_K"]).reshape(batch, length, cfg.n_kv_heads, width).transpose(1, 2)
    v = (x @ params["W_V"]).reshape(batch, length, cfg.n_kv_heads, width).transpose(1, 2)
    cos, sin = rope_angles(width, cfg.max_len, cfg.rope_base)
    q = rope(q, cos[start:start + length], sin[start:start + length])
    k = rope(k, cos[start:start + length], sin[start:start + length])
    repeats = cfg.n_heads // cfg.n_kv_heads
    k = k.repeat_interleave(repeats, dim=1)
    v = v.repeat_interleave(repeats, dim=1)
    mixed = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return mixed.transpose(1, 2).reshape(batch, length, cfg.d_model) @ params["W_O"]


def swiglu(params, x):
    return (F.silu(x @ params["W1"]) * (x @ params["W3"])) @ params["W2"]


def block(params, x, cfg, start=0):
    attn = {name.removeprefix("attn."): p for name, p in params.items() if name.startswith("attn.")}
    ffn = {name.removeprefix("ffn."): p for name, p in params.items() if name.startswith("ffn.")}
    x = x + attention(attn, rmsnorm(x, params["norm1.gamma"], cfg.eps), cfg, start)
    return x + swiglu(ffn, rmsnorm(x, params["norm2.gamma"], cfg.eps))


def forward(params, ids, cfg, *, input_embedding=None, output_embedding=None):
    """Whole decoder, with optional separate embeddings to test each tied path."""
    ids = torch.as_tensor(np.atleast_2d(ids), dtype=torch.long)
    x = (params["E"] if input_embedding is None else input_embedding)[ids]
    for i in range(cfg.n_layers):
        prefix = f"blocks.{i}."
        layer_params = {name.removeprefix(prefix): p for name, p in params.items()
                        if name.startswith(prefix)}
        x = block(layer_params, x, cfg)
    h = rmsnorm(x, params["norm_f.gamma"], cfg.eps)
    return h @ (params["E"] if output_embedding is None else output_embedding).T
