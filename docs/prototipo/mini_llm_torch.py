"""
Mini LLM do zero: cada parte corresponde a uma equacao da explicacao.

    Embedding -> [ RMSNorm -> Atencao causal (RoPE, GQA) -> soma residual
                   RMSNorm -> FFN SwiGLU               -> soma residual ] x L
              -> RMSNorm -> logits -> softmax / entropia cruzada

Rodar:  python mini_llm.py
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 64
    d_model: int = 128      # d
    n_layers: int = 4       # L
    n_heads: int = 4        # H
    n_kv_heads: int = 2     # GQA: cabecas que tem K e V proprios (= n_heads vira MHA)
    max_len: int = 128      # maior contexto suportado
    rope_base: float = 10000.0
    eps: float = 1e-6


# ---------------------------------------------------------------- 6. RMSNorm
class RMSNorm(nn.Module):
    # RMSNorm(x) = gamma * x / sqrt(mean(x^2) + eps)
    def __init__(self, d, eps):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * x / rms


# ---------------------------------------------------------------- 3b. RoPE
def rope_angles(d_k, max_len, base):
    # theta_i = base^(-2i/d_k), angulo da posicao m no par i = m * theta_i
    i = torch.arange(0, d_k // 2, dtype=torch.float32)
    theta = base ** (-2 * i / d_k)                        # (d_k/2,)
    m = torch.arange(max_len, dtype=torch.float32)        # (max_len,)
    ang = torch.outer(m, theta)                           # (max_len, d_k/2)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    # x: (B, heads, n, d_k). Gira cada par (x_2i, x_2i+1) pelo angulo m*theta_i
    n = x.shape[-2]
    cos, sin = cos[:n], sin[:n]
    x_par, x_impar = x[..., 0::2], x[..., 1::2]
    y_par = x_par * cos - x_impar * sin
    y_impar = x_par * sin + x_impar * cos
    return torch.stack([y_par, y_impar], dim=-1).flatten(-2)


# ---------------------------------------------------------------- 3 e 4. Atencao
def attention(q, k, v):
    # softmax(Q K^T / sqrt(d_k) + M) V, com M = -inf acima da diagonal
    d_k = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)     # (B, H, n, n)
    n = scores.shape[-1]
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=q.device), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)                   # cada linha soma 1
    return weights @ v


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.H, self.H_kv = cfg.n_heads, cfg.n_kv_heads
        self.d_k = cfg.d_model // cfg.n_heads
        self.W_Q = nn.Linear(cfg.d_model, self.H * self.d_k, bias=False)
        self.W_K = nn.Linear(cfg.d_model, self.H_kv * self.d_k, bias=False)
        self.W_V = nn.Linear(cfg.d_model, self.H_kv * self.d_k, bias=False)
        self.W_O = nn.Linear(self.H * self.d_k, cfg.d_model, bias=False)
        cos, sin = rope_angles(self.d_k, cfg.max_len, cfg.rope_base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x):
        B, n, _ = x.shape
        # projeta e separa em cabecas: (B, n, H*d_k) -> (B, H, n, d_k)
        q = self.W_Q(x).view(B, n, self.H, self.d_k).transpose(1, 2)
        k = self.W_K(x).view(B, n, self.H_kv, self.d_k).transpose(1, 2)
        v = self.W_V(x).view(B, n, self.H_kv, self.d_k).transpose(1, 2)
        # RoPE so em Q e K (V nao participa do produto escalar)
        q = apply_rope(q, self.cos, self.sin)
        k = apply_rope(k, self.cos, self.sin)
        # GQA: cada grupo de H/H_kv cabecas de Q usa o mesmo K e V
        rep = self.H // self.H_kv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        heads = attention(q, k, v)                        # (B, H, n, d_k)
        concat = heads.transpose(1, 2).reshape(B, n, self.H * self.d_k)
        return self.W_O(concat)


# ---------------------------------------------------------------- 5. FFN SwiGLU
class SwiGLU(nn.Module):
    # FFN(x) = W2( SiLU(W1 x) * (W3 x) ), com h = 8d/3 para ter ~8d^2 parametros
    def __init__(self, d):
        super().__init__()
        h = int(8 * d / 3)
        self.W1 = nn.Linear(d, h, bias=False)
        self.W3 = nn.Linear(d, h, bias=False)
        self.W2 = nn.Linear(h, d, bias=False)

    def forward(self, x):
        return self.W2(F.silu(self.W1(x)) * self.W3(x))


# ---------------------------------------------------------------- 2. Bloco
class Block(nn.Module):
    # H = X + MHA(Norm(X));  X' = H + FFN(Norm(H))
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.eps)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.eps)
        self.ffn = SwiGLU(cfg.d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------- 1, 7 e 8. Modelo
class MiniLLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.E = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm_f = RMSNorm(cfg.d_model, cfg.eps)
        self.apply(self._init)

    @staticmethod
    def _init(mod):
        # variancia ~ 1/d nas projecoes, para o produto escalar nao explodir
        if isinstance(mod, (nn.Linear, nn.Embedding)):
            nn.init.normal_(mod.weight, std=0.02)

    def forward(self, tokens, targets=None):
        x = self.E(tokens)                        # X0 = E[tokens]
        for block in self.blocks:
            x = block(x)
        logits = self.norm_f(x) @ self.E.weight.T  # pesos compartilhados com E
        loss = None
        if targets is not None:
            # L = -(1/n) sum log P(x_{t+1} | x_<=t)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, tokens, n_new, temperature=1.0):
        # Sem KV cache: recalcula o contexto inteiro a cada token (simples, nao eficiente)
        for _ in range(n_new):
            ctx = tokens[:, -self.cfg.max_len:]
            logits, _ = self(ctx)
            probs = F.softmax(logits[:, -1] / temperature, dim=-1)
            nxt = torch.multinomial(probs, 1)
            tokens = torch.cat([tokens, nxt], dim=1)
        return tokens


# ---------------------------------------------------------------- Conferencias
def checks():
    torch.manual_seed(0)
    print("== Conferencias das equacoes ==")

    # (a) nossa atencao == a atencao oficial do PyTorch
    q, k, v = (torch.randn(2, 4, 10, 16) for _ in range(3))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    print(f"(a) atencao manual vs PyTorch, maior diferenca: {(attention(q, k, v) - ref).abs().max():.2e}")

    # (b) RoPE: q_m . k_n so depende de m - n
    d_k = 16
    cos, sin = rope_angles(d_k, 64, 10000.0)
    qv, kv = torch.randn(d_k), torch.randn(d_k)

    def score(m, n):
        seq_q = torch.zeros(1, 1, 64, d_k); seq_q[0, 0, m] = qv
        seq_k = torch.zeros(1, 1, 64, d_k); seq_k[0, 0, n] = kv
        return (apply_rope(seq_q, cos, sin)[0, 0, m] @ apply_rope(seq_k, cos, sin)[0, 0, n]).item()

    print(f"(b) RoPE: score(5,2)={score(5, 2):.5f}  score(40,37)={score(40, 37):.5f}  (mesma distancia 3)")

    # (c) gradiente da entropia cruzada com softmax = p - y
    z = torch.randn(1, 8, requires_grad=True)
    y = torch.tensor([3])
    F.cross_entropy(z, y).backward()
    esperado = F.softmax(z, -1).detach() - F.one_hot(y, 8)
    print(f"(c) dL/dz vs (p - y), maior diferenca: {(z.grad - esperado).abs().max():.2e}")

    # (d) variancia de q.k cresce com d_k; dividir por sqrt(d_k) volta para ~1
    for d in (16, 256):
        a, b = torch.randn(10000, d), torch.randn(10000, d)
        s = (a * b).sum(-1)
        print(f"(d) d_k={d:4d}: var(q.k)={s.var():7.1f}   var(q.k/sqrt(d_k))={(s / math.sqrt(d)).var():.2f}")

    # (e) contagem de parametros vs formula (MHA, sem GQA): 12*L*d^2 + V*d
    cfg = Config(n_kv_heads=4)
    m = MiniLLM(cfg)
    real = sum(p.numel() for p in m.parameters())
    aprox = 12 * cfg.n_layers * cfg.d_model ** 2 + cfg.vocab_size * cfg.d_model
    print(f"(e) parametros reais={real:,}  formula 12Ld^2+Vd={aprox:,}")
    print()


# ---------------------------------------------------------------- Treino
TEXTO = (
    "o transformer olha para todos os tokens anteriores e decide quanto cada um importa. "
    "a atencao compara uma pergunta com varias chaves e mistura os valores. "
    "a rede aprende a prever o proximo token e a perda mede o quanto ela errou. "
) * 20


def train():
    torch.manual_seed(42)
    chars = sorted(set(TEXTO))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in TEXTO])

    cfg = Config(vocab_size=len(chars), max_len=64)
    model = MiniLLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.1)
    print(f"== Treino: {sum(p.numel() for p in model.parameters()):,} parametros, "
          f"vocabulario de {len(chars)} caracteres ==")
    print(f"perda inicial esperada ~ ln(V) = {math.log(len(chars)):.2f}")

    n, B = cfg.max_len, 16
    for step in range(301):
        ix = torch.randint(len(data) - n - 1, (B,))
        x = torch.stack([data[i:i + n] for i in ix])
        y = torch.stack([data[i + 1:i + n + 1] for i in ix])   # alvo = texto deslocado 1 posicao
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            print(f"passo {step:3d}  perda {loss.item():.3f}")

    model.eval()
    inicio = torch.tensor([[stoi[c] for c in "a atencao "]])
    for T in (0.5, 1.5):
        out = model.generate(inicio, 80, temperature=T)[0]
        print(f"\nT={T}: " + "".join(chars[i] for i in out))


if __name__ == "__main__":
    checks()
    train()
