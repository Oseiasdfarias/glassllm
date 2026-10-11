"""Catalog of traceable steps: the equation behind every `# @step <id>` marker.

The text that explains each step lives in the site chapters (one per language).
This file only holds the math, which is the same in every language.
"""
from __future__ import annotations

import json
import re
import tokenize
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StepInfo:
    latex: str
    plain: str


STEPS: dict[str, StepInfo] = {
    "embed": StepInfo(r"X_0 = E[\mathrm{ids}]", "X0 = E[ids]"),
    "norm.attn": StepInfo(r"\hat{X} = \mathrm{RMSNorm}(X)", "X̂ = RMSNorm(X)"),
    "attn.q": StepInfo(r"Q = \hat{X} W_Q", "Q = X̂·W_Q"),
    "attn.k": StepInfo(r"K = \hat{X} W_K", "K = X̂·W_K"),
    "attn.v": StepInfo(r"V = \hat{X} W_V", "V = X̂·W_V"),
    "attn.rope_q": StepInfo(r"Q' = R_{\Theta, m}\, Q", "Q' = rotate(Q, m·θ)"),
    "attn.rope_k": StepInfo(r"K' = R_{\Theta, n}\, K", "K' = rotate(K, n·θ)"),
    "attn.scores": StepInfo(r"S = \frac{Q' K'^\top}{\sqrt{d_k}}", "S = Q'·K'ᵀ / √d_k"),
    "attn.mask": StepInfo(r"S + M, \quad M_{ij} = -\infty \text{ if } j > i", "S + M   (M = −∞ above the diagonal)"),
    "attn.softmax": StepInfo(r"A = \mathrm{softmax}(S + M)", "A = softmax(S + M)"),
    "attn.mix": StepInfo(r"Y_h = A V", "Y_h = A·V"),
    "attn.out": StepInfo(r"\mathrm{MHA}(X) = \mathrm{Concat}(Y_1, \dots, Y_H)\, W_O", "MHA = Concat(Y_1..Y_H)·W_O"),
    "resid.attn": StepInfo(r"H = X + \mathrm{MHA}(\mathrm{RMSNorm}(X))", "H = X + MHA(RMSNorm(X))"),
    "norm.ffn": StepInfo(r"\hat{H} = \mathrm{RMSNorm}(H)", "Ĥ = RMSNorm(H)"),
    "ffn.gate": StepInfo(r"G = \hat{H} W_1", "G = Ĥ·W_1"),
    "ffn.up": StepInfo(r"U = \hat{H} W_3", "U = Ĥ·W_3"),
    "ffn.act": StepInfo(r"\mathrm{SiLU}(G) \odot U", "SiLU(G) ⊙ U"),
    "ffn.down": StepInfo(r"F = (\mathrm{SiLU}(G) \odot U)\, W_2", "F = (SiLU(G) ⊙ U)·W_2"),
    "resid.ffn": StepInfo(r"X' = H + F", "X' = H + F"),
    "norm.final": StepInfo(r"h = \mathrm{RMSNorm}(X_L)", "h = RMSNorm(X_L)"),
    "logits": StepInfo(r"z = h E^\top", "z = h·Eᵀ"),
    "probs": StepInfo(r"p = \mathrm{softmax}(z / T)", "p = softmax(z / T)"),
    "sample": StepInfo(r"x_{t+1} \sim p", "next ~ p"),
    "loss": StepInfo(r"\mathcal{L} = -\frac{1}{N} \sum_t \log p_t[y_t]", "L = −(1/N) Σ log p[target]"),
    "loss.grad": StepInfo(r"\frac{\partial \mathcal{L}}{\partial z} = \frac{p - y}{N}", "∂L/∂z = (p − y) / N"),
}

# Steps that run once per block; the tracer records them with a layer index.
LAYER_STEPS = {
    "norm.attn", "attn.q", "attn.k", "attn.v", "attn.rope_q", "attn.rope_k", "attn.scores",
    "attn.mask", "attn.softmax", "attn.mix", "attn.out", "resid.attn", "norm.ffn",
    "ffn.gate", "ffn.up", "ffn.act", "ffn.down", "resid.ffn",
}

# Steps whose value has a head axis first: (H, n, ...).
HEAD_STEPS = {
    "attn.q", "attn.k", "attn.v", "attn.rope_q", "attn.rope_k",
    "attn.scores", "attn.mask", "attn.softmax", "attn.mix",
}

MARKER = re.compile(r"#\s*@step\s+([\w.]+)")
PACKAGE_DIR = Path(__file__).parent
INDEX_PATH = PACKAGE_DIR / "step_index.json"


def scan_markers(package_dir: Path = PACKAGE_DIR) -> dict[str, dict]:
    """Find every `# @step <id>` comment: {id: {"file": "glassllm/x.py", "line": n}}."""
    index: dict[str, dict] = {}
    for path in sorted(package_dir.glob("*.py")):
        with path.open("rb") as source:
            for token in tokenize.tokenize(source.readline):
                if token.type != tokenize.COMMENT:
                    continue
                for match in MARKER.finditer(token.string):
                    step_id = match.group(1)
                    number = token.start[0]
                    if step_id in index:
                        raise ValueError(f"step {step_id!r} is marked twice: {index[step_id]} and {path.name}:{number}")
                    index[step_id] = {"file": f"glassllm/{path.name}", "line": number}
    return index


def load_step_index() -> dict[str, dict]:
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))
