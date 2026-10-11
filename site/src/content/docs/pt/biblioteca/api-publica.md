---
title: "Mapa da API e convenções"
description: "Importações, tipos, dimensões e contratos do motor glassllm em NumPy."
---

## Escolher o nível de entrada

O pacote `glassllm` exporta 12 nomes: `Config`, `GenStep`, `GlassLM`, `LANGUAGES`, `STEPS`, `StepRecord`, `Tensor`, `Tokenizer`, `Tracer`, `load_pretrained`, `no_grad` e `__version__`. Esses nomes cobrem a configuração, a geração, o carregamento e a observação do modelo. As funções matemáticas, camadas, otimizadores e ferramentas de treinamento são importadas de seus submódulos.

```python
import numpy as np
from glassllm import Config, GlassLM, Tensor, Tracer, no_grad

cfg = Config(vocab_size=8, d_model=8, n_layers=1,
             n_heads=2, n_kv_heads=1, max_len=4)
model = GlassLM(cfg, seed=0)
trace = Tracer()
with no_grad():
    logits = model([1, 2, 3], trace=trace)
assert logits.shape == (1, 3, 8)
assert not logits.requires_grad
np.testing.assert_allclose(
    trace.get("attn.softmax", layer=0, head=0).value.sum(-1), 1.0,
    rtol=1e-5,
)
```

O exemplo usa pesos aleatórios para verificar contratos. Para gerar com os pesos treinados, use `model, tokenizer = load_pretrained("pt")`; a publicação da biblioteca ainda está adiada, portanto a instalação documentada usa o checkout ou o wheel local.

## Caminhos de importação

| Módulo | Símbolos e finalidade |
| --- | --- |
| `glassllm.config` | `Config`: dimensões, limites e validações |
| `glassllm.model` | `GlassLM`, `GenStep`, `next_token_probs`: logits, perda e geração |
| `glassllm.tokenizer` | `Tokenizer`, `pre_tokenize`, `SPACE`, `SPECIALS`, `UNK_TEXT`: BPE próprio |
| `glassllm.trace` | `Tracer`, `StepRecord`, `record`: snapshots copiados |
| `glassllm.steps` | `StepInfo`, `STEPS`, `LAYER_STEPS`, `HEAD_STEPS`, `scan_markers`, `load_step_index`: equações e origem |
| `glassllm.autograd` | `Tensor`, `no_grad` e operações diferenciáveis: grafo reverso |
| `glassllm.nn` | `Module`, `RMSNorm`, `Attention`, `SwiGLU`, `Block`, `KVCache`, funções de inicialização, máscara e ângulos |
| `glassllm.optim` | `AdamW`, `clip_grad_norm`, `lr_schedule`: atualização dos parâmetros |
| `glassllm.train` | `get_batch`, `evaluate`, `train`: janelas, avaliação e treinamento |
| `glassllm.cli` | `main`, `build_parser`, `format_matrix`, comandos `cmd_*`: terminal |

Os nomes com prefixo `_` são auxiliares internos. Não há compatibilidade com a interface de módulos do PyTorch: `Module` não oferece `.train()`, `.eval()`, `.to()` ou `.state_dict()`. O motor executa no NumPy; o navegador usa esse mesmo Python em Pyodide.

## Convenções de forma e índice

| Símbolo | Significado | Padrão dos modelos incluídos |
| --- | --- | --- |
| `B` | batch | 1 na interface e no trace |
| `n` | posições recebidas na chamada atual | até 128 |
| `m` | posições das chaves, incluindo cache | até 128 |
| `V` | tamanho do vocabulário | 1024 |
| `d` | dimensão do modelo | 128 |
| `Hq`, `Hkv` | cabeças de consulta e de chave/valor | 4 e 2 |
| `dk = d / Hq` | dimensão de cada cabeça | 32 |

IDs e targets são índices inteiros; posições, camadas e cabeças começam em zero. `forward` aceita IDs `(n,)` ou `(B,n)` e devolve `Tensor` `(B,n,V)`. Ele mantém o eixo de batch mesmo quando recebeu uma lista. O tracer remove o batch unitário dos valores gravados: atenção completa `(Hq,n,m)`, ou `(n,m)` após selecionar uma cabeça. O tokenizer não acrescenta BOS/EOS automaticamente.

## Tipos, estado e erros

`Tensor.data` é um `numpy.ndarray`; `Tensor.grad` começa como `None`. Parâmetros normalmente são float32; exemplos de derivadas podem usar float64. Não há dispositivo GPU, tipos simbólicos ou suporte a derivadas de segunda ordem. Chamadas NumPy diretas em `.data` ficam fora do grafo.

`forward`, `loss` e `generate` não atualizam pesos. `train` e `AdamW.step` os modificam no mesmo objeto; `forward` também preenche o cache quando ele é fornecido. `generate` cria seu próprio cache e um novo tracer por token quando `trace=True`.

As validações explícitas usam principalmente `ValueError`, `TypeError`, `KeyError` e `FileNotFoundError`. Formas incompatíveis, IDs fora de faixa e arquivos malformados também podem produzir erros do NumPy ou JSON. A biblioteca não converte todo erro interno para uma classe única. Como `generate` é um generator, suas validações só são executadas quando o iterador avança.

Continue em [Configuração e carregamento](../config-carregamento/), [GlassLM](../glasslm/) ou [Tensor e autograd](../tensor-autograd/).
