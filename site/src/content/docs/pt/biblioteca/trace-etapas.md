---
title: "Tracer, registros e catálogo de etapas"
description: "Snapshots reais, seleção de camada e cabeça, 25 operações e vínculo com o código."
---

## Gravar uma execução

`from glassllm import Tracer, StepRecord, STEPS` importa as APIs de observação. `Tracer()` começa com `records: list[StepRecord]` vazio. O tracer copia valores nos pontos marcados do modelo; não suspende os cálculos, não calcula atenção por conta própria e não grava toda operação NumPy elementar. O slow motion da interface reproduz esses snapshots depois da execução.

`record(step_id: str, value, layer: int | None = None, batched: bool = True, *, head: int | None = None) -> None` aceita um Tensor ou valor convertido para array. `step_id` deve existir em `STEPS`, ou ocorre `KeyError`. `batched=True` exige um primeiro eixo de tamanho 1 e o remove; escalar ou batch maior produz `ValueError`. Use `batched=False` para escalares, vetores de probabilidade ou valores já sem batch.

O array gravado é uma cópia independente da entrada. `order` é a posição do registro na lista, começando em zero. Camada e cabeça, quando fornecidas, precisam ser inteiros não negativos; booleanos são rejeitados. Cabeça só é permitida para IDs de `HEAD_STEPS`. O tracer não valida se a camada existe em algum modelo nem exige camada para um ID de bloco: isso é responsabilidade da chamada.

```python
import numpy as np
from glassllm import Tracer

trace = Tracer()
value = np.ones((1, 2, 3), dtype=np.float32)
trace.record("embed", value)
value[:] = 7
record = trace.get("embed")
assert record.shape == (2, 3)
np.testing.assert_array_equal(record.value, np.ones((2, 3)))
assert record.order == 0
trace.record("sample", np.array(4), batched=False)
assert int(trace.get("sample").value) == 4
```

A função de submódulo `glassllm.trace.record(trace: Tracer | None, step_id, value, layer=None, batched=True, *, head=None) -> None` chama o método quando há tracer e não faz nada quando `trace=None`. É o adaptador usado pelas camadas.

## Consultar sem ambiguidades

`get(step_id: str, layer: int | None = None, head: int | None = None) -> StepRecord` retorna o **primeiro** snapshot compatível. Omitir camada não significa última camada; omitir cabeça devolve o array como foi gravado. Ao reutilizar um tracer em várias chamadas, registros são acrescentados e o primeiro valor continua sendo o mais antigo. Prefira um tracer por forward ou consulte `records` diretamente para uma sequência temporal.

Um snapshot completo de cabeça tem forma `(H,n,...)`. `get(..., head=h)` seleciona seu primeiro eixo e devolve um novo StepRecord com `head=h`, preservando `order` e a equação. O recorte pode compartilhar memória com o snapshot completo; não o altere se pretende preservar a observação. Um registro gravado com `head=h` já foi fatiado e só atende esse seletor.

Seletores negativos, booleanos ou não inteiros e cabeças fora de faixa geram `ValueError`; procurar um registro ausente gera `KeyError`. Selecionar cabeça de um ID sem eixo de cabeça também gera `ValueError`. `step_ids() -> list[str]` retorna IDs distintos na ordem de primeira ocorrência, sem repetir cada bloco.

```python
import numpy as np
from glassllm import Config, GlassLM

model = GlassLM(Config(vocab_size=8, d_model=8, n_layers=1,
                      n_heads=2, n_kv_heads=1, max_len=4))
step = next(model.generate([1, 2, 3], 1, greedy=True, trace=True))
assert step.trace is not None
attention = step.trace.get("attn.softmax", layer=0, head=1)
assert attention.shape == (3, 3)
assert attention.head == 1
np.testing.assert_allclose(attention.value.sum(-1), 1.0, rtol=1e-5)
assert len(step.trace.records) == 23
```

## StepRecord e StepInfo

`StepRecord(id: str, layer: int | None, value: np.ndarray, order: int, head: int | None = None)` é uma dataclass frozen. `shape` devolve `value.shape`; `equation` consulta o texto simples de `STEPS[id]`; `latex` consulta sua fórmula LaTeX. A classe não torna `value` imutável. `StepInfo(latex: str, plain: str)`, de `glassllm.steps`, é a dataclass frozen que guarda os dois formatos da equação.

## Os 25 IDs e os eventos repetidos

`STEPS` tem 25 IDs. `LAYER_STEPS` contém as 18 operações executadas por bloco; `HEAD_STEPS` contém as 9 que preservam um eixo de cabeça. As equações são as mesmas em PT/EN; explicações e rótulos ficam no site.

| Etapa | Forma gravada, sem batch | Cabeça |
| --- | --- | --- |
| `embed`, `norm.final` | `(n,d)` | Não |
| `norm.attn`, `resid.attn`, `norm.ffn`, `resid.ffn` | `(n,d)` por bloco | Não |
| `attn.q`, `attn.rope_q` | `(Hq,n,dk)` | Q |
| `attn.k`, `attn.v`, `attn.rope_k` | `(Hkv,n,dk)` das posições novas | K/V |
| `attn.scores`, `attn.mask`, `attn.softmax` | `(Hq,n,m)` com chaves do cache | Q |
| `attn.mix` | `(Hq,n,dk)` | Q |
| `attn.out`, `ffn.down` | `(n,d)` por bloco | Não |
| `ffn.gate`, `ffn.up`, `ffn.act` | `(n,int(8*d/3))` por bloco | Não |
| `logits` | `(n,V)` | Não |
| `probs` | `(V,)` | Não |
| `sample`, `loss` | Escalar | Não |
| `loss.grad` | `(n,V)` | Não |

Para `L` blocos, forward grava `18*L + 3` eventos. Geração acrescenta `probs` e `sample`: com 4 blocos são **77 eventos**, embora apenas 23 IDs distintos sejam executados. A perda acrescenta `loss` e `loss.grad` aos 75 eventos do forward, também totalizando 77, com outro conjunto de 23 IDs. Nenhuma execução de geração inclui perda.

## Equação e linha de origem

`scan_markers(package_dir: Path = PACKAGE_DIR) -> dict[str,dict]` examina comentários Python `# @step <id>` dos módulos `.py` imediatos e devolve `{id: {"file": "glassllm/x.py", "line": n}}`, com linhas a partir de 1. Marca repetida gera `ValueError`; strings que parecem comentários não contam. `load_step_index() -> dict[str,dict]` lê o JSON empacotado `step_index.json`; não recalcula números de linha automaticamente.

```python
from glassllm import STEPS
from glassllm.steps import HEAD_STEPS, LAYER_STEPS, load_step_index, scan_markers

assert len(STEPS) == 25
assert len(LAYER_STEPS) == 18
assert len(HEAD_STEPS) == 9
assert scan_markers() == load_step_index()
assert load_step_index()["attn.softmax"]["file"] == "glassllm/nn.py"
```

Depois de mover linhas do motor, execute `python scripts/build_step_index.py`. O índice aponta para a operação de origem, não para uma stack trace de cada chamada. A interface mostra [recortes do inspetor](../../interface/inspetor/) com valores não finitos como `−∞`; esses valores são parte real da máscara causal.
