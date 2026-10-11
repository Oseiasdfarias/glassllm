---
title: "Treinamento e otimização"
description: "Janelas de tokens, avaliação, train, AdamW, clipping e agenda de learning rate."
---

## Preparar batches

Importações: `from glassllm.train import get_batch, evaluate, train` e `from glassllm.optim import AdamW, clip_grad_norm, lr_schedule`. Essas funções não são exports da raiz.

`get_batch(data: np.ndarray, batch_size: int, seq_len: int, rng: np.random.Generator) -> tuple[np.ndarray,np.ndarray]` recebe uma sequência unidimensional de IDs. Sorteia `batch_size` janelas e devolve `x,y` de forma `(batch_size,seq_len)`, preservando o dtype dos dados. Cada target é o token seguinte: `y[:, :-1] == x[:, 1:]`. O sorteio inclui a última janela com target completo, e pode repetir janelas.

Batch e comprimento precisam ser positivos; `len(data)` precisa ser maior que `seq_len`, ou ocorre `ValueError`. A função não tokeniza texto nem valida todos os IDs. No modelo, `seq_len` também precisa caber em `cfg.max_len`.

```python
import numpy as np
from glassllm.train import get_batch

data = np.arange(12, dtype=np.int64)
x, y = get_batch(data, batch_size=2, seq_len=4, rng=np.random.default_rng(0))
assert x.shape == y.shape == (2, 4)
np.testing.assert_array_equal(y, x + 1)
```

## Avaliar

`evaluate(model: GlassLM, data: np.ndarray, batch_size: int, seq_len: int, n_batches: int = 10, seed: int = 1234) -> float` calcula a média das perdas de batches amostrados. A seed fixa reproduz as janelas; não é uma varredura de todo o corpus. Exige `n_batches>0`, além das condições de get_batch e forward.

Executa dentro de `no_grad`, sem alterar pesos ou gradientes já existentes. Não chama `.eval()`: o modelo não tem dropout nem modo de batch norm. A métrica é cross-entropy média em logaritmo natural, por token, e não percentual de acerto. Perplexidade pode ser calculada como `np.exp(loss)`, considerando o mesmo tokenizer e conjunto de dados.

## train

A assinatura completa é:

```text
train(model, data, steps, batch_size=16, seq_len=128, lr=0.003,
      warmup=100, seed=0, val_data=None, eval_every=100,
      keep_best=False, log=print) -> list[dict]
```

`model` é modificado no lugar; `data/val_data` são IDs unidimensionais. `steps` conta atualizações, não épocas. Cada iteração sorteia um batch, calcula perda, zera gradientes dos parâmetros, executa backward, limita a norma global em 1, calcula o learning rate e chama AdamW. Um novo otimizador é criado em cada chamada: chamar `train` novamente não retoma seus momentos anteriores.

| Argumento | Efeito |
| --- | --- |
| `steps` | Inteiro não negativo; zero não faz atualizações |
| `batch_size`, `seq_len` | Janelas de treino; positivos e compatíveis com corpus/janela |
| `lr` | Learning rate base; a API não faz validação completa de finitude/sinal |
| `warmup` | Passos de aquecimento; não negativo |
| `seed` | Gerador das janelas; não reinicializa os pesos do modelo |
| `val_data` | Corpus de validação opcional; precisa comportar batches |
| `eval_every` | Intervalo positivo de avaliação e log |
| `keep_best` | Com validação, restaura o melhor conjunto de pesos medido |
| `log` | Callback que recebe string; `None` desliga mensagens |

O histórico tem um dicionário por atualização concluída: `step` começa em 0, `loss` é o valor do batch antes da atualização e `lr` é a taxa aplicada. Com validação, `val_loss` aparece nos passos `step % eval_every == 0` e no último passo. A validação mede os pesos **depois** da atualização, em 10 batches com seed 1234.

```python
import numpy as np
from glassllm import Config, GlassLM
from glassllm.train import evaluate, train

cfg = Config(vocab_size=8, d_model=8, n_layers=1,
             n_heads=2, n_kv_heads=1, max_len=4)
model = GlassLM(cfg, seed=0)
data = np.tile(np.arange(8, dtype=np.int64), 5)
history = train(model, data, steps=3, batch_size=2, seq_len=4,
                warmup=1, val_data=data, eval_every=2,
                keep_best=True, log=None)
assert [entry["step"] for entry in history] == [0, 1, 2]
assert [entry["step"] for entry in history if "val_loss" in entry] == [0, 2]
best = min(entry["val_loss"] for entry in history if "val_loss" in entry)
np.testing.assert_allclose(evaluate(model, data, 2, 4), best)
```

Com `keep_best=True`, uma cópia dos pesos é guardada a cada melhora medida; no fim, eles são restaurados. Não há early stopping e isso não salva arquivo automaticamente. Sem validação, `keep_best` não tem efeito. O otimizador e os gradientes finais não são restaurados junto com os pesos.

## Prazo e falhas

O treino tem prazo fixo de duas horas, incluindo validação. Ele verifica o relógio entre operações NumPy; uma operação individual em andamento não é interrompida. Ao expirar, gera `TimeoutError`, sem devolver um histórico parcial. Um melhor conjunto de pesos já medido é restaurado no `finally`, inclusive se a execução falhar; se nenhum melhor conjunto foi concluído, os pesos ficam no último estado alcançado.

Argumentos `steps<0`, `warmup<0` ou `eval_every<=0` geram `ValueError` antes das atualizações. Erros de corpus, shapes e IDs surgem nos componentes chamados. Um callback de log que falha também interrompe a execução. Escolha argumentos finitos válidos e considere que o treinamento padrão é CPU NumPy.

## AdamW

`AdamW(params: list[Tensor], lr: float = 0.003, betas=(0.9,0.95), eps: float = 1e-8, weight_decay: float = 0.1)` guarda a lista, momentos `m/v`, contador global `t` e contadores de atualização por parâmetro. Defaults de beta2 e learning rate são escolhas deste projeto, não todos os defaults de outro framework.

`zero_grad() -> None` coloca `grad=None` em todos os parâmetros. `step(lr: float | None = None) -> None` usa o override apenas nessa atualização; não muda o `lr` base. Parâmetros sem gradiente são ignorados, inclusive para decay e correção de bias. Para cada um que participa, atualiza média móvel do gradiente e seu quadrado, corrige seus vieses pelo próprio número de atualizações e aplica o passo adaptativo.

O weight decay é desacoplado e aplicado apenas a arrays com `ndim>=2`, como matrizes de projeção e embedding. Vetores gamma de RMSNorm não recebem decay. Não há validação completa de betas, shapes, dtype ou learning rate: forneça parâmetros float e controles matematicamente válidos. A classe não oferece save/load de estado.

```python
import numpy as np
from glassllm import Tensor
from glassllm.optim import AdamW, clip_grad_norm, lr_schedule

parameter = Tensor([0.0, 0.0], requires_grad=True)
optimizer = AdamW([parameter], lr=0.1, weight_decay=0.0)
optimizer.zero_grad()
((parameter - 3.0) ** 2).sum().backward()
before = clip_grad_norm([parameter], max_norm=1.0)
assert before > 1.0
assert np.linalg.norm(parameter.grad) <= 1.0
optimizer.step()
assert np.all(parameter.data > 0.0)
assert lr_schedule(0, 10, 1.0, warmup=2) == 0.5
assert lr_schedule(20, 10, 1.0, warmup=2) == 0.1
```

## Clipping e learning rate

`clip_grad_norm(params: list[Tensor], max_norm: float) -> float` retorna a norma L2 global **antes** do clipping. Calcula a soma dos quadrados em float64 para não estourar com gradientes float32 grandes. Se a norma excede o limite, multiplica todos os gradientes existentes por `max_norm/(norm+1e-6)`; ignora `None`. Sem gradientes, retorna 0. Use `max_norm>0`; não há validação dedicada de limite negativo ou NaN.

`lr_schedule(step: int, total_steps: int, base_lr: float, warmup: int, min_ratio: float = 0.1) -> float` aplica aquecimento `base_lr*(step+1)/warmup` enquanto `step<warmup`, depois cosseno entre base e mínimo. `warmup=0` começa na taxa base. O progresso do cosseno é limitado em 1; em `step>=total_steps`, resulta no mínimo quando o aquecimento já terminou. Como as atualizações reais vão de 0 a `total_steps-1`, a última pode estar ligeiramente acima do mínimo.

As referências são [Adam](https://arxiv.org/abs/1412.6980), [AdamW](https://arxiv.org/abs/1711.05101) e o cosseno de [SGDR](https://arxiv.org/abs/1608.03983). O projeto usa um único decaimento cosseno com warmup, sem os warm restarts de SGDR. O [guia de arquivos](../cli-arquivos/) mostra como salvar os pesos e a validação do export float16.
