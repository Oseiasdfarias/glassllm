---
title: "GlassLM e GenStep"
description: "Inferência, perda, geração com cache, parâmetros e checkpoints do Transformer."
---

## Construir e listar parâmetros

`from glassllm import GlassLM, GenStep` importa a classe de modelo e o resultado de uma previsão. `GlassLM(cfg: Config, seed: int = 0, dtype=np.float32)` inicializa pesos com um gerador NumPy local. A mesma configuração, dtype e seed reproduzem os pesos iniciais. Não são os pesos treinados carregados por `load_pretrained`.

O modelo herda `Module`. `named_parameters(prefix: str = "") -> dict[str,Tensor]` percorre atributos, submódulos e listas de submódulos; nomes como `E`, `blocks.0.attn.W_Q` e `norm_f.gamma` correspondem às entradas do checkpoint. `parameters() -> list[Tensor]` retorna os mesmos objetos em ordem. Eles não são cópias: alterar `.data` altera o modelo. A matriz de saída usa `E` transposta, sem segundo parâmetro.

## forward e `__call__`

`forward(ids, trace: Tracer | None = None, cache: list[KVCache] | None = None, start: int = 0) -> Tensor` devolve logits `(B,n,V)`. `model(ids, ...)` é um alias desta chamada. IDs são convertidos a `int64`; entrada `(n,)` ganha batch 1. São rejeitados números de eixos diferentes de 1 ou 2, posições negativas ou `start+n > max_len`, e trace com batch diferente de 1.

IDs precisam pertencer ao vocabulário. A indexação NumPy admite IDs negativos; o motor não oferece validação semântica dedicada para rejeitá-los. Entrada de ponto flutuante também é convertida para inteiro, portanto valide dados externos antes da chamada. Um comprimento vazio pode atravessar partes do forward, mas não serve como entrada de geração.

`trace` aqui é um objeto **Tracer**, ao contrário do argumento booleano de `generate`. Sem `no_grad`, os logits normalmente mantêm o grafo dos parâmetros. Quando um cache é passado, cada bloco o preenche; a lista deve ter um item por bloco. `start` é o offset absoluto das posições atuais dentro da janela. Cache e offset precisam concordar; use [new_cache e camadas](../camadas-cache/) para a convenção.

```python
import numpy as np
from glassllm import Config, GlassLM, Tracer, no_grad

model = GlassLM(Config(vocab_size=8, d_model=8, n_layers=1,
                      n_heads=2, n_kv_heads=1, max_len=4), seed=0)
trace = Tracer()
logits = model.forward([1, 2], trace=trace)
assert logits.shape == (1, 2, 8)
assert trace.get("embed").shape == (2, 8)
with no_grad():
    reference = model([1, 2, 3]).data[:, -1:]
    cache = model.new_cache()
    model([1, 2], cache=cache)
    incremental = model([3], cache=cache, start=2).data
np.testing.assert_allclose(incremental, reference, rtol=1e-5, atol=1e-6)
```

## loss

`loss(ids, targets, trace: Tracer | None = None) -> Tensor` executa o forward completo, achata as posições e retorna a cross-entropy média escalar. Há um target por posição de entrada; o método não desloca o texto automaticamente. Para previsão do próximo token, forneça `ids=data[:-1]` e `targets=data[1:]`. Targets devem ser índices inteiros válidos e ter `B*n` elementos.

O valor usa log-softmax estável, sem temperatura de geração. Ele permite `loss.backward()` para acumular gradientes nos parâmetros. Com trace, registra também `loss` e `loss.grad`; o último é o gradiente analítico em relação aos logits e não significa que `backward()` já foi executado. Não há cache no treinamento da perda.

## generate e GenStep

`generate(ids, n_new: int, temperature: float = 1.0, greedy: bool = False, seed: int | None = None, trace: bool = False) -> Iterator[GenStep]` gera até `n_new` tokens. Entrada deve ser uma sequência não vazia de IDs. O prompt é truncado para os últimos `max_len` tokens. Zero tokens novos retorna um iterador sem resultados; o código atual também não cria resultados para um `n_new` negativo, sem validação dedicada.

Cada `GenStep` é uma dataclass frozen com `token: int`, `probs: np.ndarray` de forma `(V,)` e `trace: Tracer | None`. A distribuição usa float64 e inclui todo o vocabulário, não só os seis candidatos da interface. Frozen impede reatribuir campos, mas não torna o array `probs` imutável.

Temperatura precisa ser positiva mesmo com `greedy=True`; zero e negativos produzem `ValueError`. `greedy` escolhe `argmax`; sorteio usa o gerador NumPy criado com `seed`. `trace` aceita apenas bool ou produz `TypeError`: `trace=True` cria um tracer novo por resultado. Os erros são observados ao avançar o generator, não ao apenas criá-lo.

```python
import numpy as np
from glassllm import Config, GlassLM

model = GlassLM(Config(vocab_size=8, d_model=8, n_layers=1,
                      n_heads=2, n_kv_heads=1, max_len=4), seed=0)
steps = list(model.generate([1, 2], 3, seed=42, trace=True))
again = list(model.generate([1, 2], 3, seed=42))
assert [s.token for s in steps] == [s.token for s in again]
assert steps[0].trace is not steps[1].trace
assert int(steps[0].trace.get("sample").value) == steps[0].token
np.testing.assert_allclose(steps[0].probs.sum(), 1.0)
```

O prefixo é calculado uma vez; chamadas seguintes recebem o último token e reutilizam K/V. Quando a janela fica cheia, o cache é reconstruído com os últimos `max_len` IDs e posições reiniciadas em zero. Assim a saída corresponde a um forward completo nessa janela, em vez de extrapolar a tabela RoPE. Não há parada automática em EOS. `no_grad` é restaurado antes de cada `yield`, permitindo operações de treinamento do chamador entre resultados.

`from glassllm.model import next_token_probs` expõe `next_token_probs(logits: np.ndarray, temperature: float) -> np.ndarray`: softmax no último eixo de `logits / temperature`, sem modificar o array de entrada. Exige temperatura positiva; não faz amostragem nem filtragem top-k/top-p.

## new_cache, save e load

`new_cache() -> list[KVCache]` cria caches vazios independentes para os blocos. `save(path: str | Path, fp16: bool = True) -> None` grava NPZ comprimido, com `__config__` JSON e arrays por nome. Por padrão os parâmetros são convertidos para float16 **no arquivo**, sem alterar os pesos em memória. `fp16=False` preserva o dtype de cada parâmetro. O destino é sobrescrito; diretórios pais não são criados e o NumPy pode acrescentar `.npz` a caminhos sem essa extensão.

`GlassLM.load(path: str | Path, dtype=np.float32) -> GlassLM` reconstrói a Config e copia arrays convertidos para o dtype solicitado. Ausência de chaves gera `KeyError`; arquivos inválidos produzem erros do NumPy. O leitor atual não compara explicitamente todas as formas dos parâmetros com a arquitetura antes de atribuí-los; use arquivos produzidos pelo próprio `save`.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from glassllm import Config, GlassLM, no_grad

model = GlassLM(Config(vocab_size=8, d_model=8, n_layers=1,
                      n_heads=2, n_kv_heads=1, max_len=4))
with TemporaryDirectory() as directory:
    path = Path(directory) / "model.npz"
    model.save(path, fp16=False)
    restored = GlassLM.load(path)
    with no_grad():
        np.testing.assert_array_equal(model([1, 2]).data, restored([1, 2]).data)
```

O checkpoint não inclui tokenizer, corpus, cache, trace, gradientes ou estado do AdamW. Para retomar um experimento, guarde o tokenizer separado; para retomar o mesmo estado de otimização, seria necessário serializar o otimizador adicionalmente, API que não existe nesta versão. A arquitetura reúne atenção do [Transformer](https://arxiv.org/abs/1706.03762) e [embedding compartilhado](https://arxiv.org/abs/1608.05859), com as mudanças RMSNorm, RoPE, GQA e SwiGLU descritas nas páginas de fundamentos.
