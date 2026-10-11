---
title: "Camadas e cache"
description: "Module, RMSNorm, atenção GQA, RoPE, SwiGLU, blocos e estado K/V."
---

## Module e inicialização

Todos os símbolos desta página pertencem a `glassllm.nn`. `Module` fornece `named_parameters(prefix: str = "") -> dict[str,Tensor]` e `parameters() -> list[Tensor]`. A descoberta percorre atributos Tensor com `requires_grad`, atributos Module e listas de Modules. Dicionários e tuplas arbitrários não são percorridos; referências duplicadas não têm deduplicação geral. No modelo padrão, o embedding compartilhado está armazenado uma só vez.

`init_param(rng: np.random.Generator, shape: tuple[int,...], std: float, dtype) -> Tensor` gera normal com média zero e desvio `std`, converte para `dtype` e solicita gradiente. Chame fora de `no_grad` para que parâmetros sejam descobertos. Não há um seed global alterado por esta função: a sequência vem do `rng` fornecido.

## RMSNorm

`RMSNorm(d: int, eps: float, dtype)` cria `gamma` `(d,)`, inicializado com uns. `norm(x: Tensor) -> Tensor` aplica `x / sqrt(mean(x², axis=-1)+eps) * gamma`, mantendo a forma. Não subtrai média, não tem beta nem estatísticas móveis de batch. O último eixo de `x` deve corresponder a `d`.

O contrato usual usa `d` positivo e `eps` positivo; a classe isolada não repete todas as validações da Config. Com `eps=0` e vetor nulo, o denominador é zero. A [RMSNorm original](https://arxiv.org/abs/1910.07467) fundamenta esta normalização; o projeto usa a versão sobre todo o último eixo.

## Ângulos e máscara

`rope_angles(d_k: int, max_len: int, base: float) -> tuple[np.ndarray,np.ndarray]` gera cosseno e seno `(max_len,d_k/2)` para ângulos `position * base**(-2*i/d_k)`. Exige dimensão positiva par, comprimento não negativo e base positiva, ou gera `ValueError`. Os arrays NumPy saem da fórmula sem dtype solicitado; Attention os converte ao dtype dos pesos.

`causal_mask(n_queries: int, n_keys: int, start: int) -> np.ndarray` retorna booleanos `(n_queries,n_keys)`. A posição de query é `start+i`; uma chave `j > start+i` é True e será bloqueada. Contagens e início negativos geram `ValueError`. Com cache, o resultado não precisa ser quadrado.

```python
import numpy as np
from glassllm.nn import causal_mask, rope_angles

np.testing.assert_array_equal(causal_mask(2, 3, start=1),
                              [[False, False, True], [False, False, False]])
cos, sin = rope_angles(4, 3, 10000.0)
assert cos.shape == sin.shape == (3, 2)
np.testing.assert_allclose(cos[0], [1.0, 1.0])
np.testing.assert_allclose(sin[0], [0.0, 0.0])
```

## Attention

`Attention(d, n_heads, n_kv_heads, max_len, rope_base, rng, std, dtype)` cria `W_Q: (d,d)`, `W_K/W_V: (d,Hkv*dk)` e `W_O: (d,d)`, com `dk=d/n_heads`. Dimensões precisam ser positivas e divisíveis; cabeças Q precisam ser múltiplas das K/V e `dk` deve ser par. Erros de estrutura produzem `ValueError`.

`attention(x: Tensor, layer: int, trace: Tracer | None = None, cache: KVCache | None = None, start: int = 0) -> Tensor` recebe `(B,n,d)` e retorna `(B,n,d)`. Calcula projeções, divide cabeças, rotaciona Q/K, incorpora cache, repete K/V para grupos Q, calcula scores escalados por `sqrt(dk)`, mascara, normaliza, mistura V, concatena e projeta. Não usa bias, dropout ou atenção cruzada.

O índice `layer` rotula snapshots, sem escolher parâmetros de outra camada. Posições fora da tabela RoPE geram `ValueError`. Em trace, Q usa `Hq` cabeças e K/V usa `Hkv`: com padrão 4/2, cabeças Q 0–1 compartilham K/V 0, e Q 2–3 compartilham K/V 1.

## KVCache e execução incremental

`KVCache()` começa com `k=None`, `v=None`. Depois do prefixo, guarda arrays `(B,Hkv,m,dk)`: chaves já rotacionadas e valores antes da repetição de grupos. Cada chamada concatena posições novas no eixo 2 e atualiza ambos. Cache não é um parâmetro e não faz parte do checkpoint.

Use cache somente em inferência, dentro de `no_grad`. Concatenar arrays desconecta o histórico do grafo, portanto cache não é um mecanismo de treinamento por partes. Não reutilize caches entre prompts, modelos ou batches diferentes. Não há método `reset`: construa outro cache, normalmente com `model.new_cache()`.

```python
import numpy as np
from glassllm import Config, GlassLM, no_grad

model = GlassLM(Config(vocab_size=8, d_model=8, n_layers=1,
                      n_heads=2, n_kv_heads=1, max_len=4))
cache = model.new_cache()
assert cache[0].k is None and cache[0].v is None
with no_grad():
    full = model([1, 2, 3]).data[:, -1:]
    model([1, 2], cache=cache, start=0)
    assert cache[0].k.shape == (1, 1, 2, 4)
    last = model([3], cache=cache, start=2).data
assert cache[0].k.shape == (1, 1, 3, 4)
np.testing.assert_allclose(last, full, rtol=1e-5, atol=1e-6)
```

A chamada manual exige `start=m` antes de acrescentar tokens e tamanho total até `max_len`. A camada não verifica todos os desencontros de cache/offset; `generate` administra essa consistência e reconstrói a janela ao atingir o limite. K/V gravados no tracer representam as **posições novas**; scores e softmax incorporam também as posições antigas do cache.

## SwiGLU e Block

`SwiGLU(d: int, rng: np.random.Generator, std: float, dtype)` exige `d>0` e cria largura `hidden=int(8*d/3)`. `W1/W3` têm forma `(d,hidden)`; `W2`, `(hidden,d)`. `ffn(x: Tensor, layer: int, trace: Tracer | None = None) -> Tensor` calcula `SiLU(x@W1) * (x@W3)`, seguido de `@W2`. Entrada e saída são `(B,n,d)`; gate, up e act são `(B,n,hidden)`. Registra quatro eventos FFN por bloco.

`Block(cfg, rng: np.random.Generator, dtype)` reúne `norm1`, `attn`, `norm2` e `ffn`. `block(x: Tensor, layer: int, trace=None, cache=None, start=0) -> Tensor` faz duas somas residuais: `h=x+Attention(RMSNorm(x))` e `out=h+SwiGLU(RMSNorm(h))`. Mantém `(B,n,d)` e grava 18 operações ao todo com Attention e FFN. Cada bloco usa seu próprio cache.

A atenção escalada vem de [Attention Is All You Need](https://arxiv.org/abs/1706.03762), a posição de [RoPE](https://arxiv.org/abs/2104.09864), o compartilhamento K/V de [GQA](https://arxiv.org/abs/2305.13245) e o portão de [GLU Variants](https://arxiv.org/abs/2002.05202). Aqui GQA é treinada diretamente; o código não implementa a conversão de checkpoints multi-head do artigo. Veja o [grafo da interface](../../interface/arquitetura/) para relacionar cada ramo aos snapshots.
