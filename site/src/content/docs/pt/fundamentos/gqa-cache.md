---
title: "Cabeças, GQA e cache K/V"
description: "Compartilhamento de chaves e valores entre cabeças, inferência incremental e reconstrução da janela."
---

## Quatro consultas, duas coleções de K/V

A configuração padrão tem quatro cabeças Q e duas cabeças K/V, cada uma com dimensão 32. Na *grouped-query attention* (GQA), um grupo de consultas compartilha as mesmas chaves e valores, mas cada consulta produz seus próprios scores e pesos.

```text
Q cabeça 0 ─┐
            ├─ usa K/V cabeça 0
Q cabeça 1 ─┘

Q cabeça 2 ─┐
            ├─ usa K/V cabeça 1
Q cabeça 3 ─┘
```

`repeat_kv(x, repeats=2)` em `src/glassllm/autograd.py` implementa a expansão para o produto matricial: `(B,2,m,32)` vira `(B,4,m,32)`, na ordem `[0,0,1,1]`. A operação usa `np.repeat`, portanto materializa a expansão; a economia descrita abaixo está no cache armazenado, não em todas as estruturas temporárias.

No backward, as contribuições das cabeças que compartilharam K/V são somadas. Por exemplo, repetir os valores `[[1,2],[3,4]]` duas vezes por cabeça e somar toda a saída devolve gradiente `[[2,2],[2,2]]` para o tensor original. São projeções compartilhadas, não quatro parâmetros K/V independentes.

## O que o cache guarda

`KVCache` em `src/glassllm/nn.py` contém os arrays `k` e `v`, inicialmente `None`. `GlassLM.new_cache()` cria um cache para cada uma das quatro camadas.

| Tensor na etapa incremental | Shape padrão, batch 1 |
| --- | --- |
| Q novo | `(1,4,1,32)` |
| K/V novos, antes do append | `(1,2,1,32)` |
| K/V armazenados após append | `(1,2,m,32)` |
| Scores e pesos | `(1,4,1,m)` |
| Saída da atenção | `(1,1,128)` |

K é guardado **depois** de RoPE, V após sua projeção. O cache não guarda os pesos de atenção nem todos os intermediários dos blocos. As novas Q/K/V são calculadas apenas para os novos tokens; Q consulta todas as chaves preservadas. A concatenação copia os arrays antigos, portanto esta implementação didática não usa buffers pré-alocados.

## Prefill e token seguinte

Na primeira previsão, o modelo processa todo o prompt (*prefill*). Na previsão seguinte, passa somente o token acabado de escolher e informa `start` igual ao comprimento já armazenado. Compare com um forward completo:

```python
import numpy as np
from glassllm import Config, GlassLM, no_grad

cfg = Config(vocab_size=16, d_model=8, n_layers=1,
             n_heads=2, n_kv_heads=1, max_len=8)
model = GlassLM(cfg, seed=0)
with no_grad():
    cache = model.new_cache()
    model.forward([4, 5, 6], cache=cache)
    incremental = model.forward([7], cache=cache, start=3).data[0, -1]
    completo = model.forward([4, 5, 6, 7]).data[0, -1]
print(cache[0].k.shape)
print(np.allclose(incremental, completo, atol=1e-7))
```

Saída verificada: `(1,1,4,4)` e `True`; a diferença máxima foi aproximadamente `2,78 × 10⁻¹⁷`. A configuração reduzida é apenas para tornar o exemplo rápido. O cache é para inferência: ao concatenar arrays, o caminho de gradientes do histórico é desconectado; treine com forward completo e sem cache.

## Memória e limite da janela

Para K e V em todas as camadas:

```text
memória = 2 · B · L · m · H_KV · dₖ · bytes_por_elemento
```

Com batch 1, quatro camadas, 128 posições, duas cabeças K/V e dimensão 32, são 65.536 elementos. Em float32, isso corresponde a 256 KiB; em float64, a 512 KiB. Confira `cache[0].k.dtype` e some `k.nbytes + v.nbytes`: operações do motor podem promover ativações para float64 mesmo quando os parâmetros são float32. Essas contas excluem scores, gradientes e snapshots do `Tracer`. GQA reduz pela metade esse armazenamento em comparação com quatro cabeças K/V do mesmo tamanho.

Quando a próxima inserção ultrapassaria `max_len=128`, `GlassLM.generate` descarta o cache e reprocessa a janela dos últimos 128 tokens, com posições reiniciadas em zero. Não remove apenas a primeira chave do cache: isso não reproduziria os mesmos estados calculados com o histórico recortado.

## Ligação com a visualização e referência

No inspetor, `attn.k` e `attn.v` mostram as projeções **novas**, registradas antes da concatenação. `attn.scores` e `attn.softmax` incluem a extensão total `m`. O nó de cache no grafo representa essa ligação estrutural e não um snapshot dedicado.

Ainslie et al. apresentam GQA em [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245), 2023. O projeto implementa o compartilhamento de cabeças, sem o procedimento de conversão de checkpoints do artigo. Continue em [atenção causal](../atencao-causal/) ou [geração](../saida-geracao/).
