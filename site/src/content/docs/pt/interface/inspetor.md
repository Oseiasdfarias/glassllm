---
title: "Inspecionar tensores e código"
description: "Leia a equação, o shape, as estatísticas e o recorte numérico de uma operação real, com sua linha de código."
---

## Selecionar um resultado real

Gere um token e clique numa operação da arquitetura. O inspetor combina título, bloco, explicação, equação, dimensões, estatísticas, uma grade numérica e **Cálculo no código**. Os valores são cópias registradas pelo `Tracer` naquela previsão, não uma nova execução provocada pelo clique.

Antes da primeira previsão, o mapa oferece descrição e dimensões simbólicas. Depois, operações registradas passam a mostrar o shape real. A seleção pausa a geração e o fluxo; assim, você lê valores de uma mesma previsão até optar por avançar.

## Escolher bloco e cabeça

O desenho apresenta **Bloco 1–4**; a API Python usa `layer=0–3`. O seletor apresenta **Cabeça 1–4**, mas envia os índices `0–3`. Em K, V e RoPE de K, existem duas cabeças, então o seletor oferece apenas **Cabeça 1–2**. Q, scores, máscara, softmax e mistura usam quatro cabeças.

Selecionar outra cabeça usa o mesmo trace. Não recalcula os pesos, não muda a cabeça usada pelo painel resumido de atenção e não altera a próxima escolha do modelo. Operações sem eixo de cabeça, como embedding, projeção de saída e FFN, ocultam o seletor.

## Interpretar as dimensões

O rastreamento remove o eixo de batch de tamanho 1. Assim, o embedding aparece como `n × 128`, enquanto um `forward` Python retorna logits com shape `1 × n × 1024`. Quando você escolhe uma cabeça, o inspetor também retira esse eixo da seleção: pesos de atenção aparecem como `n × m`.

Na primeira previsão, `n` normalmente é o número de tokens do prefixo. Com cache, uma previsão seguinte pode ter `n = 1` e mais colunas de contexto. Um shape diferente entre previsões não implica mudança da arquitetura. Observe também a reconstrução da janela descrita em [GQA e cache](../../fundamentos/gqa-cache/).

## A grade é um recorte, as estatísticas são do tensor

Para manter a interface leve, o worker transforma o tensor selecionado em matriz e mostra **até as últimas oito linhas e as últimas oito colunas**. Se houver mais dimensões antes da última, elas são achatadas nas linhas. A legenda informa o tamanho completo e os offsets do trecho exibido.

Os valores de mínimo, média e máximo são calculados sobre todos os elementos finitos do tensor selecionado, não somente sobre a grade visível. Por isso o máximo informado pode não estar no recorte. Cabeças selecionadas têm estatísticas próprias.

Nos scores, máscara e softmax, as colunas são tokens do contexto. Na saída de logits e probabilidades, o eixo é o vocabulário. A grade dessas saídas mostra os últimos IDs, não os candidatos mais prováveis; para estes, use **Próximas possibilidades**. Linhas e dimensões de features podem aparecer como índices numéricos.

Os números exibidos são arredondados e podem usar notação científica. A biblioteca retém o valor NumPy; o formato de apresentação não altera o tensor.

## Valores não finitos e máscara causal

`−∞` na máscara é esperado: ele bloqueia posições futuras. Depois do softmax, essas posições recebem zero. O worker serializa `−∞`, `+∞` e `NaN` como rótulos, preservando o significado ao transportar JSON.

Elementos não finitos são contados e excluídos das estatísticas. Se nenhum elemento for finito, as estatísticas aparecem como `—`. Um recorte sem `−∞` pode ser válido, especialmente com uma única consulta nova: todas as chaves disponíveis são posições permitidas.

## Ligar a equação à linha de código

**Cálculo no código** mostra arquivo, linha e expressão marcada por `# @step`. O índice dessas marcações é produzido a partir do código empacotado. A linha ajuda a localizar a implementação, mas não mostra toda a função ou todas as validações: abra o módulo correspondente para ler o contexto.

O ID do catálogo, como `attn.softmax`, identifica a operação em todos os blocos. `Tracer.get("attn.softmax", layer=0, head=1)` seleciona a segunda cabeça do primeiro bloco em Python. O trace usa snapshots copiados, então uma operação posterior não modifica o valor já registrado.

## Nós estruturais e histórico

Contexto, cache K/V, expansão GQA e matriz E compartilhada explicam a estrutura do modelo, mas não possuem um evento próprio para cada ação no trace. Ao selecionar esses nós, o inspetor informa **Estrutura do modelo · dimensões simbólicas**.

O worker mantém apenas o trace mais recente. Não há seletor para recuperar uma previsão antiga depois de gerar outra. Os prints do inspetor expandem a área de matriz para mostrar o componente completo; durante o uso normal, role dentro do painel quando necessário. Use [slow motion](../slow-motion/) para manter um resultado estável enquanto percorre a sequência.
