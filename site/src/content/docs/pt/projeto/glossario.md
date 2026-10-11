---
title: Glossário e notação
description: Os termos e símbolos usados na interface, nas equações e nas assinaturas da biblioteca.
---

## Texto e vocabulário

| Termo | Significado no glassllm |
| --- | --- |
| Token | Unidade do tokenizer: caractere, subpalavra ou token especial. Não corresponde necessariamente a uma palavra |
| ID | Inteiro que identifica uma entrada no vocabulário de um tokenizer específico |
| Vocabulário | Tabela entre representações de tokens e IDs; cada modelo distribuído tem 1024 entradas |
| BPE | Byte Pair Encoding: aqui, aprende fusões de pares frequentes sobre símbolos de texto; não é um tokenizer completo de todos os bytes |
| `▁` | Representação visual do espaço dentro dos tokens |
| `↵` | Representação visual de uma quebra de linha na interface |
| UNK | Token desconhecido para símbolos que o vocabulário não representa; o decode usa a representação definida pelo tokenizer |
| BOS / EOS | Tokens especiais de início/fim presentes no vocabulário; a API não os insere automaticamente nem encerra ao gerar EOS |
| Prompt | Texto inicial fornecido à geração |
| Contexto | IDs utilizados para prever o próximo token, limitado pela janela do modelo |

O ID é local ao vocabulário. Não combine IDs de um tokenizer PT com o modelo EN. Tokenização também não determina um significado fixo de cada vetor; os embeddings são parâmetros aprendidos. Veja [tokens e BPE](../../fundamentos/tokens-bpe/).

## Dimensões e eixos

| Símbolo | Significado | Valor padrão ou relação |
| --- | --- | --- |
| `B` | Número de sequências no batch | O trace de `GlassLM.forward` requer `B = 1` |
| `n` | Posições novas processadas neste forward | Prefixo inicial ou 1 token no avanço com cache |
| `m` | Posições disponíveis nas chaves/valores | Cache anterior + posições novas |
| `d` / `d_model` | Dimensão do vetor de cada token | 128 |
| `H_Q` | Cabeças de consulta | 4 |
| `H_KV` | Cabeças de chave/valor | 2 |
| `d_k` | Dimensão por cabeça | `d / H_Q = 32` |
| `d_ff` | Dimensão interna da FFN | `floor(8d/3) = 341` |
| `V` | Tamanho do vocabulário, quando usado como dimensão | 1024 |
| `L` | Número de blocos, quando usado como contagem | 4 |
| `N` | Número de alvos da perda média | `B × n` |

Um array `B × n × d` contém um vetor de `d` números para cada posição de cada sequência. Uma matriz `n × m` de atenção relaciona posições de consulta às posições disponíveis no contexto. O tracer remove o eixo de batch; selecionar uma cabeça também remove seu eixo. Por isso a forma no inspetor pode ter menos eixos que a variável dentro da camada.

O vocabulário `V` como dimensão e a matriz de valores `V` da atenção compartilham uma letra convencional, mas são objetos diferentes. Da mesma forma, `L` pode indicar número de blocos e `𝓛` a perda; o contexto da equação identifica o uso.

## Matrizes, operações e parâmetros

| Símbolo | Leitura |
| --- | --- |
| `E` | Matriz de embedding `V × d`, também usada como transposta na saída |
| `X`, `H` | Representações das posições do texto dentro do modelo |
| `Q`, `K`, `V` | Consultas, chaves e valores da atenção |
| `W_Q`, `W_K`, `W_V`, `W_O` | Matrizes das projeções da atenção |
| `S` | Compatibilidades `QKᵀ/√d_k`, antes da máscara |
| `M` | Máscara causal: 0 nas posições permitidas, `−∞` nas futuras |
| `A` | Pesos de atenção após softmax |
| `γ` | Escala aprendida da RMSNorm, uma por dimensão |
| `ε` | Estabilizador numérico da RMSNorm: `1e-6` por padrão |
| `T` | Temperatura positiva utilizada antes do softmax de saída |
| `ᵀ` | Transposta: troca os eixos de uma matriz |
| `⊙` | Produto elemento a elemento |
| `@` / `·` | Produto matricial no código / notação das equações |

Uma projeção como `Q = X @ W_Q` combina as dimensões de cada vetor; a atenção combina posições usando `A @ V`. A FFN aplica suas projeções independentemente a cada posição. [Atenção causal](../../fundamentos/atencao-causal/) detalha essa diferença.

## Saída, aprendizado e inspeção

**Logit** é uma pontuação sem normalização para um token candidato. **Softmax** transforma pontuações em probabilidades que somam 1 ao longo do eixo escolhido. **Atenção** também usa softmax, mas sua distribuição é sobre posições de contexto, e não sobre tokens do vocabulário. **Greedy** escolhe `argmax`; **sampling** sorteia segundo a distribuição.

**Perda** mede a incompatibilidade entre previsão e alvo. **Gradiente** é a derivada da perda em relação a uma variável ou parâmetro. **Backpropagation** aplica a regra da cadeia em sentido reverso; **autograd** registra as operações e organiza essa propagação. **Parâmetro** é um `Tensor` do modelo que recebe gradiente no treino; **ativação** é um valor intermediário calculado com esses parâmetros.

**Cache K/V** guarda chaves rotacionadas e valores já calculados para evitar recomputar todo o prefixo a cada token. **Checkpoint** é o arquivo de configuração e pesos; não inclui tokenizer nem estado AdamW. **Trace** é a sequência de snapshots de uma execução. **Snapshot** é uma cópia numérica feita no ponto marcado. **Slow motion** é a reprodução visual dessa sequência no site.

## Precisão numérica e recortes

`float16`, `float32` e `float64` usam diferentes números de bits para representar valores. Um checkpoint float16 carregado como float32 não garante que todos os intermediários permaneçam float32. `−∞` é um valor não finito intencional na máscara; `NaN` indica um resultado indefinido. O inspetor conta esses valores e os exclui das estatísticas finitas.

`itemsize` informa o número de bytes por elemento de um array; `nbytes` informa os bytes de seus elementos. Esses valores permitem medir parâmetros, cache e snapshots sem supor que todos tenham o mesmo dtype. Bytes de objetos Python e do runtime não entram em `nbytes`.

As últimas 8 × 8 entradas são um recorte de leitura, não todo o tensor. Mínimo, média e máximo se referem aos elementos finitos do tensor selecionado inteiro. A numeração visível do seletor de cabeças começa em 1, enquanto o índice Python e a legenda resumida da atenção começam em 0. Consulte [o inspetor](../../interface/inspetor/) para interpretar seus rótulos e [a API](../../biblioteca/api-publica/) para as convenções de chamada.
