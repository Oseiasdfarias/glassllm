---
title: Dados, modelos e reprodutibilidade
description: A origem dos corpora PT/EN, a configuração dos checkpoints e o significado das métricas distribuídas com o pacote.
---

## Origem dos textos

Os modelos distribuídos são separados por idioma. `scripts/prepare_corpus.py` obtém oito livros do Project Gutenberg: cinco de Machado de Assis para PT e três de Lewis Carroll para EN. As páginas de [Dom Casmurro](https://www.gutenberg.org/ebooks/55752) e [Alice's Adventures in Wonderland](https://www.gutenberg.org/ebooks/11) identificam essas obras e seus arquivos. O catálogo abaixo corresponde aos IDs utilizados pelo script; ele não descreve um corpus geral da internet.

| Idioma | Obra | ID Gutenberg |
| --- | --- | --- |
| PT | Dom Casmurro | 55752 |
| PT | Memorias Posthumas de Braz Cubas | 54829 |
| PT | Quincas Borba | 55682 |
| PT | Esau e Jacob | 56737 |
| PT | Memorial de Ayres | 55797 |
| EN | Alice's Adventures in Wonderland | 11 |
| EN | Through the Looking-Glass | 12 |
| EN | Sylvie and Bruno | 620 |

O script remove cabeçalhos e rodapés Gutenberg, normaliza Unicode para NFC, comprime espaços horizontais e reduz séries de linhas vazias. Para cada livro, reserva aproximadamente os 5% finais para validação, avançando o corte até uma quebra de linha. Concatena separadamente as partes de treino e de validação de cada idioma. Essa divisão avalia trechos posteriores das mesmas obras; não mede generalização para autores ou domínios inéditos.

## Tokenizers e configuração

Cada idioma tem seu próprio tokenizer BPE de 1024 entradas. IDs iguais em PT e EN podem representar textos diferentes; carregue o tokenizer correspondente ao checkpoint. Os arquivos ficam em `src/glassllm/weights/`, junto com os pesos `.npz` e metadados `.meta.json`.

| Propriedade | Valor dos modelos distribuídos |
| --- | --- |
| Vocabulário | 1024 |
| Dimensão do modelo | 128 |
| Blocos Transformer | 4 |
| Cabeças de consulta Q | 4 |
| Cabeças K/V | 2 |
| Dimensão por cabeça | 32 |
| Dimensão interna SwiGLU | 341 |
| Janela de contexto | 128 tokens |
| Parâmetros únicos | 852.608 |

A matriz de embedding é compartilhada com a projeção de saída. Essa contagem considera os parâmetros únicos retornados por `model.parameters()`. O checkpoint exporta pesos em float16 por padrão; `load_pretrained` reconstrói o modelo com parâmetros float32. A precisão das operações intermediárias pode ser promovida pelas regras NumPy; consulte o `dtype` do array quando estimar memória.

## Métricas dos arquivos distribuídos

Os valores abaixo foram lidos dos metadados empacotados. A perda é entropia cruzada média em unidades de logaritmo natural; valores menores indicam maior probabilidade atribuída aos alvos das janelas avaliadas.

| Medida | PT | EN |
| --- | --- | --- |
| Passos solicitados | 3000 | 3000 |
| Tokens no conjunto de treino | 694.356 | 229.390 |
| Tokens no conjunto de validação | 38.009 | 12.573 |
| Perda do checkpoint float32 | 3,4348712394 | 4,1935762033 |
| Perda depois do export float16 e reload | 3,4348720647 | 4,1935892849 |

A avaliação registrada usa 10 batches de 16 sequências com 128 posições e semente 1234: são 20.480 alvos amostrados por avaliação. Esse total não é o número de tokens únicos do conjunto de validação. As janelas são sorteadas, podem se sobrepor e usam alvos deslocados uma posição; a perda média não percorre necessariamente todo o corpus.

Os metadados guardam os passos solicitados, não o índice da melhor iteração. `keep_best=True` restaura os pesos de melhor perda nas avaliações periódicas antes do export; o resultado distribuído pode vir de uma avaliação anterior ao último passo. O loop usa índices de passo começando em zero: avalia depois das atualizações dos índices 0, 100, 200 e assim por diante, além do último índice, mesmo fora desse intervalo.

As perdas PT e EN não são uma classificação direta de qualidade entre idiomas: tokenização e dados diferem. Também não são uma pontuação de conhecimento, precisão factual ou capacidade de diálogo. O modelo é pequeno e pode repetir, gerar texto incompleto ou produzir sequências pouco coerentes.

## Reproduzir o pipeline

No checkout, com o ambiente instalado, os comandos abaixo preparam os corpora e exibem as opções dos scripts de tokenizer e treinamento. Os comandos com `--help` não treinam. O download requer rede; ao executar um treino, ele altera os arquivos do diretório indicado por `--out`. Use outro diretório para preservar os checkpoints distribuídos e crie nesse diretório o tokenizer do idioma antes de iniciar `train_model.py`.

```sh
uv run python scripts/prepare_corpus.py --out data
uv run python scripts/train_tokenizer.py --help
uv run python scripts/train_model.py --help
```

`scripts/train_model.py` usa semente 0 para inicialização e amostragem de batches, taxa inicial padrão `3e-3`, warmup de 100 passos, avaliação periódica a cada 100 índices e seleção do melhor checkpoint. Compara a perda float32 com a perda do export float16 nas mesmas janelas; rejeita valores não finitos ou diferença acima de 0,01. Há verificações de um orçamento de duas horas ao redor das fases do script. `train` também verifica seu prazo entre fases de cada atualização e entre batches de validação. Essas verificações não interrompem uma operação NumPy em execução; um `TimeoutError` encerra a tentativa sem prometer que o export e os metadados serão produzidos.

Guarde configuração, versões Python/NumPy, arquivos baixados, sementes e argumentos de treino para repetir uma execução. Os arquivos obtidos da rede podem mudar; uma semente não substitui a preservação dos dados. `model.save()` salva configuração e pesos, e não estado do otimizador ou histórico de treinamento.

## Conferir um checkpoint

```python
from glassllm import load_pretrained

model, tokenizer = load_pretrained("pt")
print(tokenizer.vocab_size)
print(sum(parameter.data.size for parameter in model.parameters()))
print(model.cfg.max_len)
```

Resultado dos assets distribuídos: `1024`, `852608` e `128`. Para repetir uma continuação, mantenha idioma, frase, checkpoint, temperatura, modo de escolha e semente. O laboratório oferece modo mais provável; a API Python também aceita `seed` para sorteio. Veja [geração](../../biblioteca/glasslm/) e [treinamento](../../biblioteca/treino-otimizacao/).
