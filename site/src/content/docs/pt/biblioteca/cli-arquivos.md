---
title: "CLI, arquivos e scripts"
description: "Comandos de terminal, formatos de assets, preparação de corpus e treinamento local."
---

## Executar a CLI

A instalação local registra `glassllm = glassllm.cli:main`. No checkout com ambiente preparado, os comandos também podem ser executados por `uv run python -m glassllm.cli`. `--lang` aceita `pt` ou `en`, com padrão `en`; pode aparecer antes ou depois do subcomando que oferece esse argumento. `--model-dir` aponta para assets locais com o mesmo esquema `{lang}.npz` e `{lang}.tokenizer.json`.

```bash
uv run glassllm --lang pt tokenize "Era uma vez" --merges
uv run glassllm generate "Alice" --lang en -n 3 --greedy --seed 0
uv run glassllm trace "Alice was" --lang en --step attn.softmax --layer 0 --head 0
uv run glassllm steps
```

Os três primeiros usam pesos e tokenizer do pacote. `steps` funciona sem assets, pois lê o catálogo. `tokenize` na implementação atual também passa pelo carregador do modelo; um tokenizer isolado pode ser usado pela API Python se os pesos não estiverem disponíveis. `train` carrega apenas o tokenizer e cria pesos novos.

## Subcomandos e argumentos

| Comando | Argumentos e resultado |
| --- | --- |
| `tokenize TEXT` | Imprime ID e string por token; `--merges` mostra estados sucessivos de cada chunk |
| `generate TEXT` | `-n/--n=40`, `--temperature=0.8`, `--greedy=False`, `--seed=None`; imprime prompt e continuação |
| `trace TEXT` | `--step=attn.softmax`, `--layer=None`, `--head=None`; imprime equação, forma e valores |
| `steps` | Lista os 25 IDs de operação e suas equações |
| `train` | `--steps=200`, `--batch-size=8`, `--seq-len=64`, `--d-model=64`, `--layers=2`, `--lr=0.003`, `--seed=0`, `--log-every=20`, `--out=None`, `--corpus=None` |

`trace` aceita apenas IDs do catálogo. Etapas de bloco usam camada 0 quando omitida; etapas de cabeça usam cabeça 0. Seletores não aplicáveis geram erro. Para `probs/sample`, faz uma geração greedy; para `loss/loss.grad`, desloca a sequência e precisa de pelo menos dois tokens; nas demais etapas, faz forward completo. Matrizes mostram todas as linhas e no máximo 8 colunas, com reticências no restante. Probabilidades mostram os 10 candidatos de maior valor; isso difere dos seis da interface.

Treino da CLI lê `data/{lang}_train.txt`, ou `--corpus` explícito. Usa tokenizer já treinado, sempre 4 cabeças Q e 2 K/V, com `max_len=seq_len`; `d_model` precisa respeitar essas divisibilidades. O warmup é `min(100, steps//10+1)`. Não faz validação nem restauração do melhor modelo, ao contrário de `scripts/train_model.py`. `--out` salva o NPZ ao terminar; omitido, o modelo permanece só durante o processo.

```bash
uv run glassllm train --lang pt --steps 3 --batch-size 2 --seq-len 8 --d-model 8 --layers 1 --corpus data/pt_train.txt --out /tmp/glassllm-cli-demo.npz
```

Este último comando exige corpus preparado. Use outro destino para preservar seus próprios experimentos; `save` sobrescreve arquivos existentes. A geração da CLI não oferece trace completo por token, top-k, top-p, parada em EOS ou limite de caracteres.

## API do módulo cli

| Assinatura | Uso programático |
| --- | --- |
| `main(argv: list[str] | None = None) -> None` | Faz parsing e despacha; `None` usa argumentos do processo |
| `build_parser() -> argparse.ArgumentParser` | Constrói o parser com todos os subcomandos |
| `format_matrix(values: np.ndarray, row_labels: list[str], col_labels: list[str] | None = None) -> str` | Exige matriz 2D e contagens de rótulos compatíveis; nomes de colunas padrão são índices |
| `cmd_tokenize(args, model: GlassLM | None, tok: Tokenizer) -> None` | Imprime tokens e, opcionalmente, fusões |
| `cmd_generate(args, model: GlassLM, tok: Tokenizer) -> None` | Consome o generator e imprime texto |
| `cmd_trace(args, model: GlassLM, tok: Tokenizer) -> None` | Calcula e imprime uma etapa selecionada |
| `cmd_steps(args, model, tok) -> None` | Imprime catálogo; ignora modelo/tokenizer |
| `cmd_train(args, model: GlassLM, tok: Tokenizer) -> None` | Cria e treina um modelo novo; o argumento model não é usado |

Os handlers recebem um namespace compatível com `build_parser`; a entrada normal recomendada é `main`. `format_matrix` não altera o array e limita colunas com `MAX_COLS=8`. Seu formatador representa qualquer valor não finito como `-inf`, sem distinguir NaN e +inf; use arrays finitos ou leia o valor original quando precisar dessa distinção.

```python
import numpy as np
from glassllm.cli import build_parser, format_matrix, main

args = build_parser().parse_args(["generate", "Alice", "--lang", "en", "-n", "1"])
assert args.lang == "en" and args.n == 1
table = format_matrix(np.array([[1.0, 2.0]]), ["token"], ["f0", "f1"])
assert "token" in table and "1.000" in table
main(["steps"])
```

`main` converte erros comuns de valor, chave, índice e arquivo em `SystemExit("error: ...")`. Erros de sintaxe de argumentos ficam a cargo do argparse; timeout de treino não está nessa lista de conversão. Os handlers internos podem levantar erros diretamente se chamados fora de main.

## Arquivos produzidos

| Arquivo | Conteúdo e leitor |
| --- | --- |
| `{lang}.npz` | Matrizes por nome e `__config__` JSON; `GlassLM.load` |
| `{lang}.tokenizer.json` | `version`, `alphabet`, `merges`; `Tokenizer.load` |
| `{lang}.meta.json` | Passos, contagem de tokens, perdas FP32/FP16 e parâmetros da avaliação; JSON de metadados, não checkpoint |
| `step_index.json` | ID → arquivo/linha das operações; `load_step_index` |
| `.whl` local | Pacote Python com módulos e assets preparados; usado também pelo navegador |

Modelos não guardam corpus, tokenizer ou otimizador no NPZ. Um nome de wheel precisa seguir o formato completo de wheel, como `glassllm-0.1.0-py3-none-any.whl`; uma URL terminada apenas em `glassllm` não serve ao micropip. O pacote publicado pode ser instalado com `python -m pip install glassllm`; o laboratório no navegador continua servindo seu próprio wheel ao Pyodide.

## Scripts do checkout

Execute scripts a partir da raiz do repositório com o pacote instalado no ambiente.

```bash
uv run python scripts/prepare_corpus.py --out data
uv run python scripts/train_tokenizer.py --lang pt --vocab-size 1024 --data data --out /tmp/glassllm-training
uv run python scripts/train_model.py --lang pt --steps 3 --batch-size 2 --data data --out /tmp/glassllm-training
uv run python scripts/build_step_index.py
```

`prepare_corpus.py` baixa oito livros, remove delimitadores Gutenberg, normaliza NFC e espaços, e reserva o trecho final de aproximadamente 5% de cada livro para validação, cortando na quebra de linha seguinte. Requer rede; não mistura aleatoriamente partes do mesmo livro entre treino e validação. `train_tokenizer.py` usa apenas o corpus de treino e escreve o tokenizer no diretório de saída.

`train_model.py` exige o tokenizer nesse mesmo diretório. O padrão é 3.000 passos, batch 16 e configuração completa. Treina com validação e `keep_best=True`, salva pesos float16 e compara perda de validação do export com FP32 nas mesmas janelas; valores não finitos ou diferença absoluta maior que 0,01 geram `ValueError`. Seu prazo de duas horas cobre também avaliações e export. Metadados só são escritos após essas verificações.

`build_step_index.py` atualiza o índice de linhas após mudanças no motor. `build_api_inventory.py` recompõe o inventário de símbolos usado na documentação; ele não valida assinaturas por execução. Veja [integração do projeto](../../projeto/integracao-browser/) e [treinamento](../treino-otimizacao/) para entender os contratos usados por esses scripts.
