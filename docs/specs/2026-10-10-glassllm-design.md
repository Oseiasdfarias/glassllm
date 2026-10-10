# glassllm: design

**Data:** 2026-10-10
**Status:** aguardando revisão

## 1. Objetivo

Um ambiente para aprender as bases de um LLM na prática, olhando o fluxo real. A pessoa liga
três coisas: a **equação**, a **linha de código** que a implementa e o **valor** calculado
naquele momento, para uma frase de verdade.

São dois produtos com um núcleo só:

- **`glassllm` no PyPI:** uma lib Python (só NumPy) com um Transformer pequeno escrito do zero,
  um modo de depuração que grava cada passo, e uma CLI para estudar pelo terminal.
- **Site estático:** trilha de capítulos e painel de geração em loop, rodando a mesma lib no
  navegador via Pyodide.

**Público:** quem sabe Python e um pouco de álgebra linear e quer entender o que acontece dentro
de um LLM. Uso secundário do autor: estudo para entrevista de AI Engineer, conteúdo para
LinkedIn, portfólio.

**Critérios de sucesso:**

1. Para cada passo do modelo, o site mostra lado a lado a equação, a linha exata do código e os
   números calculados para a frase digitada.
2. O painel livre gera texto token a token, em loop, mostrando atenção e probabilidades de
   cada rodada.
3. `pip install glassllm` funciona offline depois de instalado, com modelos treinados em PT e EN.
4. Cada camada da lib bate com uma referência em PyTorch (frente e trás) em teste automatizado.
5. Interface e textos em português e inglês.

## 2. Decisões e motivos

| Decisão | Motivo |
|---|---|
| Lib só com NumPy, autograd próprio | Roda igual no CPython e no Pyodide; nada fica escondido, e o próprio gradiente vira conteúdo. |
| Site estático com Pyodide | Abre por link, sem instalar nada; roda o código Python real. |
| Mesmo wheel no PyPI e no site | Uma fonte de verdade: o site roda exatamente a versão publicada. |
| Astro + GitHub Pages | Capítulos como conteúdo MDX, i18n nativo, KaTeX no build, ilhas interativas só onde precisa. |
| Tokenizer BPE estilo SentencePiece | É o que LLMs reais usam (Llama); fica legível com `▁` no lugar do espaço. |
| Um modelo por idioma | Um modelo pequeno treinado em dois idiomas misturados piora nos dois. |
| Layout A nos capítulos (três painéis) | Melhor para ligar equação, código e valor; a dedução longa abre no estilo roteiro. |
| Nome `glassllm` | Livre no PyPI; ideia de "glass box", um LLM de vidro; curto para CLI e import. |

## 3. Estrutura do repositório

```
glassllm/
├── pyproject.toml
├── src/glassllm/
│   ├── __init__.py          API pública: Model, Tokenizer, Tracer, load_pretrained
│   ├── tokenizer.py
│   ├── autograd.py
│   ├── nn.py                camadas: Embedding, RMSNorm, RoPE, Attention, SwiGLU, Block
│   ├── model.py             GlassLM, geração com KV cache
│   ├── optim.py             AdamW
│   ├── train.py             laço de treino
│   ├── trace.py             Tracer e Step
│   ├── steps.py             catálogo de passos (equação por id)
│   ├── cli.py
│   └── weights/             pt.npz, en.npz, pt.tokenizer.json, en.tokenizer.json
├── data/                    corpus processado (domínio público)
├── scripts/                 prepare_corpus.py, train_tokenizer.py, train_model.py, build_step_index.py
├── tests/
├── site/                    projeto Astro
├── docs/equacoes.md         explicação completa das equações (PT)
├── docs/prototipo/          protótipo original em PyTorch
└── .github/workflows/       site.yml, publish.yml, tests.yml
```

## 4. A lib `glassllm`

**Requisitos:** Python ≥ 3.10, única dependência de execução `numpy`. Dependências de
desenvolvimento: `pytest`, `torch` (CPU). Build com `uv` + `hatchling`. Wheel puro
(`py3-none-any`).

### 4.1 Tokenizer (`tokenizer.py`)

- BPE sobre caracteres, estilo SentencePiece: o texto é normalizado (NFC), o espaço vira `▁`,
  o alfabeto inicial são os caracteres do corpus, e o treino junta o par adjacente mais
  frequente até o vocabulário chegar ao tamanho alvo.
- Vocabulário alvo: 1024 por idioma. Especiais: `<pad>`, `<unk>`, `<bos>`, `<eos>`.
- Caractere fora do alfabeto vira `<unk>`.
- API: `Tokenizer.train(text, vocab_size)`, `encode(text) -> list[int]`,
  `decode(ids) -> str`, `encode_verbose(text)` (devolve cada fusão aplicada, para o capítulo
  de tokenização), `save/load` em JSON.

### 4.2 Autograd (`autograd.py`)

- Classe `Tensor` que embrulha um `np.ndarray`, guarda os pais e a função de gradiente, e
  calcula `backward()` em ordem topológica reversa.
- Operações: soma, subtração, multiplicação, divisão, potência, matmul (com broadcast de
  lote), exp, log, sum, mean, max, reshape, transpose, indexação (lookup de embedding),
  concat, `where` (máscara), sigmoid, e funções compostas `softmax`, `log_softmax`,
  `cross_entropy`, `silu`.
- `softmax` e `cross_entropy` têm gradiente analítico próprio (estável e didático: a
  `cross_entropy` mostra `p − y` no código).
- Modo `no_grad` para inferência (sem guardar grafo).

### 4.3 Modelo (`nn.py`, `model.py`)

Configuração padrão:

| Campo | Valor |
|---|---|
| `vocab_size` | 1024 |
| `d_model` | 128 |
| `n_layers` | 4 |
| `n_heads` | 4 |
| `n_kv_heads` | 2 (GQA) |
| `max_len` | 128 |
| `rope_base` | 10000 |
| FFN | SwiGLU, `h = int(8d/3)` |
| Norma | RMSNorm, pre-norm |
| Saída | pesos amarrados com o embedding |

Cerca de 0,9 milhão de parâmetros.

- `GlassLM.forward(ids, trace=None)` devolve logits.
- `GlassLM.generate(ids, n_new, temperature, greedy=False, seed=None, trace=None)` usa
  **KV cache**: o prefixo roda uma vez, e cada token novo só calcula a própria linha de Q, K, V.
  Implementado como gerador, devolvendo um token por vez, para o loop do painel.
- `load_pretrained("pt" | "en")` carrega pesos e tokenizer de `glassllm/weights/`.

### 4.4 Modo de depuração (`trace.py`, `steps.py`)

- O código marca cada passo com um comentário `# @step <id>` na linha que o calcula, por
  exemplo `# @step attn.softmax`.
- `Tracer` é passado opcionalmente ao modelo. Em cada passo marcado, o modelo chama
  `trace.record("<id>", valor, layer=l, head=h)`. Sem tracer, a chamada não faz nada.
- Cada registro vira um `Step`: `id`, `layer`, `head`, `value` (cópia NumPy), `shape`, ordem
  de execução.
- `steps.py` é o catálogo: para cada `id`, a equação em LaTeX e uma versão em texto puro para
  a CLI. É neutro de idioma.
- `scripts/build_step_index.py` lê os marcadores `@step` no código-fonte e gera
  `step_index.json` com arquivo e linha de cada passo. O site usa esse índice para destacar
  a linha certa.
- Lista de passos da primeira versão: `embed`, `norm.attn`, `attn.q`, `attn.k`, `attn.v`,
  `attn.rope_q`, `attn.rope_k`, `attn.scores`, `attn.mask`, `attn.softmax`, `attn.mix`,
  `attn.out`, `resid.attn`, `norm.ffn`, `ffn.gate`, `ffn.up`, `ffn.act`, `ffn.down`,
  `resid.ffn`, `norm.final`, `logits`, `probs`, `sample`, `loss`, `loss.grad`.

### 4.5 Treino (`optim.py`, `train.py`)

- AdamW escrito em NumPy, recorte de gradiente por norma, taxa de aprendizado com warmup e
  decaimento cosseno.
- `scripts/train_model.py` treina o modelo de cada idioma com a própria lib e salva os pesos
  em fp16 (`.npz`). Semente fixa.
- Se o treino em NumPy for lento demais para ser prático (mais de 2 horas por modelo), o
  problema é levado ao autor antes de qualquer troca de abordagem.

### 4.6 CLI (`cli.py`)

- `glassllm tokenize --lang pt "texto"`: mostra os tokens e as fusões aplicadas.
- `glassllm generate --lang pt --temperature 0.8 --n 40 "texto"`: gera, imprimindo token a token.
- `glassllm trace --lang pt "texto" --step attn.softmax --layer 0 --head 0`: imprime a equação
  e a matriz do passo, com os tokens nas linhas e colunas.
- `glassllm train --lang pt --steps 200`: treino curto do zero, imprimindo a perda.

## 5. Dados

- PT: *Dom Casmurro*, Machado de Assis (Projeto Gutenberg, domínio público).
- EN: *Alice's Adventures in Wonderland*, Lewis Carroll (Projeto Gutenberg, domínio público).
- `scripts/prepare_corpus.py` baixa, remove cabeçalho e rodapé do Gutenberg, normaliza e salva
  em `data/`. 95% treino, 5% validação.

## 6. O site (`site/`)

- **Astro**, saída estática, versão estável pinada no `package.json`. Node 22 LTS, instalado no
  usuário via `fnm` (a máquina não tem Node hoje).
- **i18n nativo do Astro:** rotas `/pt/...` e `/en/...`, português como padrão. Textos de
  interface num dicionário por idioma.
- **Capítulos** numa content collection, um MDX por capítulo e idioma. O frontmatter de cada
  capítulo lista os passos (`steps: [attn.scores, attn.mask, attn.softmax, ...]`) e, para cada
  passo, título e texto curto. O schema da collection exige que cada passo exista no
  `step_index.json`.
- **Equações** renderizadas no build com `remark-math` + `rehype-katex`.
- **Ilhas interativas** em TypeScript, sem framework de UI: painel de três colunas, heatmap,
  barras de probabilidade, painel livre.
- **Pyodide** roda num Web Worker, para a interface não travar. Na build, o wheel do
  `glassllm` (gerado na mesma pipeline) é copiado para `site/public/py/`, e o worker instala
  com `micropip`. Versão do Pyodide pinada.
- O worker expõe mensagens simples: `load(lang)`, `tokenize(text)`, `trace(text)`,
  `generate_step()`, `train_steps(n)`. Os arrays voltam como listas, só dos passos pedidos.

### 6.1 Trilha

| # | Capítulo | Passos principais |
|---|---|---|
| 0 | Visão geral | fluxo completo, sem números |
| 1 | Tokenização BPE | fusões, `encode_verbose` |
| 2 | Embedding | `embed` |
| 3 | Atenção | `attn.q/k/v`, `attn.scores`, `attn.softmax`, `attn.mix` |
| 4 | Máscara causal | `attn.mask` |
| 5 | RoPE | `attn.rope_q`, `attn.rope_k` |
| 6 | Multi-head e GQA | cabeças lado a lado, `attn.out` |
| 7 | Residual e RMSNorm | `norm.*`, `resid.*` |
| 8 | FFN SwiGLU | `ffn.*` |
| 9 | Logits, perda e treino | `logits`, `probs`, `loss`, `loss.grad`, treino ao vivo de poucos passos |
| 10 | Geração em loop e KV cache | `sample`, painel livre |

### 6.2 Tela do capítulo (layout A)

Três colunas: **Teoria** (texto curto e equação, com a dedução completa num bloco que abre),
**Código real** (arquivo da lib com a linha do passo destacada) e **Valores** (matriz do passo
com tokens nos eixos, seletor de camada e cabeça). Uma cor liga o termo da equação, a linha e
os valores. Rodapé com navegação de passos. No celular, as colunas viram uma pilha.

### 6.3 Painel livre

Campo de texto inicial, botões Play, Pausa e Um passo, temperatura, sorteio ou mais provável,
velocidade. O destaque percorre Tokens → Embedding → Blocos → Logits → Sorteio; ao lado, a
atenção do último token e o top 6 de probabilidades. O token escolhido entra no texto e o
ciclo recomeça. Clicar num bloco pausa e abre aquele passo no layout A.

## 7. Publicação

- **Repositório:** público no GitHub (`Oseiasdfarias/glassllm`), criado com aprovação do autor.
- **Branches:** desenvolvimento na `develop`. Merge para a `main` e criação de releases são
  feitos pelo autor.
- **`tests.yml`:** roda `pytest` em todo push e PR.
- **`site.yml`:** em push na `main` (e disparo manual), gera o wheel, compila o Astro e publica
  no GitHub Pages com `withastro/action`.
- **`publish.yml`:** quando uma release é publicada, gera o wheel e o sdist e envia ao PyPI com
  trusted publishing (sem token no repositório).
- **Licenças:** código MIT; textos dos capítulos CC BY 4.0.

## 8. Testes

- **Paridade com PyTorch:** para cada camada e para o modelo inteiro, saída e gradientes batem
  com uma implementação em PyTorch com os mesmos pesos (tolerância `1e-5` em float64).
- **Gradiente numérico:** diferença finita nas operações do autograd.
- **Tokenizer:** `decode(encode(x)) == x` para texto dentro do alfabeto; fusões determinísticas.
- **KV cache:** gerar com cache dá os mesmos logits que rodar o contexto inteiro.
- **Catálogo de passos:** todo `@step` no código tem entrada em `steps.py` e é registrado ao
  rodar o modelo com tracer.
- **Pesos empacotados:** `load_pretrained` funciona a partir do wheel instalado, e a perda
  de validação com os pesos em fp16 fica a no máximo 0,01 da registrada ao fim do treino.
- **Site:** `astro check` na build; conferência manual no navegador ao fim de cada fase.

## 9. Fases

1. **Lib + docs:** tokenizer, autograd, modelo, trace, treino, CLI, pesos PT e EN, testes. Ao
   final, dá para estudar pelo terminal e o pacote está pronto para o PyPI.
2. **Site + painel livre:** Astro, i18n, Pyodide no worker, painel de geração em loop com o
   modelo real, deploy no GitHub Pages.
3. **Capítulos:** do 0 ao 10, um por vez, PT e EN.

Cada fase tem o seu plano de implementação.

## 10. Fora do escopo

- GPU, modelos grandes ou carregar modelos do Hugging Face.
- Treinar com corpus enviado pelo usuário.
- Contas, login ou salvar progresso no servidor.
- BPE em nível de byte (fica citado no capítulo 1 como variação).
- Exportar GIF ou vídeo do painel (pode virar fase futura para o LinkedIn).
