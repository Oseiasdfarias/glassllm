---
title: "Configuração e modelos treinados"
description: "Campos de Config, validação, modelos PT/EN e carregamento de assets locais."
---

## Config

Importação: `from glassllm import Config`. É uma dataclass mutável com a assinatura abaixo. Ela descreve a arquitetura; não contém pesos nem executa treinamento.

```text
Config(vocab_size=1024, d_model=128, n_layers=4, n_heads=4,
       n_kv_heads=2, max_len=128, rope_base=10000.0,
       eps=1e-6, init_std=0.02)
```

| Campo | Contrato e efeito |
| --- | --- |
| `vocab_size` | Inteiro positivo; número de linhas do embedding e de logits |
| `d_model` | Inteiro positivo; largura dos estados e dimensão residual |
| `n_layers` | Inteiro positivo; quantidade de blocos Transformer |
| `n_heads` | Inteiro positivo; quantidade de cabeças Q |
| `n_kv_heads` | Inteiro positivo; quantidade de cabeças K/V |
| `max_len` | Inteiro positivo; tamanho da janela e das tabelas RoPE |
| `rope_base` | Positivo; base das frequências das rotações |
| `eps` | Não negativo; estabilizador do denominador RMSNorm |
| `init_std` | Não negativo; desvio padrão dos pesos aleatórios |

Os seis campos inteiros rejeitam booleanos, valores zero, negativos e objetos que não sejam `int`. A dimensão precisa ser divisível por `n_heads`; `n_heads` precisa ser divisível por `n_kv_heads`; `d_model / n_heads` precisa ser par para rotacionar pares de coordenadas. Uma violação produz `ValueError` na construção.

A validação ocorre em `__post_init__`, e não ao alterar atributos depois. Trate a configuração de um modelo existente como fixa: mudar `cfg.d_model` não redimensiona matrizes. Valores como NaN não têm validação de finitude dedicada nos campos reais; forneça números finitos e mantenha `eps` positivo ao trabalhar com vetores nulos.

```python
from dataclasses import asdict
from glassllm import Config, GlassLM

cfg = Config(vocab_size=8, d_model=8, n_layers=1,
             n_heads=2, n_kv_heads=1, max_len=4)
model = GlassLM(cfg, seed=0)
assert asdict(cfg)["n_kv_heads"] == 1
assert model([1, 2]).shape == (1, 2, 8)
try:
    Config(d_model=127)
except ValueError:
    pass
else:
    raise AssertionError("invalid head dimensions must fail")
```

## load_pretrained

Importação: `from glassllm import load_pretrained, LANGUAGES, __version__`.

`load_pretrained(lang: str, model_dir: str | Path | None = None) -> tuple[GlassLM, Tokenizer]` lê um par de arquivos existentes e retorna um modelo com parâmetros float32 e seu tokenizer. `LANGUAGES == ("pt", "en")`; `__version__ == "0.1.0"`. `lang` é sensível a maiúsculas/minúsculas e não aceita códigos como `pt-BR`.

Se `model_dir` for `None`, a função usa os recursos `glassllm/weights/`. Um diretório alternativo deve conter `{lang}.npz` e `{lang}.tokenizer.json`. Arquivos `{lang}.meta.json` descrevem o treinamento, mas não são necessários para a leitura e não são retornados. Não há download, cadastro, API remota nem criação automática de pesos quando assets faltam.

```python
from glassllm import LANGUAGES, load_pretrained

assert LANGUAGES == ("pt", "en")
for lang in LANGUAGES:
    model, tokenizer = load_pretrained(lang)
    assert model.cfg.vocab_size == tokenizer.vocab_size == 1024
    assert sum(p.data.size for p in model.parameters()) == 852608
    step = next(model.generate(tokenizer.encode("Alice"), 1, greedy=True))
    assert step.probs.shape == (1024,)
```

O carregador rejeita idioma desconhecido com `ValueError`, ausência de um dos dois arquivos com `FileNotFoundError` e divergência entre tamanhos de vocabulário com `ValueError`. Um JSON ou NPZ inválido pode produzir erros de seus leitores. A igualdade de tamanho não verifica que dois tokenizers têm a mesma ordem de tokens: sempre preserve o par que foi usado no treinamento.

## Configuração dos assets incluídos

PT e EN usam `Config()` com 4 blocos, dimensão 128, 4 cabeças Q, 2 cabeças K/V e janela 128. Cada cabeça tem 32 coordenadas; a camada SwiGLU usa `int(8 * 128 / 3) = 341` unidades ocultas. O total de 852.608 parâmetros conta a matriz `E` uma única vez, pois ela também projeta a saída.

Carregar um modelo treinado não muda seus limites: ele continua sendo um modelo didático pequeno, treinado para continuação de texto, sem protocolo de chat ou seguimento de instruções. Para construir outra arquitetura, crie `GlassLM(Config(...), seed=...)`; para transportar seus pesos, use [save/load](../glasslm/).
