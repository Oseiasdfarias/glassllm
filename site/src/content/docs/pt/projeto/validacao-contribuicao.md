---
title: Validação e contribuição
description: Como verificar mudanças no motor, nas páginas e na visualização sem perder a ligação entre equação, tensor e código.
---

## O que os testes verificam

A biblioteca tem testes de autograd, camadas, modelo, geração, tokenizer, treino, assets, CLI e marcadores de etapas. A referência PyTorch existe apenas no ambiente de desenvolvimento para comparar valores e gradientes; o pacote de execução depende de NumPy.

| Área | Evidência conferida |
| --- | --- |
| Autograd | Gradientes comparados com PyTorch e diferenças finitas; broadcasting, indexação repetida e caminhos compartilhados |
| Camadas | RMSNorm, RoPE, atenção causal/GQA e SwiGLU: forward e backward |
| Modelo | Formas, perda, parâmetros compartilhados e save/load |
| Geração | Cache versus forward completo, reconstrução da janela, temperatura, sementes e trace por previsão |
| Tokenizer | Fusões, normalização Unicode, espaço, desconhecidos e persistência |
| Treino | Batches deslocados, AdamW, clipping, schedule, avaliação e melhor checkpoint |
| CLI e assets | Argumentos, saída, arquivos PT/EN e erros de carregamento |
| Etapas | Correspondência entre `STEPS`, marcadores de código e índice de origem |

A paridade de um caso não demonstra que qualquer configuração possível funciona. Cada teste cobre entradas e tolerâncias específicas; mudanças numéricas exigem verificar o que foi alterado, incluindo formas e dtypes.

## Executar as verificações

Na raiz do checkout:

```sh
uv sync --group dev
uv run pytest -q
uv run glassllm --help
```

O workflow `tests.yml` executa a suíte nas versões Python 3.10 e 3.13. Para uma alteração limitada, execute primeiro o arquivo relacionado; rode a suíte completa quando mudar contratos compartilhados, autograd ou o modelo. PyTorch é instalado pelo grupo de desenvolvimento, usando seu índice CPU configurado no `pyproject.toml`.

Para o site, a partir de `site/`:

```sh
npm ci
npm run check
npm run build
```

`check` prepara o wheel e verifica Astro/TypeScript; `build` gera páginas estáticas e copia os assets. Esses comandos não demonstram sozinhos que o runtime Python carregou no navegador. Gere um token real, selecione uma operação, compare forma e código, e confira pausa/retomada. Se alterar rotas ou navegação, percorra PT e EN, incluindo os links de artigos e as imagens.

## Atualizar as fontes derivadas

```sh
uv run python scripts/build_step_index.py
uv run python scripts/build_api_inventory.py
```

O primeiro comando reconstrói `src/glassllm/step_index.json`; o segundo gera `site/src/data/api-inventory.json` a partir da AST Python. Rode-os quando mudar posições de marcadores ou assinaturas. O inventário fornece nomes e assinaturas; a documentação explica o contrato, os retornos e os efeitos da função.

Conteúdo PT/EN fica em `site/src/content/docs/{pt,en}/`. As rotas e grupos são definidos em `docs-plan.ts`; o renderer importa o Markdown correspondente e rejeita conteúdo ausente. Ao alterar uma função, revise as duas páginas de referência e os fundamentos que a utilizam.

## Conferir interface e prints

Com o servidor local ativo na porta 4322, Node.js 24 e Chrome instalados, execute em `site/`:

```sh
npm run capture:docs
```

O script registra 17 capturas por idioma, usando o modelo real para previsão, trace, máscara causal e limite de 128 tokens. Também reproduz uma falha de wheel e sua recuperação. O `manifest.json` identifica idioma, frase, modo de escolha, temperatura, operação, cabeça, viewport e dimensões de cada imagem. Use `GLASSLLM_CAPTURE_URL` para outro servidor e `GLASSLLM_CHROME` para outro caminho do executável.

Confira também o teclado: foco visível, controle de abas, expansão do índice, links e tela cheia. As larguras de 390 e 320 px ajudam a detectar transbordamento, mas não substituem a leitura em um dispositivo real. Ao capturar, o inspetor é expandido para tornar matriz e fonte legíveis; os valores continuam sendo os calculados pelo modelo.

## Contribuir com uma mudança

`CONTRIBUTING.md` descreve o processo do repositório. A branch de integração é `develop`; `main` representa versões lançadas. Um PR deve explicar o problema, o comportamento resultante e a validação pertinente. Inclua prints quando mudar a apresentação visual.

Uma operação nova precisa de marcador `# @step`, equação no catálogo, índice de código atualizado e explicação nos dois idiomas. Prefira nomes que correspondam às equações e mantenha o motor executável com NumPy. Uma modificação de arquitetura deve explicitar diferenças em relação aos artigos usados como referência. A publicação da biblioteca e do site é uma etapa separada das verificações locais.
