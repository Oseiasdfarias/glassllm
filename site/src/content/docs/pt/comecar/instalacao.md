---
title: "Instalação e execução local"
description: "Execute a biblioteca, a CLI e o laboratório bilíngue a partir do checkout, usando os modelos e o wheel locais."
---

## Requisitos e arquivos do projeto

A biblioteca exige Python 3.10 ou superior e NumPy 1.26 ou superior. O ambiente de desenvolvimento usa `uv`; os testes de paridade também usam PyTorch. O site usa Node.js 22.12 ou superior e npm. O comando de captura de screenshots exige Node.js 24 e Chrome instalado.

Execute os comandos desta página a partir de um checkout do repositório. Os assets treinados ficam em `src/glassllm/weights/`: `pt.npz`, `en.npz` e os dois arquivos `*.tokenizer.json`. `load_pretrained` exige esses arquivos; não cria pesos aleatórios para substituir assets ausentes.

## Preparar o motor e testar a CLI

Na raiz do checkout:

```bash
uv sync
uv run glassllm --lang pt tokenize "Capitu olhou para"
uv run glassllm --lang pt generate "Capitu olhou para" --n 8 --greedy
uv run glassllm --lang en generate "Alice was beginning to" --n 8 --greedy
uv run glassllm steps
```

`uv sync` instala o pacote local e o grupo de desenvolvimento. A primeira sincronização pode baixar dependências grandes por incluir PyTorch para os testes. A CLI recebe `--lang pt` ou `--lang en`; sem essa opção, seu padrão é inglês. O comando `steps` lista as operações marcadas, enquanto `tokenize` mostra os IDs do vocabulário daquele idioma.

Para um ambiente dedicado apenas à execução da biblioteca, você também pode instalar o checkout em um ambiente virtual existente:

```bash
python -m pip install -e .
```

A dependência de execução é NumPy. Para instalar o pacote publicado em vez do checkout, rode `python -m pip install glassllm`. Os modelos treinados e tokenizers estão incluídos na distribuição.

## Abrir o laboratório

Na raiz do projeto, entre em `site/` e instale as versões do lockfile:

```bash
cd site
npm ci
npm run dev -- --host 127.0.0.1 --port 4322
```

Abra `http://localhost:4322/glassllm/pt/` para o laboratório em português ou troque `pt` por `en`. A documentação fica em `/glassllm/pt/docs/` e `/glassllm/en/docs/`. A base `/glassllm/` faz parte das URLs do site, inclusive em desenvolvimento.

O script `dev` executa `prepare:wheel` antes do Astro. Esse preparo chama `uv build --wheel`, copia o wheel para `site/public/py/` e `site/src/generated/` e informa seu nome ao cliente. Portanto, `uv` precisa estar disponível mesmo quando você está iniciando apenas o site.

No primeiro clique em **Gerar** ou **Um passo**, o worker carrega Pyodide 0.29.3, instala NumPy e micropip e instala o wheel local. Aguarde o estado de preparação. Após o carregamento, os cálculos da frase acontecem no navegador; o modelo não precisa de uma API de inferência.

## Verificar build e servir a saída estática

Dentro de `site/`:

```bash
npm run check
npm run build
npm run preview -- --host 127.0.0.1
```

`check` valida Astro e TypeScript. `build` recria o wheel e gera os arquivos estáticos em `site/dist/`. `preview` mostra essa saída já construída; use o endereço e a porta impressos no terminal e mantenha a base `/glassllm/`. Esses comandos não publicam o projeto.

## Quando o carregamento falha

Confira primeiro o terminal que iniciou o Astro: uma falha em `uv build` impede o preparo do wheel. Se a interface abrir mas o modelo falhar, expanda **Detalhes técnicos** e confirme o acesso à CDN do Pyodide e ao wheel da mesma origem do site.

Não renomeie um wheel para `glassllm` ou para um nome de arquivo arbitrário: micropip valida a estrutura de nomes como `glassllm-0.1.0-py3-none-any.whl`. Depois de corrigir os assets, tente **Um passo** novamente ou recarregue a página. Veja o guia de [estados e erros](../../interface/estados-erros/) para distinguir problemas de download, instalação e dados do modelo.
