# Contributing to glassllm

**English** · [Português](#contribuindo-com-o-glassllm)

Thanks for wanting to help. glassllm is a teaching project, so a clearer sentence is as
valuable as a faster function.

## Ways to help

| Kind | Examples |
|---|---|
| Explanations | Fix a derivation, add an intuition, simplify a paragraph |
| Translations | Keep English and Portuguese chapters in sync |
| Code | New tests, bug fixes, performance that keeps the code readable |
| Visuals | Better ways to show matrices, rotations, gradients |
| Bug reports | A wrong number, a broken page, an install problem |

If you plan a large change, open an issue first so we can agree on the approach.

## Ground rules

1. **Readability beats cleverness.** Someone is going to learn from this code. Prefer explicit
   NumPy over tricks, and name variables after the math (`q`, `k`, `v`, `scores`, `weights`).
2. **Every computation stays traceable.** If you add or change a step of the model:
   - mark the line with `# @step <id>`,
   - add the equation for `<id>` to `src/glassllm/steps.py`,
   - add or update the step text in both languages in the chapter that shows it.
   The test suite fails if any of the three is missing.
3. **Every layer is checked against PyTorch.** New layers come with a parity test (forward and
   backward). PyTorch is a development dependency only; the library must keep running with
   NumPy alone.
4. **Both languages move together.** A change to a chapter in one language needs the same
   change in the other. If you can only write one, say so in the PR and we will help.
5. **Plain language in explanations.** Short sentences, everyday words, and every term of art
   explained the first time it appears.

## Development setup

> The library is being built (phase 1). These commands will work once `pyproject.toml` lands.

```bash
git clone https://github.com/Oseiasdfarias/glassllm.git
cd glassllm
uv sync --group dev          # numpy, plus pytest and torch (CPU) for tests
uv run pytest                # run the test suite
uv run glassllm --help       # try the CLI
```

The site lives in `site/` and needs Node.js 22:

```bash
cd site
npm install
npm run dev
```

## Branches and pull requests

- `develop` is the integration branch. **Open pull requests against `develop`.**
- `main` holds released versions only. Maintainers merge `develop` into `main` when cutting a release.
- Name branches by intent: `feat/rope-chapter`, `fix/softmax-overflow`, `docs/attention-pt`.
- Use [Conventional Commits](https://www.conventionalcommits.org/) for messages:
  `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`.
- Keep a PR about one thing. Fill in the template, and add a screenshot for visual changes.

## Reporting bugs

Use the issue templates. A good report says what you ran, what you expected, what you got,
and your Python version (or browser, for the site).

---

# Contribuindo com o glassllm

[English](#contributing-to-glassllm) · **Português**

Obrigado por querer ajudar. O glassllm é um projeto de ensino, então uma frase mais clara vale
tanto quanto uma função mais rápida.

## Jeitos de ajudar

| Tipo | Exemplos |
|---|---|
| Explicações | Corrigir uma dedução, acrescentar uma intuição, simplificar um parágrafo |
| Traduções | Manter os capítulos em português e inglês iguais |
| Código | Testes novos, correções, desempenho sem perder a legibilidade |
| Visual | Jeitos melhores de mostrar matrizes, rotações, gradientes |
| Relatos de bug | Um número errado, uma página quebrada, problema de instalação |

Se a mudança for grande, abra uma issue antes para combinarmos a abordagem.

## Regras básicas

1. **Legibilidade vem antes de esperteza.** Alguém vai aprender com este código. Prefira NumPy
   explícito a truques, e dê às variáveis os nomes da matemática (`q`, `k`, `v`, `scores`, `weights`).
2. **Toda conta continua rastreável.** Se você criar ou mudar um passo do modelo:
   - marque a linha com `# @step <id>`,
   - acrescente a equação do `<id>` em `src/glassllm/steps.py`,
   - acrescente ou atualize o texto do passo nos dois idiomas, no capítulo que o mostra.
   Os testes falham se faltar qualquer um dos três.
3. **Toda camada é conferida contra o PyTorch.** Camadas novas vêm com teste de paridade (ida e
   volta). O PyTorch é só dependência de desenvolvimento; a lib precisa continuar rodando só
   com NumPy.
4. **Os dois idiomas andam juntos.** Mudou um capítulo num idioma, muda no outro. Se você só
   escreve em um, avise no PR que a gente ajuda.
5. **Linguagem simples nas explicações.** Frases curtas, palavras do dia a dia, e todo termo
   técnico explicado na primeira vez que aparece.

## Ambiente de desenvolvimento

> A lib está sendo construída (fase 1). Estes comandos passam a funcionar quando o
> `pyproject.toml` entrar.

```bash
git clone https://github.com/Oseiasdfarias/glassllm.git
cd glassllm
uv sync --group dev          # numpy, mais pytest e torch (CPU) para os testes
uv run pytest                # roda os testes
uv run glassllm --help       # experimenta a CLI
```

O site fica em `site/` e precisa do Node.js 22:

```bash
cd site
npm install
npm run dev
```

## Branches e pull requests

- `develop` é a branch de integração. **Abra pull requests para a `develop`.**
- `main` guarda só versões lançadas. Quem mantém o projeto faz o merge da `develop` na `main` ao lançar uma versão.
- Dê nome às branches pela intenção: `feat/capitulo-rope`, `fix/softmax-overflow`, `docs/atencao-pt`.
- Use [Conventional Commits](https://www.conventionalcommits.org/pt-br/) nas mensagens:
  `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`.
- Um PR trata de uma coisa só. Preencha o template e, em mudanças visuais, mande um print.

## Relatando bugs

Use os templates de issue. Um bom relato diz o que você rodou, o que esperava, o que
aconteceu e a sua versão do Python (ou o navegador, no caso do site).
