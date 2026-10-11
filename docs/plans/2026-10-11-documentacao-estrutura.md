# Documentação — estrutura aprovada e implementação

Estado: **redação integral implementada após autorização do autor**. São 35 páginas de
conteúdo por idioma, em Markdown PT/EN, com explicações da interface, fundamentos, referência
da API, exemplos executados, prints e bibliografia. A página inicial tem ilustração SVG,
fluxogramas de inferência/treino e caminhos de leitura ligados às páginas.

Prévia local: [índice da documentação](http://localhost:4322/glassllm/pt/docs/).
São 35 páginas de conteúdo em cinco grupos, mais o índice, com navegação PT/EN.

## Mapa de páginas

As rotas abaixo ficam sob `/glassllm/{pt,en}/docs/`. Os slugs são compartilhados pelos
idiomas; títulos, escopos e tópicos estão traduzidos na estrutura.

| Grupo | Rota | Página |
| --- | --- | --- |
| Primeiros passos | `comecar/visao-geral/` | O que é o glassllm |
| | `comecar/instalacao/` | Instalação e execução local |
| | `comecar/primeiro-experimento/` | Primeiro experimento: site e Python |
| Guia da interface | `interface/mapa-componentes/` | Mapa dos componentes |
| | `interface/geracao-tokens/` | Texto, geração e tokens |
| | `interface/ajustes/` | Temperatura e escolha do token |
| | `interface/atencao-probabilidades/` | Ler atenção e probabilidades |
| | `interface/arquitetura/` | Explorar a arquitetura inteira |
| | `interface/slow-motion/` | Estudar em slow motion |
| | `interface/inspetor/` | Inspecionar tensores e código |
| | `interface/estados-erros/` | Estados, limites e solução de problemas |
| Fundamentos | `fundamentos/tokens-bpe/` | Tokens e BPE |
| | `fundamentos/embeddings/` | Embedding e pesos compartilhados |
| | `fundamentos/atencao-causal/` | Atenção causal: Q, K e V |
| | `fundamentos/rope/` | Posição e RoPE |
| | `fundamentos/gqa-cache/` | Cabeças, GQA e cache K/V |
| | `fundamentos/normalizacao-residuos/` | RMSNorm, pre-norm e resíduos |
| | `fundamentos/swiglu/` | FFN e SwiGLU |
| | `fundamentos/saida-geracao/` | Logits, temperatura e geração |
| | `fundamentos/perda-gradientes/` | Perda e backpropagation |
| Biblioteca | `biblioteca/api-publica/` | Mapa da API e convenções |
| | `biblioteca/config-carregamento/` | Configuração e modelos treinados |
| | `biblioteca/tokenizer/` | Tokenizer: referência completa |
| | `biblioteca/glasslm/` | GlassLM e GenStep |
| | `biblioteca/trace-etapas/` | Tracer, registros e catálogo de etapas |
| | `biblioteca/tensor-autograd/` | Tensor e motor de autograd |
| | `biblioteca/operacoes-autograd/` | Operações diferenciáveis |
| | `biblioteca/camadas-cache/` | Camadas e cache |
| | `biblioteca/treino-otimizacao/` | Treinamento e otimização |
| | `biblioteca/cli-arquivos/` | CLI, arquivos e scripts |
| Projeto e referências | `projeto/integracao-browser/` | Como a interface usa o motor |
| | `projeto/dados-modelos/` | Dados, modelos e reprodutibilidade |
| | `projeto/validacao-contribuicao/` | Validação e contribuição |
| | `projeto/glossario/` | Glossário e notação |
| | `projeto/referencias/` | Artigos e mapa de referências |

## Padrão editorial adotado

**Interface:** objetivo do componente; print real e identificação dos controles no texto;
explicação de cada controle; sequência de uso; estados e limites; relação com o motor.
Os estados apresentados devem ser reproduzidos no laboratório atual.

**Fundamentos:** intuição; notação e dimensões; equação; dedução; exemplo numérico;
ligação com uma operação do código; leitura no inspetor; artigo primário. A arquitetura
do projeto combina escolhas de artigos diferentes; as diferenças estão explícitas nas páginas.
As deduções de `docs/equacoes.md` foram revisadas, incluindo os limites das simplificações
sobre invariância com ε, propagação de gradientes, GQA, cache e contagem de parâmetros.

**Referência de funções:** caminho de importação e assinatura real; finalidade; argumentos,
tipos e padrões; formas de entrada e saída; retorno; mutações de estado; exceções;
exemplo executável; resultado conferido; equação e fonte; referências e páginas relacionadas.
Operadores, propriedades e métodos herdados são descritos dentro de suas classes.

**Projeto:** fluxo de integração; arquivos e versões; procedência dos dados; limitações;
reprodutibilidade; validação e contribuição.

## Prints gerados

| Componente | Capturas disponíveis por idioma | O que explicar |
| --- | --- | --- |
| Laboratório | Visão geral desktop | Hierarquia dos painéis e sequência do primeiro uso |
| Editor | Recorte do painel com uma previsão | Frase, controles, continuação, tokens e IDs |
| Ajustes | Recorte da barra lateral | Temperatura, escolha, intervalo e runtime |
| Atenção/probabilidades | Recorte dos dois painéis | Pesos, candidatos e diferença entre os painéis |
| Arquitetura | Visão completa e bloco ampliado | Setas, ramificações, cache, resíduos e pesos E |
| Slow motion | Etapa pausada com inspetor | Tempo por cálculo, anterior/próximo e retomada |
| Inspetor | Operação selecionada e máscara causal | Cabeça, formas, recorte, estatísticas, −∞ e código |
| Estados | Carregando, pronto, pausado, limite e erro | Diagnóstico e recuperação |
| Celular/tela cheia | Larguras de 390/320 px e mapa ampliado | Leitura, navegação, rolagem e acessibilidade |

Há **34 prints reais: 17 em português e 17 em inglês**, distribuídos nas nove categorias
acima. As páginas mostram as imagens do próprio idioma, com legendas, texto alternativo e
link para abrir o original. O limite foi capturado após 128 previsões reais; o erro veio de
uma tentativa de carregar um wheel indisponível, seguida de recuperação verificada.

Os prints do inspetor expandem o painel para 600 px e removem sua rolagem vertical durante
a captura, permitindo ler a matriz e o trecho de código completos. Os dados e a operação
selecionada são os mesmos da interface. As imagens de celular usam larguras de 390 e 320 px.
Todas as capturas usam escala de pixels 2; a interface desktop usa uma janela de 1366 × 1000 px.

O arquivo [manifest.json](../../site/public/docs/screenshots/manifest.json) registra data,
idioma, frase, modo de amostragem, temperatura, contagem de tokens, operação selecionada,
cabeça, estado, fonte, dimensões e recorte. A versão da aplicação deve ser identificada pelo
commit do checkout quando os materiais forem publicados. As páginas explicam os controles,
suas ações e as limitações dos dados apresentados nos prints.

Para reproduzir, mantenha o servidor local ativo e execute, com Node.js 24 e Chrome instalados:

```sh
cd site
npm run capture:docs
```

A URL padrão é `http://127.0.0.1:4322/glassllm/`. `GLASSLLM_CAPTURE_URL` permite usar outro
servidor, e `GLASSLLM_CHROME` permite indicar o executável do Chrome. O script também verifica
as 70 páginas de conteúdo, as imagens, as fontes locais, os ícones SVG, a tela cheia e a
ausência de transbordamento horizontal nas duas larguras de celular.

`npm run check:docs` confere a redação de todas as páginas, ausência de placeholders
editoriais, âncoras do índice, links internos, imagens, BibTeX, abas dos fluxogramas,
fontes e paleta. Usa Chrome local e as mesmas variáveis de URL/executável das capturas.

## Cobertura conferida no código

A raiz exporta `Config`, `GenStep`, `GlassLM`, `LANGUAGES`, `STEPS`, `StepRecord`, `Tensor`,
`Tokenizer`, `Tracer`, `load_pretrained`, `no_grad` e `__version__`.

O inventário AST lista 64 declarações de classes/funções em 11 módulos, com métodos,
operadores, propriedades e campos declarados. A referência distingue exports da raiz
de símbolos acessíveis por submódulo. Constantes e atributos de runtime são cobertos
no escopo das páginas relacionadas; helpers privados são explicados quando necessários
para entender a implementação, sem serem apresentados como contratos públicos.

Contratos explicitados na documentação:

- `generate(trace=True)` recebe um booleano e cria um tracer por previsão.
- `forward(trace=...)` recebe um `Tracer`; não é o mesmo contrato de `generate`.
- `Tokenizer.tokens(ids)` é a API de consulta; não existe `id_to_token`.
- Embedding é uma função de `autograd`, não uma classe `Embedding`.
- O tracer remove o eixo de batch e acompanha uma sequência por vez.
- Existem 25 IDs de etapa; a geração padrão produz 77 registros. `loss` e `loss.grad`
  pertencem ao caminho de treino.
- Slow motion reproduz snapshots reais; não introduz pausas entre operações dentro do NumPy.
- O cache processa o prefixo e depois o token novo; ao encher a janela, é reconstruído.
- BOS/EOS não são inseridos automaticamente e EOS não encerra a geração atual.
- Checkpoints salvam configuração e pesos; não incluem estado do otimizador ou histórico.
- A instalação parte do checkout/wheel local, pois a publicação da lib continua adiada.

## Referências documentadas

Os metadados foram conferidos nas páginas primárias do arXiv em 2026-10-11. Os anos abaixo
são da primeira versão do preprint. A bibliografia apresenta os autores completos, distingue
notas de publicação dos anos dos preprints e oferece 14 entradas BibTeX individuais e em download.

| Artigo | Tema |
| --- | --- |
| Vaswani et al. (2017), [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Atenção escalada e Transformer |
| Sennrich et al. (2015), [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) | BPE |
| Kudo e Richardson (2018), [SentencePiece](https://arxiv.org/abs/1808.06226) | Tokenização independente de idioma; implementação própria neste projeto |
| Su et al. (2021), [RoFormer](https://arxiv.org/abs/2104.09864) | RoPE |
| Ainslie et al. (2023), [GQA](https://arxiv.org/abs/2305.13245) | Cabeças K/V compartilhadas; não reproduzimos a conversão de checkpoints |
| Zhang e Sennrich (2019), [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) | RMSNorm |
| Xiong et al. (2020), [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) | Pre-norm; o artigo analisa LayerNorm, enquanto o código usa RMSNorm |
| He et al. (2015), [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) | Contexto histórico dos resíduos |
| Shazeer (2020), [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) | SwiGLU |
| Press e Wolf (2016), [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859) | Pesos atrelados |
| Baydin et al. (2015), [Automatic differentiation in machine learning: a survey](https://arxiv.org/abs/1502.05767) | Autograd e modo reverso |
| Kingma e Ba (2014), [Adam](https://arxiv.org/abs/1412.6980) | Otimização adaptativa |
| Loshchilov e Hutter (2017), [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) | AdamW |
| Loshchilov e Hutter (2016), [SGDR](https://arxiv.org/abs/1608.03983) | Decaimento cosseno; o código não usa warm restarts |

## Arquivos e manutenção

A navegação e os metadados dos assets ficam em [docs-plan.ts](../../site/src/lib/docs-plan.ts).
As referências ficam em [docs-references.ts](../../site/src/lib/docs-references.ts).
O inventário é gerado por [build_api_inventory.py](../../scripts/build_api_inventory.py)
em [api-inventory.json](../../site/src/data/api-inventory.json).

Os textos ficam em `site/src/content/docs/{pt,en}/`. O renderer importa os arquivos Markdown,
gera índices de seções e rejeita conteúdo ausente. A página inicial é composta por
`DocumentationHome.astro`, `DocumentationFlow.astro` e `TransformerIllustration.astro`.
`DocumentationPapers.astro` conecta os artigos às páginas, e o endpoint
`docs/glassllm-references.bib` produz o download a partir dos mesmos metadados.

Para manter a documentação: acompanhe mudanças nas assinaturas e na interface, atualize PT/EN,
execute os exemplos afetados, regenere inventário/índice quando necessário e recapture os
prints após mudanças visuais. A publicação da biblioteca continua separada desta implementação.
