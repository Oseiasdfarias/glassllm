# Identidade visual do glassllm

O tema atual usa uma base preta e cinzas neutros, com o ciano original do projeto como
cor principal. A referência de sobriedade é o uso de cinza para fundos, superfícies,
texto, ícones e bordas no [sistema de cores Geist](https://vercel.com/geist/colors).
O glassllm mantém sua marca, fontes e componentes próprios.

Os arquivos [preview-loop.png](assets/preview-loop.png),
[preview-chapter.png](assets/preview-chapter.png) e os banners
[claro](assets/banner-light.svg) e [escuro](assets/banner-dark.svg) registram a demo original.
Eles são referências históricas de composição; a antiga paleta azul-marinho não define
o tema aplicado ao laboratório e à documentação.

| Uso | Cor / variável |
| --- | --- |
| Fundo da página | `#0a0a0a` · `--background` |
| Cartões | `#111111` · `--surface` |
| Superfície secundária e seleção neutra | `#141414` · `--surface-raised` |
| Código, editor e mapa | `#080808` · `--code-background` |
| Bordas | `#292929` · `--border` |
| Texto principal | `#ededed` · `--text` |
| Texto secundário | `#a1a1a1` · `--muted` |
| Legendas e informações auxiliares | `#999999` · `--muted-secondary` |
| Linhas principais do grafo | `#737373` · `--graph-edge` |
| Ações, links, seleções e valores positivos | `#4fb3c2` · `--accent` |
| Cálculo ativo e valores negativos | `#e7b04a` · `--highlight` |
| Fundo do cálculo ativo | `#1c170e` · `--highlight-background` |
| Erro de execução | `#ef8585` · `--error` |

As superfícies são sólidas, sem gradientes, glow, sombras ou textura decorativa. Os cartões
usam borda de 1 px e raio de 8 px. Áreas de código têm fundo próprio para separar números
e expressões sem adicionar cores decorativas. O cabeçalho usa o fundo da página e uma
linha de separação, preservando contraste durante a rolagem.

O ciano indica interatividade e seleção. Âmbar fica reservado à operação em estudo,
ao fluxo ativo e aos números negativos, incluindo `−∞` da máscara causal. Carregamento,
tokens escolhidos e controles de slow motion usam ciano; mensagens de falha usam vermelho.
Valores não finitos positivos ou indefinidos recebem apresentação neutra.

## Tipografia e ícones

A interface e a documentação usam **Source Sans 3**; tokens, IDs, equações, código e números
usam **Source Code Pro**. As fontes variáveis são servidas pelo próprio site, sem requisições
a serviços externos. Suas licenças OFL ficam em `site/public/fonts/`.

O corpo da documentação tem 17 px no desktop e 16 px no celular, com entrelinha de 1,7–1,8
e largura de leitura limitada. Introduções usam 19–20 px; blocos de código, 14 px no desktop
e 13 px no celular. A hierarquia de títulos e os tamanhos da interface são preservados
entre as versões do tema. Os prints mantêm as proporções e oferecem um link para abrir
a imagem original.

Os controles usam ícones SVG de contorno com geometria de 24 px, traço de 1,75 px e
extremidades arredondadas. Os ícones compartilham as definições de `site/src/lib/icons.ts`
e acompanham os rótulos dos controles. Símbolos de equações e notação de tokens mantêm
seu significado matemático ou textual.

## Laboratório, mapas e documentação

O cabeçalho reúne marca, documentação e PT/EN. O texto inicial e as ações de geração
ocupam o painel principal; os ajustes ficam numa barra lateral. O fluxo do modelo tem
uma área própria com os quatro blocos agrupados. Atenção e probabilidades aparecem
em dois painéis de leitura. No celular, as áreas se organizam em sequência.

A página principal da documentação combina uma ilustração do Transformer, percursos
de leitura, fluxogramas de geração e treinamento e uma ponte visual entre interface,
worker e biblioteca. A ilustração é esquemática: blocos, Q/K/V e cartões usam tons
neutros; vetores e pontos de atenção recebem ciano com moderação. Seus números não
substituem os tensores reais do laboratório.

A arquitetura completa ocupa uma seção própria abaixo do laboratório. Quatro colunas
mostram os blocos abertos com ramificações Q/K/V, cache e caminhos residuais. O desenho
SVG permite zoom, movimentação e tela cheia; a inspeção fica ao lado no desktop e abaixo
no celular. Cálculos usam linhas cinza; resíduos usam cinza tracejado; cache usa ciano
contínuo; pesos compartilhados usam ciano tracejado. A legenda conserva essas distinções
sem depender de uma paleta auxiliar azul, verde ou violeta.

O fluxo ativo usa âmbar. Os 77 passos rastreados exibem valores reais; estruturas sem
snapshot, como cache e GQA, são identificadas como simbólicas. **Slow motion** agrupa
tempo de leitura e acompanhamento automático. O cartão da etapa destaca número,
nome, explicação e equação. Pausa, retomada, anterior e próximo permitem ler cada
operação sem alterar a previsão estudada.

Capturas documentais devem refletir o tema vigente. O comando `npm run capture:docs`,
executado em `site/` com o servidor ativo, gera 17 prints por idioma e seu manifesto
de contexto. A regeneração das imagens acompanha alterações de paleta e componentes.
