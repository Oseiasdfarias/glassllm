---
title: "Mapa dos componentes"
description: "Oriente-se pelo editor, ajustes, painéis de leitura e arquitetura interativa do laboratório."
---

## Leia o laboratório em duas escalas

A área superior permite escrever e comparar previsões rapidamente. A arquitetura completa, abaixo, permite seguir as operações que produziram a previsão selecionada. As duas áreas usam o resultado do mesmo modelo; a passagem de uma para outra não inicia um segundo motor.

O cabeçalho oferece o laboratório, esta documentação, **Como funciona** e a troca de idioma **PT / EN**. Trocar o idioma abre outra página e carrega os assets daquele idioma quando você iniciar uma previsão. Não traduz a continuação atual nem conserva a rodada em andamento.

## Componentes e perguntas que respondem

| Componente | Pergunta respondida | O que observar |
| --- | --- | --- |
| Texto inicial | Que prefixo vou fornecer? | Frase de até 600 caracteres |
| Gerar / Um passo / Pausar / Limpar | Como avançar a rodada? | Estado de execução e quantidade gerada |
| Texto em geração | Qual continuação já foi produzida? | Prefixo seguido das decodificações dos IDs |
| Tokens BPE | Como o texto é representado? | Pedaço do vocabulário e seu ID |
| Percurso do token | Em qual parte da rede estamos? | Embedding, blocos, logits e escolha |
| Atenção | Como a última consulta distribuiu peso no contexto? | Quarto bloco, primeira cabeça |
| Próximas possibilidades | Que IDs competiam por ser a próxima saída? | Seis maiores probabilidades |
| Ajustes da geração | Como escolher e apresentar tokens? | Temperatura, sampling e intervalo |
| Rede inteira | Que cálculos e conexões formam a previsão? | Operações, resíduos, cache e pesos E |
| Inspetor | Quais números e código correspondem à seleção? | Equação, shape, estatísticas e recorte |

## Um percurso de leitura em cinco ações

1. Escolha **Mais provável** para facilitar comparações entre rodadas.
2. Pressione **Um passo** e leia o último token e seu ID.
3. Compare a distribuição de candidatos com a distribuição de atenção.
4. Abra **Ver rede completa** e amplie um bloco.
5. Selecione **Q**, **Máscara causal** ou **SwiGLU**, lendo os valores e o código ao lado.

Durante uma geração contínua, clicar numa operação pausa a geração e a reprodução. Isso estabiliza a previsão enquanto você lê. O [guia de geração](../geracao-tokens/) explica a retomada e a diferença entre editar a entrada e iniciar uma rodada nova.

## O que os prints mostram

As capturas desta página usam a previsão real do prefixo padrão, com temperatura `0.8` e escolha greedy. `lab.png` mostra a área superior; `editor.png`, texto, controles e IDs; `settings.png`, os ajustes. As capturas de 390 e 320 pixels mostram a arquitetura no celular, e `fullscreen.png` mostra o mapa ampliado.

São fotografias de estados específicos, não um histórico que a interface armazena. Os rótulos PT e EN acompanham o idioma da página. Para observar outros estados ou valores, execute o laboratório e gere uma previsão própria.

## Usar em uma tela pequena

Os painéis passam a uma leitura vertical. Os controles continuam disponíveis, mas a rede é maior que a tela: use a navegação por bloco, zoom e rolagem do mapa. As matrizes podem exigir rolagem horizontal dentro do inspetor.

Você pode alcançar controles com Tab e ativar operações focadas com Enter ou Espaço. O link **Pular para o conteúdo** permite evitar o cabeçalho. Para compreender o mapa sem seguir a animação, desmarque **Acompanhar etapa** e use a navegação manual. Leia [arquitetura](../arquitetura/) para conexões e [estados e erros](../estados-erros/) para carregamento e recuperação.
