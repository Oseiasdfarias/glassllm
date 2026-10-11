---
title: "Estudar em slow motion"
description: "Percorra lentamente os 77 snapshots de uma previsão, pausando, voltando e comparando equações, valores e código."
---

## O que fica lento

O motor primeiro calcula a previsão e registra snapshots dos resultados. O slow motion reproduz esses registros em ordem, dando tempo para ler cada operação. A seleção, explicação, equação, tensor e destaque do desenho acompanham o mesmo evento.

Ele não executa uma multiplicação NumPy aos poucos nem pausa o Python no meio da função. O token escolhido já é conhecido quando a reprodução começa e já pode aparecer na saída. Voltar uma etapa visual não desfaz a geração nem recalcula pesos.

## Preparar uma sessão de leitura

1. Abra o [laboratório](../../../) e escolha **Mais provável**.
2. Vá à arquitetura e marque **Slow motion** antes de gerar ou reproduzir.
3. Escolha **Tempo por cálculo**: 1, 2, 3, 5 ou 8 segundos. O padrão é 3 segundos.
4. Mantenha **Acompanhar etapa** ativo para o mapa localizar cada operação.
5. Use **Um passo** para uma única previsão ou **Gerar** para continuar em rodadas de cálculo e leitura.
6. Clique em **Pausar fluxo** para estabilizar o evento que deseja estudar.

Com quatro blocos, uma previsão tem 77 eventos: 1 embedding, 72 operações dos blocos, normalização final, logits, probabilidades e escolha. A 3 segundos, as 76 esperas de um percurso completo somam cerca de 3 minutos e 48 segundos, além do tempo de inspeção e cálculo. Escolha 1 segundo para um primeiro tour e aumente para estudar matrizes.

## Controles da reprodução

| Controle | Efeito sobre a previsão atual |
| --- | --- |
| **Reproduzir fluxo** | Reinicia a sequência de snapshots desde o primeiro evento. |
| **Continuar** | Retoma a reprodução depois da posição atual. Não gera um token novo. |
| **Cálculo anterior** | Pausa o fluxo e seleciona o evento anterior. |
| **Próximo cálculo** | Pausa o fluxo e seleciona o evento seguinte. |
| **Pausar fluxo** | Para a reprodução e a continuidade da geração, mantendo o evento atual. |
| **Acompanhar etapa** | Faz o mapa enquadrar a operação ativa durante a reprodução lenta. |

Anterior e próximo obedecem aos limites do trace. Antes da primeira previsão, os controles de reprodução ficam desabilitados. Durante a conta do motor, ações que dependem do trace também aguardam a conclusão.

## Pausar e retomar sem saltar a explicação

**Pausar**, na área de texto, interrompe tanto a continuidade quanto o fluxo. Selecionar um nó ou mudar a cabeça no inspetor também estabiliza a geração para que os números não mudem durante a leitura.

Para continuar somente a explicação dos registros atuais, use **Continuar** na arquitetura. Para continuar produzindo texto, use **Gerar**: quando existe um fluxo pendente, a interface termina a leitura daquela previsão antes de solicitar o token seguinte.

**Um passo** com slow motion também respeita um fluxo pendente: conclui primeiro a reprodução da previsão atual. Depois que ela termina, um novo clique calcula o próximo token. Isso evita que uma ação de estudo substitua silenciosamente os valores antes de você terminar de lê-los.

## Um roteiro para estudar atenção

Pause em **Q** e leia seu formato. Avance até **K** e observe que há menos cabeças K/V na configuração GQA. Em **RoPE**, compare a forma com a projeção anterior. Em **Scores**, confirme os eixos de consulta e chave. Em **Máscara**, encontre `−∞`; em **Softmax**, veja peso zero nas posições bloqueadas. Em **Mistura com V**, confira o retorno às dimensões de uma cabeça.

O cartão de etapa mostra a posição `i / 77`, o bloco, o título, a explicação e a equação. O inspetor exibe os valores da cabeça selecionada. Desmarque **Acompanhar etapa** quando quiser manter duas regiões próximas em vista e navegar manualmente.

## Limites de comparação

Só o trace da última previsão é mantido no worker. Gerar outro token substitui os snapshots disponíveis; a reprodução não volta a tokens anteriores do texto. O intervalo de leitura não altera probabilidades, ordem dos eventos ou resultados matemáticos.

A preferência do sistema por movimento reduzido encurta a animação automática comum. A reprodução manual e o slow motion continuam permitindo navegar pelos eventos. Para entender o recorte numérico e os nós sem snapshot, continue em [inspetor](../inspetor/).
