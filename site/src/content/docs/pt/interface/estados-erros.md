---
title: "Estados, limites e solução de problemas"
description: "Reconheça carregamento, execução, pausa e término da rodada e recupere falhas do runtime ou do wheel."
---

## Reconhecer o estado atual

O indicador em **Ajustes da geração** acompanha a execução. O primeiro uso prepara o ambiente; os demais avanços reutilizam o runtime daquela página.

| Mensagem | Significado | Próxima ação |
| --- | --- | --- |
| Modelo ainda não carregado | O worker foi criado, mas os assets ainda não foram carregados | Escreva uma frase e use **Um passo**. |
| Preparando o modelo no navegador… | Pyodide, pacotes e wheel estão sendo baixados ou instalados | Aguarde e observe eventuais erros. |
| Modelo pronto · roda localmente | O modelo está disponível, sem geração contínua | Inicie uma rodada. |
| Gerando token a token | A rodada avança e apresenta previsões | Pause para examinar um resultado. |
| Geração pausada | O gerador e o último trace permanecem disponíveis | Retome ou limpe para reiniciar. |
| Limite desta rodada alcançado | O gerador encerrou os 128 tokens novos | Inicie outra rodada ou altere a frase. |
| Não foi possível carregar o modelo… | Uma operação do worker falhou | Abra **Detalhes técnicos** e confira a causa. |

O texto geral de falha também pode aparecer para problemas de geração, não somente download. Leia a mensagem específica e os detalhes antes de supor que o modelo não existe.

## Limites diferentes, efeitos diferentes

O editor aceita 600 caracteres; o modelo usa janela de 128 tokens; a rodada produz no máximo 128 tokens novos. O painel BPE exibe os últimos 64 tokens após um avanço, e a grade do inspetor mostra até 8 × 8 elementos. Nenhum desses limites é uma contagem de palavras.

Um prefixo grande é reduzido aos últimos 128 IDs para a geração. Quando a janela enche durante a continuação, o cache é reconstruído sobre a janela mais recente. O trace só contém a previsão atual. Uma matriz `1 × m` é habitual com cache e não indica dados ausentes.

Se o contador aumentar sem um caractere novo, o modelo pode ter escolhido um token especial cuja decodificação é vazia. O gerador atual não trata `<eos>` como término automático; sua condição normal de término é a quantidade solicitada.

## Falha de download ou instalação

**Wheel indisponível / HTTP 404.** Confirme que o servidor está ativo, que a base da URL é `/glassllm/` e que o wheel foi gerado. Em desenvolvimento, `prepare:wheel` também cria o asset em `site/src/generated/`; na saída estática, ele fica em `py/`. Refazer `npm run build` ou reiniciar `npm run dev` a partir de `site/` recria os assets.

**Invalid wheel filename.** Micropip precisa do nome completo no formato de wheel, por exemplo `glassllm-0.1.0-py3-none-any.whl`. Um caminho que termina somente em `glassllm` ou um nome alterado pelo empacotador não contém os campos exigidos. Use os assets preparados pelo script do site.

**Arquivo recebido não é um wheel.** Alguns servidores devolvem HTML para um arquivo inexistente. Isso pode produzir erros de ZIP mesmo com resposta HTTP 200. Verifique o conteúdo e o caminho, não apenas o status da requisição.

**CDN ou Pyodide inacessível.** A primeira preparação depende do runtime Pyodide 0.29.3 e dos pacotes servidos pela CDN. Sem acesso inicial a esses recursos, a inferência local não começa. O laboratório não oferece um fallback de execução remota.

## Assets ou versões incompatíveis

`load_pretrained` exige pesos e tokenizer do mesmo idioma e valida o tamanho do vocabulário. Arquivos ausentes produzem erro, sem substituição automática por um modelo aleatório. Se você recompilar o wheel após modificar a biblioteca, recarregue a página para abrir um runtime com os arquivos novos.

O erro `'Tokenizer' object has no attribute 'id_to_token'` indica código que chamou um atributo fora da API atual. A interface usa `Tokenizer.tokens([id])` para rótulos e `decode([id])` para texto. Confira se o worker e o wheel vieram do mesmo build.

## Recuperar a interface

1. Expanda **Detalhes técnicos** para copiar ou ler a exceção completa. O painel mantém o traceback separado da mensagem curta.
2. Corrija o servidor, o caminho do wheel ou os assets indicados.
3. Tente **Um passo** novamente. A preparação permite uma nova tentativa depois de falha parcial.
4. Se substituiu o wheel ou alterou o runtime, recarregue a página para evitar reutilizar o módulo já importado.
5. Para uma rodada limpa, pause e use **Limpar**; isso mantém o runtime carregado, mas reinicia o experimento.

## Navegação e tela pequena

JavaScript é necessário para a geração; os textos da documentação são páginas estáticas. Use Tab, Enter e Espaço nos controles e operações. Zoom e rolagem interna permitem ler a rede no celular; **Tela cheia** depende do suporte e da permissão do navegador. Quando não estiver disponível, use a navegação por bloco.

Se o mapa se deslocar enquanto você lê, pause o fluxo ou desmarque **Acompanhar etapa**. Se o tensor mudar após um token novo, esse comportamento é esperado: a interface acompanha a última previsão. Continue em [geração e tokens](../geracao-tokens/) e [inspetor](../inspetor/) para distinguir rodada, contexto e recorte.
