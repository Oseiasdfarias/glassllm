---
title: Como a interface usa o motor
description: O percurso de uma ação no laboratório, das mensagens do worker aos tensores calculados pelo mesmo pacote Python.
---

## As três partes da aplicação

O site é gerado pelo Astro. Seus componentes definem editor, botões, arquitetura e páginas de documentação; TypeScript coordena as ações e atualiza os elementos do navegador. O modelo roda em um Web Worker separado, que carrega Python com Pyodide e instala o wheel local de `glassllm`. Dentro desse ambiente, os cálculos usam a implementação NumPy da biblioteca.

```text
editor e controles
        ↓ mensagens com requestId
Web Worker → Pyodide → glassllm → NumPy
        ↓ token, seis candidatos, resumos dos tensores
texto gerado + mapa + inspetor
```

Isso mantém a execução do modelo fora da thread que desenha a interface. O laboratório usa `load_pretrained`, `Tokenizer.encode`, `GlassLM.generate` e `Tracer.get`; o navegador não reimplementa a atenção ou o autograd em JavaScript. A [API Python](../../biblioteca/api-publica/) continua disponível sem o site.

## Carregamento e arquivos

`site/scripts/prepare-wheel.mjs` executa `uv build --wheel`, copia o wheel para os assets de desenvolvimento e para `public/py/`, e gera seu nome canônico. Ao abrir uma geração pela primeira vez, o worker carrega o runtime Pyodide **0.29.3**, `micropip` e NumPy; depois instala o wheel e seleciona os pesos PT ou EN. O download do runtime depende da rede. Depois que o modelo está carregado, a inferência usa os arquivos e a memória do navegador, sem enviar a frase a uma API de geração.

Em desenvolvimento, o wheel é servido pelo Vite a partir de `src/generated/`. No build estático, sua URL é `/glassllm/py/glassllm-0.1.0-py3-none-any.whl`. O nome do arquivo precisa preservar os campos exigidos pelo formato wheel; um caminho terminado apenas em `glassllm` não pode ser instalado pelo `micropip`. O módulo gerado `wheel-asset.ts` informa esse nome ao cliente; não há uma consulta a um manifesto remoto ou ao PyPI para localizar `glassllm`.

O worker evita inicializações simultâneas com uma promessa compartilhada. Uma falha deixa essa tentativa disponível para repetição; um runtime que já foi carregado pode ser reutilizado na tentativa seguinte. [Estados e erros](../../interface/estados-erros/) explica o diagnóstico de falhas de download e de instalação.

## O protocolo de mensagens

Cada requisição tem `requestId`, `action` e os argumentos da ação. O identificador relaciona a resposta à promessa pendente da interface; ele é separado de `id`, que identifica uma etapa matemática como `attn.q`.

| Ação | Entrada principal | Resposta e efeito |
| --- | --- | --- |
| `load` | `wheelURL`, `lang` | Carrega o modelo; retorna configuração, vocabulário e número de parâmetros |
| `tokenize` | `text` | Retorna os IDs e as representações dos tokens |
| `start` | `text`, `temperature`, `sampling` | Cria um gerador de até 128 tokens e reinicia a contagem da rodada |
| `next` | Gerador já criado | Retorna o token, seis candidatos com probabilidades, atenção resumida e resumos do trace; `done` quando o gerador termina |
| `inspect` | `step`, `layer`, `head` | Retorna um recorte e a fonte de uma operação da última previsão |
| `stop` | Nenhum argumento de modelo | Descarta o gerador atual |

Em `next`, o worker guarda o contexto anterior à escolha do novo token. Isso permite rotular os eixos das matrizes com os tokens que realmente participaram do cálculo. Também extrai os seis candidatos mais prováveis e a atenção da última posição, última camada e primeira cabeça para os painéis resumidos.

O vetor completo `GenStep.probs` permanece no worker e aparece como snapshot da etapa `probs`; ele não é enviado integralmente em `next`. `tokenize` e `stop` são ações disponíveis na ponte, mas os botões atuais não as chamam diretamente. **Pausar** interrompe o loop e a reprodução no cliente, preservando o gerador. **Limpar** reinicia a apresentação e a rodada no cliente; o próximo `start` substitui o gerador no worker.

## Snapshots e inspeção

`generate(trace=True)` cria um `Tracer` por previsão. Os quatro blocos padrão produzem 72 registros; embedding, normalização final, logits, probabilidades e escolha completam **77**. O worker envia resumos com forma, equação, primeiros valores, estatísticas e referência ao arquivo/linha. O conjunto completo de arrays da última previsão permanece no worker.

A ação `inspect` seleciona a camada e a cabeça, transforma a operação em uma matriz para visualização e retorna as últimas oito linhas e oito colunas disponíveis. As estatísticas consideram todos os valores finitos do tensor selecionado, e não apenas esse recorte. Valores como `−∞` são transportados como strings para que JSON continue válido; a quantidade de valores não finitos acompanha a resposta.

O índice `step_index.json` liga marcadores `# @step` ao código empacotado. A linha mostrada no inspetor vem do arquivo Python instalado no worker. Nós estruturais, como o compartilhamento de cabeças, podem não ter um snapshot próprio; a interface identifica essa condição em vez de criar números ilustrativos.

## Reprodução, cancelamento e memória

Slow motion percorre os registros já calculados. Seu tempo de leitura não pausa as multiplicações dentro do NumPy. A interface pode interromper a reprodução e solicitar outra inspeção; versões de seleção e de previsão impedem que uma resposta atrasada sobrescreva a seleção atual. Ao retomar a geração, ela termina o percurso da previsão atual antes de solicitar `next` novamente.

O worker mantém somente o trace da previsão mais recente. O texto gerado fica na interface, mas seus tensores anteriores deixam de ser consultáveis quando uma nova previsão substitui `_last_trace`. Para estudar uma operação, pause a geração antes de avançar. Cada snapshot é uma cópia, portanto rastrear usa mais memória que inferir sem trace. A biblioteca permite `trace=False` quando essa inspeção não é necessária.

O cache guarda dois arrays por bloco, K e V, cada um com forma `B × H_KV × m × d_k`. Se ambos usarem o mesmo `itemsize`, o armazenamento numérico é `2 × L × B × H_KV × m × d_k × itemsize` bytes. Com os padrões, batch 1, janela cheia e arrays float64, são 524.288 bytes (512 KiB); com float32 seriam 256 KiB. Consulte `cache.k.dtype`, `cache.v.dtype` e `nbytes`: pesos carregados em float32 não garantem cache float32. Essa conta exclui snapshots, cópias temporárias, objetos Python, runtime e a expansão temporária de cabeças GQA.

## Onde acompanhar o percurso

`site/src/scripts/free-lab.ts` gerencia requisições e geração; `network-view.ts` controla seleção e reprodução; `model.worker.ts` contém a ponte com Python. Para seguir a conta, comece em [primeiro experimento](../../comecar/primeiro-experimento/) e avance para [Tracer e etapas](../../biblioteca/trace-etapas/).
