export type Lang = 'pt' | 'en';

export const copy = {
  pt: {
    languageName: 'Português',
    nav: {
      home: 'Visão geral',
      method: 'Como funciona',
      lab: 'Laboratório',
      language: 'English',
    },
    meta: {
      title: 'glassllm — veja por dentro de um modelo de linguagem',
      description:
        'Um laboratório interativo para entender, token a token, como um pequeno modelo de linguagem processa e gera texto.',
    },
    hero: {
      eyebrow: 'UM MODELO DE LINGUAGEM, SEM CAIXA-PRETA',
      title: 'Abra a caixa.<br />Acompanhe cada token.',
      intro:
        'Digite uma frase e observe o caminho real do modelo: da tokenização à atenção, até a escolha da próxima palavra.',
      primary: 'Abrir o laboratório',
      secondary: 'Conhecer o percurso',
      note: 'A inferência acontece no seu navegador, com a biblioteca Python do projeto.',
      visualLabel: 'O ciclo de geração',
      tokens: ['A', 'ideia', 'ganha', 'forma'],
      stages: ['tokens', 'atenção', 'probabilidades', 'próximo token'],
      resultLabel: 'próxima escolha',
      result: 'forma',
    },
    overview: {
      kicker: 'UM MODELO PEQUENO. UM PROCESSO VISÍVEL.',
      title: 'Texto entra. Decisões ficam à mostra.',
      body:
        'glassllm é uma implementação didática de um Transformer. Em vez de esconder os cálculos atrás de uma API, ela deixa você acompanhar o que acontece em cada rodada de geração.',
      facts: [
        { mark: '01', title: 'Tokens', body: 'O texto é dividido nas unidades que o modelo realmente lê.' },
        { mark: '02', title: 'Atenção', body: 'Veja quais partes do contexto influenciam a próxima previsão.' },
        { mark: '03', title: 'Escolha', body: 'Compare as probabilidades e acompanhe o token escolhido.' },
      ],
    },
    method: {
      kicker: 'A MESMA BIBLIOTECA, DOIS JEITOS DE EXPLORAR',
      title: 'Da frase ao próximo token.',
      body:
        'O painel livre percorre o ciclo de geração em tempo real. Você controla a temperatura e o modo de escolha, avança um token por vez e inspeciona a atenção e as probabilidades.',
      steps: [
        { number: '01', title: 'Escreva', body: 'Comece com um contexto curto em português ou inglês.' },
        { number: '02', title: 'Observe', body: 'Acompanhe tokens, blocos, logits e distribuição de saída.' },
        { number: '03', title: 'Experimente', body: 'Pause, avance passo a passo e mude a temperatura.' },
      ],
      note: 'A trilha de capítulos com equações, código e valores está planejada para uma próxima fase.',
    },
    lab: {
      kicker: 'EXPERIMENTO 01 · GERAÇÃO EM LOOP',
      title: 'Seu contexto, o próximo token.',
      body:
        'Escolha uma frase inicial. O modelo acrescenta um token a cada rodada; ao lado, você vê como chegou àquela escolha.',
      loading: 'O laboratório carrega a biblioteca e os pesos sob demanda. Na primeira visita, isso pode levar alguns instantes.',
      fallback: 'O painel interativo precisa de um navegador com WebAssembly habilitado.',
    },
    footer: {
      statement: 'Feito para aprender observando o que o modelo calcula.',
      source: 'Código aberto',
      top: 'Voltar ao início ↑',
      phase: 'Transformer em NumPy · PT / EN',
    },
  },
  en: {
    languageName: 'English',
    nav: {
      home: 'Overview',
      method: 'How it works',
      lab: 'Lab',
      language: 'Português',
    },
    meta: {
      title: 'glassllm — look inside a language model',
      description:
        'An interactive lab for following, token by token, how a small language model processes and generates text.',
    },
    hero: {
      eyebrow: 'A LANGUAGE MODEL, WITHOUT THE BLACK BOX',
      title: 'Open the box.<br />Follow every token.',
      intro:
        'Enter a sentence and follow the model’s actual path: from tokenization to attention, all the way to its next-token choice.',
      primary: 'Open the lab',
      secondary: 'See how it works',
      note: 'Inference runs in your browser using this project’s Python library.',
      visualLabel: 'The generation loop',
      tokens: ['An', 'idea', 'takes', 'shape'],
      stages: ['tokens', 'attention', 'probabilities', 'next token'],
      resultLabel: 'next choice',
      result: 'shape',
    },
    overview: {
      kicker: 'A SMALL MODEL. A VISIBLE PROCESS.',
      title: 'Text goes in. Decisions come into view.',
      body:
        'glassllm is a teaching implementation of a Transformer. Instead of hiding its calculations behind an API, it lets you follow what happens at every generation step.',
      facts: [
        { mark: '01', title: 'Tokens', body: 'See how text is split into the units the model actually reads.' },
        { mark: '02', title: 'Attention', body: 'Inspect which parts of the context shape the next prediction.' },
        { mark: '03', title: 'Choice', body: 'Compare probabilities and follow the token that gets picked.' },
      ],
    },
    method: {
      kicker: 'ONE LIBRARY, TWO WAYS TO EXPLORE',
      title: 'From a sentence to the next token.',
      body:
        'The free lab follows the generation loop in real time. Set the temperature and sampling mode, step through one token at a time, and inspect attention and probabilities.',
      steps: [
        { number: '01', title: 'Write', body: 'Start with a short context in Portuguese or English.' },
        { number: '02', title: 'Observe', body: 'Follow tokens, blocks, logits, and the output distribution.' },
        { number: '03', title: 'Experiment', body: 'Pause, step through generation, and change the temperature.' },
      ],
      note: 'A chapter path connecting equations, code, and values is planned for a later phase.',
    },
    lab: {
      kicker: 'EXPERIMENT 01 · THE GENERATION LOOP',
      title: 'Your context, the next token.',
      body:
        'Choose a starting sentence. The model adds one token each round; beside it, inspect how that choice was made.',
      loading: 'The lab loads the library and weights on demand. The first visit may take a few moments.',
      fallback: 'The interactive panel requires a browser with WebAssembly enabled.',
    },
    footer: {
      statement: 'Made for learning by observing what the model computes.',
      source: 'Open source',
      top: 'Back to top ↑',
      phase: 'Transformer in NumPy · PT / EN',
    },
  },
} as const;

export type Copy = (typeof copy)[Lang];
