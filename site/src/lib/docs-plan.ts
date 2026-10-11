/** Navigation and assets for the completed PT/EN documentation. */
import { textPair as t, type Localized } from "./docs-references";
export type DocGroupId = "comecar" | "interface" | "fundamentos" | "biblioteca" | "projeto";
export interface DocPage {
  slug: string; group: DocGroupId; title: Localized;
  sources: string[]; api: string[]; screenshots: string[]; references: string[];
}
export interface ScreenshotCapture { file: string; label: Localized; compact?: boolean }
export interface ScreenshotPlan { id: string; title: Localized; focus: Localized; selector: string; captures: ScreenshotCapture[] }

export const docGroups: { id: DocGroupId; title: Localized; description: Localized }[] = [
  {
    "id": "comecar",
    "title": {
      "pt": "Primeiros passos",
      "en": "Getting started"
    },
    "description": {
      "pt": "Entender o projeto e executar o primeiro experimento.",
      "en": "Understand the project and run a first experiment."
    }
  },
  {
    "id": "interface",
    "title": {
      "pt": "Guia da interface",
      "en": "Interface guide"
    },
    "description": {
      "pt": "Componentes, prints e tarefas do laboratório.",
      "en": "Components, screenshots and lab workflows."
    }
  },
  {
    "id": "fundamentos",
    "title": {
      "pt": "Fundamentos do Transformer",
      "en": "Transformer foundations"
    },
    "description": {
      "pt": "Intuição, equações, dimensões e ligação com o código.",
      "en": "Intuition, equations, dimensions and links to the code."
    }
  },
  {
    "id": "biblioteca",
    "title": {
      "pt": "Referência da biblioteca",
      "en": "Library reference"
    },
    "description": {
      "pt": "Contratos, funções, exemplos e comportamento do motor NumPy.",
      "en": "Contracts, functions, examples and NumPy engine behavior."
    }
  },
  {
    "id": "projeto",
    "title": {
      "pt": "Projeto e referências",
      "en": "Project and references"
    },
    "description": {
      "pt": "Integração, dados, validação, glossário e artigos.",
      "en": "Integration, data, validation, glossary and papers."
    }
  }
];

export const screenshotPlans: ScreenshotPlan[] = [
  {"id":"lab","title":{"pt":"Laboratório completo","en":"Complete lab"},"focus":{"pt":"Orientação pelos painéis e ordem do primeiro uso.","en":"Panel orientation and first-use sequence."},"selector":".lab-workspace","captures":[{"file":"lab.png","label":{"pt":"Visão geral com uma previsão real","en":"Overview with an actual prediction"}}]},
  {"id":"editor","title":{"pt":"Texto, controles e tokens BPE","en":"Text, controls and BPE tokens"},"focus":{"pt":"Frase inicial, Gerar, Um passo, Pausar, Limpar, saída e IDs.","en":"Prompt, Generate, Step once, Pause, Reset, output and IDs."},"selector":".lab-text","captures":[{"file":"editor.png","label":{"pt":"Editor, continuação e IDs dos tokens","en":"Editor, continuation and token IDs"}}]},
  {"id":"settings","title":{"pt":"Ajustes do modelo","en":"Model settings"},"focus":{"pt":"Temperatura, escolha, intervalo entre tokens e estado do runtime.","en":"Temperature, selection, time between tokens and runtime status."},"selector":".lab-options","captures":[{"file":"settings.png","label":{"pt":"Ajustes e estado do modelo","en":"Settings and model status"},"compact":true}]},
  {"id":"attention","title":{"pt":"Atenção e probabilidades","en":"Attention and probabilities"},"focus":{"pt":"Contexto observado e seis candidatos; distinguir peso de atenção e probabilidade de saída.","en":"Observed context and six candidates; distinguish attention weights and output probabilities."},"selector":".lab-inspector","captures":[{"file":"attention.png","label":{"pt":"Pesos do contexto e candidatos de saída","en":"Context weights and output candidates"}}]},
  {"id":"architecture","title":{"pt":"Arquitetura completa","en":"Complete architecture"},"focus":{"pt":"Quatro blocos, ramificações, resíduos, pesos compartilhados e retorno ao contexto.","en":"Four blocks, branches, residuals, shared weights and context feedback."},"selector":".architecture","captures":[{"file":"architecture.png","label":{"pt":"Mapa completo dos quatro blocos","en":"Complete map of all four blocks"}},{"file":"architecture-block.png","label":{"pt":"Um bloco ampliado e suas ramificações","en":"One enlarged block and its branches"}}]},
  {"id":"slow","title":{"pt":"Slow motion em uma etapa","en":"Slow motion at one step"},"focus":{"pt":"Intervalo por cálculo, pausa, retomada, anterior/próximo, explicação e destaque sincronizado.","en":"Per-calculation interval, pause, resume, previous/next, explanation and synchronized highlight."},"selector":".architecture","captures":[{"file":"slow-motion.png","label":{"pt":"Etapa pausada com explicação e valores","en":"Paused step with explanation and values"}}]},
  {"id":"inspector","title":{"pt":"Inspetor de tensor e código","en":"Tensor and source inspector"},"focus":{"pt":"Cabeça, dimensões, estatísticas, recorte numérico, máscara −∞ e arquivo/linha.","en":"Head, dimensions, statistics, numerical sample, −∞ masking and file/line."},"selector":".network-inspector","captures":[{"file":"inspector-attention.png","label":{"pt":"Cabeça de atenção, matriz e código","en":"Attention head, matrix and source"},"compact":true},{"file":"inspector-mask.png","label":{"pt":"Máscara causal: valores −∞ registrados","en":"Causal mask: recorded −∞ values"},"compact":true}]},
  {"id":"states","title":{"pt":"Estados de execução","en":"Execution states"},"focus":{"pt":"Carregando, pronto, pausado, limite atingido e erro de carregamento.","en":"Loading, ready, paused, token limit reached and loading error."},"selector":".model-state","captures":[{"file":"state-loading.png","label":{"pt":"Carregamento inicial","en":"Initial loading"},"compact":true},{"file":"state-ready.png","label":{"pt":"Modelo carregado e pronto","en":"Model loaded and ready"},"compact":true},{"file":"state-paused.png","label":{"pt":"Geração pausada","en":"Paused generation"},"compact":true},{"file":"state-limit.png","label":{"pt":"Limite de 128 tokens da rodada","en":"Run limit of 128 tokens"},"compact":true},{"file":"state-error.png","label":{"pt":"Erro real com wheel indisponível","en":"Actual unavailable-wheel error"},"compact":true}]},
  {"id":"mobile","title":{"pt":"Uso no celular e tela cheia","en":"Mobile and full-screen use"},"focus":{"pt":"Leitura a 390 e 320 pixels, rolagem do mapa e navegação acessível.","en":"Reading at 390 and 320 pixels, map scrolling and accessible navigation."},"selector":".architecture","captures":[{"file":"mobile-390.png","label":{"pt":"Arquitetura no celular: 390 pixels","en":"Mobile architecture: 390 pixels"},"compact":true},{"file":"mobile-320.png","label":{"pt":"Arquitetura no celular: 320 pixels","en":"Mobile architecture: 320 pixels"},"compact":true},{"file":"fullscreen.png","label":{"pt":"Exploração em tela cheia","en":"Full-screen exploration"}}]}
];

const page = (group: DocGroupId, name: string, title: Localized, sources: string[], api: string[] = [], screenshots: string[] = [], references: string[] = []): DocPage =>
  ({ slug: `${group}/${name}`, group, title, sources, api, screenshots, references });

export const docPages: DocPage[] = [
  page("comecar", "visao-geral", t("O que é o glassllm", "What glassllm is"), ["README.pt-BR.md","src/glassllm/__init__.py"], ["load_pretrained"], ["lab"], ["transformer"]),
  page("comecar", "instalacao", t("Instalação e execução local", "Local installation and execution"), ["pyproject.toml","site/package.json","site/scripts/prepare-wheel.mjs"], [], ["states"], []),
  page("comecar", "primeiro-experimento", t("Primeiro experimento: site e Python", "First experiment: site and Python"), ["site/src/scripts/free-lab.ts","src/glassllm/model.py","src/glassllm/trace.py"], ["load_pretrained","Tokenizer.encode","GlassLM.generate","Tracer.get"], ["editor","inspector"], []),
  page("interface", "mapa-componentes", t("Mapa dos componentes", "Component map"), ["site/src/components/FreeLab.astro","site/src/components/Architecture.astro"], [], ["lab","editor","settings","mobile"], []),
  page("interface", "geracao-tokens", t("Texto, geração e tokens", "Text, generation and tokens"), ["site/src/scripts/free-lab.ts","src/glassllm/tokenizer.py","src/glassllm/model.py"], ["Tokenizer.tokens","Tokenizer.decode","GlassLM.generate"], ["editor"], ["bpe","sentencepiece"]),
  page("interface", "ajustes", t("Temperatura e escolha do token", "Temperature and token selection"), ["site/src/components/FreeLab.astro","site/src/workers/model.worker.ts","src/glassllm/model.py"], ["GlassLM.generate","next_token_probs"], ["settings"], []),
  page("interface", "atencao-probabilidades", t("Ler atenção e probabilidades", "Reading attention and probabilities"), ["site/src/workers/model.worker.ts","site/src/scripts/free-lab.ts"], ["Tracer.get","GenStep.probs"], ["attention"], ["transformer"]),
  page("interface", "arquitetura", t("Explorar a arquitetura inteira", "Exploring the complete architecture"), ["site/src/lib/network-graph.ts","site/src/scripts/network-view.ts","src/glassllm/nn.py"], [], ["architecture","mobile"], ["transformer","gqa","tied"]),
  page("interface", "slow-motion", t("Estudar em slow motion", "Studying in slow motion"), ["site/src/components/Architecture.astro","site/src/scripts/network-view.ts","site/src/scripts/free-lab.ts"], ["Tracer","StepRecord"], ["slow","mobile"], []),
  page("interface", "inspetor", t("Inspecionar tensores e código", "Inspecting tensors and code"), ["site/src/workers/model.worker.ts","site/src/scripts/network-view.ts","src/glassllm/steps.py"], ["Tracer.get","StepRecord.shape","load_step_index"], ["inspector"], ["gqa"]),
  page("interface", "estados-erros", t("Estados, limites e solução de problemas", "States, limits and troubleshooting"), ["site/src/workers/model.worker.ts","site/src/scripts/free-lab.ts","site/scripts/prepare-wheel.mjs"], [], ["states","mobile"], []),
  page("fundamentos", "tokens-bpe", t("Tokens e BPE", "Tokens and BPE"), ["src/glassllm/tokenizer.py"], ["pre_tokenize","Tokenizer.train","Tokenizer.encode_verbose"], ["editor"], ["bpe","sentencepiece"]),
  page("fundamentos", "embeddings", t("Embedding e pesos compartilhados", "Embedding and shared weights"), ["src/glassllm/autograd.py","src/glassllm/model.py"], ["embedding","GlassLM.forward"], [], ["tied"]),
  page("fundamentos", "atencao-causal", t("Atenção causal: Q, K e V", "Causal attention: Q, K and V"), ["src/glassllm/nn.py","src/glassllm/autograd.py"], ["Attention","causal_mask","softmax","masked_fill"], ["inspector"], ["transformer"]),
  page("fundamentos", "rope", t("Posição e RoPE", "Position and RoPE"), ["src/glassllm/nn.py","src/glassllm/autograd.py"], ["rope_angles","rope"], [], ["rope"]),
  page("fundamentos", "gqa-cache", t("Cabeças, GQA e cache K/V", "Heads, GQA and K/V cache"), ["src/glassllm/nn.py","src/glassllm/model.py","src/glassllm/autograd.py"], ["KVCache","repeat_kv","GlassLM.new_cache","GlassLM.generate"], [], ["gqa"]),
  page("fundamentos", "normalizacao-residuos", t("RMSNorm, pre-norm e resíduos", "RMSNorm, pre-norm and residuals"), ["src/glassllm/nn.py"], ["RMSNorm","Block"], [], ["rmsnorm","prenorm","residual"]),
  page("fundamentos", "swiglu", t("FFN e SwiGLU", "FFN and SwiGLU"), ["src/glassllm/nn.py","src/glassllm/autograd.py"], ["SwiGLU","silu","sigmoid"], [], ["swiglu"]),
  page("fundamentos", "saida-geracao", t("Logits, temperatura e geração", "Logits, temperature and generation"), ["src/glassllm/model.py"], ["next_token_probs","GenStep","GlassLM.generate"], [], ["transformer","tied"]),
  page("fundamentos", "perda-gradientes", t("Perda e backpropagation", "Loss and backpropagation"), ["src/glassllm/model.py","src/glassllm/autograd.py","src/glassllm/train.py"], ["GlassLM.loss","cross_entropy","cross_entropy_grad","Tensor.backward"], [], ["autodiff"]),
  page("biblioteca", "api-publica", t("Mapa da API e convenções", "API map and conventions"), ["src/glassllm/__init__.py","src/glassllm/"], ["Config","GenStep","GlassLM","LANGUAGES","STEPS","StepRecord","Tensor","Tokenizer","Tracer","load_pretrained","no_grad","__version__"], [], []),
  page("biblioteca", "config-carregamento", t("Configuração e modelos treinados", "Configuration and pretrained models"), ["src/glassllm/config.py","src/glassllm/__init__.py"], ["Config","load_pretrained","LANGUAGES","__version__"], [], []),
  page("biblioteca", "tokenizer", t("Tokenizer: referência completa", "Tokenizer: complete reference"), ["src/glassllm/tokenizer.py"], ["Tokenizer","Tokenizer.train","Tokenizer.encode","Tokenizer.encode_verbose","Tokenizer.tokens","Tokenizer.decode","Tokenizer.save","Tokenizer.load","Tokenizer.vocab_size","pre_tokenize","SPACE","SPECIALS","UNK_TEXT"], ["editor"], ["bpe","sentencepiece"]),
  page("biblioteca", "glasslm", t("GlassLM e GenStep", "GlassLM and GenStep"), ["src/glassllm/model.py","src/glassllm/nn.py"], ["GlassLM","GlassLM.forward","GlassLM.loss","GlassLM.generate","GlassLM.new_cache","GlassLM.save","GlassLM.load","GenStep","next_token_probs","Module.parameters","Module.named_parameters"], [], ["transformer","tied"]),
  page("biblioteca", "trace-etapas", t("Tracer, registros e catálogo de etapas", "Tracer, records and step catalog"), ["src/glassllm/trace.py","src/glassllm/steps.py","scripts/build_step_index.py"], ["Tracer","Tracer.record","Tracer.get","Tracer.step_ids","StepRecord","record","StepInfo","STEPS","LAYER_STEPS","HEAD_STEPS","scan_markers","load_step_index"], ["inspector"], []),
  page("biblioteca", "tensor-autograd", t("Tensor e motor de autograd", "Tensor and autograd engine"), ["src/glassllm/autograd.py"], ["Tensor","Tensor.backward","Tensor.sum","Tensor.mean","Tensor.max","Tensor.reshape","Tensor.transpose","no_grad"], [], ["autodiff"]),
  page("biblioteca", "operacoes-autograd", t("Operações diferenciáveis", "Differentiable operations"), ["src/glassllm/autograd.py"], ["add","neg","mul","div","power","matmul","exp","log","tsum","mean","max","reshape","transpose","concat","where","embedding","masked_fill","softmax_np","softmax","log_softmax","sigmoid","silu","cross_entropy","cross_entropy_grad","rope","repeat_kv"], [], ["autodiff","rope","swiglu"]),
  page("biblioteca", "camadas-cache", t("Camadas e cache", "Layers and cache"), ["src/glassllm/nn.py"], ["Module","init_param","RMSNorm","Attention","KVCache","SwiGLU","Block","causal_mask","rope_angles"], [], ["transformer","rmsnorm","rope","gqa","swiglu"]),
  page("biblioteca", "treino-otimizacao", t("Treinamento e otimização", "Training and optimization"), ["src/glassllm/train.py","src/glassllm/optim.py","scripts/train_model.py"], ["get_batch","evaluate","train","AdamW","AdamW.zero_grad","AdamW.step","clip_grad_norm","lr_schedule"], [], ["adam","adamw","cosine","autodiff"]),
  page("biblioteca", "cli-arquivos", t("CLI, arquivos e scripts", "CLI, files and scripts"), ["src/glassllm/cli.py","scripts/prepare_corpus.py","scripts/train_tokenizer.py","scripts/train_model.py","scripts/build_step_index.py"], ["main","build_parser","format_matrix","cmd_tokenize","cmd_generate","cmd_trace","cmd_steps","cmd_train"], [], []),
  page("projeto", "integracao-browser", t("Como a interface usa o motor", "How the interface uses the engine"), ["site/src/scripts/free-lab.ts","site/src/workers/model.worker.ts","site/scripts/prepare-wheel.mjs"], ["load_pretrained","GlassLM.generate","Tracer.get"], ["lab","inspector"], []),
  page("projeto", "dados-modelos", t("Dados, modelos e reprodutibilidade", "Data, models and reproducibility"), ["scripts/prepare_corpus.py","scripts/train_model.py","src/glassllm/weights/"], ["load_pretrained","Config"], [], []),
  page("projeto", "validacao-contribuicao", t("Validação e contribuição", "Validation and contribution"), ["tests/","CONTRIBUTING.md",".github/workflows/","site/package.json"], [], [], []),
  page("projeto", "glossario", t("Glossário e notação", "Glossary and notation"), ["docs/equacoes.md","src/glassllm/config.py","src/glassllm/steps.py"], [], [], []),
  page("projeto", "referencias", t("Artigos e mapa de referências", "Papers and reference map"), ["README.md","docs/equacoes.md"], [], [], ["transformer","bpe","sentencepiece","rope","gqa","rmsnorm","prenorm","residual","swiglu","tied","autodiff","adam","adamw","cosine"])
];
