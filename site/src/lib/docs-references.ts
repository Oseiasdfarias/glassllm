export type Localized = { pt: string; en: string };
export const textPair = (pt: string, en: string): Localized => ({ pt, en });

export interface PaperReference {
  id: string;
  title: string;
  authors: string;
  preprintYear: number;
  url: string;
  topic: Localized;
}

// Metadata checked against the primary arXiv records on 2026-10-11.
// Years refer to the first arXiv submission, not a later conference publication.
export const papers: PaperReference[] = [
  { id: "transformer", title: "Attention Is All You Need", authors: "Vaswani et al.", preprintYear: 2017, url: "https://arxiv.org/abs/1706.03762", topic: textPair("Atenção escalada e arquitetura Transformer", "Scaled attention and Transformer architecture") },
  { id: "bpe", title: "Neural Machine Translation of Rare Words with Subword Units", authors: "Sennrich, Haddow e Birch", preprintYear: 2015, url: "https://arxiv.org/abs/1508.07909", topic: textPair("BPE e representação por subpalavras", "BPE and subword representation") },
  { id: "sentencepiece", title: "SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing", authors: "Kudo e Richardson", preprintYear: 2018, url: "https://arxiv.org/abs/1808.06226", topic: textPair("Contexto para tokenização independente de idioma; a biblioteca tem implementação própria", "Language-independent tokenization context; the library has its own implementation") },
  { id: "rope", title: "RoFormer: Enhanced Transformer with Rotary Position Embedding", authors: "Su et al.", preprintYear: 2021, url: "https://arxiv.org/abs/2104.09864", topic: textPair("Posições e rotações de Q/K", "Positions and Q/K rotations") },
  { id: "gqa", title: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", authors: "Ainslie et al.", preprintYear: 2023, url: "https://arxiv.org/abs/2305.13245", topic: textPair("Compartilhamento de cabeças K/V; o projeto usa GQA, sem reproduzir o método de conversão de checkpoints do artigo", "Shared K/V heads; the project uses GQA without reproducing the paper's checkpoint conversion method") },
  { id: "rmsnorm", title: "Root Mean Square Layer Normalization", authors: "Zhang e Sennrich", preprintYear: 2019, url: "https://arxiv.org/abs/1910.07467", topic: textPair("Normalização RMSNorm", "RMSNorm normalization") },
  { id: "prenorm", title: "On Layer Normalization in the Transformer Architecture", authors: "Xiong et al.", preprintYear: 2020, url: "https://arxiv.org/abs/2002.04745", topic: textPair("Posição da normalização; o artigo analisa LayerNorm, enquanto o projeto usa RMSNorm", "Normalization placement; the paper analyzes LayerNorm while the project uses RMSNorm") },
  { id: "residual", title: "Deep Residual Learning for Image Recognition", authors: "He et al.", preprintYear: 2015, url: "https://arxiv.org/abs/1512.03385", topic: textPair("Contexto histórico de conexões residuais em visão computacional", "Historical context for residual connections in computer vision") },
  { id: "swiglu", title: "GLU Variants Improve Transformer", authors: "Shazeer", preprintYear: 2020, url: "https://arxiv.org/abs/2002.05202", topic: textPair("FFN com portão e variante SwiGLU", "Gated FFN and the SwiGLU variant") },
  { id: "tied", title: "Using the Output Embedding to Improve Language Models", authors: "Press e Wolf", preprintYear: 2016, url: "https://arxiv.org/abs/1608.05859", topic: textPair("Compartilhamento entre embedding e projeção de saída", "Sharing embedding and output projection weights") },
  { id: "autodiff", title: "Automatic differentiation in machine learning: a survey", authors: "Baydin et al.", preprintYear: 2015, url: "https://arxiv.org/abs/1502.05767", topic: textPair("Diferenciação automática e modo reverso", "Automatic differentiation and reverse mode") },
  { id: "adam", title: "Adam: A Method for Stochastic Optimization", authors: "Kingma e Ba", preprintYear: 2014, url: "https://arxiv.org/abs/1412.6980", topic: textPair("Momentos e atualização adaptativa de parâmetros", "Moments and adaptive parameter updates") },
  { id: "adamw", title: "Decoupled Weight Decay Regularization", authors: "Loshchilov e Hutter", preprintYear: 2017, url: "https://arxiv.org/abs/1711.05101", topic: textPair("Weight decay desacoplado no AdamW", "Decoupled weight decay in AdamW") },
  { id: "cosine", title: "SGDR: Stochastic Gradient Descent with Warm Restarts", authors: "Loshchilov e Hutter", preprintYear: 2016, url: "https://arxiv.org/abs/1608.03983", topic: textPair("Referência para decaimento cosseno; o projeto não implementa os warm restarts do artigo", "Cosine decay reference; the project does not implement the paper's warm restarts") },
];

// Full citation metadata transcribed from the linked primary arXiv records.
// BibTeX cites the arXiv preprints; publication notes are separate when verified there.
export const bibliography: Record<string, { authors: string[]; publication?: string }> = {
  transformer: { authors: ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit", "Llion Jones", "Aidan N. Gomez", "Lukasz Kaiser", "Illia Polosukhin"] },
  bpe: { authors: ["Rico Sennrich", "Barry Haddow", "Alexandra Birch"], publication: "ACL 2016" },
  sentencepiece: { authors: ["Taku Kudo", "John Richardson"], publication: "EMNLP 2018 · demonstrations" },
  rope: { authors: ["Jianlin Su", "Yu Lu", "Shengfeng Pan", "Ahmed Murtadha", "Bo Wen", "Yunfeng Liu"] },
  gqa: { authors: ["Joshua Ainslie", "James Lee-Thorp", "Michiel de Jong", "Yury Zemlyanskiy", "Federico Lebrón", "Sumit Sanghai"], publication: "EMNLP 2023" },
  rmsnorm: { authors: ["Biao Zhang", "Rico Sennrich"], publication: "NeurIPS 2019" },
  prenorm: { authors: ["Ruibin Xiong", "Yunchang Yang", "Di He", "Kai Zheng", "Shuxin Zheng", "Chen Xing", "Huishuai Zhang", "Yanyan Lan", "Liwei Wang", "Tie-Yan Liu"], publication: "ICML 2020" },
  residual: { authors: ["Kaiming He", "Xiangyu Zhang", "Shaoqing Ren", "Jian Sun"] },
  swiglu: { authors: ["Noam Shazeer"] },
  tied: { authors: ["Ofir Press", "Lior Wolf"], publication: "EACL 2017" },
  autodiff: { authors: ["Atilim Gunes Baydin", "Barak A. Pearlmutter", "Alexey Andreyevich Radul", "Jeffrey Mark Siskind"], publication: "Journal of Machine Learning Research 18(153):1–43, 2018" },
  adam: { authors: ["Diederik P. Kingma", "Jimmy Ba"], publication: "ICLR 2015" },
  adamw: { authors: ["Ilya Loshchilov", "Frank Hutter"], publication: "ICLR 2019" },
  cosine: { authors: ["Ilya Loshchilov", "Frank Hutter"], publication: "ICLR 2017" },
};

export function paperBibtex(paper: PaperReference): string {
  const arxiv = paper.url.split("/").at(-1)!;
  const author = bibliography[paper.id].authors.join(" and ");
  return `@misc{glassllm_${paper.id}_${paper.preprintYear},\n  title = {${paper.title}},\n  author = {${author}},\n  year = {${paper.preprintYear}},\n  eprint = {${arxiv}},\n  archivePrefix = {arXiv},\n  doi = {10.48550/arXiv.${arxiv}},\n  url = {${paper.url}}\n}`;
}
