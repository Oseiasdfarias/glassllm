/** The inference graph in model.py/nn.py. Shapes follow Tracer (batch axis removed). */
export interface ModelConfig {
  vocab_size: number;
  d_model: number;
  n_layers: number;
  n_heads: number;
  n_kv_heads: number;
  max_len: number;
  rope_base?: number;
  eps?: number;
  init_std?: number;
}

export interface GraphNode {
  key: string;
  step: string | null;
  layer: number | null;
  title: string;
  subtitle: string;
  shape: string;
  x: number;
  y: number;
  width: number;
  height: number;
  kind: 'operation' | 'addition' | 'token' | 'cache' | 'info';
}

export interface GraphEdge {
  key: string;
  from: string;
  to: string;
  path: string;
  kind: 'flow' | 'residual' | 'cache' | 'shared' | 'feedback';
  label?: string;
  labelX?: number;
  labelY?: number;
}

export interface GraphGroup {
  key: string;
  title: string;
  subtitle: string;
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface NetworkGraph {
  width: number;
  height: number;
  nodes: GraphNode[];
  edges: GraphEdge[];
  groups: GraphGroup[];
}

export const DEFAULT_CONFIG: ModelConfig = {
  vocab_size: 1024,
  d_model: 128,
  n_layers: 4,
  n_heads: 4,
  n_kv_heads: 2,
  max_len: 128,
  rope_base: 10000,
  eps: 1e-6,
  init_std: 0.02,
};

/** Exactly the ids used by Tracer, with a layer prefix only for block steps. */
export function networkKey(step: string, layer: number | null): string {
  return layer === null ? step : `${layer}:${step}`;
}

type Point = readonly [number, number];

const line = (...points: Point[]): string =>
  points.map(([x, y], index) => `${index === 0 ? 'M' : 'L'} ${x} ${y}`).join(' ');

export function createNetworkGraph(config: ModelConfig, lang: 'pt' | 'en'): NetworkGraph {
  for (const name of ['vocab_size', 'd_model', 'n_layers', 'n_heads', 'n_kv_heads', 'max_len'] as const) {
    if (!Number.isInteger(config[name]) || config[name] <= 0) {
      throw new Error(`Invalid model configuration: ${name} must be a positive integer`);
    }
  }
  if (config.d_model % config.n_heads || config.n_heads % config.n_kv_heads ||
      (config.d_model / config.n_heads) % 2) {
    throw new Error('Invalid attention configuration: integral GQA groups and an even head dimension are required');
  }

  const t = (pt: string, en: string): string => lang === 'pt' ? pt : en;
  const d = config.d_model;
  const dk = d / config.n_heads;
  const hidden = Math.floor(8 * d / 3);
  const repeats = config.n_heads / config.n_kv_heads;
  const width = Math.max(2008, config.n_layers * 480 + 88);
  const nodes: GraphNode[] = [];
  const edges: GraphEdge[] = [];
  const groups: GraphGroup[] = [];
  const byKey = new Map<string, GraphNode>();
  const sequence = `n × ${d}`;
  const queryShape = `${config.n_heads} × n × ${dk}`;
  const kvShape = `${config.n_kv_heads} × n × ${dk}`;
  const scoresShape = `${config.n_heads} × n × m`;

  function addNode(node: GraphNode): GraphNode {
    nodes.push(node);
    byKey.set(node.key, node);
    return node;
  }

  function node(key: string): GraphNode {
    const result = byKey.get(key);
    if (!result) throw new Error(`Unknown graph node: ${key}`);
    return result;
  }

  const top = (key: string): Point => { const n = node(key); return [n.x + n.width / 2, n.y]; };
  const bottom = (key: string): Point => { const n = node(key); return [n.x + n.width / 2, n.y + n.height]; };
  const left = (key: string): Point => { const n = node(key); return [n.x, n.y + n.height / 2]; };
  const right = (key: string): Point => { const n = node(key); return [n.x + n.width, n.y + n.height / 2]; };

  function edge(from: string, to: string, path: string,
    kind: GraphEdge['kind'] = 'flow', label?: string, labelX?: number, labelY?: number): void {
    edges.push({ key: `${from}->${to}`, from, to, path, kind, label, labelX, labelY });
  }

  function vertical(from: string, to: string): void {
    edge(from, to, line(bottom(from), top(to)));
  }

  function globalNode(key: string, title: string, subtitle: string, shape: string,
    x: number, y: number, w: number, h: number, kind: GraphNode['kind'], step: string | null = null): void {
    addNode({ key, step, layer: null, title, subtitle, shape, x, y, width: w, height: h, kind });
  }

  globalNode('input', t('Texto inicial', 'Initial text'),
    'Prompt', 'string', 44, 36, 180, 68, 'token');
  globalNode('tokenize', t('Tokenizador BPE', 'BPE tokenizer'),
    t('Texto → ids inteiros', 'Text → integer ids'),
    `${config.vocab_size} ${t('tokens no vocabulário', 'vocabulary tokens')}`,
    254, 36, 220, 68, 'token');
  globalNode('context', t('Contexto de tokens', 'Token context'),
    t('ids + novo id', 'ids + new id'),
    t(`Janela: até ${config.max_len}`, `Window: up to ${config.max_len}`),
    506, 36, 190, 68, 'token');
  globalNode('embed', t('Embedding de tokens', 'Token embedding'),
    'X₀ = E[ids]', sequence, 728, 36, 230, 68, 'operation', 'embed');
  globalNode('weights.E', t('Pesos compartilhados · E', 'Shared weights · E'),
    t('Embedding e saída usam E', 'Embedding and output use E'),
    `${config.vocab_size} × ${d}`, 990, 36, 250, 68, 'info');
  globalNode('dimensions', t('Uma passagem do Transformer', 'One Transformer forward pass'),
    t('n = tokens novos; m = chaves atuais + cache', 'n = new tokens; m = current + cached keys'),
    `${config.n_layers} ${t('blocos', 'blocks')} · d = ${d} · ${config.n_heads} Q / ${config.n_kv_heads} KV · dₖ = ${dk}`,
    1280, 36, width - 1324, 68, 'info');

  edge('input', 'tokenize', line(right('input'), left('tokenize')));
  edge('tokenize', 'context', line(right('tokenize'), left('context')));
  edge('context', 'embed', line(right('context'), left('embed')));
  edge('weights.E', 'embed', line(left('weights.E'), right('embed')), 'shared');

  const layerSpan = config.n_layers * 480 - 40;
  const layerLeft = (width - layerSpan) / 2;

  for (let layer = 0; layer < config.n_layers; layer += 1) {
    const x = layerLeft + layer * 480;
    const cx = x + 220;
    const k = (step: string): string => networkKey(step, layer);
    const metadata = (name: string): string => `layer.${layer}.${name}`;
    groups.push({
      key: `layer.${layer}`,
      title: t(`Bloco ${layer + 1}`, `Block ${layer + 1}`),
      subtitle: t('Atenção causal → SwiGLU · duas conexões residuais', 'Causal attention → SwiGLU · two residual paths'),
      x, y: 156, width: 440, height: 1076,
    });

    function operation(step: string, title: string, subtitle: string, shape: string,
      y: number, relativeX = 88, w = 264, kind: GraphNode['kind'] = 'operation'): void {
      addNode({ key: k(step), step, layer, title, subtitle, shape,
        x: x + relativeX, y, width: w, height: 54, kind });
    }

    operation('norm.attn', t('RMSNorm · atenção', 'RMSNorm · attention'),
      `X̂ = X${layer} / √(mean(X${layer}²) + ε) · γ₁`, sequence, 228);
    operation('attn.q', t('Q · consulta', 'Q · query'), 'Q = X̂ · W_Q', queryShape, 290, 12, 120);
    operation('attn.k', t('K · chave', 'K · key'), 'K = X̂ · W_K', kvShape, 290, 160, 120);
    operation('attn.v', t('V · valor', 'V · value'), 'V = X̂ · W_V', kvShape, 290, 308, 120);
    operation('attn.rope_q', 'RoPE · Q', 'Q′ = rotate(Q, pos)', queryShape, 352, 12, 120);
    operation('attn.rope_k', 'RoPE · K', 'K′ = rotate(K, pos)', kvShape, 352, 160, 120);
    addNode({ key: metadata('cache'), step: null, layer, title: t('Memória K/V · cache', 'K/V memory · cache'),
      subtitle: t('Anexa K′ e V aos tokens anteriores', 'Append K′ and V to previous tokens'),
      shape: `${config.n_kv_heads} × m × ${dk} · K / V`,
      x: x + 158, y: 414, width: 264, height: 54, kind: 'cache' });
    addNode({ key: metadata('gqa'), step: null, layer, title: t('Expandir cabeças · GQA', 'Expand heads · GQA'),
      subtitle: `repeat_kv(K, ${repeats}) · repeat_kv(V, ${repeats})`,
      shape: `${config.n_kv_heads} → ${config.n_heads} ${t('cabeças', 'heads')}`,
      x: x + 158, y: 476, width: 264, height: 54, kind: 'info' });
    operation('attn.scores', t('Compatibilidade Q × K', 'Q × K compatibility'),
      `S = Q′ · K′ᵀ / √${dk}`, scoresShape, 538);
    operation('attn.mask', t('Máscara causal', 'Causal mask'),
      t('Sᵢⱼ = −∞ se j > posiçãoᵢ', 'Sᵢⱼ = −∞ if j > positionᵢ'), scoresShape, 600);
    operation('attn.softmax', t('Pesos de atenção', 'Attention weights'),
      'A = softmax(S)', scoresShape, 662);
    operation('attn.mix', t('Misturar os valores', 'Mix the values'),
      'Y_h = A · V', queryShape, 724);
    operation('attn.out', t('Concatenar e projetar', 'Concatenate and project'),
      'O = Concat(Y₁ … Y_H) · W_O', sequence, 786);
    operation('resid.attn', t('Somar o caminho residual', 'Add the residual path'),
      `H = X${layer} + O`, sequence, 848, 88, 264, 'addition');
    operation('norm.ffn', 'RMSNorm · FFN',
      'Ĥ = H / √(mean(H²) + ε) · γ₂', sequence, 910);
    operation('ffn.gate', t('Projeção gate', 'Gate projection'),
      'G = Ĥ · W₁', `n × ${hidden}`, 972, 46, 162);
    operation('ffn.up', t('Projeção up', 'Up projection'),
      'U = Ĥ · W₃', `n × ${hidden}`, 972, 232, 162);
    operation('ffn.act', t('Ativação SwiGLU', 'SwiGLU activation'),
      'SiLU(G) ⊙ U', `n × ${hidden}`, 1034);
    operation('ffn.down', t('Projeção down', 'Down projection'),
      'F = (SiLU(G) ⊙ U) · W₂', sequence, 1096);
    operation('resid.ffn', t('Somar o caminho residual', 'Add the residual path'),
      `X${layer + 1} = H + F`, sequence, 1158, 88, 264, 'addition');

    const source = layer === 0 ? 'embed' : networkKey('resid.ffn', layer - 1);
    const sourceBottom = bottom(source);
    if (layer === 0) {
      edge(source, k('norm.attn'), line(sourceBottom, [sourceBottom[0], 128], [cx, 128], top(k('norm.attn'))));
      edge(source, k('resid.attn'),
        line(sourceBottom, [sourceBottom[0], 128], [x + 2, 128], [x + 2, 875], left(k('resid.attn'))),
        'residual', `X${layer}`, x + 44, 810);
    } else {
      // The previous block travels through the clear column gap before branching.
      const gapX = x - 20;
      edge(source, k('norm.attn'),
        line(sourceBottom, [sourceBottom[0], 1240], [gapX, 1240], [gapX, 214], [cx, 214], top(k('norm.attn'))));
      edge(source, k('resid.attn'),
        line(sourceBottom, [sourceBottom[0], 1240], [gapX, 1240], [gapX, 214],
          [x + 2, 214], [x + 2, 875], left(k('resid.attn'))),
        'residual', `X${layer}`, x + 44, 810);
    }

    for (const step of ['attn.q', 'attn.k', 'attn.v']) {
      const target = top(k(step));
      edge(k('norm.attn'), k(step), line(bottom(k('norm.attn')), [cx, 286], [target[0], 286], target));
    }
    vertical(k('attn.q'), k('attn.rope_q'));
    vertical(k('attn.k'), k('attn.rope_k'));
    // K enters the cache after RoPE; V is cached without a positional rotation.
    edge(k('attn.rope_k'), metadata('cache'), line(bottom(k('attn.rope_k')), [cx, 414]), 'cache');
    edge(k('attn.v'), metadata('cache'), line(bottom(k('attn.v')), [x + 368, 414]), 'cache');
    edge(metadata('cache'), metadata('cache'),
      line([x + 422, 426], [x + 446, 426], [x + 446, 456], [x + 422, 456]),
      'cache', 't−1', x + 447, 416);
    vertical(metadata('cache'), metadata('gqa'));
    edge(k('attn.rope_q'), k('attn.scores'),
      line(bottom(k('attn.rope_q')), [x + 72, 565], left(k('attn.scores'))));
    edge(metadata('gqa'), k('attn.scores'),
      line([x + 220, 530], top(k('attn.scores'))));
    edge(metadata('gqa'), k('attn.mix'),
      line(right(metadata('gqa')), [x + 432, 503], [x + 432, 751], right(k('attn.mix'))),
      'flow', 'V', x + 416, 644);
    for (const [from, to] of [
      ['attn.scores', 'attn.mask'], ['attn.mask', 'attn.softmax'],
      ['attn.softmax', 'attn.mix'], ['attn.mix', 'attn.out'],
      ['attn.out', 'resid.attn'], ['resid.attn', 'norm.ffn'],
    ]) vertical(k(from), k(to));

    for (const step of ['ffn.gate', 'ffn.up']) {
      const target = top(k(step));
      edge(k('norm.ffn'), k(step), line(bottom(k('norm.ffn')), [cx, 968], [target[0], 968], target));
      const branchBottom = bottom(k(step));
      edge(k(step), k('ffn.act'), line(branchBottom, [branchBottom[0], 1030], [cx, 1030], top(k('ffn.act'))));
    }
    vertical(k('ffn.act'), k('ffn.down'));
    vertical(k('ffn.down'), k('resid.ffn'));
    edge(k('resid.attn'), k('resid.ffn'),
      line(right(k('resid.attn')), [x + 408, 875], [x + 408, 1185], right(k('resid.ffn'))),
      'residual', 'H', x + 390, 1100);
  }

  const outputX = (width - 1536) / 2;
  globalNode('norm.final', 'RMSNorm · final', `h = RMSNorm(X${config.n_layers})`,
    sequence, outputX, 1282, 300, 72, 'operation', 'norm.final');
  globalNode('logits', t('Projetar para o vocabulário', 'Project to the vocabulary'),
    'z = h · Eᵀ', `n × ${config.vocab_size}`, outputX + 412, 1282, 300, 72, 'operation', 'logits');
  globalNode('probs', t('Probabilidades do próximo token', 'Next-token probabilities'),
    'p = softmax(z[−1] / T)', `${config.vocab_size}`, outputX + 824, 1282, 300, 72, 'operation', 'probs');
  globalNode('sample', t('Escolher o próximo token', 'Choose the next token'),
    t('Amostra ~ p ou argmax(p)', 'Sample ~ p or argmax(p)'),
    t('1 id → texto', '1 id → text'), outputX + 1236, 1282, 300, 72, 'token', 'sample');

  const lastOutput = networkKey('resid.ffn', config.n_layers - 1);
  const finalBottom = bottom(lastOutput);
  const finalTop = top('norm.final');
  edge(lastOutput, 'norm.final', line(finalBottom, [finalBottom[0], 1260], [finalTop[0], 1260], finalTop));
  edge('norm.final', 'logits', line(right('norm.final'), left('logits')));
  edge('logits', 'probs', line(right('logits'), left('probs')), 'flow', 'z[−1] / T',
    outputX + 768, 1304);
  edge('probs', 'sample', line(right('probs'), left('sample')));

  const eBottom = bottom('weights.E');
  const logitsBottom = bottom('logits');
  edge('weights.E', 'logits',
    line(eBottom, [eBottom[0], 140], [width - 20, 140], [width - 20, 1392],
      [logitsBottom[0], 1392], logitsBottom),
    'shared', t('A mesma matriz E · pesos atrelados', 'The same matrix E · tied weights'),
    width / 2 + 300, 1380);

  const sampleBottom = bottom('sample');
  const contextBottom = bottom('context');
  edge('sample', 'context',
    line(sampleBottom, [sampleBottom[0], 1428], [20, 1428], [20, 116],
      [contextBottom[0], 116], contextBottom),
    'feedback', t('Novo token → atualizar contexto → próxima passagem', 'New token → update context → next forward pass'),
    width / 2, 1452);

  return { width, height: 1468, nodes, edges, groups };
}
