import { createNetworkGraph, DEFAULT_CONFIG, networkKey, type GraphNode, type ModelConfig, type NetworkGraph } from "../lib/network-graph";
import { renderNetwork } from "../lib/network-render";
import { HEAD_STEPS, type TraceInspection, type TraceSummary, type TraceValue } from "../lib/network-trace";
import { iconMarkup } from "../lib/icons";

type Inspector = (step: string, layer: number | null, head: number) => Promise<TraceInspection>;
const descriptions: Record<string, [string, string]> = {
  input: ["A frase inicial entra no modelo como texto.", "The initial prompt enters the model as text."],
  tokenize: ["O BPE divide a frase em pedaços do vocabulário e transforma cada um em um ID.", "BPE splits the prompt into vocabulary pieces and converts each into an ID."],
  context: ["Os IDs da frase e dos tokens gerados formam a janela de contexto. Um token gerado é anexado diretamente, sem passar novamente pelo BPE.", "Prompt IDs and generated tokens form the context window. A generated ID is appended directly, without running BPE again."],
  embed: ["Cada ID busca uma linha na matriz E. Esses vetores são a entrada do primeiro bloco.", "Each ID looks up a row in E. These vectors enter the first block."],
  "weights.E": ["A matriz aprendida E é usada tanto para representar tokens na entrada quanto para calcular os logits na saída.", "The learned E matrix both represents input tokens and computes output logits."],
  dimensions: ["Na primeira previsão, n é o tamanho do prompt. Nas seguintes, o cache permite calcular apenas o token novo (n = 1). m inclui as chaves guardadas. Ao encher a janela, o modelo reconstrói o cache.", "For the first prediction, n is the prompt length. Afterwards, the cache allows computation of only the new token (n = 1). m includes stored keys. When the window fills, the model rebuilds its cache."],
  "norm.attn": ["Normaliza a escala dos vetores antes das três projeções da atenção. γ é um peso aprendido.", "Normalizes vector scale before the three attention projections. γ is a learned weight."],
  "attn.q": ["A projeção Q cria consultas: o que cada posição procura nas outras posições.", "The Q projection creates queries: what each position looks for in other positions."],
  "attn.k": ["A projeção K cria chaves que serão comparadas às consultas. Duas cabeças K são compartilhadas por quatro cabeças Q.", "The K projection creates keys to compare with queries. Two K heads are shared by four Q heads."],
  "attn.v": ["A projeção V cria os valores que serão combinados pelos pesos de atenção. V não recebe RoPE.", "The V projection creates values to combine using attention weights. V does not receive RoPE."],
  "attn.rope_q": ["Rotaciona pares de componentes de Q de acordo com a posição do token, incluindo o deslocamento do cache.", "Rotates pairs of Q components according to token position, including the cache offset."],
  "attn.rope_k": ["Aplica a mesma codificação de posição às chaves K antes de armazená-las no cache.", "Applies the same position encoding to K keys before storing them in the cache."],
  cache: ["Concatena as novas chaves após RoPE e os novos valores ao cache deste bloco. Essa memória é reutilizada na próxima previsão. É uma estrutura do modelo, sem snapshot separado no rastreamento.", "Appends new keys after RoPE and new values to this block's cache. This memory is reused for the next prediction. This model structure has no separate trace snapshot."],
  gqa: ["GQA repete as cabeças K/V para combiná-las com as cabeças Q. Os pesos K/V continuam compartilhados. Essa expansão não possui snapshot separado.", "GQA repeats K/V heads to match Q heads. K/V weights remain shared. This expansion has no separate snapshot."],
  "attn.scores": ["Cada consulta é comparada às chaves pelo produto escalar, dividido pela raiz da dimensão da cabeça.", "Each query is compared with keys using a dot product, divided by the square root of the head dimension."],
  "attn.mask": ["Posições futuras recebem −∞. Assim, a previsão não pode consultar tokens que ainda não foram vistos.", "Future positions receive −∞, so prediction cannot look at tokens it has not seen."],
  "attn.softmax": ["Converte os scores em pesos positivos. Cada linha soma aproximadamente 1; posições mascaradas recebem peso zero.", "Converts scores into positive weights. Each row sums to approximately 1; masked positions receive zero weight."],
  "attn.mix": ["Os pesos de atenção combinam os valores V de todas as posições disponíveis, em cada cabeça.", "Attention weights combine V values from all available positions within each head."],
  "attn.out": ["Reúne as cabeças e aplica W_O para voltar à dimensão do modelo.", "Joins the heads and applies W_O to return to the model dimension."],
  "resid.attn": ["Soma a saída da atenção à entrada original do bloco. O caminho residual preserva essa informação.", "Adds the attention output to the original block input. The residual path preserves that information."],
  "norm.ffn": ["Normaliza o resultado da primeira soma antes da rede de transformação SwiGLU.", "Normalizes the first residual sum before the SwiGLU transformation network."],
  "ffn.gate": ["Projeta os vetores para a dimensão interna. Esse ramo controla a ativação após SiLU.", "Projects vectors into the hidden dimension. This branch controls activation after SiLU."],
  "ffn.up": ["Cria o segundo ramo da FFN na mesma dimensão interna.", "Creates the second FFN branch in the same hidden dimension."],
  "ffn.act": ["Aplica SiLU ao ramo gate e multiplica o resultado, componente a componente, pelo ramo up.", "Applies SiLU to the gate branch and multiplies the result elementwise by the up branch."],
  "ffn.down": ["Projeta a ativação interna de volta à dimensão do modelo.", "Projects the hidden activation back into the model dimension."],
  "resid.ffn": ["Soma a saída da FFN ao resultado da atenção. Esse vetor segue para o próximo bloco ou a normalização final.", "Adds the FFN output to the attention result. This vector enters the next block or the final normalization."],
  "norm.final": ["Normaliza os vetores depois do último bloco.", "Normalizes vectors after the last block."],
  logits: ["Multiplica os vetores pela transposta de E. Cada componente é um score para um token do vocabulário.", "Multiplies vectors by E transposed. Each component is a score for a vocabulary token."],
  probs: ["Usa os logits da última posição. A temperatura controla sua escala antes do softmax.", "Uses the last position's logits. Temperature controls their scale before softmax."],
  sample: ["Escolhe um ID por sorteio ou pelo maior valor de probabilidade. Esse ID é decodificado e anexado ao contexto para a próxima previsão.", "Chooses an ID by sampling or by highest probability. That ID is decoded and appended to the context for the next prediction."],
};

export class NetworkView {
  private graph: NetworkGraph;
  private config = DEFAULT_CONFIG;
  private trace: TraceSummary[] = [];
  private snapshots = new Map<string, TraceSummary>();
  private iteration = 0;
  private selected = "embed";
  private head = 0;
  private cursor = -1;
  private zoom = 1;
  private animation = 0;
  private inspecting = 0;
  private playing = false;
  private computing = false;
  private wake: (() => void) | null = null;
  private resize: ResizeObserver;
  private q = <T extends Element = HTMLElement>(selector: string) => this.root.querySelector<T>(selector)!;
  private t = (pt: string, en: string) => this.lang === "pt" ? pt : en;

  constructor(private root: HTMLElement, private lang: "pt" | "en", private inspect: Inspector, private pauseGeneration: () => void, private onCalculation: (record: TraceSummary | null) => void, private onPlayback: (playing: boolean) => void) {
    this.graph = createNetworkGraph(this.config, lang);
    this.bind();
    this.showSelection();
    this.resize = new ResizeObserver(() => {
      // Preserve a deliberate zoom; refit only when the overview is selected.
      if (this.root.dataset.overview === "true") this.fit();
    });
    this.resize.observe(this.q("[data-network-viewport]"));
    this.fit();
  }

  configure(config: ModelConfig, parameters: number) {
    this.config = config;
    this.graph = createNetworkGraph(config, this.lang);
    this.q("[data-network-canvas]").innerHTML = renderNetwork(this.graph, this.lang);
    const count = new Intl.NumberFormat(this.lang === "pt" ? "pt-BR" : "en").format(parameters);
    this.q("[data-network-config]").textContent = this.t(
      `${config.n_layers} blocos · ${config.d_model} dimensões · ${config.n_heads} Q / ${config.n_kv_heads} K/V · ${count} parâmetros`,
      `${config.n_layers} blocks · ${config.d_model} dimensions · ${config.n_heads} Q / ${config.n_kv_heads} K/V · ${count} parameters`,
    );
    const navigation = this.q("[data-network-navigation]");
    navigation.replaceChildren(...["input", ...Array.from({ length: config.n_layers }, (_, layer) => `${layer}:norm.attn`), "logits"].map((key, i) => {
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.networkFocus = key;
      button.textContent = i === 0 ? this.t("Entrada", "Input") : i === config.n_layers + 1 ? this.t("Saída", "Output") : this.t(`Bloco ${i}`, `Block ${i}`);
      return button;
    }));
    this.showSelection();
    this.fit();
  }

  reset() {
    this.stop();
    this.trace = [];
    this.snapshots.clear();
    this.iteration = 0;
    this.cursor = -1;
    this.q<HTMLElement>("[data-network-current]").hidden = true;
    this.inspecting++;
    this.root.querySelectorAll(".has-values, .is-flowing").forEach((item) => item.classList.remove("has-values", "is-flowing"));
    this.onCalculation(null);
    this.q("[data-network-progress]").textContent = this.t("Mapa completo · aguardando a primeira previsão", "Complete map · waiting for the first prediction");
    this.showSelection();
    this.setControls();
  }

  setTrace(records: TraceSummary[], iteration: number) {
    this.stop();
    this.iteration = iteration;
    this.trace = [...records].sort((a, b) => a.order - b.order);
    this.snapshots = new Map(this.trace.map((record) => [networkKey(record.id, record.layer), record]));
    this.cursor = -1;
    this.q<HTMLElement>("[data-network-current]").hidden = true;
    this.root.querySelectorAll(".is-flowing").forEach((item) => item.classList.remove("is-flowing"));
    this.onCalculation(null);
    this.q("[data-network-progress]").textContent = this.t(`Token ${iteration} · ${this.trace.length} cálculos prontos · aguardando reprodução`, `Token ${iteration} · ${this.trace.length} calculations ready · waiting for playback`);
    this.root.querySelectorAll<SVGGElement>("[data-network-key]").forEach((el) => el.classList.toggle("has-values", this.snapshots.has(el.dataset.networkKey!)));
    this.showSelection();
    this.setControls();
  }

  get slowMotion() { return this.q<HTMLInputElement>("[data-network-slow]").checked; }
  get isPlaying() { return this.playing; }

  get hasPendingFlow() { return this.trace.length > 0 && this.cursor < this.trace.length - 1; }

  pauseFlow() { this.stop(); }
  setComputing(computing: boolean) { this.computing = computing; this.setControls(); }

  async animate(manual = false, resume = false) {
    if (!this.trace.length) return;
    this.stop();
    const version = this.animation;
    this.playing = true;
    this.setControls();
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduced && !manual && !this.slowMotion) {
      await this.untilStopped(this.activate(this.trace.length - 1, false));
    } else {
      const first = resume ? this.cursor + 1 : 0;
      for (let i = first; i < this.trace.length; i++) {
        if (version !== this.animation) return;
        await this.untilStopped(this.activate(i, manual || this.slowMotion, this.slowMotion));
        if (version !== this.animation) return;
        if (i < this.trace.length - 1) await this.wait(this.slowMotion ? Number(this.q<HTMLSelectElement>("[data-network-delay]").value) : manual ? 110 : 10);
      }
    }
    if (version !== this.animation) return;
    this.playing = false;
    this.setControls();
  }

  private stop() {
    this.animation++;
    this.playing = false;
    this.wake?.();
    this.setControls();
  }

  private wait(duration: number): Promise<void> {
    return new Promise((resolve) => {
      const finish = () => {
        window.clearTimeout(timer);
        if (this.wake === finish) this.wake = null;
        resolve();
      };
      const timer = window.setTimeout(finish, duration);
      this.wake = finish;
    });
  }

  private untilStopped(work: Promise<void>): Promise<void> {
    return new Promise((resolve, reject) => {
      const finish = () => {
        if (this.wake === finish) this.wake = null;
        resolve();
      };
      this.wake = finish;
      void work.then(finish, (error) => {
        if (this.wake === finish) this.wake = null;
        reject(error);
      });
    });
  }

  private setControls() {
    this.root.dataset.flowPlaying = String(this.playing);
    this.onPlayback(this.playing);
    this.q<HTMLButtonElement>('[data-network-action="replay"]').disabled = this.computing || !this.trace.length;
    this.q<HTMLButtonElement>('[data-network-action="resume"]').disabled = this.computing || this.playing || !this.hasPendingFlow;
    this.q<HTMLButtonElement>('[data-network-action="previous"]').disabled = this.computing || this.cursor <= 0;
    this.q<HTMLButtonElement>('[data-network-action="next"]').disabled = this.computing || !this.hasPendingFlow;
    this.q<HTMLButtonElement>('[data-network-action="stop"]').disabled = !this.playing;
  }

  private async activate(index: number, select: boolean, follow = false) {
    this.cursor = index;
    const record = this.trace[index];
    const key = networkKey(record.id, record.layer);
    this.onCalculation(record);
    this.root.querySelectorAll<SVGGElement>("[data-network-key]").forEach((el) => el.classList.toggle("is-flowing", el.dataset.networkKey === key));
    // Metadata has no trace event: highlight its branch when its result is consumed.
    const extras = record.id === "attn.scores" || record.id === "attn.mix"
      ? new Set([`layer.${record.layer}.cache`, `layer.${record.layer}.gqa`]) : new Set<string>();
    this.root.querySelectorAll<SVGGElement>("[data-network-edge]").forEach((el) => {
      el.classList.toggle("is-flowing", el.dataset.networkTo === key || extras.has(el.dataset.networkTo!));
    });
    this.q("[data-network-progress]").textContent = this.t(
      `Token ${this.iteration} · cálculo ${index + 1}/${this.trace.length} · ${record.layer === null ? "" : `bloco ${record.layer + 1} · `}${record.id}`,
      `Token ${this.iteration} · calculation ${index + 1}/${this.trace.length} · ${record.layer === null ? "" : `block ${record.layer + 1} · `}${record.id}`,
    );
    this.q<HTMLElement>("[data-network-current]").hidden = !this.slowMotion;
    if (this.slowMotion) {
      const node = this.graph.nodes.find((n) => n.key === key)!;
      this.q("[data-network-current-number]").textContent = `${index + 1} / ${this.trace.length}`;
      this.q("[data-network-current-title]").textContent = `${node.layer === null ? "" : this.t(`Bloco ${node.layer + 1} · `, `Block ${node.layer + 1} · `)}${node.title}`;
      this.q("[data-network-current-explanation]").textContent = descriptions[record.id]?.[this.lang === "pt" ? 0 : 1] ?? node.subtitle;
      this.q("[data-network-current-equation]").textContent = record.equation;
      if (follow && this.q<HTMLInputElement>("[data-network-follow]").checked) this.followNode(node);
    }
    if (select) {
      this.selected = key;
      await this.showSelection();
    }
    this.setControls();
  }

  private select(key: string) {
    if (!this.graph.nodes.some((node) => node.key === key)) return;
    this.pauseGeneration();
    this.stop();
    this.selected = key;
    const index = this.trace.findIndex((record) => networkKey(record.id, record.layer) === key);
    if (index >= 0) void this.activate(index, false);
    this.showSelection();
  }

  private node(): GraphNode { return this.graph.nodes.find((node) => node.key === this.selected)!; }

  private async showSelection() {
    const node = this.node();
    this.root.querySelectorAll<SVGGElement>("[data-network-key]").forEach((el) => {
      const active = el.dataset.networkKey === this.selected;
      el.classList.toggle("is-selected", active);
      el.setAttribute("aria-pressed", String(active));
    });
    this.q("[data-network-title]").textContent = node.title;
    this.q("[data-network-layer]").textContent = node.layer === null ? this.t("Geral", "Global") : this.t(`Bloco ${node.layer + 1}`, `Block ${node.layer + 1}`);
    const descriptionKey = node.step ?? (node.kind === "cache" ? "cache" : node.key.endsWith(".gqa") ? "gqa" : node.key);
    this.q("[data-network-description]").textContent = descriptions[descriptionKey]?.[this.lang === "pt" ? 0 : 1] ?? node.subtitle;
    this.q("[data-network-equation]").textContent = node.subtitle;
    this.q("[data-network-shape]").textContent = node.shape;
    const heads = node.step !== null && HEAD_STEPS.has(node.step);
    this.q<HTMLElement>("[data-network-head-wrap]").hidden = !heads;
    if (heads) {
      const count = ["attn.k", "attn.v", "attn.rope_k"].includes(node.step!) ? this.config.n_kv_heads : this.config.n_heads;
      this.head = Math.min(this.head, count - 1);
      const select = this.q<HTMLSelectElement>("[data-network-head]");
      select.replaceChildren(...Array.from({ length: count }, (_, i) => {
        const option = document.createElement("option");
        option.value = String(i);
        option.textContent = this.t(`Cabeça ${i + 1}`, `Head ${i + 1}`);
        return option;
      }));
      select.value = String(this.head);
    }
    const record = this.snapshots.get(node.key);
    this.clearValues();
    if (!record || !node.step) {
      this.q("[data-network-value-status]").textContent = node.step ? this.t("Gere um token para ver os valores reais.", "Generate a token to see actual values.") : this.t("Estrutura do modelo · dimensões simbólicas", "Model structure · symbolic dimensions");
      this.inspecting++;
      return;
    }
    this.q("[data-network-equation]").textContent = record.equation;
    this.q("[data-network-shape]").textContent = record.shape.length ? record.shape.join(" × ") : this.t("Escalar", "Scalar");
    this.q("[data-network-value-status]").textContent = this.t("Carregando o tensor registrado…", "Loading the recorded tensor…");
    const version = ++this.inspecting;
    await this.inspect(node.step, node.layer, this.head).then((result) => {
      if (version !== this.inspecting || result.iteration !== this.iteration) return;
      this.showValues(result);
    }).catch((error: Error) => {
      if (version === this.inspecting) this.q("[data-network-value-status]").textContent = this.t(`Falha ao inspecionar: ${error.message}`, `Inspection failed: ${error.message}`);
    });
  }

  private clearValues() {
    this.q<HTMLElement>("[data-network-stats]").hidden = true;
    this.q("[data-network-matrix]").replaceChildren();
    this.q("[data-network-matrix-caption]").textContent = "";
    this.q<HTMLElement>("[data-network-source]").hidden = true;
  }

  private number(value: TraceValue | null): string {
    if (value === null) return "—";
    if (typeof value === "string") return value;
    if (value === 0 || Number.isInteger(value) && Math.abs(value) < 10000) return String(value);
    return (Math.abs(value) < 0.001 || Math.abs(value) >= 10000 ? value.toExponential(2) : value.toFixed(3)).replace(".", this.lang === "pt" ? "," : ".");
  }

  private showValues(result: TraceInspection) {
    this.q("[data-network-shape]").textContent = result.shape.length ? result.shape.join(" × ") : this.t("Escalar", "Scalar");
    this.q("[data-network-equation]").textContent = result.equation;
    this.q<HTMLElement>("[data-network-stats]").hidden = false;
    this.q("[data-network-min]").textContent = this.number(result.min);
    this.q("[data-network-mean]").textContent = this.number(result.mean);
    this.q("[data-network-max]").textContent = this.number(result.max);
    this.q("[data-network-value-status]").textContent = this.t(
      `Valores reais · previsão ${result.iteration}${result.nonFinite ? ` · ${result.nonFinite} valores não finitos (excluídos das estatísticas)` : ""}`,
      `Actual values · prediction ${result.iteration}${result.nonFinite ? ` · ${result.nonFinite} non-finite values (excluded from statistics)` : ""}`,
    );
    const table = document.createElement("table");
    table.style.minWidth = `${(result.columnLabels.length + 1) * 62}px`;
    table.setAttribute("aria-label", this.t("Amostra numérica do tensor", "Numerical tensor sample"));
    const header = document.createElement("tr");
    header.append(document.createElement("th"));
    for (const label of result.columnLabels) {
      const cell = document.createElement("th");
      cell.scope = "col";
      cell.textContent = label.replaceAll("\n", "↵");
      cell.title = label;
      header.append(cell);
    }
    const thead = document.createElement("thead");
    thead.append(header);
    table.append(thead);
    const tbody = document.createElement("tbody");
    const magnitude = Math.max(Math.abs(result.min ?? 0), Math.abs(result.max ?? 0), 1e-9);
    result.grid.forEach((row, i) => {
      const tr = document.createElement("tr");
      const label = document.createElement("th");
      label.scope = "row";
      label.textContent = result.rowLabels[i].replaceAll("\n", "↵");
      label.title = result.rowLabels[i];
      tr.append(label);
      row.forEach((value, j) => {
        const cell = document.createElement("td");
        cell.textContent = this.number(value);
        cell.title = `[${result.rowOffset + i}, ${result.columnOffset + j}] = ${value}`;
        if (typeof value === "number") {
          const opacity = 0.05 + 0.45 * Math.min(1, Math.abs(value) / magnitude);
          cell.style.backgroundColor = value < 0 ? `rgba(231,176,74,${opacity})` : `rgba(79,179,194,${opacity})`;
        } else cell.classList.add("non-finite");
        tr.append(cell);
      });
      tbody.append(tr);
    });
    table.append(tbody);
    this.q("[data-network-matrix]").replaceChildren(table);
    const cropped = result.fullRows > 8 || result.fullColumns > 8;
    this.q("[data-network-matrix-caption]").textContent = this.t(
      `${cropped ? "Recorte: últimas " : "Matriz: "}${result.grid.length} linhas × ${result.grid[0]?.length ?? 0} colunas de ${result.fullRows} × ${result.fullColumns}. Ciano: positivo; âmbar: negativo.`,
      `${cropped ? "Sample: last " : "Matrix: "}${result.grid.length} rows × ${result.grid[0]?.length ?? 0} columns of ${result.fullRows} × ${result.fullColumns}. Cyan: positive; amber: negative.`,
    );
    if (result.source) {
      this.q<HTMLElement>("[data-network-source]").hidden = false;
      this.q("[data-network-source-location]").textContent = `${result.source.file}:${result.source.line}`;
      this.q("[data-network-source-code]").textContent = result.source.code;
    }
  }

  private fit() {
    const viewport = this.q<HTMLElement>("[data-network-viewport]");
    this.root.dataset.overview = "true";
    this.setZoom(Math.min((viewport.clientWidth - 24) / this.graph.width, (viewport.clientHeight - 24) / this.graph.height), false);
    viewport.scrollTo(0, 0);
  }

  private setZoom(value: number, center = true) {
    const viewport = this.q<HTMLElement>("[data-network-viewport]");
    const old = this.zoom;
    this.zoom = Math.max(0.08, Math.min(2, value));
    const canvas = this.q<HTMLElement>("[data-network-canvas]");
    canvas.style.width = `${this.graph.width * this.zoom}px`;
    canvas.style.height = `${this.graph.height * this.zoom}px`;
    this.q<SVGSVGElement>(".network-svg").style.transform = `scale(${this.zoom})`;
    this.q("[data-network-zoom]").textContent = `${Math.round(this.zoom * 100)}%`;
    if (center) {
      this.root.dataset.overview = "false";
      viewport.scrollLeft = (viewport.scrollLeft + viewport.clientWidth / 2) * this.zoom / old - viewport.clientWidth / 2;
      viewport.scrollTop = (viewport.scrollTop + viewport.clientHeight / 2) * this.zoom / old - viewport.clientHeight / 2;
    }
  }

  private focus(key: string) {
    const node = this.graph.nodes.find((n) => n.key === key);
    if (!node) return;
    const viewport = this.q<HTMLElement>("[data-network-viewport]");
    this.setZoom(Math.min(1.3, Math.max(0.6, (viewport.clientWidth - 24) / 460)));
    const x = node.layer !== null ? this.graph.groups[node.layer].x : node.x;
    const y = node.layer !== null ? this.graph.groups[node.layer].y : node.y - 16;
    viewport.scrollTo(Math.max(0, x * this.zoom - 12), Math.max(0, y * this.zoom - 12));
    this.select(key);
  }

  private followNode(node: GraphNode) {
    const viewport = this.q<HTMLElement>("[data-network-viewport]");
    // Zoom enough to read the operation while keeping its neighboring branches visible.
    if (this.root.dataset.overview === "true" || this.zoom < 0.6) this.setZoom(Math.min(1.1, Math.max(0.6, (viewport.clientWidth - 24) / 460)));
    const left = node.x * this.zoom;
    const top = node.y * this.zoom;
    const width = node.width * this.zoom;
    const height = node.height * this.zoom;
    const margin = 48;
    if (left < viewport.scrollLeft + margin || left + width > viewport.scrollLeft + viewport.clientWidth - margin || top < viewport.scrollTop + margin || top + height > viewport.scrollTop + viewport.clientHeight - margin) {
      viewport.scrollTo({
        left: Math.max(0, left + width / 2 - viewport.clientWidth / 2),
        top: Math.max(0, top + height / 2 - viewport.clientHeight / 2),
        behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth",
      });
    }
  }

  private bind() {
    this.root.addEventListener("click", (event) => {
      const target = event.target as Element;
      const node = target.closest<SVGGElement>("[data-network-key]");
      if (node) { this.select(node.dataset.networkKey!); return; }
      const focus = target.closest<HTMLButtonElement>("[data-network-focus]");
      if (focus) { this.focus(focus.dataset.networkFocus!); return; }
      const action = target.closest<HTMLButtonElement>("[data-network-action]")?.dataset.networkAction;
      if (action === "zoom-in") this.setZoom(this.zoom * 1.3);
      if (action === "zoom-out") this.setZoom(this.zoom / 1.3);
      if (action === "fit") this.fit();
      if (action === "replay") { this.pauseGeneration(); void this.animate(true); }
      if (action === "resume") { this.pauseGeneration(); void this.animate(true, true); }
      if (action === "stop") { this.pauseGeneration(); this.stop(); }
      if (action === "next") {
        this.pauseGeneration();
        this.stop();
        void this.activate(Math.min(this.cursor + 1, this.trace.length - 1), true, true);
      }
      if (action === "previous") {
        this.pauseGeneration();
        this.stop();
        void this.activate(Math.max(0, this.cursor - 1), true, true);
      }
      if (action === "fullscreen") {
        if (document.fullscreenElement) void document.exitFullscreen();
        else if (this.root.requestFullscreen) void this.root.requestFullscreen().catch(() => {
          this.q("[data-network-progress]").textContent = this.t("Tela cheia indisponível neste navegador.", "Full screen is unavailable in this browser.");
        });
        else this.q("[data-network-progress]").textContent = this.t("Tela cheia indisponível neste navegador.", "Full screen is unavailable in this browser.");
      }
    });
    this.root.addEventListener("keydown", (event) => {
      const node = (event.target as Element).closest<SVGGElement>("[data-network-key]");
      if (node && (event.key === "Enter" || event.key === " ")) {
        event.preventDefault();
        this.select(node.dataset.networkKey!);
      }
    });
    this.q<HTMLSelectElement>("[data-network-head]").addEventListener("change", (event) => {
      this.pauseGeneration();
      this.stop();
      this.head = Number((event.target as HTMLSelectElement).value);
      this.showSelection();
    });
    this.q<HTMLInputElement>("[data-network-slow]").addEventListener("change", () => {
      this.pauseGeneration();
      this.stop();
      this.q<HTMLSelectElement>("[data-network-delay]").disabled = !this.slowMotion;
      this.q<HTMLElement>("[data-network-current]").hidden = !this.slowMotion || this.cursor < 0;
      if (this.cursor >= 0) void this.activate(this.cursor, true, true);
    });
    document.addEventListener("fullscreenchange", () => {
      this.q('[data-network-fullscreen-label]').textContent = document.fullscreenElement ? this.t("Sair da tela cheia", "Exit full screen") : this.t("Tela cheia", "Full screen");
      const button = this.q('[data-network-action="fullscreen"]');
      button.querySelector("svg")?.remove();
      button.insertAdjacentHTML("afterbegin", iconMarkup(document.fullscreenElement ? "minimize" : "maximize"));
      this.fit();
    });
    const viewport = this.q<HTMLElement>("[data-network-viewport]");
    let drag: { x: number; y: number; left: number; top: number } | null = null;
    viewport.addEventListener("pointerdown", (event) => {
      if (event.pointerType !== "mouse" || event.button !== 0 || (event.target as Element).closest("[data-network-key]")) return;
      drag = { x: event.clientX, y: event.clientY, left: viewport.scrollLeft, top: viewport.scrollTop };
      viewport.setPointerCapture(event.pointerId);
      viewport.classList.add("is-dragging");
    });
    viewport.addEventListener("pointermove", (event) => {
      if (!drag) return;
      viewport.scrollLeft = drag.left - event.clientX + drag.x;
      viewport.scrollTop = drag.top - event.clientY + drag.y;
    });
    const end = () => { drag = null; viewport.classList.remove("is-dragging"); };
    viewport.addEventListener("pointerup", end);
    viewport.addEventListener("pointercancel", end);
  }
}
