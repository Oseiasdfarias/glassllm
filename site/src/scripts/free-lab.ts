type Message = { requestId: number; type: string; [key: string]: unknown };
import { wheelName } from "../generated/wheel-asset";
import { NetworkView } from "./network-view";
import type { ModelConfig } from "../lib/network-graph";
import type { TraceInspection, TraceSummary } from "../lib/network-trace";

for (const root of document.querySelectorAll<HTMLElement>(".lab")) {
  if (root.dataset.initialized) continue;
  root.dataset.initialized = "true";

  const lang = root.dataset.lang === "en" ? "en" : "pt";
  const copy = lang === "pt"
    ? { loading: "Preparando o modelo no navegador…", ready: "Modelo pronto · roda localmente", failed: "Não foi possível carregar o modelo", running: "Gerando token a token", paused: "Geração pausada", done: "Limite desta rodada alcançado", empty: "Escreva uma frase para começar." }
    : { loading: "Preparing the model in your browser…", ready: "Model ready · running locally", failed: "Could not load the model", running: "Generating one token at a time", paused: "Generation paused", done: "This run reached its token limit", empty: "Write a prompt to begin." };

  const query = <T extends HTMLElement>(selector: string) => root.querySelector<T>(selector)!;
  const prompt = query<HTMLTextAreaElement>("[data-prompt]");
  const output = query<HTMLDivElement>("[data-output]");
  const status = query<HTMLSpanElement>("[data-status]");
  const statusDot = query<HTMLSpanElement>("[data-status-dot]");
  const errorDetails = query<HTMLDetailsElement>("[data-error-details]");
  const errorDetail = query<HTMLPreElement>("[data-error-detail]");
  const tokenized = query<HTMLDivElement>("[data-tokenized]");
  const attention = query<HTMLDivElement>("[data-attention]");
  const probabilities = query<HTMLDivElement>("[data-probabilities]");
  const play = query<HTMLButtonElement>('[data-action="play"]');
  const pause = query<HTMLButtonElement>('[data-action="pause"]');
  const stepButton = query<HTMLButtonElement>('[data-action="step"]');
  const clear = query<HTMLButtonElement>('[data-action="clear"]');
  const temperature = query<HTMLInputElement>("[data-temperature]");
  const speed = () => Number(query<HTMLInputElement>('[name="speed"]:checked').value);
  const sampling = () => query<HTMLInputElement>('[name="sampling"]:checked').value;
  const latest = query<HTMLElement>("[data-latest-token]");
  const tokenCount = query<HTMLElement>("[data-token-count]");
  const worker = new Worker(new URL("../workers/model.worker.ts", import.meta.url), { type: "module" });
  const waiting = new Map<number, { resolve: (message: Message) => void; reject: (error: Error) => void }>();
  let messageId = 0;
  let loaded = false;
  let running = false;
  let busy = false;
  let generationStarted = false;
  let generatedCount = 0;
  let displayTokens: string[] = [];
  let displayIds: number[] = [];
  let wakePacing: (() => void) | null = null;
  const network = new NetworkView(query<HTMLElement>("[data-network]"), lang,
    async (step, layer, head) => await request("inspect", { step, layer, head }) as unknown as TraceInspection,
    () => {
      if (running) {
        running = false;
        wakePacing?.();
        status.textContent = copy.paused;
        setBusy(busy);
      }
    },
    (record) => {
      const stage = record ? (record.layer !== null ? `block-${record.layer}` : record.id === "embed" ? "embedding" : record.id === "sample" ? "sample" : "logits") : null;
      root.querySelectorAll<HTMLElement>("[data-stage]").forEach((item) => item.classList.toggle("is-active", item.dataset.stage === stage));
    },
    (playing) => { pause.disabled = !running && !playing; },
  );

  worker.addEventListener("message", (event: MessageEvent<Message>) => {
    const message = event.data;
    const request = waiting.get(message.requestId);
    if (!request) return;
    waiting.delete(message.requestId);
    if (message.type === "error") {
      errorDetails.hidden = false;
      errorDetails.open = false;
      errorDetail.textContent = String(message.details ?? message.error);
      request.reject(new Error(String(message.error)));
    }
    else request.resolve(message);
  });
  worker.addEventListener("error", (event) => {
    status.textContent = `${copy.failed}: ${event.message}`;
    statusDot.classList.add("is-error");
    errorDetails.hidden = false;
    errorDetail.textContent = event.message;
    running = false;
    busy = false;
    waiting.forEach(({ reject }) => reject(new Error(event.message)));
    waiting.clear();
    setBusy(false);
  });

  function request(action: string, payload: Record<string, unknown> = {}): Promise<Message> {
    const id = ++messageId;
    return new Promise((resolve, reject) => {
      waiting.set(id, { resolve, reject });
      worker.postMessage({ requestId: id, action, ...payload });
    });
  }

  function setBusy(isBusy: boolean) {
    play.disabled = isBusy;
    pause.disabled = !running && !network.isPlaying;
    stepButton.disabled = isBusy;
    clear.disabled = isBusy;
    prompt.disabled = isBusy;
  }

  function showTokens(tokens: string[], ids: number[]) {
    tokenized.replaceChildren(...tokens.map((token, index) => {
      const chip = document.createElement("code");
      chip.textContent = token.replaceAll("\n", "↵");
      const id = document.createElement("small");
      id.textContent = String(ids[index]);
      chip.append(id);
      return chip;
    }));
  }

  function showStep(message: Message) {
    const text = String(message.text ?? "");
    const tokenLabel = String(message.tokenLabel ?? text);
    const weights = (message.attention as number[] | undefined) ?? [];
    const contextTokens = displayTokens.slice(-weights.length);
    output.append(document.createTextNode(text));
    displayTokens.push(tokenLabel);
    displayIds.push(Number(message.token));
    showTokens(displayTokens.slice(-64), displayIds.slice(-64));
    latest.textContent = tokenLabel.replaceAll("\n", "↵");
    generatedCount += 1;
    const countLabel = lang === "pt"
      ? (generatedCount === 1 ? "token gerado" : "tokens gerados")
      : (generatedCount === 1 ? "token generated" : "tokens generated");
    tokenCount.textContent = `${generatedCount} ${countLabel}`;
    attention.replaceChildren(...weights.map((weight, index) => {
      const row = document.createElement("div");
      row.className = "attention-row";
      const label = document.createElement("code");
      label.textContent = (contextTokens[index] ?? "·").replaceAll("\n", "↵");
      const track = document.createElement("span");
      track.className = "attention-track";
      const fill = document.createElement("span");
      fill.style.width = `${Math.max(0, Math.min(100, weight * 100))}%`;
      track.append(fill);
      const value = document.createElement("small");
      value.textContent = weight.toFixed(2).replace(".", lang === "pt" ? "," : ".");
      row.append(label, track, value);
      return row;
    }));

    const top = (message.top as Array<{ token: string; text: string; prob: number }> | undefined) ?? [];
    probabilities.replaceChildren(...top.map((candidate) => {
      const row = document.createElement("div");
      row.className = "probability-row";
      const label = document.createElement("code");
      label.textContent = candidate.token.replaceAll("\n", "↵");
      const track = document.createElement("span");
      track.className = "probability-track";
      const fill = document.createElement("span");
      fill.style.width = `${Math.max(0, Math.min(100, candidate.prob * 100))}%`;
      track.append(fill);
      const value = document.createElement("small");
      value.textContent = candidate.prob.toFixed(2).replace(".", lang === "pt" ? "," : ".");
      row.append(label, track, value);
      return row;
    }));
  }

  async function ensureModel() {
    if (loaded) return;
    status.textContent = copy.loading;
    statusDot.classList.remove("is-error");
    errorDetails.hidden = true;
    statusDot.classList.add("is-loading");
    const wheelPath = import.meta.env.DEV
      ? new URL(`../generated/${wheelName}`, import.meta.url).href
      : `${import.meta.env.BASE_URL}py/${wheelName}`;
    const wheelURL = new URL(wheelPath, location.origin).href;
    try {
      const ready = await request("load", { lang, wheelURL });
      network.configure(ready.config as ModelConfig, Number(ready.parameters));
      loaded = true;
      status.textContent = `${copy.ready} · ${ready.layers} ${lang === "pt" ? "camadas" : "layers"}`;
    } finally {
      statusDot.classList.remove("is-loading");
    }
  }

  async function start() {
    if (busy || running) return;
    if (!prompt.value.trim()) {
      status.textContent = copy.empty;
      prompt.focus();
      return;
    }
    running = false;
    busy = true;
    setBusy(true);
    try {
      await ensureModel();
      if (!generationStarted) {
        network.reset();
        const started = await request("start", { text: prompt.value, temperature: Number(temperature.value), sampling: sampling() });
        displayTokens = started.tokens as string[];
        displayIds = started.ids as number[];
        generatedCount = 0;
        output.textContent = prompt.value;
        latest.textContent = "—";
        tokenCount.textContent = "0";
        showTokens(displayTokens, displayIds);
        attention.replaceChildren();
        probabilities.replaceChildren();
        generationStarted = true;
      }
      running = true;
      setBusy(true);
      status.textContent = copy.running;
      await pump();
    } catch (error) {
      running = false;
      status.textContent = `${copy.failed}: ${error instanceof Error ? error.message : String(error)}`;
      statusDot.classList.add("is-error");
    } finally {
      busy = false;
      setBusy(running);
    }
  }

  async function nextToken(continuous = false) {
    network.setComputing(true);
    let result: Message;
    try { result = await request("next"); }
    finally { network.setComputing(false); }
    if (result.type === "done") {
      running = false;
      generationStarted = false;
      status.textContent = copy.done;
      setBusy(false);
      return false;
    }
    showStep(result);
    network.setTrace(result.trace as TraceSummary[], Number(result.iteration));
    if (!continuous || running) await network.animate();
    return true;
  }

  async function pump() {
    while (running) {
      if (network.hasPendingFlow) {
        await network.animate(false, true);
        if (!running) break;
      }
      const hasNext = await nextToken(true);
      if (!hasNext || !running) break;
      await new Promise<void>((resolve) => {
        const finish = () => {
          window.clearTimeout(timer);
          if (wakePacing === finish) wakePacing = null;
          resolve();
        };
        const timer = window.setTimeout(finish, speed());
        wakePacing = finish;
      });
    }
  }

  play.addEventListener("click", () => { void start(); });
  pause.addEventListener("click", () => {
    running = false;
    wakePacing?.();
    network.pauseFlow();
    status.textContent = copy.paused;
    setBusy(busy);
  });
  stepButton.addEventListener("click", async () => {
    if (busy || running) return;
    busy = true;
    setBusy(true);
    try {
      if (!loaded) await ensureModel();
      if (!generationStarted) {
        if (!prompt.value.trim()) throw new Error(copy.empty);
        network.reset();
        const started = await request("start", { text: prompt.value, temperature: Number(temperature.value), sampling: sampling() });
        displayTokens = started.tokens as string[];
        displayIds = started.ids as number[];
        generatedCount = 0;
        output.textContent = prompt.value;
        showTokens(displayTokens, displayIds);
        generationStarted = true;
      }
      status.textContent = copy.paused;
      if (network.slowMotion && network.hasPendingFlow) await network.animate(false, true);
      else await nextToken();
    } catch (error) {
      status.textContent = `${copy.failed}: ${error instanceof Error ? error.message : String(error)}`;
      statusDot.classList.add("is-error");
    } finally {
      busy = false;
      setBusy(false);
    }
  });
  root.querySelector('[data-action="clear"]')?.addEventListener("click", () => {
    running = false;
    generatedCount = 0;
    generationStarted = false;
    network.reset();
    displayTokens = [];
    displayIds = [];
    output.textContent = "";
    latest.textContent = "—";
    tokenCount.textContent = "0";
    tokenized.replaceChildren();
    attention.innerHTML = `<p class="empty-state">${lang === "pt" ? "Gere um token para ver seus pesos." : "Generate a token to see its weights."}</p>`;
    probabilities.innerHTML = `<p class="empty-state">${lang === "pt" ? "As probabilidades aparecem aqui." : "Probabilities will appear here."}</p>`;
    status.textContent = loaded ? copy.ready : (lang === "pt" ? "Modelo ainda não carregado" : "Model not loaded yet");
    setBusy(false);
  });
  temperature.addEventListener("input", () => {
    query<HTMLOutputElement>("[data-temp-label]").value = temperature.value;
  });
}
