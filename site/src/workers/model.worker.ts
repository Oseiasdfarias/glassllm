/// <reference lib="webworker" />

const workerScope = globalThis as unknown as DedicatedWorkerGlobalScope;
const PYODIDE_VERSION = "0.29.3";
let pyodide: any;
let initialized = false;
let initializing: Promise<void> | undefined;

const python = String.raw`
import json
import numpy as np
import glassllm
from dataclasses import asdict
from importlib import resources
from glassllm.steps import HEAD_STEPS, load_step_index

_model = None
_tokenizer = None
_generation = None
_last_trace = None
_context_ids = []
_last_context_ids = []
_iteration = 0
_source_index = load_step_index()
_source_lines = {}

def _number(value):
    value = float(value)
    if np.isfinite(value):
        return value
    if np.isnan(value):
        return "NaN"
    return "−∞" if value < 0 else "+∞"

def _source(step_id):
    entry = _source_index.get(step_id)
    if entry is None:
        return None
    name = entry["file"].split("/")[-1]
    if name not in _source_lines:
        _source_lines[name] = (resources.files("glassllm") / name).read_text(encoding="utf-8").splitlines()
    return {**entry, "code":_source_lines[name][entry["line"]-1].strip()}

def _summary(record):
    array = np.asarray(record.value)
    flat = array.reshape(-1)
    finite = flat[np.isfinite(flat)]
    return {"id":record.id, "layer":record.layer, "head":record.head,
            "order":record.order, "shape":list(array.shape), "equation":record.equation,
            "preview":[_number(x) for x in flat[:16]],
            "min":float(finite.min()) if finite.size else None,
            "max":float(finite.max()) if finite.size else None,
            "mean":float(finite.mean()) if finite.size else None,
            "nonFinite":int(flat.size-finite.size), "source":_source(record.id)}

def _inspect(req):
    if _last_trace is None:
        raise ValueError("generate a token before inspecting calculated values")
    step_id = req["step"]
    head = req.get("head", 0) if step_id in HEAD_STEPS else None
    record = _last_trace.get(step_id, layer=req.get("layer"), head=head)
    array = np.asarray(record.value)
    matrix = array.reshape(-1, array.shape[-1]) if array.ndim else array.reshape(1, 1)
    row_offset = max(0, matrix.shape[0]-8)
    column_offset = max(0, matrix.shape[1]-8)
    window = matrix[row_offset:, column_offset:]
    rows = [str(i) for i in range(row_offset, matrix.shape[0])]
    columns = [str(i) for i in range(column_offset, matrix.shape[1])]
    if array.ndim >= 2 and len(_last_context_ids) >= matrix.shape[0]:
        row_tokens = _last_context_ids[-matrix.shape[0]:]
        rows = _tokenizer.tokens(row_tokens[row_offset:])
    if step_id in {"attn.scores", "attn.mask", "attn.softmax"}:
        columns = _tokenizer.tokens(_last_context_ids[column_offset:])
    elif step_id in {"probs", "logits"}:
        columns = _tokenizer.tokens(list(range(column_offset, matrix.shape[1])))
    return {"type":"inspection", "iteration":_iteration, **_summary(record),
            "grid":[[_number(x) for x in row] for row in window],
            "rowLabels":rows, "columnLabels":columns,
            "rowOffset":row_offset, "columnOffset":column_offset,
            "fullRows":int(matrix.shape[0]), "fullColumns":int(matrix.shape[1])}

def _handle(raw):
    global _model, _tokenizer, _generation, _last_trace, _context_ids, _last_context_ids, _iteration
    req = json.loads(raw)
    action = req["action"]
    if action == "load":
        _model, _tokenizer = glassllm.load_pretrained(req["lang"])
        return {"type":"ready", "vocab":_tokenizer.vocab_size, "layers":_model.cfg.n_layers,
                "max_len":_model.cfg.max_len, "config":asdict(_model.cfg),
                "parameters":sum(p.data.size for p in _model.parameters())}
    if action == "tokenize":
        ids = _tokenizer.encode(req["text"])
        return {"type":"tokens", "ids":ids, "tokens":_tokenizer.tokens(ids)}
    if action == "start":
        ids = _tokenizer.encode(req["text"])
        if not ids:
            raise ValueError("prompt needs at least one token")
        _generation = iter(_model.generate(ids, 128, temperature=req["temperature"],
                                          greedy=req.get("sampling") == "greedy", trace=True))
        _context_ids = ids[-_model.cfg.max_len:]
        _last_trace = None
        _last_context_ids = []
        _iteration = 0
        return {"type":"started", "ids":ids, "tokens":_tokenizer.tokens(ids)}
    if action == "next":
        step = next(_generation)
        _last_trace = step.trace
        _last_context_ids = list(_context_ids)
        _context_ids = (_context_ids + [step.token])[-_model.cfg.max_len:]
        _iteration += 1
        record = step.trace.get("attn.softmax", layer=_model.cfg.n_layers-1, head=0)
        weights = record.value[-1].astype(float).tolist()
        order = np.argsort(step.probs)[-6:][::-1]
        top = [{"token":_tokenizer.tokens([int(i)])[0], "text":_tokenizer.decode([int(i)]),
                "prob":float(step.probs[int(i)])} for i in order]
        return {"type":"step", "token":step.token, "tokenLabel":_tokenizer.tokens([step.token])[0], "text":_tokenizer.decode([step.token]),
                "attention":weights, "top":top, "iteration":_iteration,
                "trace":[_summary(record) for record in step.trace.records]}
    if action == "inspect":
        return _inspect(req)
    if action == "stop":
        _generation = None
        return {"type":"stopped"}
    raise ValueError("unknown action: " + action)

def handle_json(raw):
    try:
        return json.dumps({"ok":True, "data":_handle(raw)}, ensure_ascii=False, allow_nan=False)
    except StopIteration:
        return json.dumps({"ok":True, "data":{"type":"done"}}, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"ok":False, "error":str(exc)}, ensure_ascii=False)
`;

async function initialize(wheelURL: unknown) {
  if (initialized) return;
  if (!initializing) {
    initializing = (async () => {
      if (typeof wheelURL !== "string") throw new Error("Model wheel URL is required to initialize the runtime");
      const root = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
      if (!pyodide) {
        const { loadPyodide } = await import(/* @vite-ignore */ `${root}pyodide.mjs`);
        pyodide = await loadPyodide({ indexURL: root });
      }
      await pyodide.loadPackage(["micropip", "numpy"]);
      await pyodide.runPythonAsync(`import micropip\nawait micropip.install(${JSON.stringify(wheelURL)})`);
      await pyodide.runPythonAsync(python);
      initialized = true;
    })();
  }
  try { await initializing; }
  finally { initializing = undefined; }
}

workerScope.onmessage = async (event: MessageEvent) => {
  const { requestId, action, ...payload } = event.data;
  try {
    await initialize(payload.wheelURL);
    pyodide.globals.set("_rpc_payload", JSON.stringify({ action, ...payload }));
    const result = pyodide.runPython("handle_json(_rpc_payload)");
    const parsed = JSON.parse(result.toString());
    if (typeof result !== "string") result.destroy();
    if (!parsed.ok) throw new Error(parsed.error);
    workerScope.postMessage({ ...parsed.data, requestId });
  } catch (error) {
    const details = error instanceof Error ? error.message : String(error);
    const summary = details.split("\n").map((line) => line.trim()).filter(Boolean).at(-1) ?? details;
    workerScope.postMessage({ requestId, type: "error", error: summary, details });
  }
};
