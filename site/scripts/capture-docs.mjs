/** Capture actual PT/EN lab states and verify the documentation in Chrome. */
import { Buffer } from "node:buffer";
import { spawn } from "node:child_process";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const siteDir = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const baseURL = (process.env.GLASSLLM_CAPTURE_URL || "http://127.0.0.1:4322/glassllm/").replace(/\/?$/, "/");
const outputDir = resolve(siteDir, "public/docs/screenshots");
const chrome = spawn(process.env.GLASSLLM_CHROME || "google-chrome", [
  "--headless=new", "--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu",
  "--no-first-run", "--no-default-browser-check", "--remote-debugging-port=0",
  "--user-data-dir=/tmp/glassllm-visual-browser", "about:blank",
], { stdio: ["ignore", "ignore", "pipe"] });
let socket;
const images = [];
const metadata = [];

try {
  const endpoint = await new Promise((resolveEndpoint, reject) => {
    const timeout = setTimeout(() => reject(new Error("Chrome startup timed out")), 15000);
    let log = "";
    chrome.stderr.on("data", (chunk) => {
      log += chunk;
      const match = log.match(/DevTools listening on (ws:\/\/[^\s]+)/);
      if (match) { clearTimeout(timeout); resolveEndpoint(match[1]); }
    });
    chrome.once("error", reject);
    chrome.once("exit", (code) => { clearTimeout(timeout); reject(new Error("Chrome exited: " + code)); });
  });
  socket = new WebSocket(endpoint);
  await new Promise((done, reject) => { socket.addEventListener("open", done); socket.addEventListener("error", reject); });
  let sequence = 0;
  const pending = new Map();
  socket.addEventListener("message", (event) => {
    const message = JSON.parse(event.data);
    const item = pending.get(message.id);
    if (item) { pending.delete(message.id); message.error ? item.reject(new Error(JSON.stringify(message.error))) : item.resolve(message.result); }
  });
  const send = (method, params = {}, sessionId) => new Promise((done, reject) => {
    const id = ++sequence;
    pending.set(id, { resolve: done, reject });
    socket.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }));
  });
  const { targetId } = await send("Target.createTarget", { url: "about:blank" });
  const { sessionId } = await send("Target.attachToTarget", { targetId, flatten: true });
  const call = (method, params) => send(method, params, sessionId);
  const evaluate = async (expression) => {
    const result = await call("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
    if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
    return result.result.value;
  };
  const waitFor = async (expression, seconds = 30) => {
    const deadline = Date.now() + seconds * 1000;
    while (Date.now() < deadline) {
      const result = await evaluate(expression);
      if (result) return result;
      await new Promise((done) => setTimeout(done, 100));
    }
    const state = await evaluate('({status:document.querySelector("[data-status]")?.textContent,progress:document.querySelector("[data-network-progress]")?.textContent})');
    throw new Error("Timed out: " + expression + " " + JSON.stringify(state));
  };
  await call("Page.enable", {});
  await call("Runtime.enable", {});
  // Exercise a real missing-wheel error without changing application DOM or source.
  await call("Page.addScriptToEvaluateOnNewDocument", { source: `
    const CaptureWorker = window.Worker;
    window.Worker = class extends CaptureWorker {
      postMessage(message, ...rest) {
        if (message.action === 'load' && window.__captureMissingWheel) {
          window.__captureMissingWheel = false;
          message = {...message, wheelURL:message.wheelURL.replace(/glassllm-[^/]+\\.whl$/, 'glassllm-99.0.0-py3-none-any.whl')};
        }
        return super.postMessage(message, ...rest);
      }
    };
  ` });
  let viewport;
  const resize = async (width = 1366, height = 1000, mobile = false) => {
    viewport = { width, height, deviceScaleFactor: 2, mobile };
    await call("Emulation.setDeviceMetricsOverride", viewport);
    await evaluate("new Promise(done=>requestAnimationFrame(()=>requestAnimationFrame(done)))");
  };
  const navigate = async (path, selector) => {
    await call("Page.navigate", { url: baseURL + path });
    await waitFor(`!!document.querySelector(${JSON.stringify(selector)})`);
    await evaluate("document.fonts.ready");
  };
  const capture = async (lang, file, selector) => {
    await evaluate("document.fonts.ready");
    // Chrome hides the page scrollbar during capture; measure after hiding it to keep the crop aligned.
    await evaluate(`(() => {const el=document.querySelector(${JSON.stringify(selector)});const header=document.querySelector('.lab-header');const skip=document.querySelector('.skip-link');window.__captureOverflowY=document.documentElement.style.overflowY;document.documentElement.style.overflowY='hidden';window.__captureHeaderVisibility=header.style.visibility;window.__captureSkipVisibility=skip.style.visibility;header.style.visibility='hidden';skip.style.visibility='hidden';el.scrollIntoView({behavior:'instant',block:'center'});if(${JSON.stringify(selector)}==='.network-inspector'){const workspace=document.querySelector('.network-workspace');window.__captureInspectorStyle=el.getAttribute('style');window.__captureWorkspaceStyle=workspace.getAttribute('style');workspace.style.gridTemplateColumns='minmax(0,1fr) 600px';el.style.maxHeight='none';el.style.overflowY='visible';}})()`);
    await evaluate("new Promise(done=>requestAnimationFrame(()=>requestAnimationFrame(done)))");
    const context = await evaluate(`(() => {
      const el=document.querySelector(${JSON.stringify(selector)}); const r=el.getBoundingClientRect();
      const head=document.querySelector('[data-network-head]');
      return {clip:{x:r.x,y:r.y+scrollY,width:r.width,height:r.height,scale:1},
        prompt:document.querySelector('[data-prompt]')?.value,
        sampling:document.querySelector('[name=sampling]:checked')?.value,
        temperature:Number(document.querySelector('[data-temperature]')?.value),
        generated:document.querySelector('[data-token-count]')?.textContent,
        operation:document.querySelector('.network-node.is-selected')?.dataset.networkKey,
        head:document.querySelector('[data-network-head-wrap]')?.hidden?null:Number(head?.value),
        status:document.querySelector('[data-status]')?.textContent,
        font:getComputedStyle(el).fontFamily};
    })()`);
    const fullScreen = await evaluate("!!document.fullscreenElement");
    const { data } = await call("Page.captureScreenshot", fullScreen ? { format: "png", captureBeyondViewport: false } : { format: "png", captureBeyondViewport: true, clip: context.clip });
    await evaluate(`(() => {document.documentElement.style.overflowY=window.__captureOverflowY;document.querySelector('.lab-header').style.visibility=window.__captureHeaderVisibility;document.querySelector('.skip-link').style.visibility=window.__captureSkipVisibility;const el=document.querySelector(${JSON.stringify(selector)});if(${JSON.stringify(selector)}==='.network-inspector'){const workspace=document.querySelector('.network-workspace');window.__captureInspectorStyle===null?el.removeAttribute('style'):el.setAttribute('style',window.__captureInspectorStyle);window.__captureWorkspaceStyle===null?workspace.removeAttribute('style'):workspace.setAttribute('style',window.__captureWorkspaceStyle);}})()`);
    const buffer = Buffer.from(data, "base64");
    images.push({ lang, file, buffer });
    metadata.push({ lang, file, selector, viewport: { ...viewport }, width: buffer.readUInt32BE(16), height: buffer.readUInt32BE(20), expandedScrollableContent: selector === ".network-inspector", ...context });
    console.log("Captured", lang + "/" + file);
  };
  const ready = "document.querySelectorAll('.probability-row').length===6 && !document.querySelector('[data-action=step]').disabled";
  const click = (selector) => evaluate(`document.querySelector(${JSON.stringify(selector)}).click()`);
  const inspect = async (key) => {
    await evaluate(`document.querySelector('[data-network-key="${key}"]').dispatchEvent(new MouseEvent('click',{bubbles:true}))`);
    await waitFor("document.querySelector('[data-network-value-status]').textContent.match(/Valores reais|Actual values/) && document.querySelector('[data-network-matrix] td')");
  };

  for (const lang of ["pt", "en"]) {
    console.log("Capturing actual model states:", lang);
    await resize();
    await call("Emulation.setEmulatedMedia", { features: [{ name: "prefers-reduced-motion", value: "no-preference" }] });
    await navigate(lang + "/", ".lab[data-initialized]");
    await click('[name=sampling][value=greedy]');
    await click('[data-action=step]');
    await waitFor("document.querySelector('[data-status-dot]').classList.contains('is-loading')");
    await capture(lang, "state-loading.png", ".model-state");
    await waitFor(ready, 120);
    await capture(lang, "state-paused.png", ".model-state");
    await click('[data-action=clear]');
    await capture(lang, "state-ready.png", ".model-state");
    await click('[data-action=step]');
    await waitFor(ready);
    if (await evaluate("document.querySelectorAll('.network-node.has-values').length") !== 77) throw new Error("Missing real trace records");
    await capture(lang, "lab.png", ".lab-workspace");
    await capture(lang, "editor.png", ".lab-text");
    await capture(lang, "settings.png", ".lab-options");
    await capture(lang, "attention.png", ".lab-inspector");
    await click('[data-network-action=fit]');
    await capture(lang, "architecture.png", "#architecture");
    await click('[data-network-focus="0:norm.attn"]');
    await waitFor("document.querySelector('[data-network-value-status]').textContent.match(/Valores reais|Actual values/)");
    await capture(lang, "architecture-block.png", "#architecture");
    await click('[data-network-slow]');
    await evaluate("document.querySelector('[data-network-delay]').value='1000'");
    await click('[data-network-action=replay]');
    await waitFor("document.querySelector('[data-network-source-code]').textContent.includes('@step attn.q') && document.querySelector('[data-network-value-status]').textContent.match(/Valores reais|Actual values/)");
    await click('[data-network-action=stop]');
    await evaluate("document.querySelector('[data-network-delay]').value='3000'");
    await capture(lang, "slow-motion.png", "#architecture");
    await inspect("0:attn.softmax");
    await capture(lang, "inspector-attention.png", ".network-inspector");
    await inspect("0:attn.mask");
    if (!await evaluate("[...document.querySelectorAll('[data-network-matrix] td')].some(cell=>cell.textContent==='−∞')")) throw new Error("Causal mask must contain actual −∞ values");
    await capture(lang, "inspector-mask.png", ".network-inspector");
    for (const width of [390, 320]) {
      await resize(width, 844, true);
      await click('[data-network-focus="0:norm.attn"]');
      await evaluate("document.querySelector('#architecture').scrollIntoView({behavior:'instant'})");
      if (await evaluate("document.documentElement.scrollWidth") > width) throw new Error("Lab overflow at " + width);
      await capture(lang, "mobile-" + width + ".png", "#architecture");
    }
    await resize();
    await evaluate("document.querySelector('[data-network-action=fullscreen]').scrollIntoView({behavior:'instant',block:'center'})");
    const point = await evaluate("(() => {const r=document.querySelector('[data-network-action=fullscreen]').getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2}})()");
    await call("Input.dispatchMouseEvent", { type: "mousePressed", button: "left", clickCount: 1, ...point });
    await call("Input.dispatchMouseEvent", { type: "mouseReleased", button: "left", clickCount: 1, ...point });
    await waitFor("document.fullscreenElement?.id==='architecture'");
    await capture(lang, "fullscreen.png", "#architecture");
    if (!await evaluate("!!document.querySelector('[data-network-action=fullscreen] svg')")) throw new Error("Fullscreen toggle lost its SVG icon");
    await evaluate("document.exitFullscreen()");
    await waitFor("!document.fullscreenElement");
    await click('[data-network-slow]');
    await click('[data-action=clear]');
    // Reduced motion removes visualization delays; generation still runs all 128 real steps.
    await call("Emulation.setEmulatedMedia", { features: [{ name: "prefers-reduced-motion", value: "reduce" }] });
    await click('[name=speed][value="90"]');
    await click('[data-action=play]');
    await waitFor("document.querySelector('[data-status]').textContent.match(/Limite desta rodada|This run reached/) && !document.querySelector('[data-action=play]').disabled", 120);
    if (await evaluate("parseInt(document.querySelector('[data-token-count]').textContent)") !== 128) throw new Error("Token limit capture was not real");
    await capture(lang, "state-limit.png", ".model-state");
    await call("Emulation.setEmulatedMedia", { features: [{ name: "prefers-reduced-motion", value: "no-preference" }] });
    await navigate(lang + "/", ".lab[data-initialized]");
    await evaluate("window.__captureMissingWheel=true");
    await click('[data-action=step]');
    await waitFor("document.querySelector('[data-status-dot]').classList.contains('is-error') && !document.querySelector('[data-action=step]').disabled", 120);
    if (!await evaluate("!document.querySelector('[data-error-details]').hidden && !document.querySelector('[data-error-details]').open && !document.querySelector('[data-status]').textContent.includes('Traceback') && document.querySelector('[data-error-detail]').textContent.includes('Traceback')")) throw new Error("Concise error with full technical details is missing");
    await capture(lang, "state-error.png", ".model-state");
    await click('[data-action=step]');
    await waitFor(ready, 120);
    if (!await evaluate("document.querySelector('[data-error-details]').hidden && !document.querySelector('[data-status-dot]').classList.contains('is-error')")) throw new Error("Error recovery did not clear diagnostics");
  }

  // Write after capturing: public asset changes can reload the Vite page.
  for (const { lang, file, buffer } of images) {
    await mkdir(resolve(outputDir, lang), { recursive: true });
    await writeFile(resolve(outputDir, lang, file), buffer);
    if (lang === "pt" && ["editor.png", "settings.png", "slow-motion.png"].includes(file)) await writeFile(resolve(outputDir, file), buffer);
  }
  await writeFile(resolve(outputDir, "manifest.json"), JSON.stringify({ capturedAt: new Date().toISOString(), baseURL, captures: metadata }, null, 2) + "\n");
  await new Promise((done) => setTimeout(done, 1200));
  for (const lang of ["pt", "en"]) {
    await resize();
    await navigate(lang + "/docs/", ".docs-main h1");
    const check = await evaluate("({size:getComputedStyle(document.querySelector('.docs-section-intro')).fontSize,font:getComputedStyle(document.querySelector('.docs-main')).fontFamily,links:[...document.querySelectorAll('.docs-group-card li a')].map(a=>a.href),emoji:document.querySelector('.docs-main').textContent.match(/[▶Ⅱ↗▧]/u),icons:document.querySelectorAll('.docs-main svg.icon').length})");
    if (Number.parseFloat(check.size) < 16 || !check.font.includes("Source Sans 3") || check.emoji || check.icons < 35) throw new Error("Documentation typography/icons failed: " + JSON.stringify(check));
    if (!await evaluate(`document.fonts.check('17px "Source Sans 3 Variable"') && document.fonts.check('14px "Source Code Pro Variable"')`)) throw new Error("Local documentation fonts did not load");
    const pages = await evaluate(`Promise.all(${JSON.stringify(check.links)}.map(async url=>{const response=await fetch(url); const doc=new DOMParser().parseFromString(await response.text(),'text/html'); const images=[...doc.querySelectorAll('.docs-figure img')].map(img=>img.getAttribute('src')); return {status:response.status,images};}))`);
    if (pages.length !== 35 || pages.some((page) => page.status !== 200)) throw new Error("Documentation routes failed");
    const urls = [...new Set(pages.flatMap((page) => page.images))];
    if (urls.length !== 17) throw new Error("Incomplete screenshot coverage: " + urls.length);
    const statuses = await evaluate(`Promise.all(${JSON.stringify(urls)}.map(async url=>(await fetch(url)).status))`);
    if (statuses.some((status) => status !== 200)) throw new Error("Screenshot missing");
    for (const width of [390, 320]) {
      await resize(width, 844, true);
      if (await evaluate("document.documentElement.scrollWidth") > width) throw new Error("Documentation overflow at " + width);
    }
  }
  await resize();
  await navigate("pt/docs/interface/slow-motion/", ".docs-main h1");
  await evaluate("Promise.all([...document.querySelectorAll('.docs-figure img')].map(img=>{img.loading='eager';return img.decode()}))");
  await evaluate("document.documentElement.style.overflowY='hidden';document.querySelector('.lab-header').style.visibility='hidden';document.querySelector('.skip-link').style.visibility='hidden'");
  await evaluate("new Promise(done=>requestAnimationFrame(()=>requestAnimationFrame(done)))");
  const rect = await evaluate("(() => {const r=document.querySelector('.docs-main').getBoundingClientRect();return {x:r.x,y:r.y+scrollY,width:r.width,height:Math.min(r.height,2200),scale:1}})()");
  const { data } = await call("Page.captureScreenshot", { format: "png", captureBeyondViewport: true, clip: rect });
  await writeFile("/tmp/glassllm-documentation-readable.png", Buffer.from(data, "base64"));
  console.log("PASS: 34 actual captures, 70 documentation routes, local fonts, SVG icons, mobile widths and fullscreen");
} finally {
  socket?.close();
  chrome.kill("SIGTERM");
}
