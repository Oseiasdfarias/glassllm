/** Verify complete PT/EN documentation and its responsive UI in local Chrome. */
import { Buffer } from "node:buffer";
import { spawn } from "node:child_process";
import { writeFile } from "node:fs/promises";

const baseURL = (process.env.GLASSLLM_CAPTURE_URL || "http://127.0.0.1:4322/glassllm/").replace(/\/?$/, "/");
const chrome = spawn(process.env.GLASSLLM_CHROME || "google-chrome", [
  "--headless=new", "--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu",
  "--no-first-run", "--no-default-browser-check", "--remote-debugging-port=0",
  "--user-data-dir=/tmp/glassllm-visual-browser", "about:blank",
], { stdio: ["ignore", "ignore", "pipe"] });
let socket;
const report = { pages: [], viewports: [], internalLinks: 0, images: 0 };
try {
  const endpoint = await new Promise((done, reject) => {
    const timeout = setTimeout(() => reject(new Error("Chrome startup timed out")), 15000);
    let log = "";
    chrome.stderr.on("data", chunk => {
      log += chunk;
      const match = log.match(/DevTools listening on (ws:\/\/[^\s]+)/);
      if (match) { clearTimeout(timeout); done(match[1]); }
    });
    chrome.once("error", reject);
    chrome.once("exit", code => { clearTimeout(timeout); reject(new Error("Chrome exited: " + code)); });
  });
  socket = new WebSocket(endpoint);
  await new Promise((done, reject) => { socket.addEventListener("open", done); socket.addEventListener("error", reject); });
  let sequence = 0;
  const pending = new Map();
  socket.addEventListener("message", event => {
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
  const evaluate = async expression => {
    const result = await call("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
    if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
    return result.result.value;
  };
  const waitFor = async expression => {
    const deadline = Date.now() + 30000;
    while (Date.now() < deadline) {
      if (await evaluate(expression)) return;
      await new Promise(done => setTimeout(done, 100));
    }
    throw new Error("Timed out: " + expression);
  };
  await call("Page.enable", {});
  const resize = async (width = 1366, height = 1000) => {
    await call("Emulation.setDeviceMetricsOverride", { width, height, deviceScaleFactor: 2, mobile: width < 600 });
    await evaluate("new Promise(done=>requestAnimationFrame(()=>requestAnimationFrame(done)))");
  };
  const navigate = async (url, selector = ".docs-main h1") => {
    await call("Page.navigate", { url });
    await waitFor(`location.href===${JSON.stringify(url)} && !!document.querySelector(${JSON.stringify(selector)})`);
    await evaluate("document.fonts.ready");
  };
  const screenshot = async (file, selector, height = 2000) => {
    await evaluate("document.documentElement.style.overflowY='hidden';document.querySelector('.lab-header').style.visibility='hidden';document.querySelector('.skip-link').style.visibility='hidden'");
    await evaluate("new Promise(done=>requestAnimationFrame(()=>requestAnimationFrame(done)))");
    const clip = await evaluate(`(() => {const r=document.querySelector(${JSON.stringify(selector)}).getBoundingClientRect();return {x:r.x,y:r.y+scrollY,width:r.width,height:Math.min(r.height,${height}),scale:1}})()`);
    const { data } = await call("Page.captureScreenshot", { format: "png", captureBeyondViewport: true, clip });
    await writeFile(file, Buffer.from(data, "base64"));
    await evaluate("document.documentElement.style.overflowY='';document.querySelector('.lab-header').style.visibility='';document.querySelector('.skip-link').style.visibility=''");
  };
  const assertWidth = async (url, width) => {
    const actual = await evaluate("document.documentElement.scrollWidth");
    if (actual > width) {
      const overflow = await evaluate(`([...document.querySelectorAll('.docs-main *')].filter(el=>el.getBoundingClientRect().right>${width}).slice(0,5).map(el=>({tag:el.tagName,class:el.className,text:el.textContent.slice(0,80)})))`);
      throw new Error("Horizontal overflow: " + url + " " + width + " " + JSON.stringify(overflow));
    }
    report.viewports.push({ url, width, actual });
  };
  await resize();
  for (const lang of ["pt", "en"]) {
    const home = baseURL + lang + "/docs/";
    await navigate(home);
    const colors = await evaluate("({background:getComputedStyle(document.body).backgroundColor,accent:getComputedStyle(document.documentElement).getPropertyValue('--accent').trim(),gradient:getComputedStyle(document.body).backgroundImage})");
    if (colors.background !== "rgb(10, 10, 10)" || colors.accent !== "#4fb3c2" || colors.gradient !== "none") throw new Error("Minimal theme missing: " + JSON.stringify(colors));
    if (!await evaluate(`document.fonts.check('17px "Source Sans 3 Variable"') && document.fonts.check('14px "Source Code Pro Variable"')`)) throw new Error("Local fonts missing");
    await evaluate("document.querySelector('#tab-training').click()");
    if (!await evaluate("!document.querySelector('#flow-training').hidden && document.querySelector('#flow-inference').hidden && document.querySelector('#tab-training').getAttribute('aria-selected')==='true'")) throw new Error("Flow tabs failed");
    await evaluate("document.querySelector('#tab-training').dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowLeft',bubbles:true}))");
    if (!await evaluate("document.activeElement.id==='tab-inference' && !document.querySelector('#flow-inference').hidden")) throw new Error("Flow keyboard navigation failed");
    const urls = await evaluate("[...document.querySelectorAll('.docs-group-card li a')].map(link=>link.href)");
    if (urls.length !== 35) throw new Error("Expected 35 pages: " + urls.length);
    const pages = await evaluate(`Promise.all(${JSON.stringify(urls)}.map(async url=>{
      const response=await fetch(url);const doc=new DOMParser().parseFromString(await response.text(),'text/html');const body=doc.querySelector('[data-doc-content]');
      return {url,status:response.status,length:body?.textContent.trim().length??0,headings:body?.querySelectorAll('h2').length??0,placeholder:/Seção prevista para|Conteúdo proposto|Funções e símbolos a documentar|escrita completa começa|Section planned for|Proposed content|Functions and symbols to document|Full writing begins/i.test(doc.querySelector('.docs-main')?.textContent??''),anchors:[...doc.querySelectorAll('.docs-toc a')].filter(link=>!doc.getElementById(decodeURIComponent(link.getAttribute('href').slice(1)))).map(link=>link.textContent),links:[...doc.querySelectorAll('.docs-main a')].map(link=>new URL(link.getAttribute('href'),url).href),images:[...doc.querySelectorAll('.docs-figure img')].map(img=>new URL(img.getAttribute('src'),url).href)};
    }))`);
    for (const page of pages) {
      if (page.status !== 200 || page.length < 1000 || page.headings < 3 || page.placeholder || page.anchors.length) throw new Error("Incomplete documentation: " + JSON.stringify(page));
    }
    report.pages.push(...pages.map(({ url, length, headings })=>({ url, length, headings })));
    const links=[...new Set(pages.flatMap(page=>page.links).filter(url=>url.startsWith(baseURL)).map(url=>url.split('#')[0]))];
    const linkResults=await evaluate(`Promise.all(${JSON.stringify(links)}.map(async url=>({url,status:(await fetch(url)).status})))`);
    if (linkResults.some(item=>item.status!==200)) throw new Error("Broken internal link: "+JSON.stringify(linkResults.filter(item=>item.status!==200)));
    report.internalLinks+=links.length;
    const images=[...new Set(pages.flatMap(page=>page.images))];
    if(images.length!==17) throw new Error("Expected 17 screenshot assets: "+images.length);
    const imageResults=await evaluate(`Promise.all(${JSON.stringify(images)}.map(async url=>(await fetch(url)).status))`);
    if(imageResults.some(status=>status!==200)) throw new Error("Missing screenshot");
    report.images+=images.length;
    await screenshot("/tmp/glassllm-docs-home-"+lang+".png", ".docs-main", 1800);
    for(const width of [390,320]) {
      await resize(width,844);
      await assertWidth(home,width);
      if(width===390) await screenshot("/tmp/glassllm-docs-home-mobile-"+lang+".png", ".docs-main",2200);
    }
    for(const [index,url] of urls.entries()) {
      await navigate(url,"[data-doc-content]");
      await assertWidth(url,320);
      if(index%10===0) console.log("Checked responsive documentation:",lang,index+1,"/ 35");
    }
    await resize();
  }
  await navigate(baseURL+"en/docs/biblioteca/glasslm/");
  await screenshot("/tmp/glassllm-docs-article-en.png", ".docs-main",2000);
  const bib=await evaluate(`fetch(${JSON.stringify(baseURL+"docs/glassllm-references.bib")}).then(async r=>({status:r.status,text:await r.text()}))`);
  if(bib.status!==200 || (bib.text.match(/@misc\{/g)||[]).length!==14) throw new Error("BibTeX bibliography failed");
  await writeFile("/tmp/glassllm-docs-qa.json",JSON.stringify(report,null,2)+"\n");
  console.log("PASS: 70 complete pages, TOC anchors, internal links, 34 images, 14 BibTeX entries, responsive layouts, flow tabs, local fonts and minimal theme.");
} finally {
  socket?.close();
  chrome.kill("SIGTERM");
}
