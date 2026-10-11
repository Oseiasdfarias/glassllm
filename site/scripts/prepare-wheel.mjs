import { cp, mkdir, readdir, rm, stat, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { spawnSync } from "node:child_process";

const siteDir = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const repoDir = resolve(siteDir, "..");
const distDir = resolve(repoDir, "dist");
const publicDir = resolve(siteDir, "public/py");
const generatedDir = resolve(siteDir, "src/generated");

const build = spawnSync("uv", ["build", "--wheel", "--out-dir", distDir], {
  cwd: repoDir,
  stdio: "inherit",
});
if (build.error) throw new Error(`Could not run uv: ${build.error.message}`);
if (build.status !== 0) process.exit(build.status ?? 1);

const wheels = (await readdir(distDir)).filter((name) => name.startsWith("glassllm-") && name.endsWith(".whl"));
if (wheels.length === 0) throw new Error(`No glassllm wheel found in ${distDir}`);
const ordered = await Promise.all(wheels.map(async (name) => ({ name, mtime: (await stat(resolve(distDir, name))).mtimeMs })));
ordered.sort((a, b) => b.mtime - a.mtime);
const wheel = ordered[0].name;
await rm(publicDir, { recursive: true, force: true });
await mkdir(publicDir, { recursive: true });
await cp(resolve(distDir, wheel), resolve(publicDir, wheel));
await mkdir(generatedDir, { recursive: true });
await rm(generatedDir, { recursive: true, force: true });
await mkdir(generatedDir, { recursive: true });
await cp(resolve(distDir, wheel), resolve(generatedDir, wheel));
await writeFile(
  resolve(generatedDir, "wheel-asset.ts"),
  `export const wheelName = ${JSON.stringify(wheel)};\n`,
);
console.log(`Prepared ${wheel} for Vite development and the static site.`);
