import { rm } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const siteDir = resolve(dirname(fileURLToPath(import.meta.url)), "..");
await rm(resolve(siteDir, "dist"), { recursive: true, force: true });
