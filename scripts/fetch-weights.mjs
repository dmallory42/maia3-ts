// Downloads the weights this version reads into maia3-5m.safetensors and checks their SHA-256.
import { createHash } from "node:crypto";
import { existsSync, readFileSync, writeFileSync } from "node:fs";

const src = readFileSync(new URL("../src/index.ts", import.meta.url), "utf8");
const url = /url: "([^"]+)"/.exec(src)?.[1];
const sha256 = /sha256: "([0-9a-f]{64})"/.exec(src)?.[1];
const out = new URL("../maia3-5m.safetensors", import.meta.url);
const hash = (buf) => createHash("sha256").update(buf).digest("hex");

if (existsSync(out) && hash(readFileSync(out)) === sha256) process.exit(0);
const res = await fetch(url);
if (!res.ok) throw new Error(`Fetching ${url} failed: ${res.status}`);
const buf = Buffer.from(await res.arrayBuffer());
if (hash(buf) !== sha256) throw new Error(`Weights from ${url} don't match the expected SHA-256`);
writeFileSync(out, buf);
console.log(`Saved ${out.pathname}`);
