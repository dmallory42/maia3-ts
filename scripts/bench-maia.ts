// Benchmarks Maia-3 inference: npx tsx scripts/bench-maia.ts [iterations]
import { readFileSync } from "node:fs";
import { loadMaia, maiaPolicy } from "../src/index.js";

interface Case { name: string; fen: string; history?: string[]; eloSelf: number; eloOppo: number; policy: Record<string, number> }

const bin = readFileSync(new URL("../maia3-5m.safetensors", import.meta.url));
let t = performance.now();
const model = loadMaia(bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength) as ArrayBuffer);
console.log(`load: ${(performance.now() - t).toFixed(1)} ms`);

const { cases } = JSON.parse(readFileSync(new URL("../test/fixtures/maia-reference.json", import.meta.url), "utf8")) as { cases: Case[] };
const iterations = Number(process.argv[2] ?? 1);

// Warm up the JIT.
for (const c of cases.slice(0, 10)) maiaPolicy(model, c);

let worst = 0;
t = performance.now();
for (let it = 0; it < iterations; it++) {
  for (const c of cases) {
    const { policy } = maiaPolicy(model, c);
    for (const [m, p] of Object.entries(c.policy)) worst = Math.max(worst, Math.abs(policy[m]! - p));
  }
}
const ms = (performance.now() - t) / (iterations * cases.length);
console.log(`${iterations * cases.length} positions: ${ms.toFixed(1)} ms/position, max |dp| vs reference = ${worst.toExponential(2)}`);
