import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { loadMaia, maiaMove, maiaPolicy } from "../src/index.js";

interface Case {
  name: string;
  fen: string;
  history?: string[];
  eloSelf: number;
  eloOppo: number;
  policy: Record<string, number>;
  value: { win: number; draw: number; loss: number };
}

const root = new URL("..", import.meta.url);
const bin = readFileSync(fileURLToPath(new URL("maia3-5m.safetensors", root)));
const fixture = JSON.parse(readFileSync(fileURLToPath(new URL("test/fixtures/maia-reference.json", root)), "utf8")) as { cases: Case[] };
const model = loadMaia(bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength) as ArrayBuffer);

// Max absolute probability difference allowed against the PyTorch float32 reference (observed ~4e-6).
const TOLERANCE = 1e-5;

const argmax = (p: Record<string, number>) => Object.entries(p).reduce((a, b) => (b[1] > a[1] ? b : a))[0];

describe("maia-3 parity with the official model", () => {
  let worst = 0;
  for (const c of fixture.cases) {
    it(`${c.name} @ ${c.eloSelf}/${c.eloOppo}`, () => {
      const { policy, value } = maiaPolicy(model, { fen: c.fen, history: c.history, eloSelf: c.eloSelf, eloOppo: c.eloOppo });
      expect(Object.keys(policy).sort()).toEqual(Object.keys(c.policy).sort());
      expect(argmax(policy)).toBe(argmax(c.policy));
      let diff = 0;
      for (const [m, p] of Object.entries(c.policy)) diff = Math.max(diff, Math.abs(policy[m]! - p));
      worst = Math.max(worst, diff);
      expect(diff).toBeLessThan(TOLERANCE);
      expect(Math.abs(value!.win - c.value.win)).toBeLessThan(TOLERANCE);
      expect(Math.abs(value!.draw - c.value.draw)).toBeLessThan(TOLERANCE);
      expect(Math.abs(value!.loss - c.value.loss)).toBeLessThan(TOLERANCE);
    });
  }
  it("reports the worst difference", () => {
    console.log(`maia parity: ${fixture.cases.length} cases, max |dp| = ${worst.toExponential(2)}`);
  });
});

describe("maiaMove", () => {
  const input = { fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", eloSelf: 1500, eloOppo: 1500 };

  it("temperature 0 returns the argmax", () => {
    const { move, policy } = maiaMove(model, input, { temperature: 0 });
    expect(move).toBe(argmax(policy));
  });

  it("samples with a seeded rng and respects topP", () => {
    let seed = 1;
    const rng = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
    const { policy } = maiaPolicy(model, input);
    const top = argmax(policy);
    for (let i = 0; i < 20; i++) {
      // topP below the top move's probability keeps only the top move.
      expect(maiaMove(model, input, { topP: policy[top]! * 0.5, rng }).move).toBe(top);
      expect(policy[maiaMove(model, input, { rng }).move]).toBeGreaterThan(0);
    }
  });

  it("rejects history that does not reach the FEN", () => {
    expect(() => maiaPolicy(model, { ...input, history: ["e2e4"] })).toThrow();
  });
});
