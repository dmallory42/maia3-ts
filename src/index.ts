// Maia-3 human move prediction in plain TypeScript.
//
// Port of the inference path of Maia-3 by CSSLab, University of Toronto
// (https://github.com/CSSLab/maia3, paper arXiv:2605.19091), which is licensed under
// the GNU Affero General Public License v3.0. Weights: UofTCSSLab/Maia3-5M on Hugging Face.
// This port is distributed under AGPL-3.0-or-later.
//
// Inputs:
// - fen: the current position.
// - history: optional UCI moves from the standard start position that lead to `fen`.
//   The model reads the last 8 positions (current included). Without history the current
//   position is repeated, which is also what the official UCI engine does by default.
// - eloSelf / eloOppo: ratings of the side to move and its opponent. The model clamps them
//   to [0, 5000] and interpolates linearly between two learned embeddings.

import { Chess } from "chess.js";
import { encodeHistory, historyPositions, positionFromFen, squareIndex } from "./encode.js";
import { buildNet, forward, moveLogit, type Net } from "./model.js";
import { readWeights } from "./weights.js";

export type { MaiaConfig } from "./weights.js";

/** The Maia-3 5M weights this version reads, published as a release asset, and their SHA-256. */
export const WEIGHTS = {
  url: "https://github.com/dmallory42/maia3-ts/releases/download/v0.1.0/maia3-5m.safetensors",
  sha256: "c3f778e606a06d69355885a6bfb3d3dfa5c45a18cb50c3f013fe76afa5445e0a",
} as const;

export interface MaiaModel {
  readonly net: Net;
}

export interface MaiaInput {
  fen: string;
  history?: string[];
  eloSelf: number;
  eloOppo: number;
}

export interface MaiaPolicyResult {
  /** Probability per legal move, keyed by UCI (e.g. "e2e4", "e7e8q"). Sums to 1. */
  policy: Record<string, number>;
  /** Predicted game outcome for the side to move. */
  value?: { win: number; draw: number; loss: number };
}

export interface MaiaSampleOptions {
  /** Softmax temperature. 0 picks the most likely move. Default 1 (official maia3-uci default). */
  temperature?: number;
  /** Nucleus threshold; 1 disables it. Default 1 (official default). */
  topP?: number;
  /** Uniform random source in [0, 1). Default Math.random. */
  rng?: () => number;
}

const PROMO_INDEX: Record<string, number> = { q: 0, r: 1, b: 2, n: 3 };

export function loadMaia(buf: ArrayBuffer): MaiaModel {
  const { config, tensors } = readWeights(buf);
  return { net: buildNet(config, tensors) };
}

/** Legal moves as UCI strings with their raw policy logits. */
function legalLogits(model: MaiaModel, input: MaiaInput): { moves: string[]; logits: Float64Array; value: [number, number, number] } {
  const net = model.net;
  const current = positionFromFen(input.fen);
  const positions = historyPositions(current, input.history, net.cfg.history);
  encodeHistory(positions, net.cfg.history, net.s.active, net.s.counts);
  const out = forward(net, input.eloSelf, input.eloOppo);

  const chess = new Chess(input.fen);
  const verbose = chess.moves({ verbose: true });
  const flip = !current.whiteToMove;
  const moves: string[] = [];
  const logits = new Float64Array(verbose.length);
  verbose.forEach((m, k) => {
    let from = squareIndex(m.from);
    let to = squareIndex(m.to);
    if (flip) { from ^= 56; to ^= 56; }
    const promo = m.promotion ? PROMO_INDEX[m.promotion]! : -1;
    moves.push(m.from + m.to + (m.promotion ?? ""));
    logits[k] = moveLogit(net, out, from, to, promo);
  });
  return { moves, logits, value: out.value };
}

function softmax(logits: Float64Array, temperature = 1): Float64Array {
  let max = -Infinity;
  for (const l of logits) if (l > max) max = l;
  const p = new Float64Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) { p[i] = Math.exp((logits[i]! - max) / temperature); sum += p[i]!; }
  for (let i = 0; i < p.length; i++) p[i]! /= sum;
  return p;
}

/** Move probabilities over the legal moves plus the value head's win/draw/loss for the side to move. */
export function maiaPolicy(model: MaiaModel, input: MaiaInput): MaiaPolicyResult {
  const { moves, logits, value } = legalLogits(model, input);
  const probs = softmax(logits);
  const policy: Record<string, number> = {};
  moves.forEach((m, i) => { policy[m] = probs[i]!; });
  const [loss, draw, win] = softmax(Float64Array.from(value));
  return { policy, value: { win: win!, draw: draw!, loss: loss! } };
}

/**
 * Samples a move like the official engine: temperature 0 is argmax, otherwise sample from
 * softmax(logits / T), optionally restricted to the smallest prefix whose cumulative
 * probability stays within topP (the top move is always kept). The returned policy is the
 * untempered distribution.
 */
export function maiaMove(model: MaiaModel, input: MaiaInput, opts: MaiaSampleOptions = {}): { move: string; policy: Record<string, number> } {
  const temperature = opts.temperature ?? 1;
  const topP = opts.topP ?? 1;
  const rng = opts.rng ?? Math.random;
  const { moves, logits } = legalLogits(model, input);
  if (moves.length === 0) throw new Error("maia: no legal moves");
  const probs = softmax(logits);
  const policy: Record<string, number> = {};
  moves.forEach((m, i) => { policy[m] = probs[i]!; });

  let pick = 0;
  if (temperature <= 0) {
    for (let i = 1; i < logits.length; i++) if (logits[i]! > logits[pick]!) pick = i;
  } else {
    const p = softmax(logits, temperature);
    let order = Array.from(p.keys());
    if (topP < 1) {
      order.sort((a, b) => p[b]! - p[a]!);
      let cum = 0;
      order = order.filter((idx, rank) => { cum += p[idx]!; return rank === 0 || cum <= topP; });
    }
    let total = 0;
    for (const idx of order) total += p[idx]!;
    let r = rng() * total;
    pick = order[order.length - 1]!;
    for (const idx of order) { r -= p[idx]!; if (r < 0) { pick = idx; break; } }
  }
  return { move: moves[pick]!, policy };
}
