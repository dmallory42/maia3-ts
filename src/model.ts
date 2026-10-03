// Chessformer forward pass for Maia-3 in plain TypeScript. Maia-3 and the Chessformer
// architecture are by CSSLab (https://github.com/CSSLab/maia3, arXiv:2605.19091),
// licensed AGPL-3.0. This is an independent reimplementation of the inference maths.
//
// Architecture (5M preset): 64 square tokens, one-hot history planes plus two Elo
// embeddings projected to 256 dims, 8 post-norm encoder blocks (RMSNorm, GELU MLP,
// 8-head attention with a per-position "GAB" bias generated from the mean token),
// final LayerNorm. Policy logits are bilinear from/to square scores plus promotion
// biases; the value head reads the mean token.

import type { MaiaConfig } from "./weights.js";

interface Block {
  gabW2: Float32Array; gabB2: Float32Array; // [inter, dim]
  ln1W: Float32Array; ln1B: Float32Array;
  gabW3: Float32Array; gabB3: Float32Array; // [heads * gen, inter]
  ln2W: Float32Array; ln2B: Float32Array;
  qkvW: Float32Array; // [3 * dim, dim]
  outW: Float32Array; // [dim, dim]
  norm1: Float32Array;
  ffW1: Float32Array; ffB1: Float32Array; // [mlp, dim]
  ffW2: Float32Array; ffB2: Float32Array; // [dim, mlp]
  norm2: Float32Array;
}

export interface Net {
  cfg: MaiaConfig;
  eloLow: Float32Array; eloHigh: Float32Array;
  tokT: Float32Array; // token projection transposed: [inputs, dim]
  tokB: Float32Array;
  gabShared: Float32Array; // [4096, gen]
  blocks: Block[];
  normW: Float32Array; normB: Float32Array;
  lastLnW: Float32Array; lastLnB: Float32Array;
  valHidW: Float32Array; valHidB: Float32Array;
  valW: Float32Array; valB: Float32Array;
  sqFromW: Float32Array; sqToW: Float32Array; // [hid, dim]
  promoW: Float32Array; // [4, hid]
  s: Scratch;
}

interface Scratch {
  x: Float32Array; qkv: Float32Array; attn: Float32Array; tmp: Float32Array; ff: Float32Array;
  bias: Float32Array; row: Float64Array; mean: Float64Array; g1: Float32Array; g2: Float32Array;
  eloVec: Float32Array; sqFrom: Float32Array; sqTo: Float32Array; vHid: Float32Array;
  active: Int32Array; counts: Int32Array;
}

export interface NetOutput {
  /** Side-to-move perspective from/to features, used to score moves lazily. */
  sqFrom: Float32Array; sqTo: Float32Array;
  /** Promotion bias per destination file (0-7) and piece (q, r, b, n). */
  promoBias: Float64Array;
  /** Value logits [loss, draw, win] for the side to move. */
  value: [number, number, number];
}

const RMS_EPS = 1.1920928955078125e-7; // torch RMSNorm default: float32 machine epsilon
const LN_EPS = 1e-5;

export function buildNet(cfg: MaiaConfig, t: Map<string, Float32Array>): Net {
  const get = (name: string): Float32Array => {
    const v = t.get(name);
    if (!v) throw new Error(`maia: missing tensor ${name}`);
    return v;
  };
  const d = cfg.dimVit;
  const inputs = 12 * cfg.history + 2 * cfg.dimEmb;
  const tok = get("token_projection.weight");
  const tokT = new Float32Array(inputs * d);
  for (let o = 0; o < d; o++) for (let i = 0; i < inputs; i++) tokT[i * d + o] = tok[o * inputs + i]!;

  const blocks: Block[] = [];
  for (let b = 0; b < cfg.numBlocks; b++) {
    const p = `transformer.layers.${b}.`;
    blocks.push({
      gabW2: get(p + "self_attn.sm2.weight"), gabB2: get(p + "self_attn.sm2.bias"),
      ln1W: get(p + "self_attn.ln1.weight"), ln1B: get(p + "self_attn.ln1.bias"),
      gabW3: get(p + "self_attn.sm3.weight"), gabB3: get(p + "self_attn.sm3.bias"),
      ln2W: get(p + "self_attn.ln2.weight"), ln2B: get(p + "self_attn.ln2.bias"),
      qkvW: get(p + "self_attn.mha.in_proj_weight"),
      outW: get(p + "self_attn.mha.out_proj.weight"),
      norm1: get(p + "norm1.weight"),
      ffW1: get(p + "linear1.weight"), ffB1: get(p + "linear1.bias"),
      ffW2: get(p + "linear2.weight"), ffB2: get(p + "linear2.bias"),
      norm2: get(p + "norm2.weight"),
    });
  }
  const h = cfg.headHidDim;
  return {
    cfg,
    eloLow: get("elo_embedding_low.weight"), eloHigh: get("elo_embedding_high.weight"),
    tokT, tokB: get("token_projection.bias"),
    gabShared: get("gab_shared_weight"),
    blocks,
    normW: get("transformer.norm.weight"), normB: get("transformer.norm.bias"),
    lastLnW: get("last_ln.weight"), lastLnB: get("last_ln.bias"),
    valHidW: get("fc_value_hid.weight"), valHidB: get("fc_value_hid.bias"),
    valW: get("fc_value.weight"), valB: get("fc_value.bias"),
    sqFromW: get("proj_sq_from.weight"), sqToW: get("proj_sq_to.weight"),
    promoW: get("promo_bias_proj.weight"),
    s: {
      x: new Float32Array(64 * d), qkv: new Float32Array(64 * 3 * d), attn: new Float32Array(64 * d),
      tmp: new Float32Array(64 * d), ff: new Float32Array(64 * cfg.mlpDim),
      bias: new Float32Array(cfg.numHeads * 4096), row: new Float64Array(64), mean: new Float64Array(d),
      g1: new Float32Array(cfg.gabIntermediateDim), g2: new Float32Array(cfg.numHeads * cfg.gabGenSize),
      eloVec: new Float32Array(d), sqFrom: new Float32Array(64 * h), sqTo: new Float32Array(64 * h),
      vHid: new Float32Array(h), active: new Int32Array(64 * cfg.history), counts: new Int32Array(64),
    },
  };
}

/** Exact GELU, 0.5 x (1 + erf(x / sqrt 2)). */
function gelu(x: number): number {
  return 0.5 * x * (1 + erf(x * Math.SQRT1_2));
}

// erf via the W. J. Cody style rational approximation from Numerical Recipes (erfc, rel. error < 1.2e-7).
function erf(x: number): number {
  const z = Math.abs(x);
  const t = 1 / (1 + 0.5 * z);
  const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
    t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
    t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? 1 - r : r - 1;
}

/**
 * out[r, o] = sum_i x[r, i] * W[o, i] (+ bias[o]) for `rows` rows. W is [outDim, inDim].
 * Blocks 2 rows by 4 outputs so each loaded value is reused.
 */
function linear(
  out: Float32Array, x: Float32Array, rows: number, inDim: number,
  w: Float32Array, bias: Float32Array | null, outDim: number,
): void {
  for (let r = 0; r < rows; r += 2) {
    const x0 = r * inDim;
    const x1 = x0 + inDim;
    const o0 = r * outDim;
    const o1 = o0 + outDim;
    for (let o = 0; o < outDim; o += 4) {
      const w0 = o * inDim, w1 = w0 + inDim, w2 = w1 + inDim, w3 = w2 + inDim;
      let a00 = 0, a01 = 0, a02 = 0, a03 = 0, a10 = 0, a11 = 0, a12 = 0, a13 = 0;
      for (let i = 0; i < inDim; i++) {
        const u = x[x0 + i]!, v = x[x1 + i]!;
        const c0 = w[w0 + i]!, c1 = w[w1 + i]!, c2 = w[w2 + i]!, c3 = w[w3 + i]!;
        a00 += u * c0; a01 += u * c1; a02 += u * c2; a03 += u * c3;
        a10 += v * c0; a11 += v * c1; a12 += v * c2; a13 += v * c3;
      }
      if (bias) {
        const b0 = bias[o]!, b1 = bias[o + 1]!, b2 = bias[o + 2]!, b3 = bias[o + 3]!;
        a00 += b0; a01 += b1; a02 += b2; a03 += b3; a10 += b0; a11 += b1; a12 += b2; a13 += b3;
      }
      out[o0 + o] = a00; out[o0 + o + 1] = a01; out[o0 + o + 2] = a02; out[o0 + o + 3] = a03;
      out[o1 + o] = a10; out[o1 + o + 1] = a11; out[o1 + o + 2] = a12; out[o1 + o + 3] = a13;
    }
  }
}

/** Single-row matrix-vector product, out[o] = W[o] . x + bias[o]. */
function matvec(out: Float32Array, x: ArrayLike<number>, inDim: number, w: Float32Array, bias: Float32Array | null, outDim: number): void {
  for (let o = 0; o < outDim; o++) {
    let acc = bias ? bias[o]! : 0;
    const wo = o * inDim;
    for (let i = 0; i < inDim; i++) acc += x[i]! * w[wo + i]!;
    out[o] = acc;
  }
}

/** In-place LayerNorm over a vector of length n. */
function layerNormVec(v: Float32Array | Float64Array, off: number, n: number, w: Float32Array, b: Float32Array): void {
  let mean = 0;
  for (let i = 0; i < n; i++) mean += v[off + i]!;
  mean /= n;
  let varSum = 0;
  for (let i = 0; i < n; i++) { const c = v[off + i]! - mean; varSum += c * c; }
  const inv = 1 / Math.sqrt(varSum / n + LN_EPS);
  for (let i = 0; i < n; i++) v[off + i] = (v[off + i]! - mean) * inv * w[i]! + b[i]!;
}

/** x = RMSNorm(x + add) per row, in place on x. */
function addRmsNorm(x: Float32Array, add: Float32Array, d: number, w: Float32Array): void {
  for (let r = 0; r < 64; r++) {
    const off = r * d;
    let ss = 0;
    for (let i = 0; i < d; i++) { const v = x[off + i]! + add[off + i]!; x[off + i] = v; ss += v * v; }
    const inv = 1 / Math.sqrt(ss / d + RMS_EPS);
    for (let i = 0; i < d; i++) x[off + i] = x[off + i]! * inv * w[i]!;
  }
}

/** Mean over the 64 square tokens into `out`. */
function meanTokens(out: Float64Array, x: Float32Array, d: number): void {
  out.fill(0);
  for (let r = 0; r < 64; r++) for (let i = 0; i < d; i++) out[i]! += x[r * d + i]!;
  for (let i = 0; i < d; i++) out[i]! /= 64;
}

/** Fills s.bias with the per-head 64x64 attention bias generated from the mean token. */
function gabBias(net: Net, blk: Block): void {
  const { cfg, s } = net;
  const d = cfg.dimVit, inter = cfg.gabIntermediateDim, gen = cfg.gabGenSize, heads = cfg.numHeads;
  meanTokens(s.mean, s.x, d);
  matvec(s.g1, s.mean, d, blk.gabW2, blk.gabB2, inter);
  for (let i = 0; i < inter; i++) s.g1[i] = gelu(s.g1[i]!);
  layerNormVec(s.g1, 0, inter, blk.ln1W, blk.ln1B);
  matvec(s.g2, s.g1, inter, blk.gabW3, blk.gabB3, heads * gen);
  for (let i = 0; i < heads * gen; i++) s.g2[i] = gelu(s.g2[i]!);
  layerNormVec(s.g2, 0, heads * gen, blk.ln2W, blk.ln2B);
  const g = net.gabShared;
  const bias = s.bias, y = s.g2;
  for (let o = 0; o < 4096; o++) {
    const go = o * gen;
    for (let h = 0; h < heads; h++) {
      const yh = h * gen;
      let acc = 0;
      for (let i = 0; i < gen; i++) acc += y[yh + i]! * g[go + i]!;
      bias[h * 4096 + o] = acc;
    }
  }
}

/** Multi-head self-attention with additive bias; result (before out_proj) in s.attn. */
function attention(net: Net): void {
  const { cfg, s } = net;
  const d = cfg.dimVit, heads = cfg.numHeads, hd = d / heads, stride = 3 * d;
  const scale = 1 / Math.sqrt(hd);
  const qkv = s.qkv, bias = s.bias, row = s.row, out = s.attn;
  for (let h = 0; h < heads; h++) {
    const qOff = h * hd, kOff = d + h * hd, vOff = 2 * d + h * hd, bOff = h * 4096;
    for (let i = 0; i < 64; i++) {
      const qi = i * stride + qOff;
      let max = -Infinity;
      for (let j = 0; j < 64; j++) {
        const kj = j * stride + kOff;
        let dot = 0;
        for (let c = 0; c < hd; c++) dot += qkv[qi + c]! * qkv[kj + c]!;
        const v = dot * scale + bias[bOff + i * 64 + j]!;
        row[j] = v;
        if (v > max) max = v;
      }
      let sum = 0;
      for (let j = 0; j < 64; j++) { const e = Math.exp(row[j]! - max); row[j] = e; sum += e; }
      const inv = 1 / sum;
      const oi = i * d + h * hd;
      for (let c = 0; c < hd; c++) out[oi + c] = 0;
      for (let j = 0; j < 64; j++) {
        const p = row[j]! * inv;
        const vj = j * stride + vOff;
        for (let c = 0; c < hd; c++) out[oi + c]! += p * qkv[vj + c]!;
      }
    }
  }
}

/** Interpolated Elo embedding: clamp to [0, eloUpper], then mix the two learned endpoints. */
function eloEmbedding(net: Net, elo: number, out: Float32Array, off: number): void {
  const wLow = Math.min(Math.max(elo, 0), net.cfg.eloUpper) / net.cfg.eloUpper;
  for (let i = 0; i < net.cfg.dimEmb; i++) out[off + i] = wLow * net.eloLow[i]! + (1 - wLow) * net.eloHigh[i]!;
}

/**
 * Runs the network. `active`/`counts` in net.s must already hold the encoded history
 * (see encodeHistory). Returns views into the net's scratch buffers.
 */
export function forward(net: Net, eloSelf: number, eloOppo: number): NetOutput {
  const { cfg, s } = net;
  const d = cfg.dimVit, hist = cfg.history, histFeatures = 12 * hist, emb = cfg.dimEmb;

  // Token projection: the Elo part is shared by all squares, the board part is one-hot.
  const elo = new Float32Array(2 * emb);
  eloEmbedding(net, eloSelf, elo, 0);
  eloEmbedding(net, eloOppo, elo, emb);
  const base = s.eloVec;
  const tokT = net.tokT;
  for (let o = 0; o < d; o++) base[o] = net.tokB[o]!;
  for (let i = 0; i < 2 * emb; i++) {
    const e = elo[i]!, row = (histFeatures + i) * d;
    for (let o = 0; o < d; o++) base[o]! += e * tokT[row + o]!;
  }
  const x = s.x;
  for (let sq = 0; sq < 64; sq++) {
    const off = sq * d;
    for (let o = 0; o < d; o++) x[off + o] = base[o]!;
    const n = s.counts[sq]!;
    for (let k = 0; k < n; k++) {
      const row = s.active[sq * hist + k]! * d;
      for (let o = 0; o < d; o++) x[off + o]! += tokT[row + o]!;
    }
  }

  for (const blk of net.blocks) {
    gabBias(net, blk);
    linear(s.qkv, x, 64, d, blk.qkvW, null, 3 * d);
    attention(net);
    linear(s.tmp, s.attn, 64, d, blk.outW, null, d);
    addRmsNorm(x, s.tmp, d, blk.norm1);
    linear(s.ff, x, 64, d, blk.ffW1, blk.ffB1, cfg.mlpDim);
    for (let i = 0; i < s.ff.length; i++) s.ff[i] = gelu(s.ff[i]!);
    linear(s.tmp, s.ff, 64, cfg.mlpDim, blk.ffW2, blk.ffB2, d);
    addRmsNorm(x, s.tmp, d, blk.norm2);
  }
  for (let r = 0; r < 64; r++) layerNormVec(x, r * d, d, net.normW, net.normB);

  // Policy features.
  const hid = cfg.headHidDim;
  linear(s.sqFrom, x, 64, d, net.sqFromW, null, hid);
  linear(s.sqTo, x, 64, d, net.sqToW, null, hid);
  const promoBias = new Float64Array(32);
  const promoScale = Math.sqrt(hid);
  for (let f = 0; f < 8; f++) {
    const to = (56 + f) * hid;
    for (let p = 0; p < 4; p++) {
      let acc = 0;
      for (let i = 0; i < hid; i++) acc += s.sqTo[to + i]! * net.promoW[p * hid + i]!;
      promoBias[f * 4 + p] = acc * promoScale;
    }
  }

  // Value head on the mean token.
  meanTokens(s.mean, x, d);
  layerNormVec(s.mean, 0, d, net.lastLnW, net.lastLnB);
  matvec(s.vHid, s.mean, d, net.valHidW, net.valHidB, hid);
  for (let i = 0; i < hid; i++) if (s.vHid[i]! < 0) s.vHid[i] = 0;
  const v = new Float32Array(3);
  matvec(v, s.vHid, hid, net.valW, net.valB, 3);

  return { sqFrom: s.sqFrom, sqTo: s.sqTo, promoBias, value: [v[0]!, v[1]!, v[2]!] };
}

/** Policy logit for a move in the model's (side to move) frame. promo: 0-3 for q r b n, or -1. */
export function moveLogit(net: Net, out: NetOutput, from: number, to: number, promo: number): number {
  const hid = net.cfg.headHidDim;
  const fo = from * hid, tt = to * hid;
  let acc = 0;
  for (let i = 0; i < hid; i++) acc += out.sqFrom[fo + i]! * out.sqTo[tt + i]!;
  acc /= Math.sqrt(hid);
  if (promo >= 0) acc += out.promoBias[(to & 7) * 4 + promo]!;
  return acc;
}
