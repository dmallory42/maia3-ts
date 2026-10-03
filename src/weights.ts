// Maia-3 weight file reader. Maia-3 is by CSSLab (University of Toronto),
// https://github.com/CSSLab/maia3, licensed AGPL-3.0. Reads the safetensors file written by tools/convert.py.

export interface MaiaConfig {
  history: number;
  dimEmb: number;
  dimVit: number;
  numBlocks: number;
  numHeads: number;
  mlpDim: number;
  headHidDim: number;
  gabGenSize: number;
  gabIntermediateDim: number;
  eloUpper: number;
}

interface TensorInfo {
  dtype: string;
  shape: number[];
  data_offsets: [number, number];
}

/** Parses a safetensors weight file into named Float32Arrays plus the model config. */
export function readWeights(buf: ArrayBuffer): { config: MaiaConfig; tensors: Map<string, Float32Array> } {
  const view = new DataView(buf);
  const headerLen = Number(view.getBigUint64(0, true));
  if (headerLen <= 0 || 8 + headerLen > buf.byteLength) throw new Error("maia: not a safetensors file");
  const { __metadata__: meta, ...entries } = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, headerLen)));
  const config = (meta as Record<string, string> | undefined)?.config;
  if (!config) throw new Error("maia: weight file has no Maia-3 config");
  const dataStart = 8 + headerLen;

  const tensors = new Map<string, Float32Array>();
  for (const [name, t] of Object.entries(entries as Record<string, TensorInfo>)) {
    const start = dataStart + t.data_offsets[0];
    const end = dataStart + t.data_offsets[1];
    let arr: Float32Array;
    if (t.dtype === "F32") {
      // Zero-copy view when aligned; safetensors only guarantees 8-byte alignment of the data section.
      arr = start % 4 === 0 ? new Float32Array(buf, start, (end - start) / 4) : new Float32Array(buf.slice(start, end));
    } else if (t.dtype === "F16") {
      arr = halfToFloat(new Uint16Array(buf.slice(start, end)));
    } else {
      throw new Error(`maia: unsupported dtype ${t.dtype} for ${name}`);
    }
    tensors.set(name, arr);
  }
  return { config: JSON.parse(config) as MaiaConfig, tensors };
}

/** Converts IEEE 754 half precision values to float32. */
function halfToFloat(src: Uint16Array): Float32Array {
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) {
    const h = src[i]!;
    const sign = h & 0x8000 ? -1 : 1;
    const exp = (h >> 10) & 0x1f;
    const frac = h & 0x3ff;
    if (exp === 0) out[i] = sign * frac * 2 ** -24;
    else if (exp === 31) out[i] = frac ? NaN : sign * Infinity;
    else out[i] = sign * (1 + frac / 1024) * 2 ** (exp - 15);
  }
  return out;
}
