"""Export Maia-3 5M weights to maia3-5m.safetensors for maia3-ts.

The output is a standard safetensors file. Tensors keep the PyTorch names and layout
(row major, Linear weights are [out, in]). The GAB weight shared by every block is stored
once as gab_shared_weight and the unused ponder head is dropped. The model config is
stored as JSON under the "config" metadata key.

Usage: uv run python convert.py [--dtype f16|f32] [--check]
--check runs the official model with the stored (rounded) weights on the fixture and
prints the max probability difference against the float32 reference.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file

from download import checkpoint_path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "maia3-5m.safetensors"
FIXTURE = HERE.parent / "test" / "fixtures" / "maia-reference.json"

CONFIG = {
    "history": 8,
    "dimEmb": 128,
    "dimVit": 256,
    "numBlocks": 8,
    "numHeads": 8,
    "mlpDim": 512,
    "headHidDim": 256,
    "gabGenSize": 64,
    "gabIntermediateDim": 64,
    "eloUpper": 5000,
}


def load_state():
    sd = torch.load(checkpoint_path(), map_location="cpu", weights_only=True)
    if "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    sd = {k.replace("smolgen", "gab"): v.float() for k, v in sd.items()}
    shared = sd["gab_shared_weight"]
    out = {}
    for k, v in sd.items():
        if k.endswith("self_attn.gab_weight"):
            assert torch.equal(v, shared), k
            continue
        if k.startswith("fc_ponder"):
            continue
        out[k] = v
    return out


def write_safetensors(state, dtype):
    np_dtype = np.float16 if dtype == "f16" else np.float32
    arrays = {name: np.ascontiguousarray(t.numpy().astype(np_dtype)) for name, t in state.items()}
    save_file(arrays, OUT, metadata={"config": json.dumps(CONFIG, separators=(",", ":"))})
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB, {dtype}, {len(arrays)} tensors)")


def check(state, dtype):
    """Compare the official model loaded with rounded weights against the fixture."""
    import reference

    engine = reference.build_engine()
    np_dtype = np.float16 if dtype == "f16" else np.float32
    rounded = {k: torch.from_numpy(v.numpy().astype(np_dtype).astype(np.float32)) for k, v in state.items()}
    for i in range(CONFIG["numBlocks"]):
        rounded[f"transformer.layers.{i}.self_attn.gab_weight"] = rounded["gab_shared_weight"]
    engine.model.load_state_dict(rounded, strict=False)
    fixture = json.loads(FIXTURE.read_text())
    worst, argmax_bad = 0.0, 0
    for case in fixture["cases"]:
        moves = case.get("history")
        cmd = ("position startpos" + (" moves " + " ".join(moves) if moves else "")) if moves is not None \
            else f"position fen {case['fen']}"
        policy, _ = reference.evaluate(engine, cmd, case["eloSelf"], case["eloOppo"])
        ref = case["policy"]
        worst = max(worst, max(abs(policy[m] - ref[m]) for m in ref))
        if max(policy, key=policy.get) != max(ref, key=ref.get):
            argmax_bad += 1
    print(f"{dtype}: max |dp| = {worst:.2e}, argmax mismatches = {argmax_bad}/{len(fixture['cases'])}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=["f16", "f32"], default="f32")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    state = load_state()
    write_safetensors(state, args.dtype)
    if args.check:
        check(state, args.dtype)


if __name__ == "__main__":
    main()
