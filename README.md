# maia3-ts

[Maia-3](https://github.com/CSSLab/maia3) in plain TypeScript. Give it a position and two ratings and it predicts the move a human at that rating would play, including their typical mistakes.

It has no native or WebAssembly dependencies, so it runs anywhere JavaScript does: Node, browsers and small serverless workers. Its outputs match the official PyTorch model to within 0.0004% on 420 test positions.

## Use

```bash
npm install github:dmallory42/maia3-ts#v0.1.0
```

The package holds the code only. Download the weights it reads (20 MB) from the URL in `WEIGHTS` and check them against `WEIGHTS.sha256`, then:

```ts
import { loadMaia, maiaMove, maiaPolicy } from "maia3-ts";

const model = loadMaia(weights); // an ArrayBuffer holding maia3-5m.safetensors

const fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1";
const { move } = maiaMove(model, { fen, eloSelf: 1200, eloOppo: 1200 });
const { policy, value } = maiaPolicy(model, { fen, eloSelf: 1200, eloOppo: 1200 });
```

`maiaMove` samples a move at temperature 1 by default; pass `{ temperature: 0 }` for the most likely move. `maiaPolicy` returns a probability for every legal move and the predicted win, draw and loss chances for the side to move. Pass `history`, the UCI moves from the start position, for the model to see the last eight positions as it was trained to.

## Develop

```bash
npm install
npm run fetch-weights
npm test
```

The tests compare every move probability against `test/fixtures/maia-reference.json`, recorded from the official model.

## Weights and reference outputs

The weights are the official [Maia3-5M](https://huggingface.co/UofTCSSLab/Maia3-5M) checkpoint, converted from PyTorch to safetensors. The model config is stored in the file's metadata.

The Python scripts in `tools/` are only needed for a new Maia-3 checkpoint. `convert.py` writes the safetensors file and `reference.py` records the official model's outputs for the tests. They need [uv](https://docs.astral.sh/uv/):

```bash
cd tools
uv run python convert.py --check
uv run python reference.py
```

## Licence

AGPL-3.0-or-later, as a port of Maia-3 by the CSSLab at the University of Toronto, which is AGPL-3.0.
