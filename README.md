# Autofocus

[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg?style=flat-square)](https://opensource.org/licenses/MIT)

Content-aware focal-point detection for photographs: a Rust engine compiled to
WebAssembly (with a native CLI), plus the `@modufolio/autofocus` npm package
that wraps it for the browser. Given an image, it answers one question well —
*where should a crop keep its focus so the subject survives?* — so a face near
the frame edge never gets centre-cropped away.

**The detector is signals, not a neural net.** An adaptive skin model that
recalibrates its centres to the skin tones actually present in the image,
HOG-based face-likeness, edge and sharpness maps, radial symmetry, and
rule-of-thirds priors are fused per block, then a scene classifier
(portrait / group / scene / mono) picks the weight preset before the focal
point is refined. Deterministic, explainable, fast — and small enough to ship
as WASM to every visitor.

## Why it exists

- **Centre cropping mutilates portraits.** The default behaviour of every
  thumbnail pipeline is to assume the subject is centred. Photographers do not
  compose that way.
- **Saliency-only croppers stop too early.** Edge and saturation maps with
  static skin detection go wrong exactly where photographs matter most.
  This engine adds an adaptive skin model, face-likeness, eye-band
  refinement, and scene classification — a portrait, a group shot, and a
  landscape get different weightings because they *are* different problems.
- **ML croppers are heavyweight and opaque.** A model file outweighs this
  whole engine, and when it picks the wrong subject there is nothing to debug.
  Here every signal is inspectable (`detect_features`, `debug_blocks`), and
  behaviour is tuned by editing weights you can read.

## Layout

- `rust/` — the detector crate (`autofocus`): signal extraction
  (`features`, `saliency`, `segments`), scene classification (`classify`,
  `rules`, `weights`), refinement (`refine`, `face_region`, `zoom`), and a
  native CLI for batch analysis. `cdylib + rlib`, wasm-bindgen API.
- `ui/` — the npm package: a thin loader that fetches the WASM module at
  runtime and exposes the detection API.

## Quickstart

**Browser (npm):**

```js
import { detectFocus, detectFeatures } from '@modufolio/autofocus'

const { point } = await detectFocus(imgElement)    // → { x, y } in [0,1]
const features  = await detectFeatures(imgElement) // → point + category + image features
```

The WASM module ships inside the npm package and loads automatically. To
self-host it elsewhere instead, set `globalThis.AUTOFOCUS_WASM_BASE` to the
directory URL holding the two `wasm/` files, or copy them to the legacy
fallback location `/assets/wasm/autofocus/`:

```bash
cd rust && wasm-pack build --target web --out-dir <app>/public/assets/wasm/autofocus
```

**Native CLI:**

```bash
cd rust && cargo run --release -- path/to/photo.jpg
```

**Reviewing fixture accuracy:**

Render every golden fixture with the hand-placed focus point (green ring)
and the detected point (orange crosshair) drawn on the photo, a line
connecting them, and the distance encoded in the filename so a reverse
directory sort surfaces the worst detections:

```bash
cd rust && cargo run --release -- --review target/review
```

Output mirrors the `<album>/<file>` fixture layout under `target/` (already
gitignored — the photos are not MIT-licensed, keep renders out of commits).

**Evaluating an external dataset:**

Point `--eval` at any directory of photos plus a `focuspoints.json` mapping
each filename to its hand-set point (`{"img.jpg": {"x": 0.47, "y": 0.49}}`).
Prints per-photo distances and the mean; add `--review` to also render the
annotated comparisons. This is how golden sets that cannot be committed are
measured:

```bash
cd rust && cargo run --release -- --eval path/to/dataset --review target/review-dataset
```

**Rust (as a library):**

```rust
let (x, y, scene) = autofocus::detect_focus_cli(&rgba, width, height);
```

## How it is used

Born inside the Modufolio portfolio stack, where
[`modufolio/media`](https://github.com/modufolio/media) uses the focal point
for content-aware thumbnail crops. The engine has no dependency on any of
that — RGBA in, focal point out.

## License

[MIT](LICENSE) — with one carve-out: the fixture photographs under
`ui/tests/fixtures/` are copyrighted work by the author and are **not** MIT.
They may only be used to run this repository's tests — see
[`ui/tests/fixtures/LICENSE`](ui/tests/fixtures/LICENSE).
