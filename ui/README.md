# @modufolio/autofocus

Content-aware crop focus for photography sites, backed by a Rust/WASM
detector. Feature detection, face regions, radial refinement and crop
classification — so a face near the frame edge never gets centre-cropped
away.

## Install

```bash
npm install @modufolio/autofocus
```

## Use

```js
import { detectFocus, detectFeatures } from '@modufolio/autofocus'

const { point } = await detectFocus(imgElement)   // → { x, y } in [0,1]
```

## The WASM module

The detector runs as WebAssembly, fetched at runtime from your app's
`/assets/wasm/autofocus/` — it is deliberately **not** bundled in this package.
Build it per app from the Rust crate in this repository:

```bash
cd rust && wasm-pack build --target web --out-dir <app>/public/assets/wasm/autofocus
```

There is no JS implementation of the algorithm — the Rust crate is the
single implementation, and the package returns null when WASM is
unavailable.

## Develop

Apps in the Modufolio family compile this package from source by aliasing
`@modufolio/autofocus` to `ui/src/index.js`. `npm test` runs the fixture-based
detector suite; `npm run build` produces `dist/` for npm consumers.
