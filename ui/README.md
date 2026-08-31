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

The detector runs as WebAssembly, shipped inside this package (`wasm/`,
~60 KB gzipped) and loaded automatically — `npm install` is all it takes.
The loader tries, in order:

1. `globalThis.AUTOFOCUS_WASM_BASE` — set this (before the first import)
   to a directory URL if you self-host the two `wasm/` files somewhere else.
2. The package's own `wasm/` directory, relative to the loaded module.
3. `/assets/wasm/autofocus/` on the page origin — the pre-0.2 layout, so
   deploys that copy the files there keep working unchanged.

There is no JS implementation of the algorithm — the Rust crate is the
single implementation, and the package returns null when WASM is
unavailable.

## Develop

Apps in the Modufolio family compile this package from source by aliasing
`@modufolio/autofocus` to `ui/src/index.js`. `npm test` runs the fixture-based
detector suite; `npm run build` produces `dist/` for npm consumers.
