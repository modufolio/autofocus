/**
 * Native ↔ WASM parity.
 *
 * The two builds are the same Rust compiled twice, so on IDENTICAL pixel
 * buffers they must agree. The pixels are prepared once here (jpeg-js decode
 * + nearest downscale) and fed to both the WASM module and the native CLI's
 * `--raw-rgba` mode — decode and resize are deliberately outside the
 * comparison, so a failure means the builds themselves diverged
 * (toolchain, libm, wasm-opt), never the preprocessing.
 *
 * Requires both artifacts; run via `npm run test:parity`, which builds them.
 * The suite skips (with a hint) when either is missing, so plain `npm test`
 * stays runnable without the Rust toolchain.
 */

import { describe, it, expect } from 'vitest'
import { execFileSync } from 'child_process'
import { readFileSync, readdirSync, existsSync } from 'fs'
import { join, dirname } from 'path'
import { fileURLToPath } from 'url'
import { createRequire } from 'module'
import jpegJs from 'jpeg-js'

const __dir = dirname(fileURLToPath(import.meta.url))
const FIXTURES = join(__dir, 'fixtures')
const CLI = join(__dir, '../../rust/target/release/autofocus')
const WASM_DIR = join(__dir, '../wasm-node')

const ready = existsSync(CLI) && existsSync(join(WASM_DIR, 'autofocus_wasm.js'))

const MAX_DIM = 256
/** The builds agree to f32 precision; the only noise is the CLI's decimal
 *  print/parse round-trip (measured max delta: 2.9e-8 across 41 fixtures). */
const EPS = 1e-6

function decodeScaled(path) {
  const { data, width: w0, height: h0 } = jpegJs.decode(readFileSync(path), { useTArray: true })
  const scale = Math.min(1, MAX_DIM / Math.max(w0, h0))
  const w = Math.round(w0 * scale)
  const h = Math.round(h0 * scale)
  const out = new Uint8Array(w * h * 4)
  for (let y = 0; y < h; y++) {
    const sy = Math.min(h0 - 1, Math.floor((y + 0.5) / scale))
    for (let x = 0; x < w; x++) {
      const sx = Math.min(w0 - 1, Math.floor((x + 0.5) / scale))
      const s = (sy * w0 + sx) * 4
      const d = (y * w + x) * 4
      out[d] = data[s]; out[d + 1] = data[s + 1]; out[d + 2] = data[s + 2]; out[d + 3] = data[s + 3]
    }
  }
  return { rgba: out, w, h }
}

function nativePoint(rgba, w, h) {
  const stdout = execFileSync(CLI, [`--raw-rgba`, `${w}x${h}`], { input: Buffer.from(rgba) })
  const [x, y, category] = stdout.toString().trim().split(' ')
  return { x: parseFloat(x), y: parseFloat(y), category }
}

describe.skipIf(!ready)('native ↔ wasm parity (run `npm run test:parity` to build both)', () => {
  const require = createRequire(import.meta.url)
  const wasm = ready ? require(join(WASM_DIR, 'autofocus_wasm.js')) : null

  // The committed corpus only (fixtures.json), not a directory glob:
  // external golden sets (see `autofocus --eval`) may sit next to the album
  // dirs without being part of the parity invariant. On 1398701.jpg of one
  // such set the two builds genuinely diverge through a decision boundary
  // (native 0.42 vs wasm 0.41) — parity to 1e-6 is a property of the
  // committed corpus, not of every photo.
  const manifest = ready ? JSON.parse(readFileSync(join(FIXTURES, 'fixtures.json'), 'utf8')) : { albums: {} }
  const files = Object.entries(manifest.albums)
    .flatMap(([album, entry]) => entry.photos.map((p) => join(FIXTURES, album, p.file)))

  it('found the fixture set', () => {
    expect(files.length).toBeGreaterThan(0)
  })

  it('detect_focus agrees on every fixture', { timeout: 120_000 }, () => {
    let maxDelta = 0
    for (const file of files) {
      const { rgba, w, h } = decodeScaled(file)
      const native = nativePoint(rgba, w, h)
      const result = wasm.detect_focus(rgba, w, h)
      const dx = Math.abs(result[0] - native.x)
      const dy = Math.abs(result[1] - native.y)
      maxDelta = Math.max(maxDelta, dx, dy)
      expect(dx, `${file} x: wasm ${result[0]} vs native ${native.x}`).toBeLessThanOrEqual(EPS)
      expect(dy, `${file} y: wasm ${result[1]} vs native ${native.y}`).toBeLessThanOrEqual(EPS)
    }
    console.log(`parity max delta across ${files.length} fixtures: ${maxDelta}`)
  })
})
