/**
 * WASM-backed auto-focus.
 *
 * `detect_focus` plus the zoom pass compute the focus point; WASM is
 * initialised eagerly at module import so it is ready before the first
 * image is analysed. Returns null when WASM is unavailable.
 */



// ---------------------------------------------------------------------------
//  WASM loader — eager, fires at module import time
// ---------------------------------------------------------------------------

let _wasmMod = null

// _wasmReady resolves to the module on success, null on failure.
const _wasmReady = (async () => {
  const base    = location.origin + '/assets/wasm/autofocus/'
  const jsUrl   = base + 'autofocus_wasm.js'
  const wasmUrl = base + 'autofocus_wasm_bg.wasm'
  try {
    const mod = await import(jsUrl)
    // wasm-bindgen ≥0.2.93 deprecates positional init args
    await mod.default({ module_or_path: wasmUrl })
    _wasmMod = mod
    // Older builds predate build_version() — a missing tag means the browser
    // cached a stale autofocus_wasm_bg.wasm and needs a hard refresh.
    if (typeof mod.build_version !== 'function') {
      console.warn('[AutoFocus] stale cached WASM build — hard-refresh')
    }
    return mod
  } catch (err) {
    console.warn('[AutoFocus] WASM failed to load:', err)
    return null
  }
})()

// ---------------------------------------------------------------------------
//  Shared canvas helper
// ---------------------------------------------------------------------------

function getCanvasData(imgEl, maxDim) {
  const nw = imgEl.naturalWidth
  const nh = imgEl.naturalHeight
  if (!nw || !nh) return null

  const scale = Math.min(1, maxDim / Math.max(nw, nh))
  const w = Math.round(nw * scale)
  const h = Math.round(nh * scale)

  const canvas = document.createElement('canvas')
  canvas.width = w; canvas.height = h
  const ctx = canvas.getContext('2d', { willReadFrequently: true })
  ctx.drawImage(imgEl, 0, 0, w, h)

  let imageData
  try { imageData = ctx.getImageData(0, 0, w, h) } catch { return null }

  return { canvas, w, h, imageData }
}

// ---------------------------------------------------------------------------
//  Internal helpers
// ---------------------------------------------------------------------------

/** Call WASM detect_focus on already-decoded image data. Returns {x,y} or null. */
function wasmDetect(wasm, imageData, w, h) {
  if (!wasm) return null
  const result = wasm.detect_focus(imageData.data, w, h)
  return { x: result[0], y: result[1] }
}

// ---------------------------------------------------------------------------
//  Public API — WASM-backed entry points
// ---------------------------------------------------------------------------

/**
 * Two-pass zoom (ZOOM-PLAN step 4): pass 1 proposes the topmost skin
 * region; the crop is redrawn from the ORIGINAL element (real resolution
 * gain) and pass 2 verifies it (pair geometry + face_like + brow
 * co-location, thresholds baked in crop_verify / passed here).
 * Set localStorage.AF_NO_ZOOM to disable for A/B.
 */
function zoomRefine(wasm, imgEl, cd, point) {
  try {
    if (typeof wasm.zoom_region !== 'function') return point
    if (typeof localStorage !== 'undefined' && localStorage.getItem('AF_NO_ZOOM')) return point
    const zr = wasm.zoom_region(cd.imageData.data, cd.w, cd.h)
    if (!(zr[0] === 0 && zr[1] === 1)) {
      return point
    }
    const nw = imgEl.naturalWidth, nh = imgEl.naturalHeight
    const [rx0, ry0, rx1, ry1] = [zr[2], zr[3], zr[4], zr[5]]
    const padX = Math.max(0.08, rx1 - rx0) * 0.6
    const padY = Math.max(0.08, ry1 - ry0) * 0.8
    const cx0 = Math.max(0, Math.round((rx0 - padX) * nw))
    const cy0 = Math.max(0, Math.round((ry0 - padY) * nh))
    const cx1 = Math.min(nw, Math.round((rx1 + padX) * nw))
    const cy1 = Math.min(nh, Math.round((ry1 + padY) * nh))
    const cw = cx1 - cx0, ch = cy1 - cy0
    if (cw < 48 || ch < 48) return point
    const cscale = Math.min(4, 256 / Math.max(cw, ch))
    const zw = Math.max(1, Math.round(cw * cscale)), zh = Math.max(1, Math.round(ch * cscale))
    const canvas = document.createElement('canvas')
    canvas.width = zw; canvas.height = zh
    const ctx = canvas.getContext('2d', { willReadFrequently: true })
    ctx.drawImage(imgEl, cx0, cy0, cw, ch, 0, 0, zw, zh)
    const zdata = ctx.getImageData(0, 0, zw, zh)
    const baseEv = wasm.point_ev(cd.imageData.data, cd.w, cd.h, point.x, point.y)
    const v = wasm.crop_verify(zdata.data, zw, zh, baseEv, 0.06, 0.16, 0.10)
    // Why a crop was declined: the panel is the only place the production
    // pixels exist (the harness downscale can propose a different region
    // entirely), so the verdict details have to be logged here.
    if (typeof wasm.crop_verify_debug === 'function') {
      const dbg = wasm.crop_verify_debug(zdata.data, zw, zh)
      const [pairs, skinCy, peakEyeY] = [dbg[0], dbg[1], dbg[2]]
      const pairYs = [dbg[3], dbg[4], dbg[5]].filter(v => v >= 0)
      const evOk = v[3] >= 0.25 && v[3] >= baseEv + 0.10
      const coApplies = peakEyeY <= 0.50
      const flOk = pairYs.some(py => py < skinCy - 0.06)
      const coOk = !coApplies || pairYs.some(py => Math.abs(py - peakEyeY) <= 0.16)
    }
    if (v[0] !== 1) return point
    return { x: (cx0 + v[1] * cw) / nw, y: (cy0 + v[2] * ch) / nh }
  } catch (e) {
    console.warn('[AutoFocus] zoom pass failed, using base point:', e)
    return point
  }
}

/**
 * Detect the scalar image features alongside the focus point.
 *
 * Same pixels, same pipeline as detectFocusDebug — this exists so the panel
 * can persist what the analysis computes (the image_features table) instead
 * of discarding everything but the point. Returns the parsed payload
 * `{x, y, category, avg_skin, avg_sat, avg_sharp, symmetry, edge_energy,
 * ahash}` or null when WASM is unavailable.
 */
export async function detectFeatures(imgEl, maxDim = 256) {
  const cd = getCanvasData(imgEl, maxDim)
  if (!cd) return null

  const wasm = await _wasmReady
  if (!wasm || typeof wasm.detect_features !== 'function') return null

  try {
    return JSON.parse(wasm.detect_features(cd.imageData.data, cd.w, cd.h))
  } catch (e) {
    console.warn('[AutoFocus] detect_features failed:', e)
    return null
  }
}

/**
 * Detect the focus point of an image element.
 * Returns `{ point: { x, y } }` in [0,1] coords, or null when WASM is
 * unavailable or the element cannot be read.
 */
export async function detectFocus(imgEl, maxDim = 256) {
  const cd = getCanvasData(imgEl, maxDim)
  if (!cd) return null

  const wasm = await _wasmReady
  let point = wasmDetect(wasm, cd.imageData, cd.w, cd.h)
  if (point && wasm) point = zoomRefine(wasm, imgEl, cd, point)
  return point ? { point } : null
}
