/*!
 * Auto-focus detection engine (Rust). The WASM build powers browser
 * consumers (via the @modufolio/autofocus wrapper); the native build
 * powers the CLI.
 */

use wasm_bindgen::prelude::*;

#[macro_use]
pub mod trace;
mod classify;
mod face_region;
mod features;
mod refine;
pub(crate) mod rules;
mod saliency;
pub(crate) mod segments;
mod signals;
mod weights;
mod zoom;

use classify::{color_snap_gate, pick_smart_weights};
use features::analyse_pixels;
// Only debug_radial_peaks uses these, and it is native-only.
#[cfg(not(target_arch = "wasm32"))]
use features::{build_luminance_map, compute_radial_symmetry};
use saliency::saliency_top_peaks;
use segments::compute_segments_inner;

pub use zoom::{crop_face_evidence, pairs_with_mouth, point_evidence, radial_eye_pair_ys, radial_eye_pairs};
use zoom::topmost_skin_bbox;

// ---------------------------------------------------------------------------
//  Build version — bump on every algorithm change so the browser console can
//  prove which build is running (keep AF_BUILD_VERSION in analysis.js equal).
// ---------------------------------------------------------------------------
pub const AF_BUILD_VERSION: &str = "2026-08-31-snap-face-veto";

/// Exposed so browser consumers can log which build is loaded:
/// `[AutoFocus] Rust/WASM backend loaded (build …)`.
#[wasm_bindgen]
pub fn build_version() -> String {
    AF_BUILD_VERSION.to_string()
}

/// Parity-debug export: per-block metrics + category, for diffing the two
/// implementations on identical pixels. Flat array:
/// [n_cols, n_rows, then per block: skin, eye_band, face_boost, radial],
/// with category appended via last_trace.
#[wasm_bindgen]
pub fn debug_blocks(rgba_data: &[u8], width: u32, height: u32) -> Vec<f32> {
    let w = width as usize;
    let h = height as usize;
    let feat = analyse_pixels(rgba_data, w, h);
    let block_size = 8_usize.max((w.min(h) / 18).max(1));
    let (category, weights) = pick_smart_weights(&feat, w, h, block_size);
    let is_mono = category == "mono";
    let (_, _, blocks) = compute_segments_inner(&feat, w, h, block_size, &weights, is_mono, false, false);
    let mut out = vec![blocks.len() as f32];
    // encode category as a number: mono=0 portrait=1 group=2 people=3 scene=4
    out.push(match category { "mono" => 0.0, "portrait" => 1.0, "group" => 2.0, "people" => 3.0, _ => 4.0 });
    for b in &blocks {
        out.push(b.skin_density); out.push(b.eye_band); out.push(b.face_boost); out.push(b.radial);
    }
    out
}

/// Stage-by-stage trace of the most recent detect_focus call (WASM only).
/// Browser consumers log this to the console so pipeline stages can be
/// debugged without the CLI.
#[wasm_bindgen]
#[cfg(target_arch = "wasm32")]
pub fn last_trace() -> String {
    trace::TRACE.with(|t| t.borrow().clone())
}

// ---------------------------------------------------------------------------
//  Public WASM entry point
// ---------------------------------------------------------------------------

/// Detect the focus point of an image.
///
/// `rgba_data` — raw RGBA bytes (Uint8Array from `canvas.getImageData`)
/// `width`, `height` — image dimensions in pixels
///
/// Returns a flat `[x, y]` Float32Array with normalised coordinates in [0, 1].
#[wasm_bindgen]
pub fn detect_focus(rgba_data: &[u8], width: u32, height: u32) -> Vec<f32> {
    #[cfg(target_arch = "wasm32")]
    trace::TRACE.with(|t| t.borrow_mut().clear());

    let w = width  as usize;
    let h = height as usize;

    let feat       = analyse_pixels(rgba_data, w, h);
    let block_size = 8_usize.max((w.min(h) / 18).max(1));
    let (category, weights) = pick_smart_weights(&feat, w, h, block_size);
    let is_mono = category == "mono";
    let snap_saliency = is_mono;
    let (x, y, _)  = compute_segments_inner(&feat, w, h, block_size, &weights, is_mono, snap_saliency, color_snap_gate(category));

    vec![x, y]
}

/// Detect the focus point AND the scalar image features, as a JSON string.
///
/// Same pipeline as `detect_focus` — the point in the payload is identical to
/// what `detect_focus` returns for the same pixels. The difference is that the
/// per-pixel maps the analysis already computed are aggregated instead of
/// discarded, so a consumer can persist them alongside the focus point.
///
/// JSON rather than a positional Float32Array: this payload is stored, and a
/// stored format with named keys can grow without silently re-meaning index 7.
#[wasm_bindgen]
pub fn detect_features(rgba_data: &[u8], width: u32, height: u32) -> String {
    let w = width  as usize;
    let h = height as usize;

    let feat       = analyse_pixels(rgba_data, w, h);
    let block_size = 8_usize.max((w.min(h) / 18).max(1));
    let (category, weights) = pick_smart_weights(&feat, w, h, block_size);
    let is_mono = category == "mono";
    let snap_saliency = is_mono;
    let (x, y, _) = compute_segments_inner(&feat, w, h, block_size, &weights, is_mono, snap_saliency, color_snap_gate(category));

    let n = (w * h) as f32;
    let mean = |v: &[f32]| v.iter().sum::<f32>() / n;

    format!(
        concat!(
            "{{\"x\":{:.4},\"y\":{:.4},\"category\":\"{}\",",
            "\"avg_skin\":{:.4},\"avg_sat\":{:.4},\"avg_sharp\":{:.4},",
            "\"symmetry\":{:.4},\"edge_energy\":{:.4},\"ahash\":\"{}\"}}"
        ),
        x, y, category,
        mean(&feat.skin), mean(&feat.sat), mean(&feat.sharp),
        mean(&feat.radial), mean(&feat.mag),
        average_hash_hex(&feat.lum, w, h),
    )
}

/// 8x8 average hash of the luminance map, as 16 hex chars.
///
/// Comparable via Hamming distance with any aHash built the same way
/// (downsample, threshold on the mean, row-major bits).
fn average_hash_hex(lum: &[f32], w: usize, h: usize) -> String {
    const SIDE: usize = 8;

    let mut cells = [0.0_f32; SIDE * SIDE];
    for cy in 0..SIDE {
        for cx in 0..SIDE {
            // Average the source region each cell covers, so the hash sees
            // the whole image rather than 64 sampled points.
            let x0 = cx * w / SIDE;
            let x1 = ((cx + 1) * w / SIDE).max(x0 + 1).min(w);
            let y0 = cy * h / SIDE;
            let y1 = ((cy + 1) * h / SIDE).max(y0 + 1).min(h);

            let mut sum = 0.0_f32;
            for yy in y0..y1 {
                for xx in x0..x1 {
                    sum += lum[yy * w + xx];
                }
            }
            cells[cy * SIDE + cx] = sum / ((x1 - x0) * (y1 - y0)) as f32;
        }
    }

    let mean: f32 = cells.iter().sum::<f32>() / (SIDE * SIDE) as f32;

    let mut bits: u64 = 0;
    for (i, &v) in cells.iter().enumerate() {
        if v > mean {
            bits |= 1 << (63 - i);
        }
    }

    format!("{:016x}", bits)
}

/// Native-only entry point — same algorithm, also returns the image category.
///
/// Use this from the CLI binary instead of `detect_focus` to avoid
/// pulling `#[wasm_bindgen]` into a native build.
#[cfg(not(target_arch = "wasm32"))]
pub fn detect_focus_cli(rgba_data: &[u8], width: u32, height: u32) -> (f32, f32, &'static str) {
    let w = width  as usize;
    let h = height as usize;

    let feat       = analyse_pixels(rgba_data, w, h);
    let block_size = 8_usize.max((w.min(h) / 18).max(1));
    let (category, weights) = pick_smart_weights(&feat, w, h, block_size);
    let is_mono = category == "mono";
    let snap_saliency = is_mono;
    let (x, y, _) = compute_segments_inner(&feat, w, h, block_size, &weights, is_mono, snap_saliency, color_snap_gate(category));

    (x, y, category)
}

/// Debug helper (native only): top-N local maxima of the radial map, as
/// (x_norm, y_norm, value). Not part of the detection pipeline.
#[cfg(not(target_arch = "wasm32"))]
pub fn debug_radial_peaks(rgba: &[u8], width: u32, height: u32, n: usize) -> Vec<(f32, f32, f32)> {
    let (w, h) = (width as usize, height as usize);
    let lum = build_luminance_map(rgba, w * h);
    let rad = compute_radial_symmetry(&lum, w, h);
    let mut peaks = Vec::new();
    for y in 2..h - 2 {
        for x in 2..w - 2 {
            let i = y * w + x;
            let v = rad[i];
            if v < 0.15 { continue; }
            let is_max = (-2..=2_isize).all(|dy| (-2..=2_isize).all(|dx| {
                (dx == 0 && dy == 0) || rad[(y as isize + dy) as usize * w + (x as isize + dx) as usize] <= v
            }));
            if is_max { peaks.push((x as f32 / w as f32, y as f32 / h as f32, v)); }
        }
    }
    peaks.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    peaks.truncate(n);
    peaks
}

// ---------------------------------------------------------------------------
//  Zoom pass (native prototype, AF_ZOOM=1) — see FINDINGS Rounds 8/9: at
//  256 px full-frame a small face's pupils are ~1 px (invisible to FRST)
//  while fabric dots are pupil-sized. Re-analysing a crop of the top skin
//  region at crop-256px puts real pupils in FRST's 2-4 px band and blows
//  fabric dots out of it, making the existing signals trustworthy.
// ---------------------------------------------------------------------------

/// Base pass + per-block (cx_norm, cy_norm, skin, eye_band) for region
/// proposals. Native only.
#[cfg(not(target_arch = "wasm32"))]
pub fn detect_focus_cli_blocks(
    rgba_data: &[u8], width: u32, height: u32,
) -> (f32, f32, &'static str, Vec<(f32, f32, f32, f32)>) {
    let w = width as usize;
    let h = height as usize;
    let feat = analyse_pixels(rgba_data, w, h);
    let block_size = 8_usize.max((w.min(h) / 18).max(1));
    let (category, weights) = pick_smart_weights(&feat, w, h, block_size);
    let is_mono = category == "mono";
    let (x, y, blocks) = compute_segments_inner(&feat, w, h, block_size, &weights, is_mono, is_mono, color_snap_gate(category));
    let out = blocks.iter()
        .map(|b| (b.cx / w as f32, b.cy / h as f32, b.skin_density, b.eye_band))
        .collect();
    (x, y, category, out)
}

/// WASM pass-1 helper: [is_mono, has_region, rx0, ry0, rx1, ry1].
#[wasm_bindgen]
pub fn zoom_region(rgba_data: &[u8], width: u32, height: u32) -> Vec<f32> {
    let w = width as usize;
    let h = height as usize;
    let feat = analyse_pixels(rgba_data, w, h);
    let block_size = 8_usize.max((w.min(h) / 18).max(1));
    let (category, weights) = pick_smart_weights(&feat, w, h, block_size);
    let is_mono = category == "mono";
    let (_, _, blocks) = compute_segments_inner(&feat, w, h, block_size, &weights, is_mono, false, false);
    match (is_mono, topmost_skin_bbox(&blocks, w, h)) {
        (false, Some((a, b, c, d))) => vec![0.0, 1.0, a, b, c, d],
        (m, _) => vec![if m { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0, 0.0, 0.0],
    }
}

/// WASM pass-2 verifier: [accepted(0/1), zx, zy, evidence]. Gates baked:
/// evidence floor 0.25, face_like FL=0.06, brow co-location CO=0.16
/// (keep in sync with the CLI defaults in main.rs).
#[wasm_bindgen]
pub fn crop_verify(rgba_data: &[u8], width: u32, height: u32, base_ev: f32, fl: f32, co: f32, margin: f32) -> Vec<f32> {
    let (evidence, zx, zy, skin_cy, peak_eye_y) = crop_face_evidence(rgba_data, width, height);
    let pair_ys = radial_eye_pair_ys(rgba_data, width, height);
    // Brow co-location (CO): the qualifying eye pair must sit near the crop's
    // strongest skin-gated eye-band block. The assumption ("that peak IS the
    // eyes") does break sometimes — one portrait fixture has real pairs at
    // y 0.15-0.17 with peak_eye_y 0.64 on the lips, so a correct head crop
    // is rejected. Relaxing CO to upper-half peaks only was tried and
    // REVERTED: it bought that one image (portraits 0.1093 -> 0.1086) and
    // cost own-work 0.1669 -> 0.1711 and ImageNet 0.2124 -> 0.2167
    // (6W/21L). CO is
    // load-bearing specificity — a relaxation needs POSITIVE evidence that a
    // crop is a face, not the mere absence of the co-location signal.
    let face_like = pair_ys.iter().any(|&py| py < skin_cy - fl && (py - peak_eye_y).abs() <= co);
    let ok = face_like && evidence >= 0.25 && evidence >= base_ev + margin;
    vec![if ok { 1.0 } else { 0.0 }, zx, zy, evidence]
}

/// Verifier v2 (experiment): crop_verify plus the mouth-band
/// structure gate — the qualifying eye pair must also carry a dark
/// horizontal-edge band one spacing below (pairs_with_mouth). Same
/// return shape as crop_verify.
#[wasm_bindgen]
pub fn crop_verify2(rgba_data: &[u8], width: u32, height: u32, base_ev: f32, fl: f32, co: f32, margin: f32) -> Vec<f32> {
    let (evidence, zx, zy, skin_cy, peak_eye_y) = crop_face_evidence(rgba_data, width, height);
    let pairs = pairs_with_mouth(rgba_data, width, height);
    let face_like = pairs.iter().any(|p| p.cy < skin_cy - fl && (p.cy - peak_eye_y).abs() <= co);
    let ok = face_like && evidence >= 0.25 && evidence >= base_ev + margin;
    vec![if ok { 1.0 } else { 0.0 }, zx, zy, evidence]
}

/// Debug variant of crop_verify: [pairs, skin_cy, peak_eye_y, py0, py1, py2].
#[wasm_bindgen]
pub fn crop_verify_debug(rgba_data: &[u8], width: u32, height: u32) -> Vec<f32> {
    let (_, _, _, skin_cy, peak_eye_y) = crop_face_evidence(rgba_data, width, height);
    let pair_ys = radial_eye_pair_ys(rgba_data, width, height);
    let mut out = vec![pair_ys.len() as f32, skin_cy, peak_eye_y];
    for i in 0..3 { out.push(*pair_ys.get(i).unwrap_or(&-1.0)); }
    out
}

/// Anchor-verify-by-zoom candidates.
///
/// Returns the three points the caller chooses between after verifying the
/// face anchor at crop scale, plus the anchor rect to crop:
/// `[has_anchor, lx, ly, nx, ny, ux, uy, rx, ry, rw, rh]`
/// * `l*` — shipped behaviour (lock with its displacement caps)
/// * `n*` — anchor REJECTED: lock and drift budget skipped entirely
/// * `u*` — anchor VERIFIED: lock walks its full blend, no caps
/// The rect is normalised. Block scoring runs once; only the cheap rule
/// tail repeats, so this costs barely more than detect_focus.
#[wasm_bindgen]
pub fn anchor_candidates(rgba_data: &[u8], width: u32, height: u32) -> Vec<f32> {
    let w = width as usize;
    let h = height as usize;
    let feat = analyse_pixels(rgba_data, w, h);
    let block_size = 8_usize.max((w.min(h) / 18).max(1));
    let (category, weights) = pick_smart_weights(&feat, w, h, block_size);
    let is_mono = category == "mono";
    let scored = segments::score_frame(&feat, w, h, block_size, &weights, is_mono, is_mono, color_snap_gate(category));
    let (lx, ly, rect) = segments::run_rules(&scored, rules::LockMode::Normal);
    let (nx, ny, _)    = segments::run_rules(&scored, rules::LockMode::Skip);
    let (ux, uy, _)    = segments::run_rules(&scored, rules::LockMode::Uncapped);
    match rect {
        Some((rx, ry, rw, rh)) => vec![1.0, lx, ly, nx, ny, ux, uy, rx, ry, rw, rh],
        None => vec![0.0, lx, ly, nx, ny, ux, uy, 0.0, 0.0, 0.0, 0.0],
    }
}

/// Zoom-proposal candidates from saliency: up to 3 interior
/// peaks as [n, x0, y0, x1, y1, x2, y2]. Colour images only — measured:
/// the mono verifier accepts exactly the fake structure saliency
/// proposes there. The caller crops around each peak and runs the same
/// crop_verify gates as the skin proposal.
#[wasm_bindgen]
pub fn saliency_zoom_peaks(rgba_data: &[u8], width: u32, height: u32) -> Vec<f32> {
    let w = width as usize;
    let h = height as usize;
    let feat = analyse_pixels(rgba_data, w, h);
    // Same mono gate as pick_smart_weights: low average saturation.
    let avg_sat: f32 = feat.sat.iter().sum::<f32>() / (w * h) as f32;
    if avg_sat < 0.05 { return vec![0.0]; }
    let peaks = saliency_top_peaks(&feat.lum, w, h, 3);
    let mut out = vec![peaks.len() as f32];
    for (x, y, _) in peaks { out.push(x); out.push(y); }
    out
}

/// WASM wrapper for the base-point evidence used by the zoom gate.
#[wasm_bindgen]
pub fn point_ev(rgba_data: &[u8], width: u32, height: u32, px: f32, py: f32) -> f32 {
    point_evidence(rgba_data, width, height, px, py)
}
