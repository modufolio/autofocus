//! Zoom-pass helpers: crop evidence scoring, base-point evidence,
//! pixel-level radial eye pairs and the topmost-skin region proposal.

use crate::features::{Features, analyse_pixels, build_luminance_map, compute_radial_symmetry};
use crate::signals::BlockMeta;
use crate::weights::Weights;
use crate::segments::compute_segments_inner;

/// Analyse a crop and report how face-like it is at this scale.
/// Returns (evidence, x, y) with x/y normalised WITHIN the crop.
/// Evidence = strongest skin-gated eye-band block, plus a bonus per
/// confirmed pupil block (radial + skin + face gradients co-located).
pub fn crop_face_evidence(rgba_data: &[u8], width: u32, height: u32) -> (f32, f32, f32, f32, f32) {
    let w = width as usize;
    let h = height as usize;
    let feat = analyse_pixels(rgba_data, w, h);
    let block_size = 8_usize.max((w.min(h) / 18).max(1));
    let (x, y, blocks) = compute_segments_inner(&feat, w, h, block_size, &Weights::portrait(), false, false, false);
    let mut peak_eye = 0.0_f32;
    let mut peak_eye_y = 0.5_f32;
    for b in blocks.iter().filter(|b| b.skin_density >= 0.15) {
        if b.eye_band > peak_eye { peak_eye = b.eye_band; peak_eye_y = b.cy / h as f32; }
    }
    let pupils = blocks.iter()
        .filter(|b| b.radial >= 0.60 && b.skin_density >= 0.20 && b.face_boost >= 0.45)
        .count();
    let evidence = peak_eye + (pupils.min(4) as f32) * 0.10;
    // Skin-weighted vertical centroid: faces carry their eye pair ABOVE
    // this line; hands/fur crops do not (verifier-specificity signal).
    let (mut sw, mut sy) = (0.0_f32, 0.0_f32);
    for b in &blocks {
        if b.skin_density >= 0.20 { sw += b.skin_density; sy += b.skin_density * b.cy; }
    }
    let skin_cy = if sw > 0.0 { (sy / sw) / h as f32 } else { 0.5 };
    (evidence, x, y, skin_cy, peak_eye_y)
}

/// Face evidence in the neighbourhood of a point (same scorer as
/// crop_face_evidence, restricted to blocks within ~2.5 block sizes).
/// Used as the zoom gate's baseline: a crop must BEAT this, not merely
/// clear an absolute floor.
pub fn point_evidence(rgba_data: &[u8], width: u32, height: u32, px: f32, py: f32) -> f32 {
    let w = width as usize;
    let h = height as usize;
    let feat = analyse_pixels(rgba_data, w, h);
    let block_size = 8_usize.max((w.min(h) / 18).max(1));
    let (_, _, blocks) = compute_segments_inner(&feat, w, h, block_size, &Weights::portrait(), false, false, false);
    let (cx, cy) = (px * w as f32, py * h as f32);
    let r = block_size as f32 * 2.5;
    let near: Vec<&BlockMeta> = blocks.iter().filter(|b| ((b.cx - cx).powi(2) + (b.cy - cy).powi(2)).sqrt() <= r).collect();
    let peak_eye = near.iter().filter(|b| b.skin_density >= 0.15).map(|b| b.eye_band).fold(0.0_f32, f32::max);
    let pupils = near.iter().filter(|b| b.radial >= 0.60 && b.skin_density >= 0.20 && b.face_boost >= 0.45).count();
    peak_eye + (pupils.min(4) as f32) * 0.10
}

/// Count geometrically valid eye pairs among pixel-level radial peaks.
/// Applied ONLY at crop resolution, where real pupils are 2-4 px and
/// detectable; at full-frame 256px this test was measured and rejected
/// (sub-pixel pupils). A pupil pair: two comparable peaks,
/// roughly level (dy <= 0.35*dx), spacing 3px..15% of frame, and neither
/// peak has a comparable partner directly above/below (lattices do).
/// One geometrically valid eye pair (normalised midpoint + pixel spacing).
pub struct EyePair {
    pub cx: f32,
    pub cy: f32,
    pub spacing_px: f32,
    /// Member peaks in pixels (x1,y1) / (x2,y2).
    pub m1: (f32, f32),
    pub m2: (f32, f32),
}

/// Centres (normalised y) of valid eye pairs — see radial_eye_pairs.
pub fn radial_eye_pair_ys(rgba_data: &[u8], width: u32, height: u32) -> Vec<f32> {
    radial_eye_pairs_impl(rgba_data, width, height).iter().map(|p| p.cy).collect()
}

pub fn radial_eye_pairs(rgba_data: &[u8], width: u32, height: u32) -> usize {
    radial_eye_pairs_impl(rgba_data, width, height).len()
}

fn radial_eye_pairs_impl(rgba_data: &[u8], width: u32, height: u32) -> Vec<EyePair> {
    let w = width as usize;
    let h = height as usize;
    let lum = build_luminance_map(rgba_data, w * h);
    let rad = compute_radial_symmetry(&lum, w, h);
    eye_pairs_core(&rad, w, h)
}

/// Map-level pair finder shared by the crop verifier (radial_eye_pairs_impl)
/// and the full-frame pair-geometry gate (frame_eye_pairs). Identical
/// geometry to the shipped crop rule.
pub(crate) fn eye_pairs_core(rad: &[f32], w: usize, h: usize) -> Vec<EyePair> {
    // Amplitude-relative floor: bilinear (canvas) pixels flatten radial
    // peaks to roughly half the crisp-pixel amplitude, which silently
    // disabled the pair test on production crops (one reference crop:
    // evidence 0.804, face_like 0). Normalising the floor to the map's own
    // maximum makes
    // the test pixel-path invariant; the absolute minimum keeps flat
    // crops from producing noise pairs.
    let rmax = rad.iter().fold(0.0_f32, |a, &v| a.max(v));
    let floor = (0.30 * rmax).max(0.12);
    // local maxima, NMS 3px, top 40
    let mut peaks: Vec<(usize, usize, f32)> = Vec::new();
    for y in 2..h.saturating_sub(2) {
        for x in 2..w.saturating_sub(2) {
            let i = y * w + x;
            let v = rad[i];
            if v < floor { continue; }
            if v >= rad[i - 1] && v > rad[i + 1] && v >= rad[i - w] && v > rad[i + w] {
                peaks.push((x, y, v));
            }
        }
    }
    peaks.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    let mut kept: Vec<(usize, usize, f32)> = Vec::new();
    for &(x, y, v) in &peaks {
        if kept.len() >= 40 { break; }
        if kept.iter().all(|&(kx, ky, _)| {
            let (dx, dy) = (kx as i32 - x as i32, ky as i32 - y as i32);
            dx * dx + dy * dy >= 9
        }) { kept.push((x, y, v)); }
    }
    let max_spacing = (w.max(h) as f32) * 0.15;
    let mut pairs: Vec<EyePair> = Vec::new();
    for i in 0..kept.len() {
        for j in (i + 1)..kept.len() {
            let (xi, yi, vi) = kept[i];
            let (xj, yj, vj) = kept[j];
            if vj < vi * 0.45 { continue; }
            let dx = (xi as f32 - xj as f32).abs();
            let dy = (yi as f32 - yj as f32).abs();
            if !(dx >= 3.0 && dx <= max_spacing && dy <= (dx * 0.35).max(2.0)) { continue; }
            // lattice veto: a comparable peak directly above/below either member
            let vertical = kept.iter().any(|&(xk, yk, vk)| {
                if vk < vi * 0.45 { return false; }
                let near_i = (xk as f32 - xi as f32).abs() <= (dx * 0.35).max(2.0) && (yk as f32 - yi as f32).abs() >= 3.0 && (yk as f32 - yi as f32).abs() <= max_spacing;
                let near_j = (xk as f32 - xj as f32).abs() <= (dx * 0.35).max(2.0) && (yk as f32 - yj as f32).abs() >= 3.0 && (yk as f32 - yj as f32).abs() <= max_spacing;
                near_i || near_j
            });
            if !vertical {
                pairs.push(EyePair {
                    cx: ((xi + xj) as f32 / 2.0) / w as f32,
                    cy: ((yi + yj) as f32 / 2.0) / h as f32,
                    spacing_px: dx,
                    m1: (xi as f32, yi as f32),
                    m2: (xj as f32, yj as f32),
                });
            }
        }
    }
    pairs
}


/// Topmost connected skin cluster bbox (normalised) over BlockMeta —
/// shared by the CLI zoom flow and the WASM zoom_region export.
pub(crate) fn topmost_skin_bbox(blocks: &[BlockMeta], w: usize, h: usize) -> Option<(f32, f32, f32, f32)> {
    let skin: Vec<(f32, f32, f32)> = blocks.iter().filter(|b| b.skin_density >= 0.30)
        .map(|b| (b.cx / w as f32, b.cy / h as f32, b.eye_band)).collect();
    if skin.is_empty() { return None; }
    let seed = *skin.iter().min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())?;
    let mut members: Vec<(f32, f32, f32)> = vec![seed];
    let mut used = vec![false; skin.len()];
    loop {
        let mut grew = false;
        for (i, b) in skin.iter().enumerate() {
            if used[i] { continue; }
            if members.iter().any(|m| (b.0 - m.0).abs() <= 0.12 && (b.1 - m.1).abs() <= 0.10) {
                members.push(*b); used[i] = true; grew = true;
            }
        }
        if !grew { break; }
    }
    // Proposal-quality guards (iteration 12): a subject region has some
    // eye texture somewhere (a warm sky cluster: maxEye 0.00 vs a real
    // face cluster: 0.37) and is bigger than a single block
    // (gorilla's degenerate 1-block region).
    if members.len() < 3 { return None; }
    if !members.iter().any(|m| m.2 >= 0.08) { return None; }
    let x0 = members.iter().map(|m| m.0).fold(f32::MAX, f32::min);
    let y0 = members.iter().map(|m| m.1).fold(f32::MAX, f32::min);
    let x1 = members.iter().map(|m| m.0).fold(f32::MIN, f32::max);
    let y1 = members.iter().map(|m| m.1).fold(f32::MIN, f32::max);
    Some((x0, y0, x1, y1))
}

/// Verifier-v2 structure gate: a real face carries a MOUTH —
/// a dark horizontal-edge band roughly one eye-spacing below the pair
/// midpoint. Point arrays that fake pairs (chandeliers, penguin
/// colonies, fabric dots) have no such band. Returns the pairs that
/// carry mouth evidence.
pub fn pairs_with_mouth(rgba_data: &[u8], width: u32, height: u32) -> Vec<EyePair> {
    let w = width as usize;
    let h = height as usize;
    let lum = build_luminance_map(rgba_data, w * h);
    mouth_filter(&lum, w, h, radial_eye_pairs_impl(rgba_data, width, height))
}

/// Keep only the pairs that carry a mouth band (dark horizontal-edge
/// energy roughly one eye-spacing below the pair midpoint).
pub(crate) fn mouth_filter(lum: &[f32], w: usize, h: usize, pairs: Vec<EyePair>) -> Vec<EyePair> {
    // global horizontal-edge energy for normalisation
    let mut global_sum = 0.0f32;
    for y in 1..h {
        for x in 0..w {
            global_sum += (lum[y * w + x] - lum[(y - 1) * w + x]).abs();
        }
    }
    let global_mean = global_sum / ((h - 1) * w) as f32;

    pairs.into_iter().filter(|p| {
        let d = p.spacing_px;
        // mouth box: centered one spacing below the pair, one spacing wide,
        // 0.5 spacing tall
        let mx0 = ((p.cx * w as f32) - d * 0.5).max(0.0) as usize;
        let mx1 = (((p.cx * w as f32) + d * 0.5) as usize).min(w - 1);
        let my0 = ((p.cy * h as f32) + d * 0.75).max(1.0) as usize;
        let my1 = (((p.cy * h as f32) + d * 1.35) as usize).min(h - 1);
        if mx1 <= mx0 + 1 || my1 <= my0 + 1 { return false; }
        let mut sum = 0.0f32;
        let mut cnt = 0.0f32;
        for y in my0..=my1 {
            for x in mx0..=mx1 {
                sum += (lum[y * w + x] - lum[(y - 1) * w + x]).abs();
                cnt += 1.0;
            }
        }
        let mouth = sum / cnt;
        mouth >= global_mean * 1.15
    }).collect()
}

/// Full-frame pair geometry: mouth-gated eye pairs from the
/// already-computed feature maps. Full-frame pair geometry was measured
/// and rejected as a *scoring* signal because small faces have sub-pixel
/// pupils at 256px — here pairs only GATE authority (a lone pupil-like
/// block without a partner loses its anchor bonuses), so photos where
/// no pair resolves keep the ungated behaviour.
pub(crate) fn frame_eye_pairs(feat: &Features, w: usize, h: usize) -> Vec<EyePair> {
    mouth_filter(&feat.lum, w, h, eye_pairs_core(&feat.radial, w, h))
}
