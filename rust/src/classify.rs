//! Image-category classification (portrait / group / people / scene /
//! mono) and the weight-preset + saliency-snap gates keyed off it.

use crate::features::Features;
use crate::signals::{BlockMeta, compute_bg_flags, eye_band_score, face_priority_boost, block_radial_peak};
use crate::weights::Weights;

pub(crate) fn classify_image(blocks: &[BlockMeta], skin: &[f32], w: usize, h: usize) -> &'static str {
    let total_pixels = (w * h) as f32;
    let skin_density: f32 = skin.iter().sum::<f32>() / total_pixels;

    let active: Vec<&BlockMeta> = blocks.iter().filter(|b| !b.bg_suppressed).collect();
    if active.is_empty() { return "scene"; }

    let peak = active.iter().max_by(|a, b| a.eye_band.partial_cmp(&b.eye_band).unwrap()).unwrap();

    let peak_skin_eye_band = active.iter()
        .filter(|b| b.skin_density >= 0.10)
        .map(|b| b.eye_band)
        .fold(0.0_f32, f32::max);

    if peak_skin_eye_band < 0.12 {
        return if skin_density > 0.14 { "people" } else { "scene" };
    }

    let approx_block_size = (peak.bw + peak.bh) / 2.0;
    let cluster_radius    = approx_block_size * 4.5;

    let primary: Vec<&&BlockMeta> = active.iter().filter(|b| {
        b.eye_band >= 0.10 && b.skin_density >= 0.10 && b.face_boost >= 0.1 &&
        ((b.cx - peak.cx).powi(2) + (b.cy - peak.cy).powi(2)).sqrt() <= cluster_radius
    }).collect();

    let separate: Vec<&&BlockMeta> = active.iter().filter(|b| {
        if b.eye_band < 0.10 || b.skin_density < 0.10 || b.face_boost < 0.1 { return false; }
        let min_dist = if primary.is_empty() {
            ((b.cx - peak.cx).powi(2) + (b.cy - peak.cy).powi(2)).sqrt()
        } else {
            primary.iter()
                .map(|p| ((b.cx - p.cx).powi(2) + (b.cy - p.cy).powi(2)).sqrt())
                .fold(f32::MAX, f32::min)
        };
        min_dist > cluster_radius * 1.8
    }).collect();

    let primary_x_ctr = if primary.is_empty() { peak.cx }
        else { primary.iter().map(|b| b.cx).sum::<f32>() / primary.len() as f32 };
    let separate_x_ctr = if separate.is_empty() { 0.0 }
        else { separate.iter().map(|b| b.cx).sum::<f32>() / separate.len() as f32 };
    let x_separation = (separate_x_ctr - primary_x_ctr).abs();
    let primary_to_sep_ratio = if separate.is_empty() {
        f32::INFINITY
    } else {
        primary.len() as f32 / separate.len() as f32
    };
    let is_real_group = separate.len() >= 3
        && x_separation >= cluster_radius
        && primary_to_sep_ratio >= 0.40;

    // Separate cluster coherence: reject scattered noise mistaken for a second face
    let separate_x_spread = if separate.len() > 1 {
        let sep_cx = separate_x_ctr;
        separate.iter().map(|b| (b.cx - sep_cx).abs()).fold(0.0_f32, f32::max)
    } else {
        0.0
    };
    let separate_is_coherent = separate.len() <= 1
        || separate_x_spread < cluster_radius * 1.8;

    let primary_avg_eye_band = if primary.is_empty() { 0.0 }
        else { primary.iter().map(|b| b.eye_band).sum::<f32>() / primary.len() as f32 };
    let primary_strong = primary.iter().filter(|b| b.eye_band >= 0.16).count();

    let isolated_portrait_face =
        primary.len() >= 1 &&
        peak_skin_eye_band >= 0.30 &&
        peak.skin_density >= 0.22 &&
        peak.face_boost >= 0.55 &&
        (separate.len() <= 1 || !separate_is_coherent);

    let compact_portrait_face =
        !is_real_group &&
        primary.len() >= 2 &&
        primary_strong >= 2 &&
        primary_avg_eye_band >= 0.24 &&
        peak_skin_eye_band >= 0.30 &&
        peak.skin_density >= 0.18 &&
        peak.face_boost >= 0.55;

    let is_portrait = primary.len() >= 4
        || (primary.len() >= 3 && primary_strong >= 2 && primary_avg_eye_band >= 0.20)
        || (peak_skin_eye_band >= 0.24 && skin_density >= 0.18 && primary_strong >= 1 && separate.len() <= 1)
        || isolated_portrait_face
        || compact_portrait_face;

    if is_real_group  { return "group"; }
    if is_portrait    { return "portrait"; }
    if skin_density > 0.06 { "people" } else { "scene" }
}

// ---------------------------------------------------------------------------
//  Eye focus refinement
// ---------------------------------------------------------------------------


/// Colour-image saliency snap gate (variant under evaluation).
/// "scene" = variant A; widen to any colour category for
/// variant B — compute_segments_inner additionally requires
/// pupil_count == 0 before snapping. The mono snap is unconditional.
/// AF_COLOR_SNAP (native): off | scene | nopupil.
pub(crate) fn color_snap_gate(category: &str) -> bool {
    #[cfg(not(target_arch = "wasm32"))]
    {
        match std::env::var("AF_COLOR_SNAP").as_deref() {
            Ok("off") => return false,
            Ok("nopupil") => return category != "mono",
            Ok("scene") => return category == "scene",
            _ => {}
        }
    }
    category != "mono" && category != "people"
}

pub(crate) fn pick_smart_weights(feat: &Features, w: usize, h: usize, block_size: usize) -> (&'static str, Weights) {
    // Check for monochrome image (low average saturation)
    let total_pixels = (w * h) as f32;
    let avg_sat: f32 = feat.sat.iter().sum::<f32>() / total_pixels;
    if avg_sat < 0.05 {
        return ("mono", Weights::mono());
    }

    // Run a lightweight classify pass using the block metadata
    let bg = compute_bg_flags(feat, w, h, block_size);
    let n_cols = bg.n_cols;
    let n_rows = bg.n_rows;

    let mut blocks: Vec<BlockMeta> = Vec::with_capacity(n_cols * n_rows);
    for row in 0..n_rows {
        let by = row * block_size;
        let bh = block_size.min(h - by);
        for col in 0..n_cols {
            let bx = col * block_size;
            let bw = block_size.min(w - bx);
            let idx  = row * n_cols + col;
            let is_bg = bg.bg_block[idx] == 1;
            let is_geom = bg.geom_block[idx] == 1;
            let pixels = (bw * bh) as f32;
            let skin_sum: f32 = if is_bg { 0.0 } else {
                (0..bh).flat_map(|sy| (0..bw).map(move |sx| (by + sy) * w + (bx + sx)))
                    .map(|i| feat.skin[i]).sum()
            };
            let skin_density = skin_sum / pixels;
            let eye_band = if is_bg || is_geom { 0.0 } else { eye_band_score(feat, w, bx, by, bw, bh, false) };
            let face_boost = if is_bg || is_geom { 0.0 } else { face_priority_boost(feat, w, h, bx, by, bw, bh) };
            let cx = bx as f32 + bw as f32 / 2.0;
            let cy = by as f32 + bh as f32 / 2.0;
            let radial = if is_bg { 0.0 } else { block_radial_peak(feat, w, bx, by, bw, bh) };
            blocks.push(BlockMeta { cx, cy, bw: bw as f32, bh: bh as f32, eye_band, skin_density, face_boost, radial, bg_suppressed: is_bg });
        }
    }

    let category = classify_image(&blocks, &feat.skin, w, h);
    let weights = match category {
        "portrait" => Weights::portrait(),
        "group"    => Weights::group(),
        "people"   => Weights::people(),
        _          => Weights::scene(),
    };
    (category, weights)
}
