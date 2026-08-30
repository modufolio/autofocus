//! Eye-focus refinement: pull the softmax point toward the strongest
//! nearby eye-band block.

use crate::features::Features;
use crate::signals::BlockMeta;

pub(crate) fn refine_eye_focus_point(
    feat: &Features, w: usize, h: usize,
    blocks: &[BlockMeta], point: (f32, f32), block_size: usize,
) -> (f32, f32) {
    let (px, py) = (point.0 * w as f32, point.1 * h as f32);

    let eye_blocks: Vec<&BlockMeta> = blocks.iter().filter(|b| {
        b.eye_band >= 0.12 && b.skin_density >= 0.12 &&
        ((b.cx - px).powi(2) + (b.cy - py).powi(2)).sqrt() <= block_size as f32 * 7.0
    }).collect();

    if eye_blocks.len() < 3 { return point; }

    let peak_eye_band  = eye_blocks.iter().map(|b| b.eye_band).fold(0.0_f32, f32::max);
    let avg_skin       = eye_blocks.iter().map(|b| b.skin_density).sum::<f32>() / eye_blocks.len() as f32;
    if peak_eye_band < 0.16 || avg_skin < 0.22 { return point; }

    let min_bx = eye_blocks.iter().map(|b| (b.cx - b.bw / 2.0) as usize).min().unwrap_or(0);
    let min_by = eye_blocks.iter().map(|b| (b.cy - b.bh / 2.0) as usize).min().unwrap_or(0);
    let max_bx = eye_blocks.iter().map(|b| (b.cx + b.bw / 2.0) as usize).max().unwrap_or(w);
    let max_by = eye_blocks.iter().map(|b| (b.cy + b.bh / 2.0) as usize).max().unwrap_or(h);

    let min_bx = min_bx.saturating_sub(block_size).min(w);
    let min_by = min_by.saturating_sub(block_size / 2).min(h);
    let max_bx = (max_bx + block_size).min(w);
    let max_by = (max_by + block_size).min(h);

    let region_cx = (min_bx + max_bx) as f32 / 2.0;
    let region_w  = (max_bx - min_bx).max(1) as f32;
    let region_h  = (max_by - min_by).max(1) as f32;

    let mut best_score = 0.0_f32;
    let mut best_x = px;
    let mut best_y = py;

    for iy in min_by..max_by {
        for ix in min_bx..max_bx {
            let i = iy * w + ix;
            let darkness  = ((0.58 - feat.lum[i]) / 0.58).max(0.0).min(1.0);
            let edge       = (feat.mag[i] / 0.22).max(0.0).min(1.0);
            let crisp      = (feat.sharp[i] / 0.16).max(0.0).min(1.0);
            let x_bias     = (1.0 - (ix as f32 - region_cx).abs() / (region_w * 0.55)).max(0.0);
            let y_rel      = (iy - min_by) as f32 / region_h;
            let y_bias     = (1.0 - (y_rel - 0.38).abs() / 0.42).max(0.0);
            let point_bias = (1.0 - ((ix as f32 - px).powi(2) + (iy as f32 - py).powi(2)).sqrt()
                              / (block_size as f32 * 4.0)).max(0.35);
            let radial     = (feat.radial[i] * 2.2).min(1.0);
            let score = darkness * (0.40 + edge * 0.28 + crisp * 0.14 + radial * 0.45)
                      * (0.40 + x_bias * 0.20 + y_bias * 0.25 + point_bias * 0.15);
            if score > best_score { best_score = score; best_x = ix as f32; best_y = iy as f32; }
        }
    }

    if best_score < 0.22 { return point; }

    // Clamp downward drift: don't allow refine to push below point + 1.2
    // blocks — unless the winner sits on a strong radial-symmetry peak
    // (a pupil-like blob), which earns it a longer leash. This lets the
    // point escape a hat brim or hairline block sitting above the eyes.
    let best_radial = feat.radial[best_y as usize * w + best_x as usize];
    let max_down_px = if best_radial >= 0.30 {
        block_size as f32 * 3.0
    } else {
        block_size as f32 * 1.2
    };
    let clamped_best_y = best_y.min(py + max_down_px);

    (point.0 * 0.25 + (best_x / w as f32) * 0.75,
     point.1 * 0.25 + (clamped_best_y / h as f32) * 0.75)
}

// ---------------------------------------------------------------------------
