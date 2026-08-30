//! Focus-point pipeline: block scoring + softmax centroid, then the
//! ordered rule pipeline (rules.rs).

use crate::features::Features;
use crate::signals::*;
use crate::weights::{Weights, SKIN_DETAIL_BIAS, SKIN_EDGE_NORM};
use crate::rules::*;

pub(crate) const SOFTMAX_TEMPERATURE: f32 = 8.0;

/// One scored block — the input the rules read alongside BlockMeta.
pub(crate) struct Candidate {
    pub(crate) cx: f32,
    pub(crate) cy: f32,
    pub(crate) score: f32,
    pub(crate) eye_band: f32,
    pub(crate) skin_vert_bias: f32,
    pub(crate) skin_density: f32,
    #[allow(dead_code)]
    pub(crate) face_boost: f32,
}

/// Score every block and softmax the scores into the initial point.
fn score_blocks(feat: &Features, w: usize, h: usize, block_size: usize, weights: &Weights, is_mono: bool)
    -> (Vec<BlockMeta>, Vec<Candidate>, f32, f32)
{
    let half_w = w as f32 / 2.0;
    let half_h = h as f32 / 2.0;

    let bg = compute_bg_flags(feat, w, h, block_size);
    let n_cols = bg.n_cols;
    let n_rows = bg.n_rows;

    let mut max_score = f32::NEG_INFINITY;
    let mut blocks: Vec<BlockMeta> = Vec::with_capacity(n_cols * n_rows);

    let mut candidates: Vec<Candidate> = Vec::with_capacity(n_cols * n_rows);

    for row in 0..n_rows {
        let by = row * block_size;
        let bh = block_size.min(h - by);
        for col in 0..n_cols {
            let bx = col * block_size;
            let bw = block_size.min(w - bx);
            let pixels = (bw * bh) as f32;
            let idx = row * n_cols + col;
            let is_bg   = bg.bg_block[idx]   == 1;
            let is_geom = bg.geom_block[idx] == 1;

            let mut skin_sum = 0.0_f32; let mut mag_sum = 0.0_f32;
            let mut sat_sum  = 0.0_f32; let mut sharp_sum = 0.0_f32;
            let mut sharp_sum_sq = 0.0_f32;
            for sy in 0..bh {
                for sx in 0..bw {
                    let i = (by + sy) * w + (bx + sx);
                    skin_sum  += feat.skin[i];
                    sat_sum   += feat.sat[i];
                    if feat.mag[i] > 0.04 { mag_sum += feat.mag[i]; }
                    sharp_sum    += feat.sharp[i];
                    sharp_sum_sq += feat.sharp[i] * feat.sharp[i];
                }
            }

            let cx = bx as f32 + bw as f32 / 2.0;
            let cy = by as f32 + bh as f32 / 2.0;
            let dx = (cx - half_w).abs() / half_w / 2.0;
            let dy = (cy - half_h).abs() / half_h / 2.0;
            let centre_factor = 1.0 - (dx + dy);
            let avg_mag = mag_sum / pixels;

            let skin_density = if is_bg { 0.0 } else { skin_sum / pixels };
            let sat_score    = if is_bg { 0.0 } else { sat_sum  / pixels };
            let radial       = if is_bg { 0.0 } else { block_radial_peak(feat, w, bx, by, bw, bh) };
            let eye_band_raw = if is_bg || is_geom { 0.0 } else { eye_band_score(feat, w, bx, by, bw, bh, is_mono) };
            // Mono has no skin signal, so its eye-band fires on any horizontal
            // texture (ceiling trim, foliage, lace). Demand radial-symmetry
            // confirmation: real eyes are dark blobs (FRST peaks), texture
            // attractors are elongated edges whose votes smear. Applied at the
            // source so the strip search and refine pools see it too.
            let eye_band     = if is_mono {
                eye_band_raw * (0.35 + 0.65 * (radial / 0.60).min(1.0))
            } else {
                eye_band_raw
            };
            let face_boost   = if is_bg || is_geom { 0.0 } else { face_priority_boost(feat, w, h, bx, by, bw, bh) };
            let sym          = if is_mono || skin_density > 0.1 { symmetry_score(feat, w, h, cx, block_size) } else { 0.0 };

            let sharp_mean = sharp_sum / pixels;
            let sharp_var  = (sharp_sum_sq / pixels - sharp_mean * sharp_mean).max(0.0);

            let skin_vert_bias = if w as f32 / h as f32 > 1.6 {
                1.0
            } else {
                (1.0 - (cy / h as f32) * 0.85).max(0.15)
            };
            let skin_edge_gate = ((avg_mag + SKIN_DETAIL_BIAS) / SKIN_EDGE_NORM).min(1.0);
            let thirds_score = thirds((0.5 - cx / w as f32).abs() * 2.0)
                             + thirds((0.5 - cy / h as f32).abs() * 2.0);

            // Edge-column penalty: blocks whose centre is within the outermost 11%
            // of frame width are often background/border noise.
            let w_f = w as f32;
            let edge_penalty = if cx / w_f < 0.11 || cx / w_f > 0.89 { 0.78_f32 } else { 1.0_f32 };

            let score = (
                avg_mag                                      * weights.detail   +
                skin_density * skin_edge_gate * skin_vert_bias * weights.skin   +
                sat_score    * skin_edge_gate                * weights.sat      +
                sharp_var    * 4.0                           * weights.sharp    +
                centre_factor                                * weights.centre   +
                eye_band     * skin_vert_bias                * weights.hog_face +
                sym                                          * weights.symmetry +
                face_boost                                   * weights.hog_face +
                thirds_score                                 * weights.thirds
            ) * edge_penalty;

            if score > max_score { max_score = score; }
            blocks.push(BlockMeta { cx, cy, bw: bw as f32, bh: bh as f32, eye_band, skin_density, face_boost, radial, bg_suppressed: is_bg });
            candidates.push(Candidate { cx, cy, score, eye_band, skin_vert_bias, skin_density, face_boost });
        }
    }

    // Soft-max weighted centroid
    let skip_threshold = 20.0 / SOFTMAX_TEMPERATURE;
    let (mut sum_wt, mut sum_x, mut sum_y) = (0.0_f32, 0.0_f32, 0.0_f32);
    for c in &candidates {
        let delta = c.score - max_score;
        if delta < -skip_threshold { continue; }
        let wt = (delta * SOFTMAX_TEMPERATURE).exp();
        sum_wt += wt; sum_x += wt * c.cx; sum_y += wt * c.cy;
    }
    let x = sum_x / sum_wt / w as f32;
    let y = sum_y / sum_wt / h as f32;
    (blocks, candidates, x, y)
}

/// The ordered rule pipeline. Order matters: each rule refines the point
/// produced by the previous ones, and later rules read state recorded by
/// earlier ones (EyeRefine -> DriftBudget, FaceLock -> SaliencySnap).
pub(crate) fn compute_segments_inner(feat: &Features, w: usize, h: usize, block_size: usize, weights: &Weights, is_mono: bool, snap_saliency: bool, color_snap: bool)
    -> (f32, f32, Vec<BlockMeta>)
{
    let scored = score_frame(feat, w, h, block_size, weights, is_mono, snap_saliency, color_snap);
    let (x, y, _) = run_rules(&scored, LockMode::Normal);
    (x, y, scored.blocks)
}

/// Frame scored once: blocks, candidates, pairs and the post-strip point.
/// The lock-mode search runs the cheap rule tail over this several times
/// under different LockModes, so the expensive block pass happens only once.
pub(crate) struct ScoredFrame<'a> {
    pub(crate) feat: &'a Features,
    pub(crate) w: usize,
    pub(crate) h: usize,
    pub(crate) block_size: usize,
    pub(crate) snap_saliency: bool,
    pub(crate) color_snap: bool,
    pub(crate) blocks: Vec<BlockMeta>,
    pub(crate) candidates: Vec<Candidate>,
    pub(crate) pairs: Vec<crate::zoom::EyePair>,
    /// Point after StripFaceBlend + PortraitUpwardBias.
    pub(crate) x: f32,
    pub(crate) y: f32,
    /// Set when AF_RAW asked to stop early (native experiment hook).
    pub(crate) raw_stop: bool,
}

pub(crate) fn score_frame<'a>(feat: &'a Features, w: usize, h: usize, block_size: usize, weights: &Weights, is_mono: bool, snap_saliency: bool, color_snap: bool) -> ScoredFrame<'a> {
    let (blocks, candidates, x, y) = score_blocks(feat, w, h, block_size, weights, is_mono);
    let pairs = crate::zoom::frame_eye_pairs(feat, w, h);
    let mut sf = ScoredFrame {
        feat, w, h, block_size, snap_saliency, color_snap,
        blocks, candidates, pairs, x, y, raw_stop: false,
    };
    let ctx = RuleCtx {
        feat: sf.feat, w: sf.w, h: sf.h, block_size: sf.block_size,
        snap_saliency: sf.snap_saliency, color_snap: sf.color_snap,
        blocks: &sf.blocks, candidates: &sf.candidates, pairs: &sf.pairs,
        lock_mode: LockMode::Normal,
    };
    let mut st = PointState { x, y, refined: (x, y), face: None, face_clamp: None, lock_skipped: false };
    StripFaceBlend.apply(&ctx, &mut st);
    PortraitUpwardBias.apply(&ctx, &mut st);
    dbg_focus!("softmax point: {:.3},{:.3}", st.x, st.y);
    sf.x = st.x;
    sf.y = st.y;
    // Experiment hook (native only): AF_RAW=1 stops after softmax+strip to
    // measure how much the downstream rules help or hurt on a corpus.
    #[cfg(not(target_arch = "wasm32"))]
    if std::env::var_os("AF_RAW").is_some() { sf.raw_stop = true; }
    sf
}

/// Run the ordered rule tail. Order matters: each rule refines the point
/// produced by the previous ones, and later rules read state recorded by
/// earlier ones (EyeRefine -> DriftBudget, FaceLock -> SaliencySnap).
/// Returns the point plus the inferred face rect (normalised) when one exists.
pub(crate) fn run_rules(sf: &ScoredFrame, lock_mode: LockMode) -> (f32, f32, Option<(f32, f32, f32, f32)>) {
    let snap = |v: f32| (v * 100.0).round() / 100.0;
    if sf.raw_stop {
        return (sf.x.max(0.0).min(1.0), sf.y.max(0.0).min(1.0), None);
    }
    let ctx = RuleCtx {
        feat: sf.feat, w: sf.w, h: sf.h, block_size: sf.block_size,
        snap_saliency: sf.snap_saliency, color_snap: sf.color_snap,
        blocks: &sf.blocks, candidates: &sf.candidates, pairs: &sf.pairs,
        lock_mode,
    };
    let mut st = PointState { x: sf.x, y: sf.y, refined: (sf.x, sf.y), face: None, face_clamp: None, lock_skipped: false };
    let rules: [&dyn FocusRule; 5] = [&EyeRefine, &SubjectMassBlend, &FaceLock, &DriftBudget, &SaliencySnap];
    for rule in rules {
        rule.apply(&ctx, &mut st);
        dbg_focus!("after {}: {:.3},{:.3}", rule.name(), st.x, st.y);
    }
    let rect = st.face.as_ref().map(|f| {
        let (rx, ry, rw, rh) = f.rect;
        (rx / sf.w as f32, ry / sf.h as f32, rw / sf.w as f32, rh / sf.h as f32)
    });
    (snap(st.x).max(0.05).min(0.95), snap(st.y).max(0.08).min(0.92), rect)
}
