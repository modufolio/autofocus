//! Face-region inference: cluster face-quality blocks around the
//! current point, pick the anchor, derive the eye target and the face
//! rect. Consumed by the FaceLock / DriftBudget / SaliencySnap rules.

use crate::signals::{BlockMeta, pupil_like};
use crate::zoom::EyePair;

/// True when one of the pair's member peaks lies inside the block.
pub(crate) fn block_has_pair_member(b: &BlockMeta, pairs: &[EyePair]) -> bool {
    pairs.iter().any(|p| {
        let inside = |(mx, my): (f32, f32)| {
            (mx - b.cx).abs() <= b.bw * 0.5 && (my - b.cy).abs() <= b.bh * 0.5
        };
        inside(p.m1) || inside(p.m2)
    })
}

pub(crate) struct FaceRegion {
    pub(crate) face_blocks: Vec<BlockMeta>,
    pub(crate) anchor: Option<BlockMeta>,
    pub(crate) anchor_cx: f32,
    pub(crate) anchor_cy: f32,
    pub(crate) eye_target_cx: f32,
    pub(crate) eye_target_cy: f32,
    pub(crate) eye_band_ceiling: f32,
    /// Face rect in pixels: (x, y, w, h).
    pub(crate) rect: (f32, f32, f32, f32),
    /// Eye-target pool size (trace only).
    pub(crate) pool_len: usize,
    pub(crate) pool_trace: Vec<BlockMeta>,
    /// Pupil-like face blocks confirmed by a mouth-gated eye pair.
    pub(crate) paired_pupil_count: usize,
}

impl FaceRegion {
    pub(crate) fn pupil_count(&self) -> usize {
        self.face_blocks.iter().filter(|b| pupil_like(b)).count()
    }
    pub(crate) fn anchor_eye(&self) -> f32 {
        self.anchor.map(|a| a.eye_band).unwrap_or(0.0)
    }
}

pub(crate) fn eff_eye_band(b: &BlockMeta, w: usize, h: usize) -> f32 {
    let svb = if w as f32 / h as f32 > 1.6 {
        1.0
    } else {
        (1.0 - (b.cy / h as f32) * 0.85).max(0.15)
    };
    (b.eye_band) * svb
}

pub(crate) fn face_signal(b: &BlockMeta, w: usize, h: usize) -> f32 {
    eff_eye_band(b, w, h) * 0.60 + (b.face_boost) * 0.30 + (b.skin_density) * 0.10
}

pub(crate) fn upper_face_bias(b: &BlockMeta, h: usize) -> f32 {
    (1.18 - (b.cy / h as f32) * 0.45).max(0.72)
}

pub(crate) fn infer_face_region(blocks: &[BlockMeta], w: usize, h: usize, block_size: usize, x: f32, y: f32, pairs: &[EyePair]) -> FaceRegion {

    // Build face anchor candidates
    let point_px = x * w as f32;
    let point_py = y * h as f32;
    let cluster_distance = block_size as f32 * 1.75;
    let mut face_anchor_candidates = Vec::new();
    for b in blocks {
        let signal = face_signal(b, w, h);
        let dist_to_point = ((b.cx - point_px).powi(2) + (b.cy - point_py).powi(2)).sqrt();
        let point_bias = if point_px > 0.0 || point_py > 0.0 {
            (1.0 - dist_to_point / (block_size as f32 * 10.0)).max(0.82)
        } else { 1.0 };
        let eye_anchor_signal = eff_eye_band(b, w, h) * 0.72 + (b.face_boost) * 0.18 + (b.skin_density) * 0.10;
        let anchor_score = eye_anchor_signal * point_bias * upper_face_bias(b, h);
        if signal >= 0.08 && b.skin_density >= 0.08 {
            face_anchor_candidates.push((b, signal, dist_to_point, eye_anchor_signal, anchor_score));
        }
    }

    // Cluster face anchor candidates
    let mut clusters: Vec<Vec<&BlockMeta>> = Vec::new();
    let mut visited = vec![false; face_anchor_candidates.len()];
    for i in 0..face_anchor_candidates.len() {
        if visited[i] { continue; }
        let mut queue = vec![i];
        visited[i] = true;
        let mut cluster = Vec::new();
        while let Some(idx) = queue.pop() {
            let (b, _, _, _, _) = face_anchor_candidates[idx];
            cluster.push(b);
            for (j, (b2, _, _, _, _)) in face_anchor_candidates.iter().enumerate() {
                if visited[j] { continue; }
                let dist = ((b2.cx - b.cx).powi(2) + (b2.cy - b.cy).powi(2)).sqrt();
                if dist <= cluster_distance {
                    visited[j] = true;
                    queue.push(j);
                }
            }
        }
        clusters.push(cluster);
    }

    // Score clusters
    let score_cluster = |cluster: &Vec<&BlockMeta>| {
        let min_cx = cluster.iter().map(|b| b.cx).fold(f32::MAX, f32::min);
        let max_cx = cluster.iter().map(|b| b.cx).fold(f32::MIN, f32::max);
        let min_cy = cluster.iter().map(|b| b.cy).fold(f32::MAX, f32::min);
        let max_cy = cluster.iter().map(|b| b.cy).fold(f32::MIN, f32::max);
        let span_x = (max_cx - min_cx).max(block_size as f32);
        let span_y = (max_cy - min_cy).max(block_size as f32);
        let elongation = (span_x.max(span_y) / span_x.min(span_y).max(1.0)).max(1.0);
        let avg_y = cluster.iter().map(|b| b.cy).sum::<f32>() / cluster.len() as f32;
        let avg_point_dist = cluster.iter().map(|b| ((b.cx - point_px).powi(2) + (b.cy - point_py).powi(2)).sqrt()).sum::<f32>() / cluster.len() as f32;
        let signal_sum = cluster.iter().map(|b| eff_eye_band(b, w, h) * 0.72 + (b.face_boost) * 0.18 + (b.skin_density) * 0.10).sum::<f32>();
        let strong_eye_count = cluster.iter().filter(|b| b.eye_band >= 0.14).count();
        // Confirmed pupil blocks are the strongest per-cluster face evidence
        // (V326188: the real face carries radial 1.00 + skin but eye-band
        // 0.00, and lost the anchor to a glittery belt scoring eye 0.47).
        let cluster_pupil_blocks: Vec<&&BlockMeta> = cluster.iter().filter(|b| b.radial >= 0.60 && b.skin_density >= 0.20 && b.face_boost >= 0.45).collect();
        // Pair geometry: a pupil block backed by a mouth-gated eye
        // pair is near-certain face evidence; a lone one is usually texture
        // (hair curls, hay dapples, glitter) and gets a token bonus only.
        let paired_pupils = cluster_pupil_blocks.iter().filter(|b| block_has_pair_member(b, pairs)).count();
        let lone_pupils = cluster_pupil_blocks.len() - paired_pupils;
        // Hard shape veto: a face is never a long thin horizontal bar low in
        // the frame (wl_koala: a skin-toned log across the bottom formed a
        // 6x1-block "face"). Soft penalties lose to signal mass; this cannot.
        // Weak-skin bars are vetoed at ANY height: brass railings and warm
        // trim read as skin ~0.3 with textured "eyes" on one fixture, while a real
        // face lineup — the reason the veto was bottom-only — carries dense
        // skin and keeps its protection.
        let avg_skin = cluster.iter().map(|b| b.skin_density).sum::<f32>() / cluster.len() as f32;
        if span_x / span_y >= 3.5 && (avg_y / h as f32 >= 0.62 || avg_skin <= 0.40) {
            return 0.0;
        }
        let compactness = (1.45 - (elongation - 1.0) * 0.55).max(0.45);
        // Steeper top-of-frame prior: among SKIN clusters the topmost is almost
        // always the face — hands, midriff and legs sit below it (V326188:
        // the belt cluster at y 0.37 out-scored the face at y 0.10 on the
        // old nearly-flat slope).
        let upper_bias = (1.35 - (avg_y / h as f32) * 0.90).max(0.55);
        let point_bias = if point_px > 0.0 || point_py > 0.0 {
            (1.0 - avg_point_dist / (block_size as f32 * 9.0)).max(0.60)
        } else { 1.0 };
        // Lone-pupil demotion targets the SINGLE fake (hair curl, hay
        // dapple — texture never fakes two co-located pupils): 2+ lone
        // pupil blocks keep near-full authority (one fixture: a real face with 4
        // blur-unpaired pupils must not lose its cluster to decor).
        let lone_bonus = if lone_pupils >= 2 { 0.45 } else { 0.15 };
        signal_sum * compactness * upper_bias * point_bias
            * (1.0 + strong_eye_count as f32 * 0.18)
            * (1.0 + paired_pupils as f32 * 0.60 + lone_pupils as f32 * lone_bonus)
    };

    let selected_cluster = clusters.iter().max_by(|a, b| score_cluster(a).partial_cmp(&score_cluster(b)).unwrap()).cloned();
    let candidate_pool: Vec<&BlockMeta> = if let Some(ref cluster) = selected_cluster {
        if !cluster.is_empty() { cluster.clone() } else { face_anchor_candidates.iter().map(|(b, _, _, _, _)| *b).collect() }
    } else {
        face_anchor_candidates.iter().map(|(b, _, _, _, _)| *b).collect()
    };

    // Find face anchor
    let face_anchor = candidate_pool.iter().max_by(|a, b| {
        let a_score = eff_eye_band(a, w, h) * 0.72 + (a.face_boost) * 0.18 + (a.skin_density) * 0.10;
        let a_point_bias = if point_px > 0.0 || point_py > 0.0 {
            (1.0 - (((a.cx - point_px).powi(2) + (a.cy - point_py).powi(2)).sqrt() / (block_size as f32 * 10.0))).max(0.82)
        } else { 1.0 };
        let a_anchor_score = a_score * a_point_bias * upper_face_bias(a, h);
        let b_score = eff_eye_band(b, w, h) * 0.72 + (b.face_boost) * 0.18 + (b.skin_density) * 0.10;
        let b_point_bias = if point_px > 0.0 || point_py > 0.0 {
            (1.0 - (((b.cx - point_px).powi(2) + (b.cy - point_py).powi(2)).sqrt() / (block_size as f32 * 10.0))).max(0.82)
        } else { 1.0 };
        let b_anchor_score = b_score * b_point_bias * upper_face_bias(b, h);
        a_anchor_score.partial_cmp(&b_anchor_score).unwrap()
    });

    // Face blocks
    let face_anchor_ref = face_anchor.map(|b| *b);
    let face_blocks: Vec<&BlockMeta> = if let Some(anchor) = face_anchor_ref {
        candidate_pool.iter().filter(|b|
            face_signal(b, w, h) >= face_signal(anchor, w, h).max(0.08) * 0.55 &&
            ((b.cx - anchor.cx).powi(2) + (b.cy - anchor.cy).powi(2)).sqrt() <= block_size as f32 * 6.0
        ).map(|b| *b).collect()
    } else {
        Vec::new()
    };

    // Eye anchor blocks
    let eye_anchor_blocks: Vec<&BlockMeta> = face_blocks.iter().filter(|b| b.eye_band >= 0.10 || pupil_like(b)).map(|b| *b).collect();
    let anchor_pool: Vec<&BlockMeta> = if eye_anchor_blocks.len() >= 3 { eye_anchor_blocks } else { face_blocks.clone() };
    let anchor_weight = |b: &BlockMeta| {
        let eye_weight = (eff_eye_band(b, w, h) * 0.72 + (b.face_boost) * 0.12 + (b.skin_density) * 0.06).max(0.02);
        eye_weight * eye_weight
    };
    let anchor_weight_sum: f32 = anchor_pool.iter().map(|b| anchor_weight(b)).sum();
    let anchor_cx = if anchor_weight_sum > 0.0 {
        anchor_pool.iter().map(|b| b.cx * anchor_weight(b)).sum::<f32>() / anchor_weight_sum
    } else if let Some(anchor) = face_anchor_ref {
        anchor.cx
    } else {
        0.5 * w as f32
    };
    let anchor_cy = if anchor_weight_sum > 0.0 {
        anchor_pool.iter().map(|b| b.cy * anchor_weight(b)).sum::<f32>() / anchor_weight_sum
    } else if let Some(anchor) = face_anchor_ref {
        anchor.cy
    } else {
        0.5 * h as f32
    };

    // Eye target pool
    let min_bx = face_blocks.iter().map(|b| b.cx - b.bw / 2.0).fold(f32::MAX, f32::min);
    let min_by = face_blocks.iter().map(|b| b.cy - b.bh / 2.0).fold(f32::MAX, f32::min);
    let max_br = face_blocks.iter().map(|b| b.cx + b.bw / 2.0).fold(f32::MIN, f32::max);
    let max_bb = face_blocks.iter().map(|b| b.cy + b.bh / 2.0).fold(f32::MIN, f32::max);
    let face_span_y = (max_bb - min_by).max(block_size as f32);
    let eye_band_ceiling = min_by + face_span_y * 0.52;
    // A strong radial-symmetry peak (pupil-like blob) overrides the vertical
    // ceiling: when a hat or hair pushes the face rect upward, the ceiling can
    // land above the eyes and exclude them from the pool.
    let eye_target_pool: Vec<&BlockMeta> = anchor_pool.iter().filter(|b|
        (b.cy <= eye_band_ceiling || pupil_like(b)) &&
        (b.eye_band >= (face_anchor_ref.map(|a| a.eye_band).unwrap_or(0.0) * 0.45).max(0.12) || pupil_like(b))
    ).map(|b| *b).collect();
    let eye_target_source: Vec<&BlockMeta> = if eye_target_pool.len() >= 2 { eye_target_pool } else { anchor_pool.clone() };
    let eye_target_weight = |b: &BlockMeta| {
        let eye_weight = (eff_eye_band(b, w, h) * 0.82 + (b.face_boost) * 0.12 + (b.skin_density) * 0.06
                          + if pupil_like(b) { b.radial * 0.45 } else { 0.0 }).max(0.04);
        eye_weight * eye_weight
    };
    let eye_target_weight_sum: f32 = eye_target_source.iter().map(|b| eye_target_weight(b)).sum();
    let eye_target_cx = if eye_target_weight_sum > 0.0 {
        eye_target_source.iter().map(|b| b.cx * eye_target_weight(b)).sum::<f32>() / eye_target_weight_sum
    } else {
        anchor_cx
    };
    let eye_target_cy = if eye_target_weight_sum > 0.0 {
        eye_target_source.iter().map(|b| b.cy * eye_target_weight(b)).sum::<f32>() / eye_target_weight_sum
    } else {
        anchor_cy.min(min_by + face_span_y * 0.42)
    };

    // Face rect
    let pad = block_size as f32 * 0.5;
    let rx = (min_bx - pad).max(0.0);
    let ry = (min_by - pad).max(0.0);
    let rw = (max_br - min_bx + pad * 2.0).min(w as f32 - rx);
    let rh = (max_bb - min_by + pad * 2.0).min(h as f32 - ry);

    let paired_pupil_count = face_blocks.iter()
        .filter(|b| pupil_like(b) && block_has_pair_member(b, pairs))
        .count();

    FaceRegion {
        paired_pupil_count,
        face_blocks: face_blocks.iter().map(|b| **b).collect(),
        anchor: face_anchor_ref.map(|b| *b),
        anchor_cx,
        anchor_cy,
        eye_target_cx,
        eye_target_cy,
        eye_band_ceiling,
        rect: (rx, ry, rw, rh),
        pool_len: eye_target_source.len(),
        pool_trace: eye_target_source.iter().map(|b| **b).collect(),
    }
}
