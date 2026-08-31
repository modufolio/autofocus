//! The focus-point rule pipeline.
//!
//! After block scoring + softmax (segments.rs), the point walks through
//! an ordered list of rules. Each rule owns one behaviour, reads the
//! shared `RuleCtx`, and adjusts `PointState`. Order matters and is
//! fixed in segments::compute_segments_inner; every rule is a refinement
//! with its own gate — a rule that does not apply leaves the state
//! untouched.

use crate::face_region::{FaceRegion, infer_face_region, face_signal};
use crate::features::Features;
use crate::refine::refine_eye_focus_point;
use crate::saliency::saliency_peak;
use crate::segments::Candidate;
use crate::signals::{BlockMeta, pupil_like};
use crate::zoom::EyePair;

/// How much authority the FaceLock is granted this pass.
/// The caller runs the pipeline under several modes and picks between the
/// resulting points after verifying the anchor at crop scale.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum LockMode {
    /// Shipped behaviour: lock applies with its displacement caps.
    Normal,
    /// Anchor rejected: FaceLock and DriftBudget do not run at all.
    Skip,
    /// Anchor verified: FaceLock walks the full blend, no caps, no budget.
    Uncapped,
}

/// Read-only context shared by every rule.
pub(crate) struct RuleCtx<'a> {
    pub(crate) feat: &'a Features,
    pub(crate) w: usize,
    pub(crate) h: usize,
    pub(crate) block_size: usize,
    pub(crate) snap_saliency: bool,
    pub(crate) color_snap: bool,
    pub(crate) blocks: &'a [BlockMeta],
    pub(crate) candidates: &'a [Candidate],
    /// Full-frame mouth-gated eye pairs — gate pupil authority.
    pub(crate) pairs: &'a [EyePair],
    /// How much authority the face lock gets this pass.
    pub(crate) lock_mode: LockMode,
}

/// Mutable state the rules thread through the pipeline.
pub(crate) struct PointState {
    pub(crate) x: f32,
    pub(crate) y: f32,
    /// Point right after EyeRefine — the drift budget is measured from here.
    pub(crate) refined: (f32, f32),
    /// Face region inferred by FaceLock; later rules read pupil evidence.
    pub(crate) face: Option<FaceRegion>,
    /// Set when the face lock fires: the final point may never leave a
    /// pupil-confirmed face region.
    pub(crate) face_clamp: Option<(f32, f32, f32, f32)>,
    /// The anchor failed the plausibility gate (or LockMode::Skip),
    /// so the lock never moved the point and the drift budget has nothing
    /// to contain.
    pub(crate) lock_skipped: bool,
}

pub(crate) trait FocusRule {
    fn name(&self) -> &'static str;
    fn apply(&self, ctx: &RuleCtx, st: &mut PointState);
}

/// Strip-based face search: split the frame into 8 horizontal strips,
/// find the one with the strongest eye-band response, and blend the
/// point toward that strip's face centroid.
pub(crate) struct StripFaceBlend;

impl FocusRule for StripFaceBlend {
    fn name(&self) -> &'static str { "StripFaceBlend" }

    fn apply(&self, ctx: &RuleCtx, st: &mut PointState) {
        let (w, h) = (ctx.w, ctx.h);
        let candidates = ctx.candidates;
        let mut x = st.x;
        let mut y = st.y;
        const N_STRIPS: usize = 8;
        let mut strip_peak    = [0.0_f32; N_STRIPS];
        let mut strip_skin_pk = [0.0_f32; N_STRIPS];
        let mut strip_joint_pk = [0.0_f32; N_STRIPS];
    
        for c in candidates {
            let strip = ((c.cy / h as f32 * N_STRIPS as f32) as usize).min(N_STRIPS - 1);
            let eff = c.eye_band * c.skin_vert_bias;
            if eff > strip_peak[strip]    { strip_peak[strip]    = eff; }
            if c.skin_density > strip_skin_pk[strip] { strip_skin_pk[strip] = c.skin_density; }
            let joint = c.eye_band * c.skin_density;
            if joint > strip_joint_pk[strip] { strip_joint_pk[strip] = joint; }
        }
    
        let face_qual: Vec<&Candidate> = candidates.iter()
            .filter(|c| c.eye_band >= 0.10 && c.skin_density >= 0.10)
            .collect();
        let face_y_min = face_qual.iter().map(|c| c.cy).fold(f32::MAX, f32::min);
        let face_y_max = face_qual.iter().map(|c| c.cy).fold(f32::MIN, f32::max);
        let (face_y_min, face_y_max) = if face_qual.is_empty() { (0.0, h as f32) } else { (face_y_min, face_y_max) };
        let face_vert_span = (face_y_max - face_y_min) / h as f32;
        let face_top_frac  = face_y_min / h as f32;
        let is_full_body_top = face_vert_span < 0.20 && face_top_frac < 0.15;
    
        if (w as f32 / h as f32) < 0.8 && is_full_body_top && strip_skin_pk[0] >= 0.15 {
            strip_peak[0] *= 1.4;
        }
    
        let face_strip    = strip_peak.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).map(|(i,_)| i).unwrap_or(0);
        let face_strip_pk = strip_peak[face_strip];
    
        if face_strip_pk >= 0.05 {
            let sy0 = (face_strip as f32 / N_STRIPS as f32) * h as f32;
            let sy1 = ((face_strip + 1) as f32 / N_STRIPS as f32) * h as f32;
            let (mut fwt, mut fsx, mut fsy) = (0.0_f32, 0.0_f32, 0.0_f32);
            for c in candidates {
                if c.cy >= sy0 && c.cy < sy1 {
                    let eff = c.eye_band * c.skin_vert_bias;
                    if eff >= face_strip_pk * 0.5 {
                        let w2 = eff * eff;
                        fwt += w2; fsx += w2 * c.cx; fsy += w2 * c.cy;
                    }
                }
            }
            if fwt > 0.0 {
                let alpha = if face_strip == 0 && (w as f32 / h as f32) < 0.8 && is_full_body_top && strip_skin_pk[0] >= 0.15 { 0.15 } else { 0.40 };
                x = x * alpha + (fsx / fwt / w as f32) * (1.0 - alpha);
                y = y * alpha + (fsy / fwt / h as f32) * (1.0 - alpha);
            }
        }
        st.x = x;
        st.y = y;
    }
}

/// Portrait upward bias: on portrait-ish aspect ratios with significant
/// skin, nudge the point up toward the face and away from the body.
pub(crate) struct PortraitUpwardBias;

impl FocusRule for PortraitUpwardBias {
    fn name(&self) -> &'static str { "PortraitUpwardBias" }

    fn apply(&self, ctx: &RuleCtx, st: &mut PointState) {
        let (feat, w, h) = (ctx.feat, ctx.w, ctx.h);
        let mut y = st.y;
        if (w as f32 / h as f32) <= 1.6 {
            let skin_mean = feat.skin.iter().sum::<f32>() / (w * h) as f32;
            if skin_mean > 0.18 { y = y * 0.91 - 0.04; }
        }
        st.y = y;
    }
}

/// Eye-focus refinement: pull the point toward the strongest nearby
/// eye-band block. Also records the refined point that the drift budget
/// is measured from.
pub(crate) struct EyeRefine;

impl FocusRule for EyeRefine {
    fn name(&self) -> &'static str { "EyeRefine" }

    fn apply(&self, ctx: &RuleCtx, st: &mut PointState) {
        let (rx, ry) = refine_eye_focus_point(ctx.feat, ctx.w, ctx.h, ctx.blocks, (st.x, st.y), ctx.block_size);
        dbg_focus!("refined point: {:.3},{:.3}", rx, ry);
        st.x = rx;
        st.y = ry;
        st.refined = (rx, ry);
    }
}

/// Subject-mass blend: cluster skin/eye/face signal blocks, score the
/// clusters, and blend the point toward the best cluster's centroid —
/// displacement-capped so background texture cannot teleport the point.
pub(crate) struct SubjectMassBlend;

impl FocusRule for SubjectMassBlend {
    fn name(&self) -> &'static str { "SubjectMassBlend" }

    fn apply(&self, ctx: &RuleCtx, st: &mut PointState) {
        let (w, h, block_size) = (ctx.w, ctx.h, ctx.block_size);
        let blocks = ctx.blocks;
        let mut x = st.x;
        let mut y = st.y;
        // 1. Build mass candidates (signal = skin + eye + face)
        let point_px = x * w as f32;
        let point_py = y * h as f32;
        let cluster_distance = block_size as f32 * 1.6;
        let mut mass_candidates = Vec::new();
        for b in blocks {
            let skin_signal = (b.skin_density - 0.08).max(0.0) * 1.4;
            let eye_signal = b.eye_band.max(0.0) * 0.9;
            let face_signal = b.face_boost.max(0.0) * 0.45;
            let signal = skin_signal + eye_signal + face_signal;
            let dist_to_point = ((b.cx - point_px).powi(2) + (b.cy - point_py).powi(2)).sqrt();
            if signal >= 0.18 || (b.skin_density >= 0.12 && b.eye_band >= 0.08) {
                mass_candidates.push((b, signal, dist_to_point));
            }
        }
        // 2. Cluster mass candidates
        let mut clusters: Vec<Vec<&BlockMeta>> = Vec::new();
        let mut visited = vec![false; mass_candidates.len()];
        for i in 0..mass_candidates.len() {
            if visited[i] { continue; }
            let mut queue = vec![i];
            visited[i] = true;
            let mut cluster = Vec::new();
            while let Some(idx) = queue.pop() {
                let (b, _, _) = mass_candidates[idx];
                cluster.push(b);
                for (j, (b2, _, _)) in mass_candidates.iter().enumerate() {
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
        // 3. Score clusters
        let mut best_cluster = None;
        let mut best_score = 0.0_f32;
        for cluster in &clusters {
            if cluster.len() < 4 { continue; }
            let min_cx = cluster.iter().map(|b| b.cx).fold(f32::MAX, f32::min);
            let max_cx = cluster.iter().map(|b| b.cx).fold(f32::MIN, f32::max);
            let min_cy = cluster.iter().map(|b| b.cy).fold(f32::MAX, f32::min);
            let max_cy = cluster.iter().map(|b| b.cy).fold(f32::MIN, f32::max);
            let span_x = (max_cx - min_cx).max(block_size as f32);
            let span_y = (max_cy - min_cy).max(block_size as f32);
            let aspect = (span_x.max(span_y) / span_x.min(span_y).max(1.0)).max(1.0);
            let avg_dist = cluster.iter().map(|b| ((b.cx - point_px).powi(2) + (b.cy - point_py).powi(2)).sqrt()).sum::<f32>() / cluster.len() as f32;
            let avg_skin = cluster.iter().map(|b| b.skin_density).sum::<f32>() / cluster.len() as f32;
            let avg_eye = cluster.iter().map(|b| b.eye_band).sum::<f32>() / cluster.len() as f32;
            let avg_face = cluster.iter().map(|b| b.face_boost).sum::<f32>() / cluster.len() as f32;
            let signal_sum = cluster.iter().map(|b| {
                let skin_signal = (b.skin_density - 0.08).max(0.0) * 1.4;
                let eye_signal = b.eye_band.max(0.0) * 0.9;
                let face_signal = b.face_boost.max(0.0) * 0.45;
                skin_signal + eye_signal + face_signal
            }).sum::<f32>();
            let weight_sum = cluster.iter().map(|b| {
                let skin_signal = (b.skin_density - 0.08).max(0.0) * 1.4;
                let eye_signal = b.eye_band.max(0.0) * 0.9;
                let face_signal = b.face_boost.max(0.0) * 0.45;
                let s = skin_signal + eye_signal + face_signal;
                s * s
            }).sum::<f32>();
            let cx = if weight_sum > 0.0 {
                cluster.iter().map(|b| {
                    let skin_signal = (b.skin_density - 0.08).max(0.0) * 1.4;
                    let eye_signal = b.eye_band.max(0.0) * 0.9;
                    let face_signal = b.face_boost.max(0.0) * 0.45;
                    let s = skin_signal + eye_signal + face_signal;
                    b.cx * s * s
                }).sum::<f32>() / weight_sum
            } else {
                cluster.iter().map(|b| b.cx).sum::<f32>() / cluster.len() as f32
            };
            let cy = if weight_sum > 0.0 {
                cluster.iter().map(|b| {
                    let skin_signal = (b.skin_density - 0.08).max(0.0) * 1.4;
                    let eye_signal = b.eye_band.max(0.0) * 0.9;
                    let face_signal = b.face_boost.max(0.0) * 0.45;
                    let s = skin_signal + eye_signal + face_signal;
                    b.cy * s * s
                }).sum::<f32>() / weight_sum
            } else {
                cluster.iter().map(|b| b.cy).sum::<f32>() / cluster.len() as f32
            };
            let area_bias = (cluster.len() as f32).sqrt();
            let compactness = (1.35 - (aspect - 1.0) * 0.30).max(0.35);
            let point_bias = if point_px > 0.0 || point_py > 0.0 {
                (1.0 - avg_dist / (block_size as f32 * 10.0)).max(0.68)
            } else { 1.0 };
            let bar_penalty = if span_y <= block_size as f32 * 1.25 && span_x >= block_size as f32 * 3.0 { 0.28 } else { 1.0 };
            let subject_bias = (avg_skin * 1.9 + avg_eye * 1.3 + avg_face * 0.35).max(0.45);
            let score = signal_sum * area_bias * compactness * point_bias * bar_penalty * subject_bias;
            if score > best_score {
                best_score = score;
                best_cluster = Some((cx, cy, cluster.len(), span_x, span_y, avg_skin, avg_eye, avg_face));
            }
        }
        // 4. Blend with subject mass if strong enough
            if let Some((mcx, mcy, _block_count, _span_x, _span_y, _avg_skin, avg_eye, _avg_face)) = best_cluster {
            if best_score >= 1.5 {
                let subject_mass_x = mcx / w as f32;
                let subject_mass_y = mcy / h as f32;
                let dist = ((subject_mass_x - x).powi(2) + (subject_mass_y - y).powi(2)).sqrt();
                // Eye-evidence gate: a cluster worth re-aiming at should show
                // at least a trace of eye-band. Skin-toned bokeh, rocks and
                // brass score high on colour alone but keep avg_eye below
                // block-noise level (dress speckle reads ~0.015) — exactly the
                // false subjects the drift caps only mitigate. Binary, not
                // proportional: scaling the blend by avg_eye also weakens
                // legitimate soft/small faces (0.005-0.03 range), and
                // face_boost is no help — it is colour-driven and maxes out
                // on bokeh.
                //
                // Body-mass gate: a mass that sits far BELOW the point AND is
                // densely bare skin is a torso/legs, not a subject to re-aim
                // at — re-aiming walks the point off the face and down the
                // body (one fixture: refine landed on the face, the mass
                // centroid sat on the jeans 0.52 lower). Density separates it
                // from legitimate downward pulls, which all carry
                // avg_skin <= 0.58.
                let mass_drop = subject_mass_y - y;
                let body_mass = mass_drop > 0.25 && _avg_skin > 0.60;
                if dist >= 0.08 && avg_eye >= 0.005 && !body_mass {
                    let blend = (0.22 + dist * 0.85).max(0.28).min(0.62);
                    let nx = x * (1.0 - blend) + subject_mass_x * blend;
                    let ny = y * (1.0 - blend) + subject_mass_y * blend;
                    // Cap the displacement like the face-region lock does: the
                    // largest skin/eye mass can be background texture (rocks,
                    // brass, wood), and an uncapped blend rides most of the way
                    // to it no matter how far away it is.
                    let (dx, dy) = (nx - x, ny - y);
                    let drift = (dx * dx + dy * dy).sqrt();
                    let max_drift = 0.16_f32;
                    if drift > max_drift {
                        let scale = max_drift / drift;
                        x += dx * scale;
                        y += dy * scale;
                    } else {
                        x = nx;
                        y = ny;
                    }
                }
            }
        }
        st.x = x;
        st.y = y;
    }
}

/// Face-region lock: infer the face region around the current point and,
/// when the evidence is strong enough, walk the point to the eye target
/// inside the face rect — displacement-capped, and recording the clamp
/// bounds a pupil-confirmed region imposes on every later rule.
pub(crate) struct FaceLock;

impl FocusRule for FaceLock {
    fn name(&self) -> &'static str { "FaceLock" }

    fn apply(&self, ctx: &RuleCtx, st: &mut PointState) {
        let (w, h, block_size) = (ctx.w, ctx.h, ctx.block_size);
        let (x, y) = (st.x, st.y);
        let region = infer_face_region(ctx.blocks, w, h, block_size, x, y, ctx.pairs);
        // Skip: the anchor failed crop-scale verification, so the
        // region is still inferred (callers read its rect) but never moves
        // the point.
        if ctx.lock_mode == LockMode::Skip {
            st.face = Some(region);
            st.face_clamp = None;
            st.lock_skipped = true;
            return;
        }
        // Anchor plausibility gate. A face anchor sitting LOW in
        // the frame with weak eye-band evidence is body, fabric or bedding
        // that passed the skin model, not a face: skipping the lock (and the
        // budget) entirely beats letting it pull. Thresholds swept on
        // the tuning corpora and validated on the held-out sets (no
        // regression). Only under Normal — Uncapped means a
        // caller verified the anchor and is overriding.
        // NOTE: this replaced the planned
        // anchor-verify-by-zoom, which was measured and REJECTED — crop-scale
        // face tests are ANTI-correlated with anchor correctness here (body
        // texture at 4x manufactures more pupil pairs than soft real faces).
        if ctx.lock_mode == LockMode::Normal
            && region.anchor_cy / h as f32 > 0.40
            && region.anchor_eye() < 0.15
        {
            dbg_focus!("anchor gate: REJECT (cy {:.2} > 0.40, anchor_eye {:.2} < 0.15)",
                region.anchor_cy / h as f32, region.anchor_eye());
            st.face = Some(region);
            st.face_clamp = None;
            st.lock_skipped = true;
            return;
        }
        let (_, ry_, _, rh_) = region.rect;
        dbg_focus!("anchor: {:.1},{:.1}  eye_target: {:.1},{:.1}  ceiling: {:.1}  face_rect y:{:.1} h:{:.1}  face_blocks: {}  pool: {}",
            region.anchor_cx, region.anchor_cy, region.eye_target_cx, region.eye_target_cy, region.eye_band_ceiling, ry_, rh_, region.face_blocks.len(), region.pool_len);
        for b in &region.pool_trace {
            dbg_focus!("  pool block cx {:.0} cy {:.0} eye {:.2} radial {:.2} skin {:.2}", b.cx, b.cy, b.eye_band, b.radial, b.skin_density);
        }
        let face_blocks = &region.face_blocks;
        let face_anchor_ref = region.anchor.as_ref();
        let (eye_target_cx, eye_target_cy) = (region.eye_target_cx, region.eye_target_cy);
        let (rx, ry, rw, rh) = region.rect;
        let mut locked_x = st.x;
        let mut locked_y = st.y;
        // Set when the face lock fires; the final point is re-clamped into these
        // bounds AFTER the drift budget — once a face region is confirmed on a
        // colour image the point must never escape it (measured: budget
        // interplay left the point mid-body while the face rect sat on the
        // head).
        let mut face_clamp: Option<(f32, f32, f32, f32)> = None;
        // Eye-evidence gate (same principle as the subject-mass blend): a face
        // region in which not a single block shows eye-band or pupil evidence is
        // colour mass (skin-toned bokeh, wood, rocks), not a face — do not let
        // it pull the point.
        let face_has_eye = face_blocks.iter().any(|b| b.eye_band >= 0.10 || pupil_like(b));
        // Two or more confirmed pupil blocks mean the anchor IS a face — let
        // the lock walk further toward it (one 4-pupil fixture had its rescue
        // from the sweater clipped mid-chest by the flat 0.16 cap). Weak
        // anchors keep the short cap that stops false-subject teleports.
        let lock_pupils = face_blocks.iter().filter(|b| pupil_like(b)).count();
        // No !is_mono guard: pupil_like requires real skin + face_boost, which
        // are zero in true B&W — so mono protection is inherent. An explicit
        // guard only blocked desaturated COLOUR photos (measured: avg_sat
        // < 0.05 with skin 0.86).
        let lock_max_drift = if ctx.lock_mode == LockMode::Uncapped { f32::INFINITY }
            else if lock_pupils >= 2 { 0.30_f32 } else { 0.16_f32 };
        if face_has_eye && face_blocks.len() >= 4 && face_anchor_ref.map(|a| face_signal(a, w, h)).unwrap_or(0.0) >= 0.18 {
            let target_x = eye_target_cx / w as f32;
            let target_y = eye_target_cy / h as f32;
            let mut blended_x = x * 0.32 + target_x * 0.68;
            let mut blended_y = y * 0.22 + target_y * 0.78;
    
            // Cap the displacement so a bad anchor cannot yank the focus point
            // more than 0.16 normalised units from the incoming estimate.
            let max_drift = lock_max_drift;
            let drift_x = blended_x - x;
            let drift_y = blended_y - y;
            let drift   = (drift_x * drift_x + drift_y * drift_y).sqrt();
            if drift > max_drift {
                let scale  = max_drift / drift;
                blended_x  = x + drift_x * scale;
                blended_y  = y + drift_y * scale;
            }
    
            let face_rect_x = rx / w as f32;
            let face_rect_y = ry / h as f32;
            let face_rect_w = rw / w as f32;
            let face_rect_h = rh / h as f32;
            // The 0.52 cap assumes the face rect hugs the face; a hat or big hair
            // stretches the rect upward and pushes the cap above the real eyes.
            // Confirmed pupil blocks below the cap extend it to their row.
            let pupil_cap = face_blocks.iter().filter(|b| pupil_like(b))
                .map(|b| (b.cy + b.bh * 0.5) / h as f32)
                .fold(0.0_f32, f32::max);
            let y_cap = (face_rect_y + face_rect_h * 0.52).max(pupil_cap.min(face_rect_y + face_rect_h * 0.80));
            locked_x = blended_x.max(face_rect_x + face_rect_w * 0.12).min(face_rect_x + face_rect_w * 0.88);
            locked_y = blended_y.max(face_rect_y + face_rect_h * 0.06).min(y_cap);
            // Only a pupil-confirmed region is inescapable: eye-band-only
            // rects can sit on dress prints or wood grain, and hard-clamping
            // into those regressed the own-work and wedding sets. The pupil
            // must be PAIRED — a lone hair-curl/hay-dapple "pupil" no longer
            // makes its region a prison.
            if lock_pupils >= 2 || region.paired_pupil_count >= 1 {
                face_clamp = Some((
                    face_rect_x + face_rect_w * 0.12,
                    face_rect_x + face_rect_w * 0.88,
                    face_rect_y + face_rect_h * 0.06,
                    y_cap,
                ));
            }
    
            // The rect clamp above can override the max_drift cap: when the
            // face rect sits far from the incoming point (a false face in
            // skin-toned rocks, brass, wood), clamping into it teleports the
            // point across the frame. Re-apply the cap as a hard invariant —
            // a face region that far from the evidence doesn't own the point.
            let drift_x = locked_x - x;
            let drift_y = locked_y - y;
            let drift   = (drift_x * drift_x + drift_y * drift_y).sqrt();
            if drift > max_drift {
                let scale = max_drift / drift;
                locked_x = x + drift_x * scale;
                locked_y = y + drift_y * scale;
            }
        }
        st.x = locked_x;
        st.y = locked_y;
        st.face_clamp = face_clamp;
        st.face = Some(region);
    }
}

/// Shared drift budget: bound the TOTAL displacement from the refined
/// point so stacked stage caps cannot teleport the point, then re-clamp
/// into a pupil-confirmed face region.
pub(crate) struct DriftBudget;

impl FocusRule for DriftBudget {
    fn name(&self) -> &'static str { "DriftBudget" }

    fn apply(&self, _ctx: &RuleCtx, st: &mut PointState) {
        // Under Skip the lock never moved the point, and under
        // Uncapped the anchor is verified — in both cases the shared budget
        // has nothing to contain.
        if _ctx.lock_mode != LockMode::Normal || st.lock_skipped { return; }
        let (refined_x, refined_y) = st.refined;
        let mut locked_x = st.x;
        let mut locked_y = st.y;
        let face_clamp = st.face_clamp;
        let pupil_count = st.face.as_ref().map(|f| f.pupil_count()).unwrap_or(0);
        let paired_pupils = st.face.as_ref().map(|f| f.paired_pupil_count).unwrap_or(0);
        let anchor_eye = st.face.as_ref().map(|f| f.anchor_eye()).unwrap_or(0.0);
        // Shared drift budget: the mass blend and the face lock each cap their
        // own displacement at 0.16, but when both pull toward the same false
        // subject (skin-toned rocks, brass) the caps stack to 0.32. Bound the
        // TOTAL displacement from the refined point, so no combination of
        // downstream stages can teleport the point across the frame.
        let total_dx = locked_x - refined_x;
        let total_dy = locked_y - refined_y;
        let total_drift = (total_dx * total_dx + total_dy * total_dy).sqrt();
        // A high-quality anchor — confirmed pupil blocks plus a real eye-band
        // response — earns a longer pull: when the softmax point gets lost in
        // background texture (dark foliage, night scenes) the face cluster is
        // the better evidence. Weak anchors (rocks, rope, brass) stay on the
        // short leash. Multiple confirmed pupils count as much as a strong
        // band (a 4-pupil anchor at anchor_eye 0.199 deserves the long walk
        // too), and pupil_count >= 1 already implies colour evidence (see
        // pupil_like), so true B&W can never reach the long arm.
        //
        // Deliberately NOT paired-only: that tightening was measured and
        // REVERTED — a real lone pupil (anchor_eye 0.21) legitimately earns
        // the long walk, while the fake it aimed at never touched the budget.
        // Pair authority lives in the cluster scores and the clamp, not here.
        let _ = paired_pupils;
        let max_total_drift = if pupil_count >= 2 || (pupil_count >= 1 && anchor_eye >= 0.20) { 0.34_f32 } else { 0.18_f32 };
        dbg_focus!("pupil_count: {}  paired: {}  anchor_eye: {:.2}  budget: {:.2}", pupil_count, paired_pupils, anchor_eye, max_total_drift);
        if total_drift > max_total_drift {
            let scale = max_total_drift / total_drift;
            locked_x = refined_x + total_dx * scale;
            locked_y = refined_y + total_dy * scale;
        }
        // Face-region invariant: the budget may shorten the walk but must not
        // carry the point back OUT of a confirmed face region.
        if let Some((fx0, fx1, fy0, fy1)) = face_clamp {
            locked_x = locked_x.max(fx0).min(fx1);
            locked_y = locked_y.max(fy0).min(fy1);
        }
    
        st.x = locked_x;
        st.y = locked_y;
    }
}

/// Saliency snap: when the point nearly agrees with the spectral-residual
/// peak, snap to the peak (mono <= 0.17; colour <= 0.135 and only with no
/// pupil-confirmed face). Refinement only — it cannot teleport.
pub(crate) struct SaliencySnap;

impl FocusRule for SaliencySnap {
    fn name(&self) -> &'static str { "SaliencySnap" }

    fn apply(&self, ctx: &RuleCtx, st: &mut PointState) {
        let pupil_count = st.face.as_ref().map(|f| f.pupil_count()).unwrap_or(0);
        let (snap_saliency, color_snap) = (ctx.snap_saliency, ctx.color_snap);
        let mut locked_x = st.x;
        let mut locked_y = st.y;
        let feat = ctx.feat;
        let (w, h) = (ctx.w, ctx.h);
        // Mono: B&W images have no skin path, and the spectral-residual peak
        // sat on/near the golden point across the mono failure set in the
        // browser experiment (fp16 err 0.03 vs pipeline 0.14). When
        // the pipeline's point and the peak nearly agree, the peak is the more
        // precise of the two — snap to it. Refinement only: gated by distance,
        // it cannot teleport. Threshold tunable via AF_SNAP_T (native) for
        // suite A/Bs; keep the shipped default in sync with the JS side.
        // Colour snap: same rule, but only when the pipeline found no
        // pupil-confirmed face — that gate is what separates the wildlife
        // false-portraits (tiger fur, fox, ladybug: warm colour reads as skin,
        // pupil_count == 0) from the group-shot rejection cases, where real
        // faces sit near legitimate bright anomalies.
        if snap_saliency || (color_snap && pupil_count == 0) {
            let (px, py) = saliency_peak(&feat.lum, w, h);
            // Face-region veto (colour snap only): when the pipeline found a
            // face region — even one too weak for the lock to act on — a
            // saliency peak far OUTSIDE that region is a prop or print
            // out-saliencing a person, not a better subject. Peaks near the
            // region (the group-shot bright-anomaly cases) still snap.
            if !snap_saliency {
                if let Some(f) = st.face.as_ref() {
                    if !f.face_blocks.is_empty() {
                        let (rx, ry, rw, rh) = f.rect;
                        let m = 0.12 * w.max(h) as f32;
                        let (pxp, pyp) = (px * w as f32, py * h as f32);
                        let outside = pxp < rx - m || pxp > rx + rw + m
                            || pyp < ry - m || pyp > ry + rh + m;
                        dbg_focus!("colour snap face-region gate: peak ({:.2},{:.2}) rect ({:.0},{:.0},{:.0},{:.0}) outside={}", px, py, rx, ry, rw, rh, outside);
                        if outside {
                            st.x = locked_x;
                            st.y = locked_y;
                            return;
                        }
                    }
                }
            }
            // Colour peaks are lower-quality than mono (measured: scene top-1
            // hit 21% vs mono's snap wins), so the colour snap gets a shorter
            // leash: the measured wins all snapped from <= 0.112 away, the
            // losses from 0.14+.
            // 0.135 not 0.12: browser-canvas pixels shift knife-edge points a few
            // hundredths vs the harness decoder (same reason mono went 0.15 ->
            // 0.17); the corpora have no snap distance in (0.104, 0.141), so
            // anything in that band is measurement-identical on the golden suites;
            // 0.135 covers the ladybug's production dist of 0.131 with margin.
            // With NO face evidence at all (no anchor, no face blocks) the
            // colour point is pure softmax prior — as signal-starved as mono —
            // and earns a longer 0.145 leash (an insect scene: anchor none,
            // browser dist 0.136). 0.145 still excludes the known no-anchor
            // long-snap loss (0.153); face-carrying photos (0.141) keep the
            // short leash.
            let no_face_evidence = st.face.as_ref().map_or(true, |f| f.face_blocks.is_empty());
            let default_t = if snap_saliency { 0.17_f32 }
                else if no_face_evidence { 0.145_f32 }
                else { 0.135_f32 };
            let snap_t = {
                #[cfg(not(target_arch = "wasm32"))]
                { std::env::var("AF_SNAP_T").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(default_t) }
                #[cfg(target_arch = "wasm32")]
                { default_t }
            };
            let d = ((locked_x - px).powi(2) + (locked_y - py).powi(2)).sqrt();
            dbg_focus!("mono saliency peak: {:.3},{:.3}  dist: {:.3}  snap_t: {:.2}", px, py, d, snap_t);
            if d <= snap_t {
                locked_x = px;
                locked_y = py;
            }
        }

        st.x = locked_x;
        st.y = locked_y;
    }
}

