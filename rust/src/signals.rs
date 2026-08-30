//! Block-level signals derived from the feature maps: rule-of-thirds,
//! HOG face texture (eye band), symmetry, face-priority boost,
//! background/geometry suppression and per-block metadata.

use crate::features::Features;

pub(crate) const HOG_BINS: usize       = 9;
pub(crate) const BIN_WIDTH: f32        = std::f32::consts::PI / HOG_BINS as f32;

#[inline]
pub(crate) fn thirds(d: f32) -> f32 {
    // Composition prior peaks at the GOLDEN-SECTION lines (p = 0.382/0.618
    // -> d = 0.236), not the thirds lines (d = 1/3): the 355 golden marks
    // sit at median 0.113 from the nearest phi-point vs 0.147 from the
    // nearest thirds power point, and the A/B is decisive on composed
    // photography (own-work 0.1737 -> 0.1675, nature -> 0.1974, the
    // wedding sets neutral) at an accepted +0.005 cost on the ImageNet
    // snapshot corpus. A dual-peak curve was measured and is worse than
    // either pure prior.
    let x = (((d - 0.236 + 1.0) % 2.0) * 0.5 - 0.5) * 16.0;
    (1.0 - x * x).max(0.0)
}

// ---------------------------------------------------------------------------
//  Adaptive skin centres (per-image k-means)
// ---------------------------------------------------------------------------

pub(crate) fn build_hog_histogram(feat: &Features, w: usize, bx: usize, by: usize, bw: usize, bh: usize) -> [f32; 9] {
    let mut hist = [0.0_f32; 9];
    for sy in 0..bh {
        for sx in 0..bw {
            let i = (by + sy) * w + (bx + sx);
            hist[feat.hog_bin[i] as usize] += feat.mag[i];
        }
    }
    hist
}

pub(crate) fn eye_band_score(feat: &Features, w: usize, bx: usize, by: usize, bw: usize, bh: usize, allow_no_skin: bool) -> f32 {
    let pixels = (bw * bh) as f32;
    let mut h_sum = 0.0_f32;
    let mut m_sum = 0.0_f32;
    let mut skin_count = 0_usize;
    let mut lum_sum = 0.0_f32;
    let mut dark_count = 0_usize;
    let mut bright_count = 0_usize;
    for sy in 0..bh {
        for sx in 0..bw {
            let i = (by + sy) * w + (bx + sx);
            h_sum += feat.h_ratio[i];
            m_sum += feat.mag[i];
            if feat.skin[i] > 0.05 { skin_count += 1; }
            lum_sum += feat.lum[i];
            if feat.lum[i] < 0.42 { dark_count += 1; }
            if feat.lum[i] > 0.78 { bright_count += 1; }
        }
    }
    let avg_h       = h_sum / pixels;
    let avg_m       = m_sum / pixels;
    let skin_ratio  = skin_count as f32 / pixels;
    let avg_lum     = lum_sum / pixels;
    let dark_ratio  = dark_count as f32 / pixels;
    let bright_ratio = bright_count as f32 / pixels;

    if avg_h < 0.55 || avg_m < 0.05 { return 0.0; }
    if !allow_no_skin && skin_ratio < 0.08 { return 0.0; }

    let skin_gate = if allow_no_skin { 1.0 } else { (skin_ratio * 2.0).min(1.0) };
    let darkness_gate = (dark_ratio * 3.2 + (0.70 - avg_lum).max(0.0) * 1.6).min(1.0).max(0.18);
    let highlight_penalty = if bright_ratio > 0.30 && dark_ratio < 0.14 {
        (1.0 - (bright_ratio - 0.30) * 2.4).max(0.08)
    } else {
        1.0
    };

    (avg_h * avg_m * 4.0).min(1.0) * skin_gate * darkness_gate * highlight_penalty
}

pub(crate) fn symmetry_score(feat: &Features, w: usize, h: usize, cx: f32, block_size: usize) -> f32 {
    let cx_i = cx as usize;
    let max_offset = cx_i.min(w - cx_i).min(block_size * 4);
    if max_offset < block_size { return 0.0; }

    let half_h = h.min(block_size * 4);
    let mut match_sum = 0.0_f32;
    let mut total = 0_usize;

    let mut offset = block_size;
    while offset <= max_offset {
        let lx = cx_i.saturating_sub(offset);
        let rx = cx_i + offset - block_size;
        if rx + block_size > w { break; }

        let start_y = h / 2 - (half_h / 2).min(h / 2);
        let end_y = (start_y + half_h).min(h.saturating_sub(block_size));

        let mut by = start_y;
        while by < end_y {
            let bh = block_size.min(h - by);
            let hl = build_hog_histogram(feat, w, lx, by, block_size, bh);
            let hr = build_hog_histogram(feat, w, rx, by, block_size, bh);
            let mut dot = 0.0_f32; let mut ml = 0.0_f32; let mut mr = 0.0_f32;
            for b in 0..HOG_BINS {
                dot += hl[b] * hr[b];
                ml  += hl[b] * hl[b];
                mr  += hr[b] * hr[b];
            }
            let denom = ml.sqrt() * mr.sqrt();
            if denom > 0.0 { match_sum += dot / denom; }
            total += 1;
            by += block_size;
        }
        offset += block_size;
    }

    if total == 0 { return 0.0; }
    let raw = match_sum / total as f32;
    if raw > 0.65 { raw } else { raw * 0.35 }
}

pub(crate) fn face_priority_boost(feat: &Features, w: usize, _h: usize, bx: usize, by: usize, bw: usize, bh: usize) -> f32 {
    let upper_half = (bh as f32 * 0.45) as usize;
    if upper_half == 0 { return 0.0; }
    let mut skin_upper = 0.0_f32;
    let mut edge_upper = 0.0_f32;
    let mut lum_upper  = 0.0_f32;
    let mut dark_upper = 0_usize;
    let mut bright_upper = 0_usize;
    let px = (bw * upper_half) as f32;
    for sy in 0..upper_half {
        for sx in 0..bw {
            let i = (by + sy) * w + (bx + sx);
            if feat.skin[i] > 0.1 { skin_upper += feat.skin[i]; }
            edge_upper += feat.h_ratio[i];
            lum_upper  += feat.lum[i];
            if feat.lum[i] < 0.42 { dark_upper += 1; }
            if feat.lum[i] > 0.80 { bright_upper += 1; }
        }
    }
    let skin_avg         = skin_upper / px;
    let edge_avg         = edge_upper / px;
    let avg_lum_upper    = lum_upper  / px;
    let dark_upper_ratio = dark_upper as f32 / px;
    let bright_upper_ratio = bright_upper as f32 / px;
    let darkness_gate    = (dark_upper_ratio * 3.6 + (0.68 - avg_lum_upper).max(0.0) * 1.8).min(1.0).max(0.12);
    let highlight_penalty = if bright_upper_ratio > 0.28 && dark_upper_ratio < 0.15 {
        (1.0 - (bright_upper_ratio - 0.28) * 2.7).max(0.05)
    } else {
        1.0
    };
    (skin_avg * edge_avg * 4.5 * darkness_gate * highlight_penalty).min(1.8)
}

// ---------------------------------------------------------------------------
//  Background suppression + geometric run detection
// ---------------------------------------------------------------------------

pub(crate) struct BgFlags {
    pub(crate) bg_block:   Vec<u8>,
    pub(crate) geom_block: Vec<u8>,
    pub(crate) n_cols:     usize,
    pub(crate) n_rows:     usize,
}

pub(crate) fn compute_bg_flags(feat: &Features, w: usize, h: usize, block_size: usize) -> BgFlags {
    let n_cols = (w + block_size - 1) / block_size;
    let n_rows = (h + block_size - 1) / block_size;
    let n = n_cols * n_rows;

    let mut block_skin = vec![0.0_f32; n];
    let mut block_eye  = vec![0.0_f32; n];
    let mut block_mag  = vec![0.0_f32; n];

    for row in 0..n_rows {
        let by = row * block_size;
        let bh = block_size.min(h - by);
        for col in 0..n_cols {
            let bx = col * block_size;
            let bw = block_size.min(w - bx);
            let px = (bw * bh) as f32;
            let idx = row * n_cols + col;
            let mut s = 0.0_f32; let mut m = 0.0_f32;
            for sy in 0..bh {
                for sx in 0..bw {
                    let i = (by + sy) * w + (bx + sx);
                    s += feat.skin[i];
                    if feat.mag[i] > 0.04 { m += feat.mag[i]; }
                }
            }
            block_skin[idx] = s / px;
            block_eye[idx]  = eye_band_score(feat, w, bx, by, bw, bh, false);
            block_mag[idx]  = m / px;
        }
    }

    const SKIN_THRESHOLD:    f32 = 0.35;
    const EYE_BG_THRESHOLD:  f32 = 0.05;
    const MAG_BG_THRESHOLD:  f32 = 0.06;
    const MAX_SUPPRESS_DEPTH: i32 = 3;
    const SKIN_HIGH:         f32 = 0.65;
    const EYE_TRIM:          f32 = 0.15;
    const MAG_NEAR_EDGE:     f32 = 0.035;

    let mut bg_block = vec![0_u8; n];
    let mut queued   = vec![0_u8; n];
    let mut queue: Vec<usize> = Vec::new();

    // Seed: boundary blocks with high skin + low eye-band
    for row in 0..n_rows {
        for col in 0..n_cols {
            if row != 0 && row != n_rows - 1 && col != 0 && col != n_cols - 1 { continue; }
            let idx = row * n_cols + col;
            if block_skin[idx] > SKIN_THRESHOLD && block_eye[idx] < EYE_BG_THRESHOLD {
                bg_block[idx] = 1; queued[idx] = 1; queue.push(idx);
            }
        }
    }

    let dirs: [(i32, i32); 4] = [(-1,0),(1,0),(0,-1),(0,1)];

    // Two near-edge expansion passes (FIX: gate on blockMag < MAG_NEAR_EDGE)
    for _pass in 0..2 {
        let snapshot: Vec<usize> = queue.clone();
        for &idx in &snapshot {
            let row = idx / n_cols;
            let col = idx % n_cols;
            let edge_dist = row.min(n_rows - 1 - row).min(col).min(n_cols - 1 - col);
            if edge_dist > 2 { continue; }
            for &(dr, dc) in &dirs {
                let nr = row as i32 + dr;
                let nc = col as i32 + dc;
                if nr < 0 || nr >= n_rows as i32 || nc < 0 || nc >= n_cols as i32 { continue; }
                let ni = nr as usize * n_cols + nc as usize;
                if queued[ni] != 0 { continue; }
                if block_skin[ni] > SKIN_HIGH && block_eye[ni] < EYE_TRIM && block_mag[ni] < MAG_NEAR_EDGE {
                    bg_block[ni] = 1; queued[ni] = 1; queue.push(ni);
                }
            }
        }
    }

    // BFS flood-fill depth-limited to MAX_SUPPRESS_DEPTH
    let mut head = 0;
    while head < queue.len() {
        let idx = queue[head]; head += 1;
        let row = idx / n_cols;
        let col = idx % n_cols;
        for &(dr, dc) in &dirs {
            let nr = row as i32 + dr;
            let nc = col as i32 + dc;
            if nr < 0 || nr >= n_rows as i32 || nc < 0 || nc >= n_cols as i32 { continue; }
            let ni = nr as usize * n_cols + nc as usize;
            if queued[ni] != 0 { continue; }
            let nr_u = nr as usize; let nc_u = nc as usize;
            let edge_dist = nr_u.min(n_rows - 1 - nr_u).min(nc_u).min(n_cols - 1 - nc_u);
            if edge_dist as i32 > MAX_SUPPRESS_DEPTH { continue; }
            if block_skin[ni] > SKIN_THRESHOLD && block_eye[ni] < EYE_BG_THRESHOLD && block_mag[ni] < MAG_BG_THRESHOLD {
                bg_block[ni] = 1; queued[ni] = 1; queue.push(ni);
            }
        }
    }

    // Geometric run detection: horizontal runs of >=5 blocks with high mag + eye-band
    let min_run_len       = 5;
    let run_mag_threshold = 0.07_f32;
    let run_eye_threshold = 0.05_f32;
    let mut geom_block = vec![0_u8; n];

    for row in 0..n_rows {
        let mut run_start: Option<usize> = None;
        let mut run_end = 0;
        for col in 0..=n_cols {
            let active = col < n_cols
                && block_mag[row * n_cols + col] >= run_mag_threshold
                && block_eye[row * n_cols + col] >= run_eye_threshold;
            if active {
                if run_start.is_none() { run_start = Some(col); }
                run_end = col;
            } else if let Some(rs) = run_start {
                let run_len = run_end - rs + 1;
                if run_len >= min_run_len {
                    let quiet_count = |r: usize| -> usize {
                        (rs..=run_end).filter(|&c| block_eye[r * n_cols + c] < run_eye_threshold).count()
                    };
                    let above_quiet = row == 0 || quiet_count(row - 1) > run_len * 2 / 3;
                    let below_quiet = row + 1 >= n_rows || quiet_count(row + 1) > run_len * 2 / 3;
                    if above_quiet || below_quiet {
                        for c in rs..=run_end { geom_block[row * n_cols + c] = 1; }
                    }
                }
                run_start = None;
            }
        }
    }

    BgFlags { bg_block, geom_block, n_cols, n_rows }
}

// ---------------------------------------------------------------------------
//  Image classification
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub(crate) struct BlockMeta {
    pub(crate) cx:           f32,
    pub(crate) cy:           f32,
    pub(crate) bw:           f32,
    pub(crate) bh:           f32,
    pub(crate) eye_band:     f32,
    pub(crate) skin_density: f32,
    pub(crate) face_boost:   f32,
    pub(crate) radial:       f32,
    pub(crate) bg_suppressed: bool,
}

/// Peak radial-symmetry response inside a block — high for blocks containing
/// pupil-like dark blobs, low for elongated edges (brims, hairlines).
pub(crate) fn block_radial_peak(feat: &Features, w: usize, bx: usize, by: usize, bw: usize, bh: usize) -> f32 {
    let mut peak = 0.0_f32;
    for sy in 0..bh {
        for sx in 0..bw {
            let v = feat.radial[(by + sy) * w + (bx + sx)];
            if v > peak { peak = v; }
        }
    }
    peak
}


/// Pupil evidence: a sharp radial-symmetry peak co-located with skin and
/// face-shaped gradients marks an eye even when the eye-band texture gate
/// fails (glasses, deep shadow, strong highlights on the iris).
#[inline]
pub(crate) fn pupil_like(b: &BlockMeta) -> bool {
    b.radial >= 0.60 && b.skin_density >= 0.20 && b.face_boost >= 0.45
}
