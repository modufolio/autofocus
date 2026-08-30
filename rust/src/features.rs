//! Pixel-level feature extraction: colour transforms, adaptive skin
//! model, luminance/saturation/gradient/sharpness maps and the fast
//! radial-symmetry transform (pupil detector).

use crate::signals::{HOG_BINS, BIN_WIDTH};

// ---------------------------------------------------------------------------
//  Colour helpers
// ---------------------------------------------------------------------------

#[inline]
pub(crate) fn linearize(c: u8) -> f32 {
    let n = c as f32 / 255.0;
    if n <= 0.04045 {
        n / 12.92
    } else {
        ((n + 0.055) / 1.055_f32).powf(2.4)
    }
}

#[inline]
pub(crate) fn luminance(r: u8, g: u8, b: u8) -> f32 {
    0.2126 * linearize(r) + 0.7152 * linearize(g) + 0.0722 * linearize(b)
}

#[inline]
pub(crate) fn saturation_score(r: u8, g: u8, b: u8) -> f32 {
    let max = r.max(g).max(b);
    if max == 0 { return 0.0; }
    (max - r.min(g).min(b)) as f32 / max as f32
}

#[derive(Clone, Copy)]
pub(crate) struct SkinCenters {
    c1_rn: f32, c1_gn: f32,
    c2_rn: f32, c2_gn: f32,
}

impl SkinCenters {
    fn static_priors() -> Self {
        Self { c1_rn: 0.370, c1_gn: 0.320, c2_rn: 0.410, c2_gn: 0.270 }
    }
}

pub(crate) fn adapt_skin_centers(src: &[u8], n_pixels: usize) -> SkinCenters {
    let stat = SkinCenters::static_priors();

    // Collect candidates within 1.5σ of either static centre
    let mut candidates: Vec<(f32, f32)> = Vec::new();
    for i in 0..n_pixels {
        let o = i * 4;
        let r = src[o]; let g = src[o + 1]; let b = src[o + 2];
        let sum = r as f32 + g as f32 + b as f32;
        if sum < 30.0 { continue; }
        let rn = r as f32 / sum;
        let gn = g as f32 / sum;
        if rn - gn < 0.04 { continue; }
        let d1 = ((rn - 0.370) / 0.045).powi(2) + ((gn - 0.320) / 0.040).powi(2);
        let d2 = ((rn - 0.410) / 0.055).powi(2) + ((gn - 0.270) / 0.045).powi(2);
        if d1.min(d2) < 2.25 { candidates.push((rn, gn)); }
    }

    let min_count = (200_usize).max(n_pixels / 100);
    if candidates.len() < min_count { return stat; }

    // Mini k-means: 5 iterations warm-started from static centres
    let (mut c1r, mut c1g) = (stat.c1_rn, stat.c1_gn);
    let (mut c2r, mut c2g) = (stat.c2_rn, stat.c2_gn);

    for _ in 0..5 {
        let (mut s1r, mut s1g, mut n1) = (0.0_f32, 0.0_f32, 0_u32);
        let (mut s2r, mut s2g, mut n2) = (0.0_f32, 0.0_f32, 0_u32);
        for &(rn, gn) in &candidates {
            let d1 = (rn - c1r).powi(2) + (gn - c1g).powi(2);
            let d2 = (rn - c2r).powi(2) + (gn - c2g).powi(2);
            if d1 < d2 { s1r += rn; s1g += gn; n1 += 1; }
            else        { s2r += rn; s2g += gn; n2 += 1; }
        }
        if n1 > 0 { c1r = s1r / n1 as f32; c1g = s1g / n1 as f32; }
        if n2 > 0 { c2r = s2r / n2 as f32; c2g = s2g / n2 as f32; }
    }

    // Variance sanity check: adapted centres must reduce intra-cluster variance by >10%
    let intra_var = |cr1: f32, cg1: f32, cr2: f32, cg2: f32| -> f32 {
        candidates.iter().map(|&(rn, gn)| {
            let d1 = (rn - cr1).powi(2) + (gn - cg1).powi(2);
            let d2 = (rn - cr2).powi(2) + (gn - cg2).powi(2);
            d1.min(d2)
        }).sum::<f32>() / candidates.len() as f32
    };
    if intra_var(c1r, c1g, c2r, c2g) >= intra_var(stat.c1_rn, stat.c1_gn, stat.c2_rn, stat.c2_gn) * 0.90 {
        return stat;
    }

    // Hard clamp as belt-and-braces backstop
    let clamp = |v: f32, lo: f32, hi: f32| v.max(lo).min(hi);
    SkinCenters {
        c1_rn: clamp(c1r, 0.340, 0.400), c1_gn: clamp(c1g, 0.290, 0.350),
        c2_rn: clamp(c2r, 0.380, 0.440), c2_gn: clamp(c2g, 0.240, 0.300),
    }
}

#[inline]
pub(crate) fn skin_score_with_centers(rn: f32, gn: f32, sc: &SkinCenters) -> f32 {
    let d1r = (rn - sc.c1_rn) / 0.038;
    let d1g = (gn - sc.c1_gn) / 0.035;
    let p1  = (-0.5 * (d1r * d1r + d1g * d1g)).exp();

    let d2r = (rn - sc.c2_rn) / 0.048;
    let d2g = (gn - sc.c2_gn) / 0.040;
    let p2  = (-0.5 * (d2r * d2r + d2g * d2g)).exp();

    p1.max(p2)
}

// ---------------------------------------------------------------------------
//  Feature extraction
// ---------------------------------------------------------------------------

pub(crate) struct Features {
    pub(crate) lum:    Vec<f32>,
    pub(crate) mag:    Vec<f32>,
    pub(crate) h_ratio: Vec<f32>,
    pub(crate) hog_bin: Vec<u8>,
    pub(crate) skin:   Vec<f32>,
    pub(crate) sat:    Vec<f32>,
    pub(crate) sharp:  Vec<f32>,
    pub(crate) radial: Vec<f32>,
}

/// Fast radial symmetry transform (Loy & Zelinsky), dark-blob variant.
///
/// Pupils are small dark radially-symmetric blobs: their edge gradients all
/// point away from a common centre. Each strong-gradient pixel votes for a
/// dark centre `r` pixels against its gradient direction; peaks in the vote
/// map mark pupil-like blobs. Hat brims, glasses frames and hair edges are
/// elongated, so their votes smear along a line instead of stacking.
pub(crate) const RADIAL_RADII: [usize; 3] = [2, 3, 4];
pub(crate) const RADIAL_MIN_GRAD: f32 = 0.06;

pub(crate) fn compute_radial_symmetry(lum: &[f32], w: usize, h: usize) -> Vec<f32> {
    let n = w * h;
    let mut acc = vec![0.0_f32; n];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let i = y * w + x;
            let gx = lum[i + 1] - lum[i - 1];
            let gy = lum[i + w] - lum[i - w];
            let m = (gx * gx + gy * gy).sqrt();
            if m < RADIAL_MIN_GRAD { continue; }
            let ux = gx / m;
            let uy = gy / m;
            for r in RADIAL_RADII {
                let vx = (x as f32 - ux * r as f32).round();
                let vy = (y as f32 - uy * r as f32).round();
                if vx >= 0.0 && vy >= 0.0 && (vx as usize) < w && (vy as usize) < h {
                    acc[vy as usize * w + vx as usize] += m;
                }
            }
        }
    }
    // Two 3x3 box-blur passes turn coincident votes into smooth peaks.
    let mut blurred = vec![0.0_f32; n];
    for _ in 0..2 {
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let i = y * w + x;
                let mut s = 0.0_f32;
                for dy in 0..3 {
                    let row = (y + dy - 1) * w + x;
                    s += acc[row - 1] + acc[row] + acc[row + 1];
                }
                blurred[i] = s / 9.0;
            }
        }
        std::mem::swap(&mut acc, &mut blurred);
    }
    let max = acc.iter().fold(0.0_f32, |a, &v| a.max(v));
    if max > 0.0 {
        let inv = 1.0 / max;
        for v in acc.iter_mut() { *v *= inv; }
    }
    acc
}

pub(crate) fn build_luminance_map(src: &[u8], n: usize) -> Vec<f32> {
    let mut lum = vec![0.0_f32; n];
    for i in 0..n {
        let o = i * 4;
        lum[i] = luminance(src[o], src[o + 1], src[o + 2]);
    }
    lum
}

pub(crate) fn analyse_pixels(src: &[u8], w: usize, h: usize) -> Features {
    let n = w * h;
    let sc = adapt_skin_centers(src, n);
    let lum = build_luminance_map(src, n);

    let mut mag     = vec![0.0_f32; n];
    let mut h_ratio = vec![0.0_f32; n];
    let mut hog_bin = vec![0_u8;    n];
    let mut skin    = vec![0.0_f32; n];
    let mut sat     = vec![0.0_f32; n];
    let mut sharp   = vec![0.0_f32; n];

    let mag_norm   = 1.0 / (4.0 * std::f32::consts::SQRT_2);
    let sharp_norm = 0.25;

    for y in 0..h {
        for x in 0..w {
            let i  = y * w + x;
            let px = i * 4;

            // Sobel + Laplacian (interior only)
            if x > 0 && x < w - 1 && y > 0 && y < h - 1 {
                let tl = lum[(y-1)*w + (x-1)]; let tc = lum[(y-1)*w + x]; let tr = lum[(y-1)*w + (x+1)];
                let ml = lum[y*w   + (x-1)];                               let mr = lum[y*w   + (x+1)];
                let bl = lum[(y+1)*w + (x-1)]; let bc = lum[(y+1)*w + x]; let br = lum[(y+1)*w + (x+1)];

                let gx = -tl + tr - 2.0*ml + 2.0*mr - bl + br;
                let gy = -tl - 2.0*tc - tr + bl + 2.0*bc + br;

                mag[i] = (gx*gx + gy*gy).sqrt() * mag_norm;
                if mag[i] > 1.0 { mag[i] = 1.0; }

                let agx = gx.abs(); let agy = gy.abs();
                h_ratio[i] = if agx + agy > 0.0 { agy / (agx + agy) } else { 0.0 };

                let mut angle = gy.atan2(gx);
                if angle < 0.0 { angle += std::f32::consts::PI; }
                hog_bin[i] = ((angle / BIN_WIDTH) as usize).min(HOG_BINS - 1) as u8;

                let d2x = (2.0 * lum[i] - ml - mr).abs();
                let d2y = (2.0 * lum[i] - tc - bc).abs();
                sharp[i] = ((d2x + d2y) * sharp_norm).min(1.0);
            }

            // Colour features
            let r = src[px]; let g = src[px+1]; let b = src[px+2];
            let sum = r as f32 + g as f32 + b as f32;
            if sum >= 15.0 {
                let max_c = r.max(g).max(b); let min_c = r.min(g).min(b);
                let rn = r as f32 / sum;
                let gn = g as f32 / sum;
                if (max_c - min_c) as f32 / max_c as f32 >= 0.10 && rn - gn >= 0.04 {
                    skin[i] = skin_score_with_centers(rn, gn, &sc);
                }
            }
            sat[i] = saturation_score(r, g, b);
        }
    }

    let radial = compute_radial_symmetry(&lum, w, h);
    Features { lum, mag, h_ratio, hog_bin, skin, sat, sharp, radial }
}

// ---------------------------------------------------------------------------
//  Block-level helpers
// ---------------------------------------------------------------------------

