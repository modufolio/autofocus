//! Spectral-residual saliency (Hou & Zhang 2007).

// ---------------------------------------------------------------------------
//  Spectral-residual saliency (Hou & Zhang 2007) — kept in sync with the
//  JS implementation in ui/src/saliency.js.
//
//  Used by the mono snap rule: on B&W images the pipeline is signal-starved
//  (no skin path), and the browser experiment showed the saliency peak on
//  or near the golden point across the mono failure set (fp16 peak err
//  0.03 vs pipeline 0.14 on bride-class fixtures). The peak only ever
//  REFINES a nearby point — it cannot teleport.
// ---------------------------------------------------------------------------

pub(crate) const SAL_N: usize = 64;

fn sal_fft1d(re: &mut [f32; SAL_N], im: &mut [f32; SAL_N], inverse: bool) {
    let n = SAL_N;
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 { j ^= bit; bit >>= 1; }
        j ^= bit;
        if i < j { re.swap(i, j); im.swap(i, j); }
    }
    let mut len = 2usize;
    while len <= n {
        let ang = if inverse { 2.0 } else { -2.0 } * std::f32::consts::PI / len as f32;
        let (w_re, w_im) = (ang.cos(), ang.sin());
        let mut i = 0usize;
        while i < n {
            let (mut cur_re, mut cur_im) = (1.0f32, 0.0f32);
            for k in 0..len / 2 {
                let (u_re, u_im) = (re[i + k], im[i + k]);
                let v_re = re[i + k + len / 2] * cur_re - im[i + k + len / 2] * cur_im;
                let v_im = re[i + k + len / 2] * cur_im + im[i + k + len / 2] * cur_re;
                re[i + k] = u_re + v_re;
                im[i + k] = u_im + v_im;
                re[i + k + len / 2] = u_re - v_re;
                im[i + k + len / 2] = u_im - v_im;
                let next_re = cur_re * w_re - cur_im * w_im;
                cur_im = cur_re * w_im + cur_im * w_re;
                cur_re = next_re;
            }
            i += len;
        }
        len <<= 1;
    }
    if inverse {
        for i in 0..n { re[i] /= n as f32; im[i] /= n as f32; }
    }
}

fn sal_fft2d(re: &mut [f32], im: &mut [f32], inverse: bool) {
    let mut row_re = [0.0f32; SAL_N];
    let mut row_im = [0.0f32; SAL_N];
    for y in 0..SAL_N {
        for x in 0..SAL_N { row_re[x] = re[y * SAL_N + x]; row_im[x] = im[y * SAL_N + x]; }
        sal_fft1d(&mut row_re, &mut row_im, inverse);
        for x in 0..SAL_N { re[y * SAL_N + x] = row_re[x]; im[y * SAL_N + x] = row_im[x]; }
    }
    for x in 0..SAL_N {
        for y in 0..SAL_N { row_re[y] = re[y * SAL_N + x]; row_im[y] = im[y * SAL_N + x]; }
        sal_fft1d(&mut row_re, &mut row_im, inverse);
        for y in 0..SAL_N { re[y * SAL_N + x] = row_re[y]; im[y * SAL_N + x] = row_im[y]; }
    }
}

/// Border-damped saliency peak from the full-resolution luminance map.
/// Returns (x, y) normalised to [0,1].
/// Damped 64x64 spectral-residual map from a full-resolution luminance
/// buffer. Shared by the snap (global peak) and the zoom proposals
/// (top-N interior peaks).
fn saliency_map(lum: &[f32], w: usize, h: usize) -> Vec<f32> {
    // Bilinear downsample to 64x64 (same result as the JS canvas path)
    let mut small = vec![0.0f32; SAL_N * SAL_N];
    for y in 0..SAL_N {
        for x in 0..SAL_N {
            let fx = ((x as f32 + 0.5) / SAL_N as f32) * w as f32 - 0.5;
            let fy = ((y as f32 + 0.5) / SAL_N as f32) * h as f32 - 0.5;
            let x0 = fx.floor().max(0.0) as usize;
            let y0 = fy.floor().max(0.0) as usize;
            let x1 = (x0 + 1).min(w - 1);
            let y1 = (y0 + 1).min(h - 1);
            let ax = (fx - x0 as f32).max(0.0);
            let ay = (fy - y0 as f32).max(0.0);
            small[y * SAL_N + x] = lum[y0 * w + x0] * (1.0 - ax) * (1.0 - ay)
                + lum[y0 * w + x1] * ax * (1.0 - ay)
                + lum[y1 * w + x0] * (1.0 - ax) * ay
                + lum[y1 * w + x1] * ax * ay;
        }
    }

    let mut re = small;
    let mut im = vec![0.0f32; SAL_N * SAL_N];
    sal_fft2d(&mut re, &mut im, false);

    let mut log_amp = vec![0.0f32; SAL_N * SAL_N];
    let mut ph_re = vec![0.0f32; SAL_N * SAL_N];
    let mut ph_im = vec![0.0f32; SAL_N * SAL_N];
    for i in 0..SAL_N * SAL_N {
        let amp = (re[i] * re[i] + im[i] * im[i]).sqrt();
        log_amp[i] = (amp + 1e-9).ln();
        ph_re[i] = if amp > 1e-12 { re[i] / amp } else { 1.0 };
        ph_im[i] = if amp > 1e-12 { im[i] / amp } else { 0.0 };
    }
    for y in 0..SAL_N {
        for x in 0..SAL_N {
            let (mut sum, mut cnt) = (0.0f32, 0.0f32);
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                    if nx < 0 || ny < 0 || nx >= SAL_N as i32 || ny >= SAL_N as i32 { continue; }
                    sum += log_amp[ny as usize * SAL_N + nx as usize];
                    cnt += 1.0;
                }
            }
            let amp = (log_amp[y * SAL_N + x] - sum / cnt).exp();
            re[y * SAL_N + x] = amp * ph_re[y * SAL_N + x];
            im[y * SAL_N + x] = amp * ph_im[y * SAL_N + x];
        }
    }
    sal_fft2d(&mut re, &mut im, true);

    let mut sal = vec![0.0f32; SAL_N * SAL_N];
    for i in 0..SAL_N * SAL_N { sal[i] = re[i] * re[i] + im[i] * im[i]; }
    for _ in 0..2 {
        let src = sal.clone();
        for y in 0..SAL_N {
            for x in 0..SAL_N {
                let (mut sum, mut cnt) = (0.0f32, 0.0f32);
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                        if nx < 0 || ny < 0 || nx >= SAL_N as i32 || ny >= SAL_N as i32 { continue; }
                        sum += src[ny as usize * SAL_N + nx as usize];
                        cnt += 1.0;
                    }
                }
                sal[y * SAL_N + x] = sum / cnt;
            }
        }
    }
    // Border damping + interior peak (same as saliencyMapFromLuminance)
    const M: usize = 3;
    for y in 0..SAL_N {
        for x in 0..SAL_N {
            let d = x.min(y).min(SAL_N - 1 - x).min(SAL_N - 1 - y);
            if d < M { sal[y * SAL_N + x] *= d as f32 / M as f32; }
        }
    }
    sal
}

/// Peak search excludes a wider border (P > M in saliency_map): edge
/// hotspots are watermarks, letterbox edges and frame-cut objects, never
/// the subject. Same algorithm as saliencyMapFromLuminance in saliency.js.
const SAL_P: usize = 6;

pub(crate) fn saliency_peak(lum: &[f32], w: usize, h: usize) -> (f32, f32) {
    let sal = saliency_map(lum, w, h);
    let mut peak_i = SAL_P * SAL_N + SAL_P;
    for y in SAL_P..SAL_N - SAL_P {
        for x in SAL_P..SAL_N - SAL_P {
            if sal[y * SAL_N + x] > sal[peak_i] { peak_i = y * SAL_N + x; }
        }
    }
    (
        ((peak_i % SAL_N) as f32 + 0.5) / SAL_N as f32,
        ((peak_i / SAL_N) as f32 + 0.5) / SAL_N as f32,
    )
}

/// Top-N interior local maxima of the saliency map, greedily NMS'd with a
/// minimum spacing so the peaks describe distinct hotspots. Returns
/// (x, y, strength) in normalised coords, strongest first. Measured: the
/// subject is AMONG the top-3 peaks for 59% of golden photos even when
/// the global peak is wrong — which is what the zoom-proposal stage exploits.
pub(crate) fn saliency_top_peaks(lum: &[f32], w: usize, h: usize, n: usize) -> Vec<(f32, f32, f32)> {
    let sal = saliency_map(lum, w, h);
    // local maxima in the interior band
    let mut maxima: Vec<(usize, usize, f32)> = Vec::new();
    for y in SAL_P..SAL_N - SAL_P {
        for x in SAL_P..SAL_N - SAL_P {
            let v = sal[y * SAL_N + x];
            let is_max = (-1i32..=1).all(|dy| (-1i32..=1).all(|dx| {
                if dx == 0 && dy == 0 { return true; }
                let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                sal[ny as usize * SAL_N + nx as usize] <= v
            }));
            if is_max { maxima.push((x, y, v)); }
        }
    }
    maxima.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    // NMS: 10 cells (~16% of the frame) apart, so three peaks are three
    // genuinely different places to look
    const MIN_DIST2: i32 = 10 * 10;
    let mut kept: Vec<(usize, usize, f32)> = Vec::new();
    for &(x, y, v) in &maxima {
        if kept.len() >= n { break; }
        if kept.iter().all(|&(kx, ky, _)| {
            let (dx, dy) = (kx as i32 - x as i32, ky as i32 - y as i32);
            dx * dx + dy * dy >= MIN_DIST2
        }) { kept.push((x, y, v)); }
    }
    kept.iter().map(|&(x, y, v)| (
        (x as f32 + 0.5) / SAL_N as f32,
        (y as f32 + 0.5) / SAL_N as f32,
        v,
    )).collect()
}
