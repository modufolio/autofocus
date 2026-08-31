//! Probe: full-frame radial eye pairs + top radial peaks for one image.
fn main() {
    let path = std::env::args().nth(1).expect("usage: dbg_pairs <image>");
    let img = image::open(&path).expect("decode");
    let (w0, h0) = (img.width(), img.height());
    let scale = (256.0 / w0.max(h0) as f32).min(1.0);
    let w = (w0 as f32 * scale).round() as u32;
    let h = (h0 as f32 * scale).round() as u32;
    let rgba = img.resize_exact(w, h, image::imageops::FilterType::Nearest).to_rgba8();
    let n = autofocus::radial_eye_pairs(rgba.as_raw(), w, h);
    let ys = autofocus::radial_eye_pair_ys(rgba.as_raw(), w, h);
    println!("pairs: {n}  pair_ys: {ys:?}");
    for (x, y, v) in autofocus::debug_radial_peaks(rgba.as_raw(), w, h, 6) {
        println!("  radial peak ({x:.2},{y:.2}) v={v:.2}");
    }
}
