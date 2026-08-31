// Debug: print top radial-symmetry peaks for an image at analysis scale.
use image::imageops::FilterType;

fn main() {
    let path = std::env::args().nth(1).expect("usage: dbg_radial <img>");
    let img = image::open(&path).unwrap();
    let (w0, h0) = (img.width(), img.height());
    let scale = (256.0 / w0.max(h0) as f32).min(1.0);
    let w = (w0 as f32 * scale).round() as u32;
    let h = (h0 as f32 * scale).round() as u32;
    let rgba = img.resize_exact(w, h, FilterType::Nearest).to_rgba8();
    let peaks = autofocus::debug_radial_peaks(rgba.as_raw(), w, h, 12);
    for (x, y, v) in peaks {
        println!("peak {:.2},{:.2}  radial={:.3}", x, y, v);
    }
}
