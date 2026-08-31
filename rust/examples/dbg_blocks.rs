//! Print per-block (x, y, skin, eye) for one image — diagnosis aid for the
//! zoom proposal guards.
fn main() {
    let path = std::env::args().nth(1).expect("usage: dbg_blocks <image>");
    let img = image::open(&path).expect("decode");
    let (w0, h0) = (img.width(), img.height());
    let scale = (256.0 / w0.max(h0) as f32).min(1.0);
    let w = (w0 as f32 * scale).round() as u32;
    let h = (h0 as f32 * scale).round() as u32;
    let rgba = img.resize_exact(w, h, image::imageops::FilterType::Nearest).to_rgba8();
    let (_, _, cat, blocks) = autofocus_wasm::detect_focus_cli_blocks(rgba.as_raw(), w, h);
    println!("category: {cat}  blocks: {}", blocks.len());
    let mut top: Vec<_> = blocks.iter().filter(|b| b.1 <= 0.45).collect();
    top.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    for b in top.iter().filter(|b| b.2 >= 0.15) {
        println!("  x {:.2} y {:.2} skin {:.2} eye {:.2}", b.0, b.1, b.2, b.3);
    }
}
