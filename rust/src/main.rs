/*!
 * autofocus CLI
 *
 * Fast focus-point detection for images, for batch analysis and
 * backfills.
 *
 * USAGE
 *   autofocus [OPTIONS] <FILE>...
 *   autofocus [OPTIONS] -          # read file paths from stdin, one per line
 *
 * OUTPUT (--format pct, default)
 *   portrait01.jpg: 42.5%,31%
 *
 * The "x%,y%" output is the focus-string format `modufolio/media` stores
 * in its `focus` field, so CLI output can be persisted or diffed
 * byte-for-byte against values the web pipeline produced.
 */

use std::{
    io::{self, BufRead},
    path::PathBuf,
    process,
    time::Instant,
};

use clap::{Parser, ValueEnum};
use image::imageops::FilterType;

// Pull in the core detection function from the library.
use autofocus_wasm::{detect_focus_cli, detect_focus_cli_blocks, crop_face_evidence, point_evidence, radial_eye_pair_ys};

// ---------------------------------------------------------------------------
//  CLI definition
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(
    name    = "autofocus",
    version,
    about   = "Detect the focus point of images",
    long_about = None,
)]
struct Cli {
    /// Image files to analyse (JPEG, PNG, WebP, GIF).
    /// Pass '-' to read file paths from stdin, one per line.
    files: Vec<PathBuf>,

    /// Downscale the long edge to this many pixels before analysis.
    /// Lower = faster; higher = more precise for large images.
    #[arg(short = 'm', long, default_value = "256", value_name = "N")]
    max_dim: u32,

    /// Output format.
    #[arg(short, long, default_value = "pct", value_name = "FORMAT")]
    format: Format,

    /// Print only the result on each line — no filename prefix.
    /// Useful when piping output to another program.
    #[arg(short, long)]
    quiet: bool,

    /// Print per-image timings to stderr.
    #[arg(long)]
    time: bool,

    /// Parity harness: read one raw RGBA frame of the given size from stdin
    /// and print "x y category". Bypasses decode/scale entirely so native
    /// and WASM builds can be compared on identical pixels.
    #[arg(long, value_name = "WxH", hide = true)]
    raw_rgba: Option<String>,
}

#[derive(ValueEnum, Clone)]
enum Format {
    /// "42.5%,31%" — the modufolio/media focus-string format (default)
    Pct,
    /// {"file":"…","x":0.42,"y":0.31,"category":"portrait"}
    Json,
    /// "0.4200 0.3100" — raw normalised coordinates
    Xy,
    /// One JSON object per line with the full image-features payload,
    /// for bulk import into a consumer's feature store. The point includes
    /// the zoom pass, matching the in-browser detection exactly.
    Features,
}

// ---------------------------------------------------------------------------
//  Entry point
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();

    if let Some(dims) = &cli.raw_rgba {
        run_raw_rgba(dims);
        return;
    }

    let paths = collect_paths(&cli.files);
    if paths.is_empty() {
        eprintln!("autofocus: no input files");
        process::exit(1);
    }

    let mut any_error = false;

    for path in &paths {
        let t = cli.time.then(Instant::now);

        let outcome = if matches!(cli.format, Format::Features) {
            process_image_features(path, cli.max_dim).map(|line| {
                println!("{line}");
            })
        } else {
            process_image(path, cli.max_dim).map(|(x, y, category)| {
                let line = format_result(path, x, y, category, &cli.format, cli.quiet);
                println!("{line}");
            })
        };

        match outcome {
            Ok(()) => {
                if let Some(t) = t {
                    eprintln!("{}: {:.1} ms", path.display(), t.elapsed().as_secs_f64() * 1000.0);
                }
            }
            Err(e) => {
                eprintln!("autofocus: {}: {e}", path.display());
                any_error = true;
            }
        }
    }

    if any_error {
        process::exit(1);
    }
}

// ---------------------------------------------------------------------------
//  Path collection — args or stdin ('-')
// ---------------------------------------------------------------------------

fn collect_paths(input: &[PathBuf]) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    for p in input {
        if p.to_str() == Some("-") {
            let stdin = io::stdin();
            for line in stdin.lock().lines() {
                let line = line.expect("stdin read error");
                let t = line.trim();
                if !t.is_empty() {
                    paths.push(PathBuf::from(t));
                }
            }
        } else {
            paths.push(p.clone());
        }
    }
    paths
}

// ---------------------------------------------------------------------------
//  Features output — the image_features backfill
// ---------------------------------------------------------------------------

/// The full stored-features payload for one image, as a single JSON line.
///
/// The scalar aggregates come from `detect_features` on the downscaled frame;
/// the point comes from `process_image`, which includes the zoom pass — so a
/// backfilled row matches what the in-browser detection would have stored
/// for the same image.
fn process_image_features(
    path: &PathBuf,
    max_dim: u32,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut decoder = image::ImageReader::open(path)?
        .with_guessed_format()?
        .into_decoder()?;
    let orientation = image::ImageDecoder::orientation(&mut decoder)
        .unwrap_or(image::metadata::Orientation::NoTransforms);
    let mut img = image::DynamicImage::from_decoder(decoder)?;
    img.apply_orientation(orientation);
    let (w0, h0) = (img.width(), img.height());

    if w0 < 3 || h0 < 3 {
        return Err(format!("image too small ({w0}\u{d7}{h0})").into());
    }

    let scale = (max_dim as f32 / w0.max(h0) as f32).min(1.0);
    let w = (w0 as f32 * scale).round() as u32;
    let h = (h0 as f32 * scale).round() as u32;
    let rgba = if scale < 1.0 {
        img.resize_exact(w, h, FilterType::Nearest)
    } else {
        img
    }
    .to_rgba8();

    let mut payload: serde_json::Value =
        serde_json::from_str(&autofocus_wasm::detect_features(rgba.as_raw(), w, h))?;

    let (x, y, _) = process_image(path, max_dim)?;
    payload["x"] = serde_json::json!(x);
    payload["y"] = serde_json::json!(y);
    payload["file"] = serde_json::json!(path.display().to_string());

    Ok(payload.to_string())
}

// ---------------------------------------------------------------------------
//  Image loading + detection
// ---------------------------------------------------------------------------

fn process_image(
    path: &PathBuf,
    max_dim: u32,
) -> Result<(f32, f32, &'static str), Box<dyn std::error::Error>> {
    // Decode with EXIF orientation applied, so phone photos (orientation 5-8
    // swap width/height) are analysed upright — matching how browsers and the
    // canvas path in the media editor render them.
    let mut decoder = image::ImageReader::open(path)?
        .with_guessed_format()?
        .into_decoder()?;
    let orientation = image::ImageDecoder::orientation(&mut decoder)
        .unwrap_or(image::metadata::Orientation::NoTransforms);
    let mut img = image::DynamicImage::from_decoder(decoder)?;
    img.apply_orientation(orientation);
    let (w0, h0) = (img.width(), img.height());
    let orig = img.clone();

    if w0 < 3 || h0 < 3 {
        return Err(format!("image too small ({w0}×{h0})").into());
    }

    // Same scale logic as the JS/WASM pipeline, so points are comparable
    let scale = (max_dim as f32 / w0.max(h0) as f32).min(1.0);
    let w = (w0 as f32 * scale).round() as u32;
    let h = (h0 as f32 * scale).round() as u32;

    let scaled = if scale < 1.0 {
        img.resize_exact(w, h, FilterType::Nearest)
    } else {
        img
    };

    let rgba = scaled.to_rgba8();

    // Zoom pass (default on; AF_NO_ZOOM=1 disables): re-analyse the topmost skin region at
    // crop resolution, where real pupils enter FRST's 2-4 px band and
    // fabric prints leave it. Skips mono (needs the skin map).
    if std::env::var_os("AF_NO_ZOOM").is_none() {
        let (bx, by, category, blocks) = detect_focus_cli_blocks(rgba.as_raw(), w, h);
        if category != "mono" {
            if let Some((rx0, ry0, rx1, ry1)) = topmost_skin_region(&blocks) {
                // Pad the region: faces need context, and skin clusters
                // usually stop at the chin.
                let pad_x = (rx1 - rx0).max(0.08) * 0.6;
                let pad_y = (ry1 - ry0).max(0.08) * 0.8;
                let cx0 = ((rx0 - pad_x).max(0.0) * w0 as f32) as u32;
                let cy0 = ((ry0 - pad_y).max(0.0) * h0 as f32) as u32;
                let cx1 = (((rx1 + pad_x).min(1.0)) * w0 as f32) as u32;
                let cy1 = (((ry1 + pad_y).min(1.0)) * h0 as f32) as u32;
                let (cw, ch) = (cx1.saturating_sub(cx0), cy1.saturating_sub(cy0));
                if cw >= 48 && ch >= 48 {
                    let crop = orig.crop_imm(cx0, cy0, cw, ch);
                    let cscale = (max_dim as f32 / cw.max(ch) as f32).min(4.0);
                    let (zw, zh) = (((cw as f32 * cscale) as u32).max(1), ((ch as f32 * cscale) as u32).max(1));
                    let zoomed = crop.resize_exact(zw, zh, FilterType::Nearest).to_rgba8();
                    let (evidence, zx, zy, skin_cy, peak_eye_y) = crop_face_evidence(zoomed.as_raw(), zw, zh);
                    // Pair-geometry requirement (ZOOM-PLAN step 2): lace and
                    // prints fake eye-band at crop scale too, but they fail
                    // the eye-pair test (lattice veto); real faces pass it
                    // at this resolution.
                    // Specificity: at least one pair must sit ABOVE the
                    // crop's skin centroid (eyes ride high on a face's skin
                    // mass; hands/fur pairs do not).
                    let pair_ys = radial_eye_pair_ys(zoomed.as_raw(), zw, zh);
                    let pairs = pair_ys.len();
                    let fl = std::env::var("AF_FL").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.06);
                    // Brow co-location (iteration 9): the crop's strongest
                    // skin-gated eye-band row must sit near a valid pair —
                    // brows/lashes make band texture AT the eye line; fur
                    // and hands scatter it.
                    let co = std::env::var("AF_CO").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.16);
                    let face_like = pair_ys.iter().any(|&py| py < skin_cy - fl && (py - peak_eye_y).abs() <= co);
                    if std::env::var_os("AUTOFOCUS_DEBUG").is_some() {
                        eprintln!("zoom: region=({:.2},{:.2})-({:.2},{:.2}) evidence={:.3} crop_pt=({:.2},{:.2})", rx0, ry0, rx1, ry1, evidence, zx, zy);
                    }
                    // Relative gate (ZOOM-PLAN step 2): the crop must BEAT
                    // the evidence already present around the base point by
                    // a margin, not merely clear an absolute floor.
                    let base_ev = point_evidence(rgba.as_raw(), w, h, bx, by);
                    let margin = std::env::var("AF_ZOOM_MARGIN").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.10);
                    if std::env::var_os("AUTOFOCUS_DEBUG").is_some() {
                        eprintln!("zoom gate: crop_ev={:.3} base_ev={:.3} margin={:.2} pairs={} face_like={} skin_cy={:.2}", evidence, base_ev, margin, pairs, face_like, skin_cy);
                    }
                    if face_like && evidence >= 0.25 && evidence >= base_ev + margin {
                        let fx = (cx0 as f32 + zx * cw as f32) / w0 as f32;
                        let fy = (cy0 as f32 + zy * ch as f32) / h0 as f32;
                        return Ok(((fx * 100.0).round() / 100.0, (fy * 100.0).round() / 100.0, category));
                    }
                }
            }
        }
        return Ok((bx, by, category));
    }

    let (x, y, category) = detect_focus_cli(rgba.as_raw(), w, h);

    Ok((x, y, category))
}

/// Bounding box (normalised) of the topmost skin cluster: seed at the
/// skin block with the smallest cy, extended downward while rows stay
/// connected. Faces are the highest skin mass in almost all photos.
fn topmost_skin_region(blocks: &[(f32, f32, f32, f32)]) -> Option<(f32, f32, f32, f32)> {
    let skin: Vec<(f32, f32, f32)> = blocks.iter().filter(|b| b.2 >= 0.30).map(|b| (b.0, b.1, b.3)).collect();
    if skin.is_empty() { return None; }
    let seed = *skin.iter().min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())?;
    let mut members: Vec<(f32, f32, f32)> = vec![seed];
    let mut used = vec![false; skin.len()];
    loop {
        let mut grew = false;
        for (i, b) in skin.iter().enumerate() {
            if used[i] { continue; }
            if members.iter().any(|m| (b.0 - m.0).abs() <= 0.12 && (b.1 - m.1).abs() <= 0.10) {
                members.push(*b); used[i] = true; grew = true;
            }
        }
        if !grew { break; }
    }
    // Proposal-quality guards — keep in sync with topmost_skin_bbox in lib.rs.
    if members.len() < 3 { return None; }
    if !members.iter().any(|m| m.2 >= 0.08) { return None; }
    let x0 = members.iter().map(|m| m.0).fold(f32::MAX, f32::min);
    let x1 = members.iter().map(|m| m.0).fold(f32::MIN, f32::max);
    let y0 = members.iter().map(|m| m.1).fold(f32::MAX, f32::min);
    let y1 = members.iter().map(|m| m.1).fold(f32::MIN, f32::max);
    Some((x0, y0, x1, y1))
}

// ---------------------------------------------------------------------------
//  Output formatting
// ---------------------------------------------------------------------------

fn format_result(
    path: &PathBuf,
    x: f32,
    y: f32,
    category: &str,
    fmt: &Format,
    quiet: bool,
) -> String {
    let prefix = if quiet { String::new() } else { format!("{}: ", path.display()) };

    match fmt {
        Format::Pct => {
            // The focus-string contract: round(x * 100) to one decimal,
            // then strip a trailing ".0"
            format!("{prefix}{}{}", pct_str(x), {
                let s = pct_str(y);
                format!(",{s}")
            })
        }
        Format::Json => {
            // Compact JSON, one object per line — easy to parse with `jq`
            format!(
                r#"{prefix}{{"file":{},"x":{},"y":{},"category":"{}"}}"#,
                serde_json::to_string(path.to_str().unwrap_or("")).unwrap(),
                round2(x),
                round2(y),
                category,
            )
        }
        Format::Xy => {
            format!("{prefix}{:.4} {:.4}", x, y)
        }
        // Handled by process_image_features before format_result is reached.
        Format::Features => unreachable!("features format bypasses format_result"),
    }
}

/// Format a [0,1] float as a focus-string percentage:
///   0.425 → "42.5%"
///   0.30  → "30%"   (no trailing ".0")
fn pct_str(v: f32) -> String {
    // Round to one decimal place
    let pct = (v * 1000.0).round() / 10.0;
    if pct.fract() < 0.0001 {
        format!("{}%", pct as i32)
    } else {
        format!("{pct}%")
    }
}

/// Round to 2 decimal places for JSON output.
fn round2(v: f32) -> f32 {
    (v * 100.0).round() / 100.0
}

/// Read `w * h * 4` RGBA bytes from stdin, run the core detector (no zoom —
/// zoom re-reads the original image, which raw mode does not have), and
/// print `x y category`.
fn run_raw_rgba(dims: &str) {
    let (w, h) = match dims.split_once('x') {
        Some((w, h)) => (
            w.parse::<u32>().unwrap_or(0),
            h.parse::<u32>().unwrap_or(0),
        ),
        None => (0, 0),
    };
    if w < 3 || h < 3 {
        eprintln!("autofocus: --raw-rgba expects WxH, got '{dims}'");
        process::exit(1);
    }

    let mut buf = Vec::with_capacity((w * h * 4) as usize);
    io::Read::read_to_end(&mut io::stdin().lock(), &mut buf).expect("read stdin");
    if buf.len() != (w * h * 4) as usize {
        eprintln!(
            "autofocus: expected {} bytes for {w}x{h}, got {}",
            w * h * 4,
            buf.len()
        );
        process::exit(1);
    }

    let (x, y, category) = autofocus_wasm::detect_focus_cli(&buf, w, h);
    println!("{x} {y} {category}");
}
