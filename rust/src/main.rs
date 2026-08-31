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

mod render;

use std::{
    io::{self, BufRead},
    path::{Path, PathBuf},
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

    /// Review mode: render every fixture photo into this directory with the
    /// hand-set golden focus point and the detected point drawn on it. The
    /// normalised distance is encoded in each output filename, so sorting a
    /// directory surfaces the worst detections first.
    #[arg(long, value_name = "OUT_DIR")]
    review: Option<PathBuf>,

    /// Fixture root for --review (contains fixtures.json and the album dirs).
    #[arg(long, value_name = "DIR", default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/../ui/tests/fixtures"))]
    fixtures: PathBuf,

    /// Evaluate an external dataset: a directory of photos plus a
    /// focuspoints.json mapping each filename to its hand-set point
    /// ({"img.jpg": {"x": 0.47, "y": 0.49}, …}). Prints the per-photo
    /// distance and the mean; combine with --review OUT_DIR to also render
    /// the annotated comparison photos. For golden sets that cannot live in
    /// the committed fixture tree.
    #[arg(long, value_name = "DIR", conflicts_with = "fixtures")]
    eval: Option<PathBuf>,

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

    if let Some(dir) = &cli.eval {
        if !cli.files.is_empty() {
            eprintln!("autofocus: --eval ignores positional files (it walks focuspoints.json)");
        }
        run_eval(dir, cli.review.as_deref(), cli.max_dim);
        return;
    }

    if let Some(out) = &cli.review {
        if !cli.files.is_empty() {
            eprintln!("autofocus: --review ignores positional files (it walks the fixture manifest)");
        }
        run_review(&cli.fixtures, out, cli.max_dim);
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
    process_image_with_orig(path, max_dim).map(|(point, _)| point)
}

/// Like process_image, but also hands back the full-res oriented image —
/// review mode draws on it instead of decoding a second time.
fn process_image_with_orig(
    path: &Path,
    max_dim: u32,
) -> Result<((f32, f32, &'static str), image::DynamicImage), Box<dyn std::error::Error>> {
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
                    // Two verified pairs is structural confirmation the base
                    // point never had to earn — body texture at crop scale
                    // manufactures evidence, not paired-eye geometry. With
                    // >= 2 pairs the crop only has to match base evidence;
                    // single-pair crops still owe the full margin.
                    let clears = if pairs >= 2 { evidence >= base_ev } else { evidence >= base_ev + margin };
                    // Desperation arm: when the incumbent point carries NO
                    // face evidence at crop scale, a very strong proposal is
                    // accepted even without resolved pairs (a turned or
                    // hair-covered head can defeat pair geometry while still
                    // measuring overwhelmingly face-like). Never fires when
                    // the incumbent has anything at all. 0.35 swept over the
                    // full fixture corpus: only crops with a zero-evidence
                    // incumbent reach this arm at all, and the two that do
                    // are both real faces whose hooded/averted eyes never
                    // resolve pair geometry even at head-crop scale.
                    let desp = std::env::var("AF_DESPERATE").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.35);
                    let desperate = base_ev < 0.05 && evidence >= desp;
                    if (face_like && evidence >= 0.25 && clears) || desperate {
                        let fx = (cx0 as f32 + zx * cw as f32) / w0 as f32;
                        let fy = (cy0 as f32 + zy * ch as f32) / h0 as f32;
                        return Ok((((fx * 100.0).round() / 100.0, (fy * 100.0).round() / 100.0, category), orig));
                    }
                }
            }
        }
        return Ok(((bx, by, category), orig));
    }

    let (x, y, category) = detect_focus_cli(rgba.as_raw(), w, h);

    Ok(((x, y, category), orig))
}

/// Review mode: annotate every fixture photo with its golden and detected
/// focus points and write the result under `out`, mirroring the album dirs.
fn run_review(fixtures: &Path, out: &Path, max_dim: u32) {
    let manifest_path = fixtures.join("fixtures.json");
    let raw = match std::fs::read_to_string(&manifest_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("autofocus: cannot read {}: {e}", manifest_path.display());
            process::exit(1);
        }
    };
    let manifest: serde_json::Value = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("autofocus: {} is not valid JSON: {e}", manifest_path.display());
            process::exit(1);
        }
    };

    let mut any_error = false;
    let mut n = 0usize;
    let mut sum = 0.0f32;

    let albums = manifest["albums"].as_object().cloned().unwrap_or_default();
    for (album, entry) in &albums {
        let Some(photos) = entry["photos"].as_array() else { continue };
        let out_dir = out.join(album);
        if let Err(e) = std::fs::create_dir_all(&out_dir) {
            eprintln!("autofocus: cannot create {}: {e}", out_dir.display());
            process::exit(1);
        }
        for p in photos {
            let file = p["file"].as_str().unwrap_or_default();
            let gx = p["focus"]["x"].as_f64().unwrap_or(0.5) as f32;
            let gy = p["focus"]["y"].as_f64().unwrap_or(0.5) as f32;
            let src = fixtures.join(album).join(file);

            match review_photo(&src, gx, gy, Some(&out_dir), max_dim) {
                Ok(dist) => {
                    println!("{album}/{file}: d={dist:.3}");
                    n += 1;
                    sum += dist;
                }
                Err(e) => {
                    eprintln!("{}/{file}: {e}", album);
                    any_error = true;
                }
            }
        }
    }

    if n > 0 {
        println!("reviewed {n} photos, mean distance {:.4} -> {}", sum / n as f32, out.display());
    }
    if any_error {
        process::exit(1);
    }
}

/// Detect one golden photo, measure the distance to the hand-set point and,
/// when `out_dir` is given, write the annotated comparison render there with
/// the distance encoded in the filename (reverse sort = worst first).
fn review_photo(src: &Path, gx: f32, gy: f32, out_dir: Option<&Path>, max_dim: u32) -> Result<f32, String> {
    /// Long edge of the written review copies — big enough to judge a face,
    /// small enough that the whole directory stays skimmable.
    const REVIEW_DIM: u32 = 1200;

    let ((dx, dy, _category), orig) =
        process_image_with_orig(src, max_dim).map_err(|e| format!("ERROR {e}"))?;

    let dist = ((dx - gx).powi(2) + (dy - gy).powi(2)).sqrt();

    if let Some(out_dir) = out_dir {
        // Downscale first so marker strokes are uniform across the set.
        let scale = (REVIEW_DIM as f32 / orig.width().max(orig.height()) as f32).min(1.0);
        let rw = (orig.width() as f32 * scale).round() as u32;
        let rh = (orig.height() as f32 * scale).round() as u32;
        let mut canvas = orig
            .resize_exact(rw, rh, FilterType::Triangle)
            .to_rgba8();
        render::draw_comparison(&mut canvas, (gx, gy), (dx, dy));

        let file = src.file_name().unwrap_or_default().to_string_lossy();
        let out_file = out_dir.join(format!("d{dist:.3}-{file}"));
        std::fs::File::create(&out_file).map_err(|e| format!("WRITE ERROR {e}")).and_then(|f| {
            let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(std::io::BufWriter::new(f), 85);
            // JPEG has no alpha channel; the markers are opaque anyway.
            let rgb = image::DynamicImage::ImageRgba8(canvas).to_rgb8();
            enc.encode_image(&rgb).map_err(|e| format!("WRITE ERROR {e}"))
        })?;
    }

    Ok(dist)
}

/// Eval mode: walk an external dataset directory — photos plus a
/// focuspoints.json of {"file.jpg": {"x": …, "y": …}} — and print per-photo
/// distances and the mean. With `out`, also render the annotated comparisons.
fn run_eval(dir: &Path, out: Option<&Path>, max_dim: u32) {
    let manifest_path = dir.join("focuspoints.json");
    let raw = match std::fs::read_to_string(&manifest_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("autofocus: cannot read {}: {e}", manifest_path.display());
            process::exit(1);
        }
    };
    let manifest: serde_json::Value = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("autofocus: {} is not valid JSON: {e}", manifest_path.display());
            process::exit(1);
        }
    };
    let Some(points) = manifest.as_object() else {
        eprintln!("autofocus: {} must be an object of filename -> {{x, y}}", manifest_path.display());
        process::exit(1);
    };

    if let Some(out) = out {
        if let Err(e) = std::fs::create_dir_all(out) {
            eprintln!("autofocus: cannot create {}: {e}", out.display());
            process::exit(1);
        }
    }

    let mut any_error = false;
    let mut n = 0usize;
    let mut sum = 0.0f32;

    // BTreeMap-style order: deterministic output regardless of JSON order.
    let mut files: Vec<&String> = points.keys().collect();
    files.sort();
    for file in files {
        let p = &points[file];
        let gx = p["x"].as_f64().unwrap_or(0.5) as f32;
        let gy = p["y"].as_f64().unwrap_or(0.5) as f32;
        match review_photo(&dir.join(file), gx, gy, out, max_dim) {
            Ok(dist) => {
                println!("{file}: d={dist:.3}");
                n += 1;
                sum += dist;
            }
            Err(e) => {
                eprintln!("{file}: {e}");
                any_error = true;
            }
        }
    }

    if n > 0 {
        println!("evaluated {n} photos, mean distance {:.4}", sum / n as f32);
    }
    if any_error {
        process::exit(1);
    }
}

/// Bounding box (normalised) of the topmost skin cluster: seed at the
/// skin block with the smallest cy, extended downward while rows stay
/// connected. Faces are the highest skin mass in almost all photos.
fn topmost_skin_region(blocks: &[(f32, f32, f32, f32)]) -> Option<(f32, f32, f32, f32)> {
    let skin: Vec<(f32, f32, f32, f32)> = blocks.iter().filter(|b| b.2 >= 0.30).map(|b| (b.0, b.1, b.3, b.2)).collect();
    if skin.is_empty() { return None; }
    let seed = *skin.iter().min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())?;
    let mut members: Vec<(f32, f32, f32, f32)> = vec![seed];
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
    // 2, not 3: a distant face at 256px is often two skin blocks. The crop
    // gates (pair geometry, mouth band, brow co-location) do the real
    // filtering; this guard only screens degenerate one-block regions.
    if members.len() < 2 { return None; }
    // Eye-band OR strong skin: a small real face at 256px can peak as low as
    // 0.06 eye-band, and a *distant* head can show none at all — but a real
    // head is strong skin (>= 0.55), while the warm-sky false clusters this
    // guard exists for read weak. The crop gates (pair geometry, mouth band,
    // brow co-location) remain the deciding filter either way.
    if !members.iter().any(|m| m.2 >= 0.05 || m.3 >= 0.55) { return None; }
    // Shape guard: a head cluster is compact; a raised arm is a one-block
    // sliver (measured ~10:1) whose hand/bracelet manufactures pair geometry
    // at crop scale. Reject extreme aspect ratios before the crop gates ever
    // see them. 3.5 keeps real narrow heads (measured up to ~2.3).
    {
        let bw = members.iter().map(|m| m.0).fold(f32::MIN, f32::max)
            - members.iter().map(|m| m.0).fold(f32::MAX, f32::min);
        let bh = members.iter().map(|m| m.1).fold(f32::MIN, f32::max)
            - members.iter().map(|m| m.1).fold(f32::MAX, f32::min);
        let (long, short) = (bw.max(bh), bw.min(bh).max(0.03));
        if long / short > 3.5 { return None; }
    }
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
