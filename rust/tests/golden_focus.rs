//! Golden focus tests.
//!
//! Runs the shipped CLI pipeline (decode → EXIF orientation → downscale →
//! detect → zoom pass) over the fixture photos in `ui/tests/fixtures/` and
//! measures the distance from each detected point to the focus point the
//! photographer set by hand. The manifest (`fixtures.json`) is the ground
//! truth; photos stay grouped by album so a regression shows up as *which
//! group* drifted, not just a global average.
//!
//! Thresholds are regression guards, not aspirations: they are the measured
//! baseline (2026-08-31, commit of first import) plus a small margin.
//! When the algorithm improves, tighten them.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Per-album mean-distance baselines (measured) + margin.
const MARGIN: f32 = 0.05;
const BASELINE_MEAN: &[(&str, f32)] = &[
    ("album-01", 0.155),
    ("album-02", 0.050),
    ("album-03", 0.111),
    ("album-04", 0.020),
    ("album-05", 0.207),
    ("album-06", 0.034),
    ("album-07", 0.106),
    ("album-08", 0.218),
    ("album-09", 0.027),
    ("album-10", 0.030),
];
const BASELINE_OVERALL_MEAN: f32 = 0.088;
/// No single photo may drift beyond this (measured max: 0.381).
const WORST_CASE: f32 = 0.45;

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../ui/tests/fixtures")
}

struct Golden {
    album: String,
    path: PathBuf,
    x: f32,
    y: f32,
}

fn load_manifest() -> Vec<Golden> {
    let root = fixtures_dir();
    let raw = std::fs::read_to_string(root.join("fixtures.json"))
        .expect("ui/tests/fixtures/fixtures.json missing");
    let json: serde_json::Value = serde_json::from_str(&raw).expect("manifest is not valid JSON");

    let mut out = Vec::new();
    for (album, entry) in json["albums"].as_object().expect("albums object") {
        for p in entry["photos"].as_array().expect("photos array") {
            out.push(Golden {
                album: album.clone(),
                path: root.join(album).join(p["file"].as_str().expect("file")),
                x: p["focus"]["x"].as_f64().expect("focus.x") as f32,
                y: p["focus"]["y"].as_f64().expect("focus.y") as f32,
            });
        }
    }
    out
}

/// Run the real CLI binary over all fixture photos in one invocation and
/// return detected points keyed by filename.
fn detect_all(goldens: &[Golden]) -> HashMap<String, (f32, f32)> {
    let output = Command::new(env!("CARGO_BIN_EXE_autofocus"))
        .arg("--format")
        .arg("json")
        .args(goldens.iter().map(|g| &g.path))
        .output()
        .expect("failed to spawn the autofocus binary");
    assert!(
        output.status.success(),
        "autofocus CLI failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut points = HashMap::new();
    for line in stdout.lines() {
        let Some(start) = line.find('{') else { continue };
        let j: serde_json::Value = serde_json::from_str(&line[start..]).expect("CLI json line");
        let file = Path::new(j["file"].as_str().expect("file"))
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string();
        points.insert(
            file,
            (
                j["x"].as_f64().expect("x") as f32,
                j["y"].as_f64().expect("y") as f32,
            ),
        );
    }
    points
}

#[test]
fn detected_points_stay_within_the_golden_baselines() {
    let goldens = load_manifest();
    assert!(!goldens.is_empty(), "no fixtures found");
    let points = detect_all(&goldens);

    let mut per_album: HashMap<&str, Vec<f32>> = HashMap::new();
    let mut failures = Vec::new();

    for g in &goldens {
        let file = g.path.file_name().unwrap().to_string_lossy().to_string();
        let (dx, dy) = points
            .get(&file)
            .unwrap_or_else(|| panic!("CLI produced no point for {file}"));
        let dist = ((dx - g.x).powi(2) + (dy - g.y).powi(2)).sqrt();
        per_album.entry(g.album.as_str()).or_default().push(dist);
        if dist > WORST_CASE {
            failures.push(format!("{}/{file}: dist {dist:.3} > {WORST_CASE}", g.album));
        }
    }

    let mut report = String::new();
    let mut all = Vec::new();
    let baselines: HashMap<_, _> = BASELINE_MEAN.iter().copied().collect();
    let mut albums: Vec<_> = per_album.iter().collect();
    albums.sort_by_key(|(a, _)| *a);
    for (album, dists) in albums {
        let mean = dists.iter().sum::<f32>() / dists.len() as f32;
        all.extend_from_slice(dists);
        let budget = baselines
            .get(album)
            .copied()
            .unwrap_or_else(|| panic!("no baseline for {album} — add it to BASELINE_MEAN"))
            + MARGIN;
        report.push_str(&format!(
            "  {album}: mean {mean:.4} (budget {budget:.3}, n={})\n",
            dists.len()
        ));
        if mean > budget {
            failures.push(format!("{album}: mean {mean:.4} > budget {budget:.3}"));
        }
    }
    let overall = all.iter().sum::<f32>() / all.len() as f32;
    report.push_str(&format!("  overall: mean {overall:.4} (n={})\n", all.len()));
    println!("golden focus distances:\n{report}");

    if overall > BASELINE_OVERALL_MEAN + MARGIN {
        failures.push(format!(
            "overall mean {overall:.4} > {:.3}",
            BASELINE_OVERALL_MEAN + MARGIN
        ));
    }
    assert!(failures.is_empty(), "golden regressions:\n{}", failures.join("\n"));
}
