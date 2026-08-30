//! The panel persists detect_features output into the image_features table,
//! so the payload shape is a storage contract, not just a debug string.

use autofocus_wasm::detect_features;

/// A synthetic image: dark left half, bright right half, 64x64.
fn synthetic(w: usize, h: usize) -> Vec<u8> {
    let mut rgba = vec![0u8; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let v = if x < w / 2 { 30 } else { 220 };
            let o = (y * w + x) * 4;
            rgba[o] = v;
            rgba[o + 1] = v;
            rgba[o + 2] = v;
            rgba[o + 3] = 255;
        }
    }
    rgba
}

#[test]
fn payload_is_valid_json_with_every_stored_field() {
    let json = detect_features(&synthetic(64, 64), 64, 64);
    let v: serde_json::Value = serde_json::from_str(&json).expect("payload must parse as JSON");

    for key in ["x", "y", "avg_skin", "avg_sat", "avg_sharp", "symmetry", "edge_energy"] {
        let n = v[key].as_f64().unwrap_or_else(|| panic!("{key} must be numeric"));
        assert!((0.0..=1.0).contains(&n), "{key} = {n} must be within [0, 1]");
    }

    let category = v["category"].as_str().expect("category must be a string");
    assert!(!category.is_empty());

    let ahash = v["ahash"].as_str().expect("ahash must be a string");
    assert_eq!(ahash.len(), 16, "ahash is 64 bits as 16 hex chars");
    assert!(ahash.chars().all(|c| c.is_ascii_hexdigit()));
}

#[test]
fn ahash_reflects_luminance_structure() {
    // Half dark / half bright: exactly the bright half's 32 bits are set,
    // and mirroring the image mirrors the hash rather than reproducing it.
    let json_a = detect_features(&synthetic(64, 64), 64, 64);
    let a: serde_json::Value = serde_json::from_str(&json_a).unwrap();

    let mut mirrored = synthetic(64, 64);
    for y in 0..64 {
        for x in 0..32 {
            for c in 0..4 {
                mirrored.swap((y * 64 + x) * 4 + c, (y * 64 + (63 - x)) * 4 + c);
            }
        }
    }
    let json_b = detect_features(&mirrored, 64, 64);
    let b: serde_json::Value = serde_json::from_str(&json_b).unwrap();

    let bits_a = u64::from_str_radix(a["ahash"].as_str().unwrap(), 16).unwrap();
    let bits_b = u64::from_str_radix(b["ahash"].as_str().unwrap(), 16).unwrap();

    assert_eq!(bits_a.count_ones(), 32);
    assert_ne!(bits_a, bits_b, "a mirrored image must not share the hash");
}

#[test]
fn focus_point_matches_detect_focus() {
    // The stored features row must describe the same point detect_focus
    // returns — the two exports share one pipeline by construction.
    let img = synthetic(96, 96);
    let point = autofocus_wasm::detect_focus_cli(&img, 96, 96);
    let v: serde_json::Value =
        serde_json::from_str(&detect_features(&img, 96, 96)).unwrap();

    assert!((v["x"].as_f64().unwrap() - point.0 as f64).abs() < 1e-3);
    assert!((v["y"].as_f64().unwrap() - point.1 as f64).abs() < 1e-3);
}
