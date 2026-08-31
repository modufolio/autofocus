//! Marker drawing for the fixture review mode (native only).
//!
//! Hand-rolled on `RgbaImage` — the crate deliberately carries no drawing or
//! font dependency. Every shape is drawn twice: a black pass one pixel wider
//! first, then the colour on top, so markers stay legible on any photograph.

use image::{Rgba, RgbaImage};

const GOLDEN: Rgba<u8> = Rgba([0, 220, 90, 255]);
const DETECTED: Rgba<u8> = Rgba([255, 90, 0, 255]);
const HALO: Rgba<u8> = Rgba([0, 0, 0, 255]);

/// Draw the golden (hand-set) point, the detected point and a line between
/// them. Coordinates are normalised [0,1]; sizes scale with the image.
pub(crate) fn draw_comparison(
    img: &mut RgbaImage,
    golden: (f32, f32),
    detected: (f32, f32),
) {
    let (w, h) = (img.width() as f32, img.height() as f32);
    let long_edge = w.max(h);
    let r = (long_edge / 90.0).max(7.0);

    let g = (golden.0 * w, golden.1 * h);
    let d = (detected.0 * w, detected.1 * h);

    // Connecting line under both markers.
    draw_line(img, g, d, 2.0, HALO);
    draw_line(img, g, d, 1.0, Rgba([255, 255, 255, 255]));

    // Golden: ring with a small filled centre.
    draw_ring(img, g, r, 3.0, HALO);
    draw_ring(img, g, r, 1.5, GOLDEN);
    draw_disc(img, g, r * 0.25 + 1.0, HALO);
    draw_disc(img, g, r * 0.25, GOLDEN);

    // Detected: crosshair.
    draw_crosshair(img, d, r * 1.2, 2.5, HALO);
    draw_crosshair(img, d, r * 1.2, 1.2, DETECTED);
}

/// Filled disc.
fn draw_disc(img: &mut RgbaImage, c: (f32, f32), radius: f32, color: Rgba<u8>) {
    stroke_region(img, c, radius + 1.0, |dx, dy| {
        (dx * dx + dy * dy).sqrt() <= radius
    }, color);
}

/// Circle outline of the given stroke width.
fn draw_ring(img: &mut RgbaImage, c: (f32, f32), radius: f32, stroke: f32, color: Rgba<u8>) {
    stroke_region(img, c, radius + stroke + 1.0, |dx, dy| {
        let d = (dx * dx + dy * dy).sqrt();
        (d - radius).abs() <= stroke * 0.5
    }, color);
}

/// Plus-shaped crosshair with a gap at the exact centre so the pixel under
/// the point stays visible.
fn draw_crosshair(img: &mut RgbaImage, c: (f32, f32), arm: f32, stroke: f32, color: Rgba<u8>) {
    let gap = arm * 0.25;
    stroke_region(img, c, arm + stroke + 1.0, |dx, dy| {
        let (ax, ay) = (dx.abs(), dy.abs());
        let on_h = ay <= stroke * 0.5 && ax <= arm && ax >= gap;
        let on_v = ax <= stroke * 0.5 && ay <= arm && ay >= gap;
        on_h || on_v
    }, color);
}

/// Thick line via distance-to-segment test over the segment's bounding box.
fn draw_line(img: &mut RgbaImage, a: (f32, f32), b: (f32, f32), stroke: f32, color: Rgba<u8>) {
    let (min_x, max_x) = (a.0.min(b.0) - stroke, a.0.max(b.0) + stroke);
    let (min_y, max_y) = (a.1.min(b.1) - stroke, a.1.max(b.1) + stroke);
    let (abx, aby) = (b.0 - a.0, b.1 - a.1);
    let len2 = (abx * abx + aby * aby).max(1e-6);

    let x0 = min_x.max(0.0) as u32;
    let x1 = (max_x.min(img.width() as f32 - 1.0)).max(0.0) as u32;
    let y0 = min_y.max(0.0) as u32;
    let y1 = (max_y.min(img.height() as f32 - 1.0)).max(0.0) as u32;

    for y in y0..=y1 {
        for x in x0..=x1 {
            let (px, py) = (x as f32 - a.0, y as f32 - a.1);
            let t = ((px * abx + py * aby) / len2).clamp(0.0, 1.0);
            let (dx, dy) = (px - t * abx, py - t * aby);
            if (dx * dx + dy * dy).sqrt() <= stroke * 0.5 {
                img.put_pixel(x, y, color);
            }
        }
    }
}

/// Iterate the bounding box around `c` and paint pixels where `hit` says so.
fn stroke_region(
    img: &mut RgbaImage,
    c: (f32, f32),
    extent: f32,
    hit: impl Fn(f32, f32) -> bool,
    color: Rgba<u8>,
) {
    let x0 = (c.0 - extent).max(0.0) as u32;
    let x1 = ((c.0 + extent).min(img.width() as f32 - 1.0)).max(0.0) as u32;
    let y0 = (c.1 - extent).max(0.0) as u32;
    let y1 = ((c.1 + extent).min(img.height() as f32 - 1.0)).max(0.0) as u32;

    for y in y0..=y1 {
        for x in x0..=x1 {
            if hit(x as f32 - c.0, y as f32 - c.1) {
                img.put_pixel(x, y, color);
            }
        }
    }
}
