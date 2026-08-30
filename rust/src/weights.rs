//! Block-score weights and the per-category presets.

pub(crate) const WEIGHT_DETAIL: f32   = 0.11;
const WEIGHT_SKIN: f32     = 0.59;
const WEIGHT_SAT: f32      = 2.4;
const WEIGHT_SHARP: f32    = 0.99;
const WEIGHT_CENTRE: f32   = 1.67;
const WEIGHT_HOG_FACE: f32 = 7.56;
const WEIGHT_SYMMETRY: f32 = 2.35;
const WEIGHT_THIRDS: f32   = 0.60;

pub(crate) const SKIN_DETAIL_BIAS: f32 = 0.01;
pub(crate) const SKIN_EDGE_NORM: f32   = 0.05;

// ---------------------------------------------------------------------------
//  Weight presets
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub(crate) struct Weights {
    pub(crate) detail:   f32,
    pub(crate) skin:     f32,
    pub(crate) sat:      f32,
    pub(crate) sharp:    f32,
    pub(crate) centre:   f32,
    pub(crate) hog_face: f32,
    pub(crate) symmetry: f32,
    pub(crate) thirds:   f32,
}

impl Weights {
    pub(crate) fn default() -> Self {
        Self {
            detail:   WEIGHT_DETAIL,
            skin:     WEIGHT_SKIN,
            sat:      WEIGHT_SAT,
            sharp:    WEIGHT_SHARP,
            centre:   WEIGHT_CENTRE,
            hog_face: WEIGHT_HOG_FACE,
            symmetry: WEIGHT_SYMMETRY,
            thirds:   WEIGHT_THIRDS,
        }
    }

    pub(crate) fn portrait() -> Self { Self::default() }

    pub(crate) fn group() -> Self {
        Self { centre: 1.9, thirds: 0.85, ..Self::default() }
    }

    pub(crate) fn people() -> Self {
        Self {
            hog_face: 2.8,
            symmetry: 1.1,
            skin: 0.45,
            detail: 0.18,
            ..Self::default()
        }
    }

    pub(crate) fn scene() -> Self {
        Self {
            detail:   0.28,
            skin:     0.12,
            sat:      2.4,
            sharp:    1.35,
            centre:   1.67,
            hog_face: 0.4,
            symmetry: 0.6,
            thirds:   1.05,
        }
    }

    pub(crate) fn mono() -> Self {
        Self {
            detail:   0.42,
            skin:     0.0,
            sat:      0.0,
            sharp:    1.75,
            centre:   1.45,
            hog_face: 2.4,
            symmetry: 1.25,
            thirds:   0.95,
        }
    }
}
