//! Prototype launcher for the kitetsu DescriptionCard primitive.
//! Shows the card on a wlr-layer-shell overlay surface (bottom-left, click/keyboard
//! pass-through) via layer-shika, which loads `ui/card.slint` through the Slint
//! interpreter. No IPC, no LLM, no daemon — surface + layout only.
//!
//! The layer-shell surface belongs in `presenter` in the real architecture; this
//! scaffold just proves it renders. Dummy content lives in the `.slint` defaults;
//! feeding a Rust spec via `set_property` is the next (spec-layer) iteration.

use std::path::PathBuf;

use layer_shika::prelude::*;

fn main() -> Result<()> {
    let ui = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("ui/card.slint");

    Shell::from_file(ui)
        .surface("CardWindow")
        .size(900, 420)
        .anchor(AnchorEdges::empty().with_bottom().with_left())
        .margin((0, 0, 16, 16)) // top, right, bottom, left
        .layer(Layer::Overlay)
        .keyboard_interactivity(KeyboardInteractivity::None)
        .namespace("kitetsu")
        .run()
}
