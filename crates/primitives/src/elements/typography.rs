//! Shared typography: the embedded JetBrains Mono faces, the family name, a
//! weight helper, and the default body size. Every preset renders text in this
//! family so the overlay looks consistent. Deliberately no layout, no colour.

use iced::{Font, font};

/// JetBrains Mono (Nerd Font build) — the family name to select on text nodes.
/// The bytes are registered by the host app once; this just names the family.
pub const FONT_NAME: &str = "JetBrainsMono Nerd Font";

/// Regular + bold faces, exposed so the host can register them with iced once.
pub const FONT_REGULAR: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/JetBrainsMonoNerdFont-Regular.ttf"
));
pub const FONT_BOLD: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/JetBrainsMonoNerdFont-Bold.ttf"
));

/// Base body font size used when a preset does not override it.
pub const DEFAULT_FONT_SIZE: f32 = 19.0;

/// JetBrains Mono at the given weight. Explicit family on every text node because
/// the iced default font wins only when no font is set, and bold spans must still
/// resolve to JetBrains Mono rather than the generic bold face.
pub fn jb(weight: font::Weight) -> Font {
    Font {
        weight,
        ..Font::with_name(FONT_NAME)
    }
}
