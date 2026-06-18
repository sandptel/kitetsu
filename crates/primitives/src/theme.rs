//! The colour palette the overlay paints from: a base16 scheme plus the few
//! semantic accessors the presets actually read. Pure data — the host loads the
//! file and builds a [`Base16`]; this module never touches the filesystem or TOML
//! (it only knows how to turn one `#rrggbb` string into a [`Color`]).

use iced::Color;

/// A base16 palette: `base00` (darkest background) … `base07` (lightest text),
/// then `base08`…`base0F` accent hues. Slots follow the base16 convention so an
/// existing scheme file drops in unchanged; the presets read it via the semantic
/// accessors below, not the raw slots.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Base16 {
    pub base00: Color,
    pub base01: Color,
    pub base02: Color,
    pub base03: Color,
    pub base04: Color,
    pub base05: Color,
    pub base06: Color,
    pub base07: Color,
    pub base08: Color,
    pub base09: Color,
    pub base0a: Color,
    pub base0b: Color,
    pub base0c: Color,
    pub base0d: Color,
    pub base0e: Color,
    pub base0f: Color,
}

impl Base16 {
    /// Card background (base00).
    pub fn background(&self) -> Color {
        self.base00
    }
    /// Raised surface / chrome fill (base01).
    pub fn surface(&self) -> Color {
        self.base01
    }
    /// Muted secondary text, e.g. the model line (base03).
    pub fn muted(&self) -> Color {
        self.base03
    }
    /// Primary body text (base05).
    pub fn text(&self) -> Color {
        self.base05
    }
    /// Accent / highlight (base0D).
    pub fn accent(&self) -> Color {
        self.base0d
    }
}

/// Opaque greyscale slot at the given 0–255 level / 255.
const fn grey(v: f32) -> Color {
    Color {
        r: v,
        g: v,
        b: v,
        a: 1.0,
    }
}

/// Opaque colour from 0.0–1.0 components.
const fn rgb(r: f32, g: f32, b: f32) -> Color {
    Color { r, g, b, a: 1.0 }
}

/// Baked-in default palette (base16 "default dark"), used when no valid
/// `colors.toml` is present.
pub const DEFAULT: Base16 = Base16 {
    base00: grey(0.094),
    base01: grey(0.157),
    base02: grey(0.220),
    base03: grey(0.345),
    base04: grey(0.722),
    base05: grey(0.847),
    base06: grey(0.910),
    base07: grey(0.973),
    base08: rgb(0.671, 0.275, 0.259),
    base09: rgb(0.863, 0.588, 0.337),
    base0a: rgb(0.969, 0.792, 0.533),
    base0b: rgb(0.631, 0.710, 0.424),
    base0c: rgb(0.525, 0.757, 0.725),
    base0d: rgb(0.486, 0.686, 0.761),
    base0e: rgb(0.729, 0.545, 0.686),
    base0f: rgb(0.631, 0.412, 0.275),
};

/// Parse one `#rrggbb` (or bare `rrggbb`) hex string into a [`Color`].
/// `None` on any malformed input so the host can fall back to [`DEFAULT`].
pub fn hex(s: &str) -> Option<Color> {
    let s = s.trim().strip_prefix('#').unwrap_or_else(|| s.trim());
    if s.len() != 6 {
        return None;
    }
    let n = u32::from_str_radix(s, 16).ok()?;
    Some(Color::from_rgb8(
        (n >> 16) as u8,
        (n >> 8) as u8,
        n as u8,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_hex_with_and_without_hash() {
        let c = hex("#7cafc2").expect("valid hex");
        assert_eq!(hex("7cafc2"), Some(c));
        assert_eq!(c, Color::from_rgb8(0x7c, 0xaf, 0xc2));
    }

    #[test]
    fn rejects_malformed_hex() {
        assert_eq!(hex("#fff"), None);
        assert_eq!(hex("#gggggg"), None);
        assert_eq!(hex(""), None);
    }
}
