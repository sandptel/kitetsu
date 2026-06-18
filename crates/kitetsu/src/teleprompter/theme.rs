//! Load the overlay colour palette from `<config_dir>/colors.toml`.
//!
//! The file is a flat base16 table (`base00 = "#rrggbb"` … `base0F`). Any
//! problem — file absent, bad TOML, a malformed hex slot — falls back to the
//! baked [`primitives::theme::DEFAULT`] so the overlay always has colours.
//! Parsing lives here (the host already has serde/toml); the primitives crate
//! stays filesystem- and TOML-free.

use std::path::Path;

use kitetsu_primitives::theme::{self, Base16};
use serde::Deserialize;
use tracing::{debug, warn};

/// The on-disk schema: sixteen `#rrggbb` hex strings, all required.
#[derive(Deserialize)]
struct Raw {
    base00: String,
    base01: String,
    base02: String,
    base03: String,
    base04: String,
    base05: String,
    base06: String,
    base07: String,
    base08: String,
    base09: String,
    base0a: String,
    base0b: String,
    base0c: String,
    base0d: String,
    base0e: String,
    base0f: String,
}

impl Raw {
    /// Convert to a palette, or `None` if any slot is not valid hex.
    fn into_palette(self) -> Option<Base16> {
        Some(Base16 {
            base00: theme::hex(&self.base00)?,
            base01: theme::hex(&self.base01)?,
            base02: theme::hex(&self.base02)?,
            base03: theme::hex(&self.base03)?,
            base04: theme::hex(&self.base04)?,
            base05: theme::hex(&self.base05)?,
            base06: theme::hex(&self.base06)?,
            base07: theme::hex(&self.base07)?,
            base08: theme::hex(&self.base08)?,
            base09: theme::hex(&self.base09)?,
            base0a: theme::hex(&self.base0a)?,
            base0b: theme::hex(&self.base0b)?,
            base0c: theme::hex(&self.base0c)?,
            base0d: theme::hex(&self.base0d)?,
            base0e: theme::hex(&self.base0e)?,
            base0f: theme::hex(&self.base0f)?,
        })
    }
}

/// Resolve `<config_dir>/colors.toml` into a palette, falling back to the baked
/// default (and warning) on any malformed file. A missing file is normal.
pub fn load(config_dir: &Path) -> Base16 {
    let path = config_dir.join("colors.toml");
    let raw = match std::fs::read_to_string(&path) {
        Ok(raw) => raw,
        Err(e) => {
            debug!(path = %path.display(), %e, "no colors.toml — using default palette");
            return theme::DEFAULT;
        }
    };
    match toml::from_str::<Raw>(&raw).ok().and_then(Raw::into_palette) {
        Some(palette) => palette,
        None => {
            warn!(path = %path.display(), "invalid colors.toml — using default palette");
            theme::DEFAULT
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_table_parses_to_palette() {
        let toml = (0..16)
            .map(|i| format!("base{i:02x} = \"#101010\"\n"))
            .collect::<String>();
        let raw: Raw = toml::from_str(&toml).expect("valid toml");
        let p = raw.into_palette().expect("valid hex");
        assert_eq!(p.text(), theme::hex("#101010").unwrap());
    }

    #[test]
    fn malformed_hex_slot_rejected() {
        let mut toml = (0..16)
            .map(|i| format!("base{i:02x} = \"#101010\"\n"))
            .collect::<String>();
        toml = toml.replace("base05 = \"#101010\"", "base05 = \"nope\"");
        let raw: Raw = toml::from_str(&toml).expect("valid toml");
        assert!(raw.into_palette().is_none());
    }
}
