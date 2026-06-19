//! Card geometry persistence: where the card sits and how big it is, remembered
//! across restarts.
//!
//! Config supplies the initial geometry; once the user drags or resizes the card,
//! the result is saved here and overrides the config defaults on the next launch.
//! A missing or unreadable file is not an error — it just means "use the config
//! defaults". Not here: the card itself, or when to save (that is `ui`).

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tracing::warn;

/// The card's saved position and size (surface-relative pixels).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Geometry {
    pub pos_x: f32,
    pub pos_y: f32,
    pub width: f32,
    pub height: f32,
}

/// Saved card geometry. An absent entry falls back to the config defaults.
#[derive(Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Layout {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub card: Option<Geometry>,
}

/// The layout file path: `$XDG_STATE_HOME/kitetsu/layout.toml`, falling back to
/// `~/.local/state/kitetsu/layout.toml`, then the temp dir if neither is set.
pub fn layout_path() -> PathBuf {
    let dir = std::env::var_os("XDG_STATE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/state")))
        .unwrap_or_else(std::env::temp_dir);
    dir.join("kitetsu").join("layout.toml")
}

/// Load the saved layout. A missing file yields an empty layout; a corrupt file
/// is warned about and treated as empty (config defaults win) rather than failing
/// the launch.
pub fn load(path: &Path) -> Layout {
    match std::fs::read_to_string(path) {
        Ok(text) => toml::from_str(&text).unwrap_or_else(|e| {
            warn!(path = %path.display(), error = %e, "ignoring unreadable layout file");
            Layout::default()
        }),
        Err(_) => Layout::default(),
    }
}

/// Write the layout, creating the parent directory if needed.
pub fn save(path: &Path, layout: &Layout) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let text = toml::to_string_pretty(layout).map_err(std::io::Error::other)?;
    std::fs::write(path, text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_toml() {
        let layout = Layout {
            card: Some(Geometry { pos_x: 40.0, pos_y: 40.0, width: 612.0, height: 380.0 }),
        };
        let text = toml::to_string_pretty(&layout).expect("serialise");
        let back: Layout = toml::from_str(&text).expect("deserialise");
        assert_eq!(layout, back);
    }

    #[test]
    fn missing_file_is_empty_layout() {
        let layout = load(Path::new("/nonexistent/kitetsu/layout.toml"));
        assert_eq!(layout, Layout::default());
        assert_eq!(layout.card, None);
    }

    #[test]
    fn corrupt_file_falls_back_to_empty() {
        let dir = std::env::temp_dir().join(format!("kitetsu-layout-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let path = dir.join("layout.toml");
        std::fs::write(&path, "this is not valid toml = = =").expect("write");
        assert_eq!(load(&path), Layout::default());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn path_ends_with_kitetsu_layout_toml() {
        let p = layout_path();
        assert!(p.ends_with("kitetsu/layout.toml"), "got {}", p.display());
    }
}
