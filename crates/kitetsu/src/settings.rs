//! Central kitetsu configuration: which tools the supervisor runs.
//!
//! Parses `<config_dir>/kitetsu.toml` — tool on/off toggles only — and resolves
//! the config directory (XDG, with an explicit override). Each tool's own
//! settings live under `<config_dir>/<tool>/` and are loaded by that tool, not
//! here. Deliberately tiny: this is the registry, not any tool's config.

use std::path::{Path, PathBuf};

use serde::Deserialize;

/// Failure to read or parse the central config.
#[derive(Debug, thiserror::Error)]
pub enum SettingsError {
    /// The file exists but could not be read.
    #[error("reading {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    /// The file was not valid TOML or did not match the schema.
    #[error("parsing {path}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },
}

/// On/off toggle for one tool. Disabled unless the file opts it in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ToolToggle {
    pub enabled: bool,
}

impl Default for ToolToggle {
    fn default() -> Self {
        Self { enabled: false }
    }
}

/// The central registry: which tools the supervisor should start.
#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Settings {
    pub teleprompter: ToolToggle,
}

impl Settings {
    /// Read `<config_dir>/kitetsu.toml`. A missing file means "nothing enabled"
    /// (all defaults) so a bare install is valid; a present-but-malformed file
    /// is an error rather than silently ignored.
    pub fn load(config_dir: &Path) -> Result<Self, SettingsError> {
        let path = config_dir.join("kitetsu.toml");
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Self::default()),
            Err(source) => return Err(SettingsError::Read { path, source }),
        };
        toml::from_str(&text).map_err(|source| SettingsError::Parse { path, source })
    }
}

/// Resolve the config directory: an explicit override wins, else
/// `$XDG_CONFIG_HOME/kitetsu`, else `$HOME/.config/kitetsu`, else (no env) a
/// cwd-relative `.config/kitetsu` as a last resort.
pub fn config_dir(override_dir: Option<&Path>) -> PathBuf {
    if let Some(dir) = override_dir {
        return dir.to_path_buf();
    }
    if let Some(xdg) = std::env::var_os("XDG_CONFIG_HOME").filter(|v| !v.is_empty()) {
        return PathBuf::from(xdg).join("kitetsu");
    }
    if let Some(home) = std::env::var_os("HOME").filter(|v| !v.is_empty()) {
        return PathBuf::from(home).join(".config").join("kitetsu");
    }
    PathBuf::from(".config").join("kitetsu")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_file_means_all_tools_disabled() {
        let dir = std::env::temp_dir().join("kitetsu-settings-test-absent");
        // Ensure the file does not exist.
        let _ = std::fs::remove_file(dir.join("kitetsu.toml"));
        let settings = Settings::load(&dir).expect("missing file is not an error");
        assert_eq!(settings, Settings::default());
        assert!(!settings.teleprompter.enabled);
    }

    #[test]
    fn empty_section_defaults_to_disabled() {
        let settings: Settings = toml::from_str("[teleprompter]\n").expect("parses");
        assert!(!settings.teleprompter.enabled);
    }

    #[test]
    fn enabled_toggle_parses() {
        let settings: Settings =
            toml::from_str("[teleprompter]\nenabled = true\n").expect("parses");
        assert!(settings.teleprompter.enabled);
    }

    #[test]
    fn unknown_tool_key_is_rejected() {
        let err = toml::from_str::<Settings>("[teleprompter]\nenable = true\n").unwrap_err();
        assert!(err.to_string().contains("enable") || err.to_string().contains("unknown"));
    }

    #[test]
    fn override_dir_wins() {
        let explicit = Path::new("/tmp/kitetsu-explicit");
        assert_eq!(config_dir(Some(explicit)), explicit.to_path_buf());
    }

    #[test]
    fn config_dir_ends_in_kitetsu() {
        // Whatever the environment, the resolved default lives in a `kitetsu` dir.
        assert_eq!(
            config_dir(None).file_name().and_then(|s| s.to_str()),
            Some("kitetsu")
        );
    }
}
