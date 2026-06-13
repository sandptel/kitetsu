//! API key resolution from process environment and `.env` files.
//!
//! Named constants label each service's env-var name. `load_dotenv` merges a
//! `.env` file into the process environment without overriding already-set vars
//! (shell wins over file). `require` fetches a named key or returns a typed error.
//! Not here: HTTP logic, model config, or value parsing beyond plain strings.

use std::path::Path;

/// OpenAI REST and Realtime API key (`OPENAI_API_KEY`).
pub const OPENAI_API_KEY: &str = "OPENAI_API_KEY";
/// Anthropic Claude API key (`ANTHROPIC_API_KEY`).
pub const ANTHROPIC_API_KEY: &str = "ANTHROPIC_API_KEY";

/// Failure to resolve a required API key.
#[derive(Debug, thiserror::Error)]
pub enum KeyError {
    /// The variable is not present in the process environment.
    #[error("env var {0:?} is not set")]
    Missing(&'static str),
    /// The variable is set but blank after trimming whitespace.
    #[error("env var {0:?} is set but blank")]
    Empty(&'static str),
}

/// Parse `path` as a `.env` file and insert each key into the process
/// environment, skipping keys that are already set (shell wins over file).
///
/// Format supported: `KEY=VALUE`, `KEY="VALUE"`, `KEY='VALUE'`; lines starting
/// with `#` and blank lines are skipped. A missing or unreadable file is
/// silently ignored — expected in production or CI environments.
///
/// Call once at program start, before spawning threads, to avoid data races on
/// the process environment.
pub fn load_dotenv(path: &Path) {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(_) => return,
    };
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, val)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim();
        let val = val.trim().trim_matches('"').trim_matches('\'');
        if std::env::var_os(key).is_none() {
            // Safety: called before any threads are spawned (see doc comment).
            #[allow(deprecated)]
            unsafe {
                std::env::set_var(key, val);
            }
        }
    }
}

/// Return the value of `key` from the process environment, or an error if it
/// is absent or blank after trimming.
pub fn require(key: &'static str) -> Result<String, KeyError> {
    match std::env::var(key) {
        Ok(v) if !v.trim().is_empty() => Ok(v),
        Ok(_) => Err(KeyError::Empty(key)),
        Err(_) => Err(KeyError::Missing(key)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn require_missing_key_returns_missing() {
        assert!(matches!(
            require("__KITETSU_ABSENT_KEY_A1B2C3__"),
            Err(KeyError::Missing(_))
        ));
    }

    #[test]
    fn load_dotenv_ignores_missing_file() {
        load_dotenv(Path::new("/tmp/__kitetsu_nonexistent_dotenv__"));
    }
}
