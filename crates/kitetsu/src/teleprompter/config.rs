//! Typed teleprompter configuration loaded from `config.toml`.
//!
//! [`Config`] mirrors the on-disk schema: `[output]`, `[prompts]`, `[audio]`,
//! and one section per pipe (`[pipe1]`/`[pipe2]`/`[pipe3]`). Every field has a
//! default so a minimal file works; [`Config::validate`] rejects nonsensical
//! combinations (enabled pipe with a blank model, zero window cap, …).
//! Not here: capture, HTTP, or the actual LLM backends — only parsing + checks.

use std::path::{Path, PathBuf};

use serde::Deserialize;

/// Failure to load or validate the configuration.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// The config file (or a referenced prompt/context file) could not be read.
    #[error("reading {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    /// The config file was not valid TOML or did not match the schema.
    #[error("parsing {path}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },
    /// The config parsed but is semantically invalid.
    #[error("invalid config: {0}")]
    Invalid(String),
}

/// How pipe output files are written.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputMode {
    /// Append each window's block, keeping history.
    #[default]
    Append,
    /// Overwrite the file each window, keeping only the latest.
    Overwrite,
}

/// Which LLM provider a pipe calls. The concrete client is built in a later
/// iteration; here it is only a config tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LlmBackendKind {
    Openai,
    Anthropic,
}

/// Output-file location and write mode.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OutputConfig {
    /// Directory for `pipe{1,2,3}.md`.
    pub dir: PathBuf,
    /// Append (keep history) or overwrite (latest only).
    pub mode: OutputMode,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            dir: PathBuf::from("out"),
            mode: OutputMode::Append,
        }
    }
}

/// Paths to the instruction and personal-context Markdown files.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PromptsConfig {
    /// Instructions for the LLM (the "how to respond").
    pub prompt: PathBuf,
    /// Personal/background context (the "what it should know").
    pub context: PathBuf,
}

impl Default for PromptsConfig {
    fn default() -> Self {
        Self {
            prompt: PathBuf::from("prompt.md"),
            context: PathBuf::from("context.md"),
        }
    }
}

/// Audio capture settings shared by all pipes.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AudioConfig {
    /// Capture the microphone (your own voice).
    pub mic: bool,
    /// Capture system audio (the other party).
    pub system: bool,
    /// Label for the mic transcript in the LLM prompt.
    pub mic_label: String,
    /// Label for the system transcript in the LLM prompt.
    pub system_label: String,
    /// Safety cap: drop the oldest samples once a window exceeds this many
    /// seconds (bounds growth if you never trigger).
    pub max_window_secs: u64,
    /// Explicit PulseAudio mic source name; `None` → auto-discover.
    pub mic_source: Option<String>,
    /// Explicit PulseAudio system-monitor source name; `None` → auto-discover.
    pub system_source: Option<String>,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            mic: true,
            system: true,
            mic_label: "Me".to_owned(),
            system_label: "Them".to_owned(),
            max_window_secs: 300,
            mic_source: None,
            system_source: None,
        }
    }
}

/// Pipe 1 — live (Realtime WS) transcription → LLM. Fastest path.
///
/// The `font_size`/`opacity`/`pos_*` keys style and place this pipe's overlay
/// card; `bg_opacity`/`text_opacity` are `None` → inherit `opacity`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Pipe1Config {
    pub enabled: bool,
    /// Realtime transcription model.
    pub stt_model: String,
    /// ISO-639-1 language hint; `None` → auto-detect.
    pub stt_language: Option<String>,
    /// Commit the WS audio buffer every N chunks (gpt-realtime-whisper has no VAD).
    pub commit_every_chunks: usize,
    pub llm_backend: LlmBackendKind,
    pub llm_model: String,
    pub font_size: f32,
    pub opacity: f32,
    pub bg_opacity: Option<f32>,
    pub text_opacity: Option<f32>,
    pub width: f32,
    pub height: f32,
    pub pos_x: f32,
    pub pos_y: f32,
}

impl Default for Pipe1Config {
    fn default() -> Self {
        Self {
            enabled: true,
            stt_model: "gpt-realtime-whisper".to_owned(),
            stt_language: Some("en".to_owned()),
            commit_every_chunks: 8,
            llm_backend: LlmBackendKind::Openai,
            llm_model: "gpt-4o".to_owned(),
            font_size: 19.0,
            opacity: 1.0,
            bg_opacity: None,
            text_opacity: None,
            width: 600.0,
            height: 400.0,
            pos_x: 40.0,
            pos_y: 40.0,
        }
    }
}

/// Pipe 2 — on-trigger REST chunk transcription → LLM. Slower, more accurate.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Pipe2Config {
    pub enabled: bool,
    /// REST transcription model.
    pub stt_model: String,
    /// ISO-639-1 language hint; `None` → auto-detect.
    pub stt_language: Option<String>,
    pub llm_backend: LlmBackendKind,
    pub llm_model: String,
    pub font_size: f32,
    pub opacity: f32,
    pub bg_opacity: Option<f32>,
    pub text_opacity: Option<f32>,
    pub width: f32,
    pub height: f32,
    pub pos_x: f32,
    pub pos_y: f32,
}

impl Default for Pipe2Config {
    fn default() -> Self {
        Self {
            enabled: true,
            stt_model: "gpt-4o-transcribe".to_owned(),
            stt_language: Some("en".to_owned()),
            llm_backend: LlmBackendKind::Openai,
            llm_model: "gpt-4o".to_owned(),
            font_size: 19.0,
            opacity: 1.0,
            bg_opacity: None,
            text_opacity: None,
            width: 600.0,
            height: 400.0,
            pos_x: 40.0,
            pos_y: 460.0,
        }
    }
}

/// Pipe 3 — raw audio sent directly to an audio-capable LLM (OpenAI only).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Pipe3Config {
    pub enabled: bool,
    pub llm_backend: LlmBackendKind,
    pub llm_model: String,
}

impl Default for Pipe3Config {
    fn default() -> Self {
        Self {
            enabled: true,
            llm_backend: LlmBackendKind::Openai,
            llm_model: "gpt-4o-audio-preview".to_owned(),
        }
    }
}

/// The full teleprompter configuration.
#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub output: OutputConfig,
    pub prompts: PromptsConfig,
    pub audio: AudioConfig,
    pub pipe1: Pipe1Config,
    pub pipe2: Pipe2Config,
    pub pipe3: Pipe3Config,
}

impl Config {
    /// Read and validate `config.toml` at `path`.
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        let text = std::fs::read_to_string(path).map_err(|source| ConfigError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        let config: Config = toml::from_str(&text).map_err(|source| ConfigError::Parse {
            path: path.to_path_buf(),
            source,
        })?;
        config.validate()?;
        Ok(config)
    }

    /// Reject semantically invalid configurations.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.audio.max_window_secs == 0 {
            return Err(ConfigError::Invalid(
                "audio.max_window_secs must be greater than 0".to_owned(),
            ));
        }
        if !self.audio.mic && !self.audio.system {
            return Err(ConfigError::Invalid(
                "at least one of audio.mic / audio.system must be enabled".to_owned(),
            ));
        }
        if self.pipe1.enabled {
            require_nonblank("pipe1.stt_model", &self.pipe1.stt_model)?;
            require_nonblank("pipe1.llm_model", &self.pipe1.llm_model)?;
            if self.pipe1.commit_every_chunks == 0 {
                return Err(ConfigError::Invalid(
                    "pipe1.commit_every_chunks must be greater than 0".to_owned(),
                ));
            }
            require_card_style(
                "pipe1",
                self.pipe1.font_size,
                self.pipe1.width,
                self.pipe1.height,
                self.pipe1.opacity,
                self.pipe1.bg_opacity,
                self.pipe1.text_opacity,
            )?;
        }
        if self.pipe2.enabled {
            require_nonblank("pipe2.stt_model", &self.pipe2.stt_model)?;
            require_nonblank("pipe2.llm_model", &self.pipe2.llm_model)?;
            require_card_style(
                "pipe2",
                self.pipe2.font_size,
                self.pipe2.width,
                self.pipe2.height,
                self.pipe2.opacity,
                self.pipe2.bg_opacity,
                self.pipe2.text_opacity,
            )?;
        }
        if self.pipe3.enabled {
            require_nonblank("pipe3.llm_model", &self.pipe3.llm_model)?;
            if self.pipe3.llm_backend != LlmBackendKind::Openai {
                return Err(ConfigError::Invalid(
                    "pipe3 (audio-direct) supports only llm_backend = \"openai\"".to_owned(),
                ));
            }
        }
        Ok(())
    }

    /// Read the prompt + context files and compose the stable system prompt
    /// (`prompt` then a blank line then `context`). Read once at daemon start so
    /// the prefix stays constant for prompt caching.
    ///
    /// Relative paths resolve against `base_dir` (the config file's directory).
    pub fn load_system_prompt(&self, base_dir: &Path) -> Result<String, ConfigError> {
        let prompt = read_relative(base_dir, &self.prompts.prompt)?;
        let context = read_relative(base_dir, &self.prompts.context)?;
        Ok(format!("{}\n\n{}", prompt.trim_end(), context.trim_end()))
    }
}

fn require_nonblank(field: &str, value: &str) -> Result<(), ConfigError> {
    if value.trim().is_empty() {
        Err(ConfigError::Invalid(format!("{field} must not be blank")))
    } else {
        Ok(())
    }
}

/// Reject an alpha value outside `0.0..=1.0`.
fn require_unit(field: &str, value: f32) -> Result<(), ConfigError> {
    if !(0.0..=1.0).contains(&value) {
        return Err(ConfigError::Invalid(format!(
            "{field} must be between 0.0 and 1.0"
        )));
    }
    Ok(())
}

/// Validate a pipe's overlay-card style keys.
fn require_card_style(
    prefix: &str,
    font_size: f32,
    width: f32,
    height: f32,
    opacity: f32,
    bg_opacity: Option<f32>,
    text_opacity: Option<f32>,
) -> Result<(), ConfigError> {
    if font_size <= 0.0 {
        return Err(ConfigError::Invalid(format!(
            "{prefix}.font_size must be greater than 0"
        )));
    }
    if width <= 0.0 {
        return Err(ConfigError::Invalid(format!(
            "{prefix}.width must be greater than 0"
        )));
    }
    if height <= 0.0 {
        return Err(ConfigError::Invalid(format!(
            "{prefix}.height must be greater than 0"
        )));
    }
    require_unit(&format!("{prefix}.opacity"), opacity)?;
    if let Some(v) = bg_opacity {
        require_unit(&format!("{prefix}.bg_opacity"), v)?;
    }
    if let Some(v) = text_opacity {
        require_unit(&format!("{prefix}.text_opacity"), v)?;
    }
    Ok(())
}

fn read_relative(base_dir: &Path, path: &Path) -> Result<String, ConfigError> {
    let resolved = if path.is_absolute() {
        path.to_path_buf()
    } else {
        base_dir.join(path)
    };
    std::fs::read_to_string(&resolved).map_err(|source| ConfigError::Read {
        path: resolved,
        source,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_config_uses_all_defaults() {
        let cfg: Config = toml::from_str("").expect("empty toml parses");
        assert_eq!(cfg, Config::default());
        assert!(cfg.validate().is_ok());
        assert_eq!(cfg.audio.max_window_secs, 300);
        assert_eq!(cfg.output.mode, OutputMode::Append);
        assert_eq!(cfg.pipe3.llm_model, "gpt-4o-audio-preview");
    }

    #[test]
    fn parses_full_schema() {
        let toml = r#"
            [output]
            dir = "results"
            mode = "overwrite"
            [prompts]
            prompt = "p.md"
            context = "c.md"
            [audio]
            mic = true
            system = false
            mic_label = "I"
            system_label = "They"
            max_window_secs = 120
            [pipe1]
            enabled = true
            stt_model = "gpt-realtime-whisper"
            stt_language = "en"
            commit_every_chunks = 4
            llm_backend = "anthropic"
            llm_model = "claude-opus-4-8"
            [pipe2]
            enabled = false
            stt_model = "whisper-1"
            llm_backend = "openai"
            llm_model = "gpt-4o-mini"
            [pipe3]
            enabled = true
            llm_backend = "openai"
            llm_model = "gpt-4o-audio-preview"
        "#;
        let cfg = toml::from_str::<Config>(toml).expect("parses");
        cfg.validate().expect("valid");
        assert_eq!(cfg.output.mode, OutputMode::Overwrite);
        assert!(!cfg.audio.system);
        assert_eq!(cfg.pipe1.llm_backend, LlmBackendKind::Anthropic);
        assert_eq!(cfg.pipe1.commit_every_chunks, 4);
        assert!(!cfg.pipe2.enabled);
    }

    #[test]
    fn unknown_key_is_rejected() {
        let err = toml::from_str::<Config>("[audio]\nmicc = true\n").unwrap_err();
        assert!(err.to_string().contains("micc") || err.to_string().contains("unknown"));
    }

    #[test]
    fn zero_window_cap_is_invalid() {
        let cfg = toml::from_str::<Config>("[audio]\nmax_window_secs = 0\n").unwrap();
        assert!(matches!(cfg.validate(), Err(ConfigError::Invalid(_))));
    }

    #[test]
    fn no_audio_sources_is_invalid() {
        let cfg = toml::from_str::<Config>("[audio]\nmic = false\nsystem = false\n").unwrap();
        assert!(matches!(cfg.validate(), Err(ConfigError::Invalid(_))));
    }

    #[test]
    fn enabled_pipe_with_blank_model_is_invalid() {
        let cfg = toml::from_str::<Config>("[pipe1]\nllm_model = \"  \"\n").unwrap();
        assert!(matches!(cfg.validate(), Err(ConfigError::Invalid(_))));
    }

    #[test]
    fn pipe3_non_openai_backend_is_invalid() {
        let cfg = toml::from_str::<Config>("[pipe3]\nllm_backend = \"anthropic\"\n").unwrap();
        assert!(matches!(cfg.validate(), Err(ConfigError::Invalid(_))));
    }

    #[test]
    fn disabled_pipe_skips_model_validation() {
        let cfg =
            toml::from_str::<Config>("[pipe1]\nenabled = false\nllm_model = \"\"\n").unwrap();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn parses_card_style_keys() {
        let cfg = toml::from_str::<Config>(
            "[pipe1]\nfont_size = 26.0\nopacity = 0.8\nbg_opacity = 0.5\npos_x = 100.0\npos_y = 200.0\n",
        )
        .unwrap();
        cfg.validate().expect("valid");
        assert_eq!(cfg.pipe1.font_size, 26.0);
        assert_eq!(cfg.pipe1.opacity, 0.8);
        assert_eq!(cfg.pipe1.bg_opacity, Some(0.5));
        assert_eq!(cfg.pipe1.text_opacity, None);
        assert_eq!(cfg.pipe1.pos_x, 100.0);
        assert_eq!(cfg.pipe1.pos_y, 200.0);
    }

    #[test]
    fn out_of_range_opacity_is_invalid() {
        let cfg = toml::from_str::<Config>("[pipe2]\nopacity = 1.5\n").unwrap();
        assert!(matches!(cfg.validate(), Err(ConfigError::Invalid(_))));
    }

    #[test]
    fn nonpositive_font_size_is_invalid() {
        let cfg = toml::from_str::<Config>("[pipe1]\nfont_size = 0.0\n").unwrap();
        assert!(matches!(cfg.validate(), Err(ConfigError::Invalid(_))));
    }
}
