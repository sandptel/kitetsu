//! Model selection and unified dispatch for speech-recognition backends.
//!
//! `AudioModel` is the public selector callers use to choose a backend and point
//! to its model files. `LoadedModel` (crate-internal) is the live handle returned
//! by `load()`; its `transcribe()` method is the single call site for inference
//! regardless of backend. Adding a new backend is: new file + new `LoadedModel`
//! variant + new `AudioModel` variant + wiring in `load()`. No existing paths change.

use std::path::PathBuf;
use std::sync::mpsc::Sender;

use tracing::warn;

use crate::listener::ListenerError;

pub(crate) mod moonshine;
pub(crate) mod whisper;

// ── Public selector ────────────────────────────────────────────────────────────

/// Moonshine model size. Both variants are English-only.
///
/// `Tiny` (26 M params) is the fastest choice; `Base` (58 M params) trades a
/// few extra milliseconds for slightly lower WER. Either is several times faster
/// than Whisper base.en on CPU alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoonshineVariant {
    /// 26 M params — fastest, lowest memory footprint.
    Tiny,
    /// 58 M params — slightly better WER, still very fast.
    Base,
}

/// Selects which speech-recognition backend and model files to use.
///
/// Variants are always present regardless of which Cargo features are enabled;
/// `cfg` gates only the `load` / `available` bodies. Call [`resolve`][Self::resolve]
/// before [`load`][Self::load] to get the automatic fallback-to-Default behaviour.
///
/// **Default resolution order** (first available wins):
/// 1. Moonshine base — when the `onnx` feature is compiled in and
///    the default dir exists (`$KITETSU_MOONSHINE_MODEL` or
///    `$XDG_DATA_HOME/kitetsu/models/moonshine-base/`).
/// 2. Whisper ggml  — when the `whisper` feature is compiled in and
///    the default file exists (`$KITETSU_WHISPER_MODEL` or
///    `$XDG_DATA_HOME/kitetsu/models/ggml-base.en.bin`).
#[derive(Debug)]
pub enum AudioModel {
    /// Resolve the best available backend at load time (see above).
    Default,
    /// Moonshine ONNX directory containing `encoder_model.onnx`,
    /// `decoder_model_merged.onnx`, and `tokenizer.json`.
    Moonshine {
        model_dir: PathBuf,
        variant: MoonshineVariant,
    },
    /// Whisper GGML model file (requires the `whisper` feature).
    Whisper { model_path: PathBuf },
    /// Whisperfile server model (reserved — see PLAN §6.6; requires `whisperfile`).
    Whisperfile { model_path: PathBuf },
}

// ── Internal engine handle ─────────────────────────────────────────────────────

/// A loaded speech-recognition engine ready for synchronous inference.
///
/// Variants are cfg-gated; the enum is uninhabited when no backend feature is
/// enabled, making it impossible to construct and the inference path unreachable.
pub(crate) enum LoadedModel {
    #[cfg(feature = "whisper")]
    Whisper(whisper::WhisperModel),
    #[cfg(feature = "onnx")]
    Moonshine(moonshine::MoonshineBackend),
}

impl LoadedModel {
    /// Run inference on `samples` (16 kHz mono f32, range −1.0..1.0).
    ///
    /// `progress_tx` receives 0–100 during inference. Whisper fires incremental
    /// updates; Moonshine sends a single `100` on completion (inference is ~50 ms,
    /// intermediate steps are not exposed by the backend). Blocks the calling
    /// thread — wrap with `tokio::task::spawn_blocking` from async contexts.
    pub(crate) fn transcribe(
        &mut self,
        samples: &[f32],
        progress_tx: Option<Sender<u8>>,
    ) -> Result<String, ListenerError> {
        match self {
            #[cfg(feature = "whisper")]
            LoadedModel::Whisper(m) => m.transcribe(samples, progress_tx),
            #[cfg(feature = "onnx")]
            LoadedModel::Moonshine(m) => m.transcribe(samples, progress_tx),
        }
    }
}

// ── AudioModel impl ────────────────────────────────────────────────────────────

impl AudioModel {
    /// Returns `true` when the selected backend feature is compiled in AND
    /// the model file / directory exists on disk.
    pub fn available(&self) -> bool {
        match self {
            AudioModel::Default => {
                (cfg!(feature = "onnx") && Self::default_moonshine_dir().is_dir())
                    || (cfg!(feature = "whisper") && Self::default_whisper_path().exists())
            }
            AudioModel::Moonshine { model_dir, .. } => cfg!(feature = "onnx") && model_dir.is_dir(),
            AudioModel::Whisper { model_path } => cfg!(feature = "whisper") && model_path.exists(),
            // reserved: see PLAN §6.6
            AudioModel::Whisperfile { model_path } => {
                cfg!(feature = "whisperfile") && model_path.exists()
            }
        }
    }

    /// If this model is unavailable, falls back to `Default` with a warning.
    ///
    /// `Default` is returned as-is even when unavailable; `load()` will then
    /// return a descriptive error so the caller knows what is missing.
    pub(crate) fn resolve(self) -> Self {
        if self.available() {
            return self;
        }
        match &self {
            AudioModel::Default => self,
            _ => {
                warn!(
                    model = ?self,
                    "selected AudioModel unavailable — falling back to AudioModel::Default",
                );
                AudioModel::Default
            }
        }
    }

    /// Load the backend and return a live engine handle.
    ///
    /// Call `resolve()` first if you want the automatic fallback-to-Default
    /// behaviour when the selected backend is unavailable.
    pub(crate) fn load(self) -> Result<LoadedModel, ListenerError> {
        match self {
            AudioModel::Default => {
                // Prefer Moonshine (onnx) when the feature is compiled in.
                #[cfg(feature = "onnx")]
                return moonshine::load(Self::default_moonshine_dir(), MoonshineVariant::Base);

                // Fall back to Whisper when only that feature is compiled in.
                #[cfg(all(not(feature = "onnx"), feature = "whisper"))]
                return whisper::load(Self::default_whisper_path());

                // Neither feature compiled — nothing to load.
                #[cfg(not(any(feature = "onnx", feature = "whisper")))]
                return Err(ListenerError::BackendNotCompiled(
                    "no STT backend compiled — enable the 'onnx' or 'whisper' feature",
                ));
            }
            AudioModel::Moonshine { model_dir, variant } => moonshine::load(model_dir, variant),
            AudioModel::Whisper { model_path } => whisper::load(model_path),
            // reserved: see PLAN §6.6 — wired in a future iteration
            AudioModel::Whisperfile { .. } => Err(ListenerError::BackendNotCompiled(
                "whisperfile — enable the 'whisperfile' feature",
            )),
        }
    }

    // ── Default path helpers ───────────────────────────────────────────────────

    /// Default Moonshine model directory.
    ///
    /// Checks `$KITETSU_MOONSHINE_MODEL` first, then falls back to
    /// `$XDG_DATA_HOME/kitetsu/models/moonshine-base/`.
    /// Expected contents: `encoder_model.onnx`, `decoder_model_merged.onnx`,
    /// `tokenizer.json` (from `onnx-community/moonshine-base-ONNX` on HuggingFace).
    pub(crate) fn default_moonshine_dir() -> PathBuf {
        if let Ok(p) = std::env::var("KITETSU_MOONSHINE_MODEL") {
            return PathBuf::from(p);
        }
        xdg_data_base().join("kitetsu/models/moonshine-base")
    }

    /// Default Whisper model file path.
    ///
    /// Checks `$KITETSU_WHISPER_MODEL` first, then falls back to
    /// `$XDG_DATA_HOME/kitetsu/models/ggml-base.en.bin`.
    pub(crate) fn default_whisper_path() -> PathBuf {
        if let Ok(p) = std::env::var("KITETSU_WHISPER_MODEL") {
            return PathBuf::from(p);
        }
        xdg_data_base().join("kitetsu/models/ggml-base.en.bin")
    }
}

fn xdg_data_base() -> PathBuf {
    std::env::var("XDG_DATA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            std::env::var("HOME")
                .map(|h| PathBuf::from(h).join(".local/share"))
                .unwrap_or_else(|_| PathBuf::from("/tmp"))
        })
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moonshine_unavailable_when_dir_absent() {
        let m = AudioModel::Moonshine {
            model_dir: PathBuf::from("/nonexistent/moonshine-base-test-sentinel"),
            variant: MoonshineVariant::Base,
        };
        assert!(!m.available());
    }

    #[test]
    fn whisper_unavailable_when_file_absent() {
        let m = AudioModel::Whisper {
            model_path: PathBuf::from("/nonexistent/ggml-base.en-test-sentinel.bin"),
        };
        assert!(!m.available());
    }

    #[test]
    fn moonshine_default_dir_respects_env_var() {
        // SAFETY: test-only env mutation; the env var name is unique to this test.
        // Parallel test execution could race if another test also writes this key,
        // but no other test in this file touches KITETSU_MOONSHINE_MODEL.
        let custom = "/tmp/kitetsu-test-moonshine-sentinel";
        unsafe { std::env::set_var("KITETSU_MOONSHINE_MODEL", custom) };
        let path = AudioModel::default_moonshine_dir();
        unsafe { std::env::remove_var("KITETSU_MOONSHINE_MODEL") };
        assert_eq!(path, PathBuf::from(custom));
    }

    #[test]
    fn whisper_default_path_respects_env_var() {
        let custom = "/tmp/kitetsu-test-whisper-sentinel.bin";
        unsafe { std::env::set_var("KITETSU_WHISPER_MODEL", custom) };
        let path = AudioModel::default_whisper_path();
        unsafe { std::env::remove_var("KITETSU_WHISPER_MODEL") };
        assert_eq!(path, PathBuf::from(custom));
    }
}
