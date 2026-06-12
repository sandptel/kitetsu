//! `AudioModel` — selects and loads a speech-recognition backend.
//!
//! Defines the public enum that callers use to choose a backend, plus the
//! internal `LoadedModel` that wraps the live engine handle. `load()` is the
//! only place that touches backend constructors; `engine.rs` owns the
//! inference call. Does no audio I/O.

use std::path::PathBuf;

use tracing::warn;

use crate::listener::ListenerError;

// ── Public API ────────────────────────────────────────────────────────────────

/// Selects which speech-recognition backend and model file to use.
///
/// `Default` resolves to the local Whisper base.en model at
/// `$KITETSU_WHISPER_MODEL` or `$XDG_DATA_HOME/kitetsu/models/ggml-base.en.bin`.
/// `Whisperfile` and `Onnx` variants are compiled but not yet wired;
/// calling `load()` on them returns [`ListenerError::BackendNotCompiled`].
#[derive(Debug, Clone)]
pub enum AudioModel {
    /// Local Whisper model at the default XDG path or `$KITETSU_WHISPER_MODEL`.
    Default,
    /// Whisper GGML model at a specific path (requires the `whisper` feature).
    Whisper { model_path: PathBuf },
    /// Whisperfile model (reserved — see PLAN §6.6; requires the `whisperfile` feature).
    Whisperfile { model_path: PathBuf },
    /// SenseVoice ONNX model directory (reserved — see PLAN §6.6; requires the `onnx` feature).
    Onnx { model_dir: PathBuf },
}

impl AudioModel {
    /// Returns `true` if the selected backend feature is compiled in AND the
    /// model file / directory exists on disk.
    pub fn available(&self) -> bool {
        match self {
            AudioModel::Default => cfg!(feature = "whisper") && Self::default_model_path().exists(),
            AudioModel::Whisper { model_path } => cfg!(feature = "whisper") && model_path.exists(),
            // reserved: see PLAN §6.6
            AudioModel::Whisperfile { model_path } => {
                cfg!(feature = "whisperfile") && model_path.exists()
            }
            AudioModel::Onnx { model_dir } => cfg!(feature = "onnx") && model_dir.is_dir(),
        }
    }

    /// If this model is unavailable, falls back to `AudioModel::Default` and
    /// logs a warning. `Default` itself is returned as-is even when unavailable
    /// (load will return `BackendNotCompiled` in that case).
    pub(crate) fn resolve(self) -> Self {
        if self.available() {
            return self;
        }
        match &self {
            AudioModel::Default => self,
            _ => {
                warn!(
                    model = ?self,
                    "selected AudioModel is unavailable; falling back to AudioModel::Default",
                );
                AudioModel::Default
            }
        }
    }

    /// Load the backend and return a live engine handle.
    ///
    /// Call `resolve()` first if you want automatic fallback-to-Default
    /// behaviour when the selected backend is unavailable.
    pub(crate) fn load(self) -> Result<LoadedModel, ListenerError> {
        self.load_inner(None)
    }

    /// Like [`load`] but caps the inference thread count.
    ///
    /// Use when running multiple streams concurrently to avoid over-subscribing
    /// the CPU — typically `min(cores/2, 4)` when two sources are active.
    pub(crate) fn load_with_threads(self, n_threads: i32) -> Result<LoadedModel, ListenerError> {
        self.load_inner(Some(n_threads))
    }

    fn load_inner(self, n_threads_override: Option<i32>) -> Result<LoadedModel, ListenerError> {
        match self {
            AudioModel::Default => load_whisper(Self::default_model_path(), n_threads_override),
            AudioModel::Whisper { model_path } => load_whisper(model_path, n_threads_override),
            // reserved: see PLAN §6.6 — wired in a future iteration
            AudioModel::Whisperfile { .. } => Err(ListenerError::BackendNotCompiled(
                "whisperfile — enable the 'whisperfile' feature",
            )),
            AudioModel::Onnx { .. } => Err(ListenerError::BackendNotCompiled(
                "onnx — enable the 'onnx' feature",
            )),
        }
    }

    fn default_model_path() -> PathBuf {
        if let Ok(p) = std::env::var("KITETSU_WHISPER_MODEL") {
            return PathBuf::from(p);
        }
        let base = std::env::var("XDG_DATA_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                std::env::var("HOME")
                    .map(|h| PathBuf::from(h).join(".local/share"))
                    .unwrap_or_else(|_| PathBuf::from("/tmp"))
            });
        base.join("kitetsu/models/ggml-base.en.bin")
    }
}

// ── Internal engine handle ────────────────────────────────────────────────────

/// A loaded speech-recognition engine ready for inference.
/// Variants are cfg-gated; when no backend feature is enabled this enum is
/// uninhabited (impossible to construct, so inference paths are unreachable).
pub(crate) enum LoadedModel {
    #[cfg(feature = "whisper")]
    Whisper(WhisperModel),
}

/// Owns a whisper.cpp inference state. `WhisperState` holds an `Arc` to the
/// inner context, so the C allocations stay alive independently of the
/// `WhisperContext` that created it.
#[cfg(feature = "whisper")]
pub(crate) struct WhisperModel {
    pub(crate) state: whisper_rs::WhisperState,
    pub(crate) n_threads: i32,
}

// ── Backend constructors (cfg-gated pairs) ────────────────────────────────────

#[cfg(feature = "whisper")]
fn load_whisper(
    path: PathBuf,
    n_threads_override: Option<i32>,
) -> Result<LoadedModel, ListenerError> {
    use whisper_rs::{WhisperContext, WhisperContextParameters};

    // Route all C-level whisper.cpp / GGML logs through the Rust `log` crate.
    // With no log_backend or tracing_backend features enabled in whisper-rs,
    // the hook is a no-op sink — silences the verbose beam-search C output.
    // `install_logging_hooks` is idempotent; no Once guard needed.
    whisper_rs::install_logging_hooks();

    let n_threads = n_threads_override.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4)
            .min(8)
    });

    tracing::info!(path = %path.display(), n_threads, "loading whisper model");

    let ctx = WhisperContext::new_with_params(&path, WhisperContextParameters::default())
        .map_err(|e| ListenerError::ModelUnavailable(e.to_string()))?;

    // `create_state` clones the Arc inside `ctx`; dropping `ctx` here is safe.
    let state = ctx
        .create_state()
        .map_err(|e| ListenerError::ModelUnavailable(e.to_string()))?;

    tracing::info!(path = %path.display(), "whisper model ready");

    Ok(LoadedModel::Whisper(WhisperModel { state, n_threads }))
}

#[cfg(not(feature = "whisper"))]
fn load_whisper(
    path: PathBuf,
    _n_threads_override: Option<i32>,
) -> Result<LoadedModel, ListenerError> {
    let _ = path;
    Err(ListenerError::BackendNotCompiled(
        "whisper — enable the 'whisper' feature",
    ))
}
