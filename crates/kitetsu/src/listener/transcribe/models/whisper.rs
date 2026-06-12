//! Whisper GGML backend via whisper-rs (direct whisper.cpp bindings).
//!
//! `load` constructs a `WhisperModel` from a GGML model file on disk.
//! `WhisperModel::transcribe` runs synchronous inference and reports incremental
//! progress (0–100) via the optional `Sender`. Does no audio capture.
//!
//! The `#[cfg(feature = "whisper")]` / `#[cfg(not(feature = "whisper"))]` twin
//! pattern keeps call sites in `models/mod.rs` unconditional — the stub returns
//! `BackendNotCompiled` when the feature is off.

use std::path::PathBuf;
#[cfg(feature = "whisper")]
use std::sync::mpsc::Sender;

use crate::listener::ListenerError;

use super::LoadedModel;

// ── Model handle (feature-gated) ───────────────────────────────────────────────

/// Owns a whisper.cpp inference state.
///
/// `WhisperState` holds an `Arc` to the parent context's C allocations, so
/// dropping the originating `WhisperContext` after `create_state()` is safe.
/// This struct is `Send`; the state is not `Sync` — each concurrent stream
/// should own its own `WhisperModel` (or serialize behind a `Mutex`).
#[cfg(feature = "whisper")]
pub(crate) struct WhisperModel {
    state: whisper_rs::WhisperState,
    n_threads: i32,
}

#[cfg(feature = "whisper")]
impl WhisperModel {
    /// Run synchronous Whisper inference on `samples` (16 kHz mono f32).
    ///
    /// Reports greedy-decode progress (0–100) through `progress_tx` if provided.
    /// The sender is dropped on return; `try_recv` returning `Disconnected` is
    /// the completion signal for a polling loop.
    pub(crate) fn transcribe(
        &mut self,
        samples: &[f32],
        progress_tx: Option<Sender<u8>>,
    ) -> Result<String, ListenerError> {
        use whisper_rs::{FullParams, SamplingStrategy};

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        // Pin to English; skips the language-detection pass (~0.5 s per call).
        params.set_language(Some("en"));
        params.set_n_threads(self.n_threads);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_suppress_blank(true);
        params.set_suppress_nst(true);

        if let Some(tx) = progress_tx {
            params.set_progress_callback_safe(move |p: i32| {
                let _ = tx.send(p.clamp(0, 100) as u8);
            });
        }

        self.state
            .full(params, samples)
            .map_err(|e| ListenerError::Transcription(e.to_string()))?;

        let n = self.state.full_n_segments();
        let mut parts: Vec<String> = Vec::with_capacity(n as usize);
        for i in 0..n {
            if let Some(seg) = self.state.get_segment(i)
                && let Ok(txt) = seg.to_str()
            {
                let trimmed = txt.trim();
                if !trimmed.is_empty() {
                    parts.push(trimmed.to_owned());
                }
            }
        }

        Ok(parts.join(" "))
    }
}

// ── Backend entry point (cfg-gated twin) ──────────────────────────────────────

/// Load a Whisper GGML model from `path` and return a live engine handle.
///
/// Installs whisper.cpp's C-level logging hooks on the first call (idempotent)
/// so the verbose beam-search output is routed through the `log` crate rather
/// than printed directly to stderr.
#[cfg(feature = "whisper")]
pub(crate) fn load(path: PathBuf) -> Result<LoadedModel, ListenerError> {
    use whisper_rs::{WhisperContext, WhisperContextParameters};

    // Route all C-level whisper.cpp / GGML logs through the Rust `log` crate.
    // With no log backend wired, this silences the verbose beam-search output.
    // `install_logging_hooks` is idempotent; no Once guard needed.
    whisper_rs::install_logging_hooks();

    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(4)
        .min(8);

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

/// Stub: returns `BackendNotCompiled` when the `whisper` feature is off.
#[cfg(not(feature = "whisper"))]
pub(crate) fn load(path: PathBuf) -> Result<LoadedModel, ListenerError> {
    let _ = path;
    Err(ListenerError::BackendNotCompiled(
        "whisper — enable the 'whisper' feature",
    ))
}
