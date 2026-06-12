//! Speech-recognition sub-package: model loading and one-shot inference.
//!
//! `models/` defines `AudioModel` (the public backend selector), `MoonshineVariant`,
//! and the internal `LoadedModel` handle — one file per backend, unified by enum
//! dispatch. `Transcriber` is defined here as the load-once / reuse integration
//! point. Does no audio capture.

mod models;

pub use models::{AudioModel, MoonshineVariant, Quantization};

use std::sync::mpsc::Sender;

use crate::listener::ListenerError;
use models::LoadedModel;

// ── Transcriber ────────────────────────────────────────────────────────────────

/// A speech-recognition model loaded once and reused across transcription calls.
///
/// Load at startup with [`Transcriber::load`]; then call [`Transcriber::transcribe`]
/// as many times as needed without paying the model-load cost on each call.
///
/// `Transcriber` is `Send`; for true concurrent inference across two audio streams
/// load two separate instances (one per stream) rather than sharing one behind a
/// `Mutex` — the parallel gain is worth the extra memory for small models.
pub struct Transcriber {
    model: LoadedModel,
}

impl Transcriber {
    /// Load the model into memory.
    ///
    /// This is the expensive startup step — ONNX sessions or GGML buffers are
    /// allocated here. Call it once when the daemon starts and hold on to the
    /// returned `Transcriber`.
    pub fn load(model: AudioModel) -> Result<Self, ListenerError> {
        let loaded = model.resolve().load()?;
        Ok(Self { model: loaded })
    }

    /// Run inference on `samples` (16 kHz mono f32) using the already-loaded model.
    ///
    /// `progress_tx`, when provided, receives progress values 0–100 during
    /// inference. For Whisper this fires incrementally; for Moonshine a single
    /// `100` is sent on completion. The sender is dropped at the end of inference;
    /// `try_recv` returning `Err(Disconnected)` is the completion signal.
    ///
    /// Blocks the calling thread for the duration of inference. Run inside
    /// `tokio::task::spawn_blocking` when called from async code.
    pub fn transcribe(
        &mut self,
        samples: &[f32],
        progress_tx: Option<Sender<u8>>,
    ) -> Result<String, ListenerError> {
        self.model.transcribe(samples, progress_tx)
    }
}
