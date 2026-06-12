//! Speech-recognition sub-package: model loading and one-shot inference.
//!
//! `model` defines `AudioModel` (the public backend selector) and the internal
//! `LoadedModel` handle. `engine` runs synchronous inference on f32 samples.
//! `Transcriber` is the integration point: loads a model once, reuses across calls.
//! `stream` adds the live VAD-gated path on top. Does no audio capture.

mod engine;
pub(crate) mod model;
pub(crate) mod stream;

pub(crate) use engine::transcribe_samples;
pub use model::AudioModel;

use std::sync::mpsc::Sender;

use crate::listener::ListenerError;
use model::LoadedModel;

// ── Transcriber ───────────────────────────────────────────────────────────────

/// A speech-recognition model loaded once and reused across transcription calls.
///
/// Load at startup with [`Transcriber::load`]; then call [`Transcriber::transcribe`]
/// as many times as needed without paying the model-load cost again. Each call
/// only runs inference — the model stays in memory for the lifetime of the struct.
///
/// `Transcriber` is `Send`; wrap in `Arc<Mutex<Transcriber>>` to share across
/// async tasks, or move into individual `tokio::task::spawn_blocking` closures
/// when calls are sequential.
pub struct Transcriber {
    model: LoadedModel,
}

impl Transcriber {
    /// Load the model into memory using all available threads (capped at 8).
    ///
    /// This is the expensive startup step — model files are read from disk and
    /// inference buffers are allocated. Call it once when the daemon starts and
    /// hold on to the returned `Transcriber`.
    pub fn load(model: AudioModel) -> Result<Self, ListenerError> {
        let loaded = model.resolve().load()?;
        Ok(Self { model: loaded })
    }

    /// Like [`load`] but caps the thread count for concurrent streaming use.
    ///
    /// When two `Transcriber` instances run on parallel threads (one per audio
    /// source), halving threads avoids CPU over-subscription.
    pub(crate) fn load_with_threads(
        model: AudioModel,
        n_threads: i32,
    ) -> Result<Self, ListenerError> {
        let loaded = model.resolve().load_with_threads(n_threads)?;
        Ok(Self { model: loaded })
    }

    /// Run inference on `samples` (16 kHz mono f32) using the already-loaded model.
    ///
    /// `progress_tx`, when provided, receives progress values 0–100 during
    /// inference. Read from a polling loop on another thread to drive a progress
    /// display. The sender is dropped at the end of inference; `try_recv` will
    /// drain and then return `Err(Disconnected)`, which is the completion signal.
    ///
    /// Blocks the calling thread for the duration of inference. Run inside
    /// `tokio::task::spawn_blocking` when called from async code.
    pub fn transcribe(
        &mut self,
        samples: &[f32],
        progress_tx: Option<Sender<u8>>,
    ) -> Result<String, ListenerError> {
        engine::transcribe_samples(&mut self.model, samples, "", progress_tx)
    }

    /// Like [`transcribe`] but seeds the decoder with prior committed text.
    ///
    /// `initial_prompt` should be the last ~200 chars of transcript so far;
    /// whisper.cpp uses it to bias the decoder for better cross-utterance
    /// continuity. The string is copied internally; the caller may drop it after.
    pub fn transcribe_with_context(
        &mut self,
        samples: &[f32],
        initial_prompt: &str,
        progress_tx: Option<Sender<u8>>,
    ) -> Result<String, ListenerError> {
        engine::transcribe_samples(&mut self.model, samples, initial_prompt, progress_tx)
    }
}
