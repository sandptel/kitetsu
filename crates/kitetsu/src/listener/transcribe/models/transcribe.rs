//! Load-once speech-recognition handle for local model inference.
//!
//! [`LocalTranscriber`] wraps a loaded model and exposes a blocking `transcribe`
//! method. Drive it from a dedicated thread or `tokio::task::spawn_blocking`.
//! Not here: network backends, async I/O, capture.

use std::sync::mpsc::Sender;

use crate::listener::ListenerError;

use super::{AudioModel, LoadedModel};

/// A speech-recognition model loaded once and reused across transcription calls.
///
/// Load at startup with [`LocalTranscriber::load`]; call
/// [`transcribe`][Self::transcribe] as many times as needed without paying the
/// model-load cost again.
///
/// `LocalTranscriber` is `Send`; for concurrent inference across two streams
/// load two separate instances — the parallel gain is worth the extra memory.
pub struct LocalTranscriber {
    model: LoadedModel,
}

impl LocalTranscriber {
    /// Load the model into memory (ONNX sessions or GGML buffers allocated here).
    pub fn load(model: AudioModel) -> Result<Self, ListenerError> {
        let loaded = model.resolve().load()?;
        Ok(Self { model: loaded })
    }

    /// Run inference on `samples` (16 kHz mono f32, −1.0..1.0).
    ///
    /// `progress_tx` receives 0–100 during inference (Moonshine sends a single
    /// `100` on completion; Whisper fires incrementally). Blocks the calling
    /// thread — wrap with `tokio::task::spawn_blocking` from async code.
    pub fn transcribe(
        &mut self,
        samples: &[f32],
        progress_tx: Option<Sender<u8>>,
    ) -> Result<String, ListenerError> {
        self.model.transcribe(samples, progress_tx)
    }
}
