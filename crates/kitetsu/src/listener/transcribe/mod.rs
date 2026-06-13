//! Speech-recognition sub-package: model loading, API transcription, live streaming.
//!
//! `models/` holds the local inference backends (`LocalTranscriber`, `AudioModel`).
//! `api/` (feature-gated) holds the OpenAI REST backend (`ApiTranscriber`).
//! `live` provides rolling-window live transcription on top of `LocalTranscriber`.
//! This module exposes the unified [`Transcriber`] that dispatches to either backend.

#[cfg(feature = "api")]
pub(crate) mod api;
mod live;
pub(crate) mod models;

#[cfg(feature = "api")]
pub use api::{ApiConfig, ApiError, ApiTranscriber};
pub use live::{LiveConfig, LiveTranscriber};
pub use models::{AudioModel, LocalTranscriber, MoonshineVariant, Quantization};

use std::sync::{Arc, Mutex};

use crate::listener::ListenerError;

// ── TranscribeError ──────────────────────────────────────────────────────────────

/// Unified error type for [`Transcriber::transcribe`].
#[derive(Debug, thiserror::Error)]
pub enum TranscribeError {
    #[error("local inference failed: {0}")]
    Local(#[from] ListenerError),
    #[cfg(feature = "api")]
    #[error("API transcription failed: {0}")]
    Api(api::ApiError),
    /// The `spawn_blocking` task panicked during local inference.
    #[error("transcription task panicked")]
    Panicked,
}

// ── TranscribeMethod ─────────────────────────────────────────────────────────────

/// Selects between a local model and a REST API backend.
pub enum TranscribeMethod {
    /// Blocking local inference via a loaded model.
    Local(LocalTranscriber),
    /// Async REST transcription via the OpenAI API (requires the `api` feature).
    #[cfg(feature = "api")]
    Api(ApiTranscriber),
}

// ── Transcriber ──────────────────────────────────────────────────────────────────

/// Unified transcription handle that dispatches to either [`TranscribeMethod`].
///
/// Construct with [`Transcriber::local`] or, when the `api` feature is enabled,
/// [`Transcriber::api`]. Call [`transcribe`][Self::transcribe] with owned samples;
/// the local path runs inference in `spawn_blocking` so callers never block the
/// async runtime.
pub struct Transcriber {
    inner: TranscriberInner,
}

// The local transcriber needs Arc<Mutex> so it can be cloned into spawn_blocking
// without moving out of &self. The Mutex is never held across .await.
enum TranscriberInner {
    Local(Arc<Mutex<LocalTranscriber>>),
    #[cfg(feature = "api")]
    Api(ApiTranscriber),
}

impl Transcriber {
    /// Load a local model and wrap it in a `Transcriber`.
    pub fn local(model: AudioModel) -> Result<Self, ListenerError> {
        let lt = LocalTranscriber::load(model)?;
        Ok(Self {
            inner: TranscriberInner::Local(Arc::new(Mutex::new(lt))),
        })
    }

    /// Wrap an already-built [`ApiTranscriber`] (requires the `api` feature).
    #[cfg(feature = "api")]
    pub fn api(transcriber: ApiTranscriber) -> Self {
        Self {
            inner: TranscriberInner::Api(transcriber),
        }
    }

    /// Transcribe `samples` (16 kHz mono f32, owned) and return the recognised text.
    ///
    /// - Local path: runs inference in `tokio::task::spawn_blocking`; the calling
    ///   future is not blocked while the OS thread works.
    /// - API path: sends a REST request and awaits the response.
    pub async fn transcribe(&self, samples: Vec<f32>) -> Result<String, TranscribeError> {
        match &self.inner {
            TranscriberInner::Local(arc) => {
                let arc = arc.clone();
                tokio::task::spawn_blocking(move || {
                    arc.lock()
                        .expect("LocalTranscriber mutex poisoned")
                        .transcribe(&samples, None)
                        .map_err(TranscribeError::Local)
                })
                .await
                .map_err(|_| TranscribeError::Panicked)?
            }
            #[cfg(feature = "api")]
            TranscriberInner::Api(t) => t.transcribe(&samples).await.map_err(TranscribeError::Api),
        }
    }
}
