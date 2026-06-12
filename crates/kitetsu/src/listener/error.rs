//! Shared error type for the listener module.
//!
//! Centralised here so both `capture` and `transcribe` sub-packages can use
//! `crate::listener::ListenerError` without circular imports.
//! Audio I/O errors and transcription errors live in the same enum because
//! the pipe in `listener/mod.rs` composes both operations.

#[derive(Debug, thiserror::Error)]
pub enum ListenerError {
    #[error("PulseAudio introspection failed: {0}")]
    IntrospectionFailed(&'static str),

    #[error("PulseAudio record stream could not be opened: {0}")]
    ConnectFailed(&'static str),

    #[error("PulseAudio read error: {0}")]
    CaptureFailed(&'static str),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("audio model is not available: {0}")]
    ModelUnavailable(String),

    #[error("backend not compiled in — {0}")]
    BackendNotCompiled(&'static str),

    #[error("transcription failed: {0}")]
    Transcription(String),

    #[error("blocking task panicked")]
    JoinError,

    // reserved: see PLAN Decision #21
    #[error("not yet implemented: {0}")]
    NotImplemented(&'static str),
}

impl ListenerError {
    pub(crate) fn join(_: tokio::task::JoinError) -> Self {
        Self::JoinError
    }
}
