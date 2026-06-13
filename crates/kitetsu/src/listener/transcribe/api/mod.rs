//! API-backed transcription sub-package: OpenAI REST client and transcription adapters.
//!
//! `openai` owns the HTTP building blocks (config, errors, WAV encoding).
//! `transcribe` provides the high-level [`ApiTranscriber`] and re-exports
//! [`LocalTranscriber`] so both backends are importable from one path.

pub(super) mod openai;
pub(super) mod transcribe;

pub use openai::{ApiConfig, ApiError};
pub use transcribe::{ApiTranscriber, LocalTranscriber};
