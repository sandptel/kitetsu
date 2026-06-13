//! API-backed transcription sub-package: OpenAI REST client and transcription adapters.
//!
//! `openai` owns the HTTP building blocks (config, errors, WAV encoding).
//! `transcribe` provides the high-level [`ApiTranscriber`] and re-exports
//! [`LocalTranscriber`] so both backends are importable from one path.
//! `env` resolves named API keys from the process environment and `.env` files.

pub(super) mod openai;
pub(super) mod transcribe;
pub(super) mod env;

pub use env::{KeyError, ANTHROPIC_API_KEY, OPENAI_API_KEY, load_dotenv, require};
pub use openai::{ApiConfig, ApiError};
pub use transcribe::ApiTranscriber;
