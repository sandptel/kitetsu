//! API-backed transcription sub-package: OpenAI REST and Realtime WebSocket clients.
//!
//! `openai` owns the REST building blocks (config, errors, WAV encoding).
//! `transcribe` provides the high-level [`ApiTranscriber`] (REST, one-shot).
//! `realtime` provides [`RealtimeSession`] (WebSocket, streaming live).
//! `env` resolves named API keys from the process environment and `.env` files.

pub(super) mod openai;
pub(super) mod transcribe;
pub(super) mod realtime;
pub(super) mod env;

pub use env::{KeyError, ANTHROPIC_API_KEY, OPENAI_API_KEY, load_dotenv, require};
pub use openai::{ApiConfig, ApiError};
pub use realtime::{RealtimeError, RealtimeSession, SessionSink, SessionStream, TranscriptEvent};
pub use transcribe::ApiTranscriber;
