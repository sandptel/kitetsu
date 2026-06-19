//! Minimal chat-LLM client for the teleprompter pipes.
//!
//! [`Backend`] is an enum over the two providers ([`OpenAiClient`],
//! [`AnthropicClient`]) — an enum rather than a trait object so no `async-trait`
//! dependency is needed. Both expose [`Backend::chat`] (text conversation) and
//! [`Backend::chat_audio`] (audio-in, OpenAI only — Anthropic returns
//! [`LlmError::Unsupported`]). Conversations are stateless on the wire: the caller
//! resends the stable `system` prompt plus the accumulated [`Turn`]s each call,
//! relying on provider-side prompt caching for the constant prefix.
//!
//! Not here: the pipes themselves, audio capture, or window state — this module
//! only speaks HTTP to the chat endpoints.

pub mod anthropic;
pub mod openai;

pub use anthropic::AnthropicClient;
pub use openai::OpenAiClient;

use super::config::LlmBackendKind;

/// Who authored a conversation turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// The operator / transcript side (what we send up).
    User,
    /// A previous model suggestion (kept for continuity).
    Assistant,
}

impl Role {
    /// Wire name used by both the OpenAI and Anthropic message schemas.
    pub fn as_str(self) -> &'static str {
        match self {
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

/// One conversation turn (role + text content).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Turn {
    pub role: Role,
    pub content: String,
}

impl Turn {
    /// A user turn (the transcript / instruction we send to the model).
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
        }
    }

    /// An assistant turn (a prior suggestion, kept for continuity).
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
        }
    }
}

/// A labeled audio clip for [`Backend::chat_audio`] (WAV-encoded bytes).
#[derive(Debug, Clone)]
pub struct AudioPart {
    /// Speaker label shown to the model (e.g. "Me", "Them").
    pub label: String,
    /// WAV (PCM16) file bytes.
    pub wav: Vec<u8>,
}

/// Failures talking to a chat LLM.
#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    /// The supplied API key was empty after trimming.
    #[error("API key is empty")]
    MissingKey,
    /// HTTP request failed (DNS, TLS, connection, decode).
    #[error("LLM request failed")]
    Request(#[from] reqwest::Error),
    /// Non-2xx response from the endpoint.
    #[error("LLM API returned {status}: {body}")]
    Status { status: u16, body: String },
    /// The response JSON did not match the expected shape.
    #[error("malformed LLM response: {0}")]
    Malformed(String),
    /// The selected backend cannot accept audio input.
    #[error("{backend} backend does not support audio input")]
    Unsupported { backend: &'static str },
}

/// A chat backend: one concrete provider client.
#[derive(Debug)]
pub enum Backend {
    OpenAi(OpenAiClient),
    Anthropic(AnthropicClient),
}

impl Backend {
    /// Build the backend named by `kind` with the given API key.
    pub fn new(kind: LlmBackendKind, api_key: String) -> Result<Self, LlmError> {
        match kind {
            LlmBackendKind::Openai => Ok(Backend::OpenAi(OpenAiClient::new(api_key)?)),
            LlmBackendKind::Anthropic => Ok(Backend::Anthropic(AnthropicClient::new(api_key)?)),
        }
    }

    /// Human-readable provider name (for logs and error context).
    pub fn name(&self) -> &'static str {
        match self {
            Backend::OpenAi(_) => "openai",
            Backend::Anthropic(_) => "anthropic",
        }
    }

    /// Send a text conversation and return the model's reply.
    ///
    /// `system` is the stable prefix (prompt.md + context.md); `turns` is the
    /// accumulated user/assistant history. Both are resent every call.
    pub async fn chat(
        &self,
        model: &str,
        system: &str,
        turns: &[Turn],
    ) -> Result<String, LlmError> {
        match self {
            Backend::OpenAi(c) => c.chat(model, system, turns).await,
            Backend::Anthropic(c) => c.chat(model, system, turns).await,
        }
    }

    /// Send labeled audio clips plus a text prompt to an audio-capable model.
    ///
    /// Only OpenAI supports this; the Anthropic backend returns
    /// [`LlmError::Unsupported`].
    pub async fn chat_audio(
        &self,
        model: &str,
        system: &str,
        prompt: &str,
        parts: &[AudioPart],
    ) -> Result<String, LlmError> {
        match self {
            Backend::OpenAi(c) => c.chat_audio(model, system, prompt, parts).await,
            Backend::Anthropic(_) => Err(LlmError::Unsupported {
                backend: "anthropic",
            }),
        }
    }
}
