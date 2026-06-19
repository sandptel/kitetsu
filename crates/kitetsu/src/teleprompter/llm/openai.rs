//! OpenAI Chat Completions client (`/v1/chat/completions`).
//!
//! Builds a `messages` array (system + accumulated turns) for text chat, and a
//! user message with `input_audio` content parts for audio chat. Request-body
//! construction and response parsing are pure functions so they can be unit
//! tested without a network. Not here: provider dispatch (`mod`), capture, WAV
//! encoding.

use base64::{Engine as _, engine::general_purpose::STANDARD as B64};
use serde_json::{Value, json};

use super::{AudioPart, LlmError, Turn};

/// Default Chat Completions endpoint (overridable for compatible hosts).
const DEFAULT_ENDPOINT: &str = "https://api.openai.com/v1/chat/completions";

/// Reusable OpenAI chat client (pools connections via `reqwest::Client`).
#[derive(Debug)]
pub struct OpenAiClient {
    client: reqwest::Client,
    api_key: String,
    endpoint: String,
}

impl OpenAiClient {
    /// Returns [`LlmError::MissingKey`] if `api_key` is blank after trimming.
    pub fn new(api_key: String) -> Result<Self, LlmError> {
        if api_key.trim().is_empty() {
            return Err(LlmError::MissingKey);
        }
        Ok(Self {
            client: reqwest::Client::new(),
            api_key,
            endpoint: DEFAULT_ENDPOINT.to_owned(),
        })
    }

    /// Override the endpoint (OpenAI-compatible hosts / tests).
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    /// Text chat: send `system` + `turns`, return the reply text.
    pub async fn chat(
        &self,
        model: &str,
        system: &str,
        turns: &[Turn],
    ) -> Result<String, LlmError> {
        let body = build_chat_body(model, system, turns);
        self.post(body).await
    }

    /// Audio chat: send `system` + a user message of `prompt` text followed by
    /// one `input_audio` part per clip; return the reply text.
    pub async fn chat_audio(
        &self,
        model: &str,
        system: &str,
        prompt: &str,
        parts: &[AudioPart],
    ) -> Result<String, LlmError> {
        let body = build_audio_body(model, system, prompt, parts);
        self.post(body).await
    }

    /// POST a prepared body and parse `choices[0].message.content`.
    async fn post(&self, body: Value) -> Result<String, LlmError> {
        let response = self
            .client
            .post(&self.endpoint)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(LlmError::Status {
                status: status.as_u16(),
                body,
            });
        }

        let json: Value = response.json().await?;
        parse_chat_response(&json)
    }
}

/// Build the text Chat Completions request body.
fn build_chat_body(model: &str, system: &str, turns: &[Turn]) -> Value {
    let mut messages = vec![json!({ "role": "system", "content": system })];
    for turn in turns {
        messages.push(json!({ "role": turn.role.as_str(), "content": turn.content }));
    }
    json!({ "model": model, "messages": messages })
}

/// Build the audio Chat Completions request body. The user message is a content
/// array: the text prompt first, then one `input_audio` part per labeled clip
/// (the label is folded into the text so the model knows who is who).
fn build_audio_body(model: &str, system: &str, prompt: &str, parts: &[AudioPart]) -> Value {
    let mut content = Vec::with_capacity(parts.len() * 2 + 1);
    content.push(json!({ "type": "text", "text": prompt }));
    for part in parts {
        content.push(json!({ "type": "text", "text": format!("[{}]", part.label) }));
        content.push(json!({
            "type": "input_audio",
            "input_audio": { "data": B64.encode(&part.wav), "format": "wav" }
        }));
    }
    json!({
        "model": model,
        "modalities": ["text"],
        "messages": [
            { "role": "system", "content": system },
            { "role": "user", "content": content }
        ]
    })
}

/// Extract `choices[0].message.content` from a Chat Completions response.
fn parse_chat_response(json: &Value) -> Result<String, LlmError> {
    json.get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .map(|s| s.trim().to_owned())
        .ok_or_else(|| LlmError::Malformed("missing choices[0].message.content".to_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_key_is_rejected() {
        assert!(matches!(
            OpenAiClient::new("  ".to_owned()),
            Err(LlmError::MissingKey)
        ));
    }

    #[test]
    fn chat_body_has_system_first_then_turns() {
        let turns = [Turn::user("Them: hi"), Turn::assistant("Hello!")];
        let body = build_chat_body("gpt-4o", "be brief", &turns);
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "be brief");
        assert_eq!(msgs[1]["role"], "user");
        assert_eq!(msgs[1]["content"], "Them: hi");
        assert_eq!(msgs[2]["role"], "assistant");
        assert_eq!(body["model"], "gpt-4o");
    }

    #[test]
    fn audio_body_carries_text_then_input_audio_parts() {
        let parts = [
            AudioPart { label: "Me".into(), wav: vec![1, 2, 3] },
            AudioPart { label: "Them".into(), wav: vec![4, 5] },
        ];
        let body = build_audio_body("gpt-4o-audio-preview", "sys", "what to say?", &parts);
        assert_eq!(body["modalities"][0], "text");
        let content = body["messages"][1]["content"].as_array().unwrap();
        // prompt text + (label + audio) * 2 = 5 entries.
        assert_eq!(content.len(), 5);
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "what to say?");
        assert_eq!(content[1]["text"], "[Me]");
        assert_eq!(content[2]["type"], "input_audio");
        assert_eq!(content[2]["input_audio"]["format"], "wav");
        // base64 of [1,2,3] is "AQID".
        assert_eq!(content[2]["input_audio"]["data"], "AQID");
    }

    #[test]
    fn parses_choice_content() {
        let json = json!({
            "choices": [ { "message": { "role": "assistant", "content": "  hi there  " } } ]
        });
        assert_eq!(parse_chat_response(&json).unwrap(), "hi there");
    }

    #[test]
    fn malformed_response_is_rejected() {
        let json = json!({ "choices": [] });
        assert!(matches!(
            parse_chat_response(&json),
            Err(LlmError::Malformed(_))
        ));
    }
}
