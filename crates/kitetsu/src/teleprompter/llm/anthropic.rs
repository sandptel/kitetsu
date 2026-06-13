//! Anthropic Messages client (`/v1/messages`).
//!
//! Text chat builds a top-level `system` string plus a `messages` array.
//! `chat_audio` is unsupported (Anthropic's API takes no audio input) and the
//! dispatch in `mod` returns [`LlmError::Unsupported`] before reaching here.
//! Request-body construction and response parsing are pure functions, unit
//! tested offline. Not here: provider dispatch (`mod`), capture.

use serde_json::{Value, json};

use super::{LlmError, Turn};

/// Default Messages endpoint.
const DEFAULT_ENDPOINT: &str = "https://api.anthropic.com/v1/messages";
/// Anthropic requires a version header.
const API_VERSION: &str = "2023-06-01";
/// Reply cap. Teleprompter suggestions are short; this bounds cost/latency.
const MAX_TOKENS: u32 = 1024;

/// Reusable Anthropic chat client.
#[derive(Debug)]
pub struct AnthropicClient {
    client: reqwest::Client,
    api_key: String,
    endpoint: String,
}

impl AnthropicClient {
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

    /// Override the endpoint (tests / compatible hosts).
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
        let body = build_messages_body(model, system, turns);

        let response = self
            .client
            .post(&self.endpoint)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", API_VERSION)
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
        parse_messages_response(&json)
    }
}

/// Build the Messages request body (top-level `system`, then turns).
fn build_messages_body(model: &str, system: &str, turns: &[Turn]) -> Value {
    let messages: Vec<Value> = turns
        .iter()
        .map(|t| json!({ "role": t.role.as_str(), "content": t.content }))
        .collect();
    json!({
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": system,
        "messages": messages,
    })
}

/// Extract the first text block from `content[]`.
fn parse_messages_response(json: &Value) -> Result<String, LlmError> {
    json.get("content")
        .and_then(|c| c.as_array())
        .and_then(|blocks| blocks.iter().find_map(|b| b.get("text")?.as_str()))
        .map(|s| s.trim().to_owned())
        .ok_or_else(|| LlmError::Malformed("no text block in content[]".to_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_key_is_rejected() {
        assert!(matches!(
            AnthropicClient::new(String::new()),
            Err(LlmError::MissingKey)
        ));
    }

    #[test]
    fn body_has_top_level_system_and_turns() {
        let turns = [Turn::user("Them: hello")];
        let body = build_messages_body("claude-opus-4-8", "be brief", &turns);
        assert_eq!(body["model"], "claude-opus-4-8");
        assert_eq!(body["system"], "be brief");
        assert!(body["max_tokens"].is_number());
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[0]["content"], "Them: hello");
    }

    #[test]
    fn parses_first_text_block() {
        let json = json!({
            "content": [ { "type": "text", "text": "  say hi  " } ]
        });
        assert_eq!(parse_messages_response(&json).unwrap(), "say hi");
    }

    #[test]
    fn malformed_response_is_rejected() {
        let json = json!({ "content": [] });
        assert!(matches!(
            parse_messages_response(&json),
            Err(LlmError::Malformed(_))
        ));
    }
}
