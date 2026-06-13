//! Transcription adapters: API-backed (async REST) and local (sync, blocking).
//!
//! [`ApiTranscriber`] posts captured audio to the OpenAI REST endpoint and
//! returns text. [`LocalTranscriber`] is re-exported here from `models` so
//! callers can choose either backend from a single import path.
//! Not here: WAV encoding details (`openai`), capture, session state.

pub use super::super::models::LocalTranscriber;

use super::openai::{ApiConfig, ApiError, wav_bytes};

/// The `response_format=json` response shape.
#[derive(Debug, serde::Deserialize)]
struct TranscriptionResponse {
    text: String,
}

/// Reusable async client that transcribes audio via the OpenAI REST API.
///
/// Build once with [`ApiTranscriber::new`]; the underlying `reqwest::Client`
/// pools connections so reuse is cheaper than rebuilding per utterance.
#[derive(Debug)]
pub struct ApiTranscriber {
    client: reqwest::Client,
    api_key: String,
    config: ApiConfig,
}

impl ApiTranscriber {
    /// Returns [`ApiError::MissingKey`] if `api_key` is blank after trimming.
    pub fn new(api_key: String, config: ApiConfig) -> Result<Self, ApiError> {
        if api_key.trim().is_empty() {
            return Err(ApiError::MissingKey);
        }
        Ok(Self {
            client: reqwest::Client::new(),
            api_key,
            config,
        })
    }

    /// Transcribe `samples` (16 kHz mono f32, −1.0..1.0) and return the text.
    ///
    /// Samples are encoded to an in-memory WAV and sent as a multipart upload;
    /// the API resamples server-side, so no local resampler is needed.
    pub async fn transcribe(&self, samples: &[f32]) -> Result<String, ApiError> {
        if samples.is_empty() {
            return Err(ApiError::EmptyAudio);
        }

        let part = reqwest::multipart::Part::bytes(wav_bytes(samples))
            .file_name("audio.wav")
            .mime_str("audio/wav")?;
        let mut form = reqwest::multipart::Form::new()
            .part("file", part)
            .text("model", self.config.model.clone())
            .text("response_format", "json");
        if let Some(language) = &self.config.language {
            form = form.text("language", language.clone());
        }

        let response = self
            .client
            .post(&self.config.endpoint)
            .bearer_auth(&self.api_key)
            .multipart(form)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(ApiError::Status {
                status: status.as_u16(),
                body,
            });
        }

        let parsed: TranscriptionResponse = response.json().await?;
        Ok(parsed.text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_key_is_rejected() {
        let err = ApiTranscriber::new("   ".to_owned(), ApiConfig::default()).unwrap_err();
        assert!(matches!(err, ApiError::MissingKey));
    }

    #[test]
    fn parses_json_text_field() {
        let parsed: TranscriptionResponse =
            serde_json::from_str(r#"{"text":"hello world"}"#).expect("valid response JSON");
        assert_eq!(parsed.text, "hello world");
    }
}
