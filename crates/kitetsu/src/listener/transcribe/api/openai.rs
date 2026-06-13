//! Raw OpenAI REST building blocks for audio transcription.
//!
//! Provides configuration, error types, and the WAV encoder for uploading audio
//! to `/v1/audio/transcriptions`. Called by [`super::transcribe`].
//! Not here: high-level transcription logic, local model inference, capture.

const SAMPLE_RATE: u32 = 16_000;
const CHANNELS: u16 = 1;
const BITS_PER_SAMPLE: u16 = 16;

/// Failures specific to the OpenAI REST transcription backend.
///
/// Kept separate from `ListenerError` — feature-gating this module keeps
/// `reqwest` out of the default build.
#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    /// The supplied API key was empty after trimming.
    #[error("API key is empty")]
    MissingKey,
    /// No samples were supplied.
    #[error("no audio to transcribe")]
    EmptyAudio,
    /// HTTP request failed (DNS, TLS, connection, decode).
    #[error("transcription request failed")]
    Request(#[from] reqwest::Error),
    /// Non-2xx response from the endpoint.
    #[error("transcription API returned {status}: {body}")]
    Status { status: u16, body: String },
}

/// Tunables for [`super::transcribe::ApiTranscriber`].
#[derive(Debug, Clone)]
pub struct ApiConfig {
    /// Transcription model (e.g. `gpt-4o-transcribe`, `whisper-1`).
    pub model: String,
    /// Full endpoint URL — overridable for OpenAI-compatible hosts.
    pub endpoint: String,
    /// Optional ISO-639-1 language hint.
    pub language: Option<String>,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            model: "gpt-4o-transcribe".to_owned(),
            endpoint: "https://api.openai.com/v1/audio/transcriptions".to_owned(),
            language: None,
        }
    }
}

/// Encode 16 kHz mono f32 `samples` into an in-memory S16LE PCM WAV.
pub(super) fn wav_bytes(samples: &[f32]) -> Vec<u8> {
    let pcm_len = samples.len() * 2;
    let byte_rate = SAMPLE_RATE * u32::from(CHANNELS) * u32::from(BITS_PER_SAMPLE / 8);
    let block_align = CHANNELS * (BITS_PER_SAMPLE / 8);

    let mut out = Vec::with_capacity(44 + pcm_len);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + pcm_len as u32).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&CHANNELS.to_le_bytes());
    out.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&BITS_PER_SAMPLE.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(pcm_len as u32).to_le_bytes());
    for &sample in samples {
        let scaled = (sample.clamp(-1.0, 1.0) * f32::from(i16::MAX)) as i16;
        out.extend_from_slice(&scaled.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_header_describes_16khz_mono_pcm() {
        let bytes = wav_bytes(&[0.0, 0.5, -0.5]);
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(&bytes[12..16], b"fmt ");
        assert_eq!(u16::from_le_bytes([bytes[20], bytes[21]]), 1);
        assert_eq!(u16::from_le_bytes([bytes[22], bytes[23]]), 1);
        assert_eq!(
            u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]),
            SAMPLE_RATE
        );
        assert_eq!(&bytes[36..40], b"data");
    }

    #[test]
    fn wav_length_is_header_plus_two_bytes_per_sample() {
        let bytes = wav_bytes(&[0.0; 10]);
        assert_eq!(bytes.len(), 44 + 10 * 2);
        assert_eq!(
            u32::from_le_bytes([bytes[40], bytes[41], bytes[42], bytes[43]]),
            20
        );
    }
}
