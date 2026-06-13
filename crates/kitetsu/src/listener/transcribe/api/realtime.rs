//! OpenAI Realtime WebSocket transcription backend.
//!
//! [`RealtimeSession`] streams 16 kHz mono f32 audio to the OpenAI Realtime
//! transcription endpoint and surfaces incremental transcript text via
//! [`TranscriptEvent`]. Audio is resampled 16 kHz → 24 kHz internally.
//!
//! Not here: capture, device discovery, session config — callers own all of
//! that and pass a `session.update` JSON value to [`RealtimeSession::connect`].

use base64::{Engine as _, engine::general_purpose::STANDARD as B64};
use futures_util::{SinkExt as _, StreamExt as _};
use tokio_tungstenite::tungstenite::client::IntoClientRequest as _;
use tokio_tungstenite::tungstenite::http::header::HeaderValue;
use tracing::{debug, info, warn};

// ── WebSocket type aliases ────────────────────────────────────────────────────

type WsConn = tokio_tungstenite::WebSocketStream<
    tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
>;
type WsSink = futures_util::stream::SplitSink<
    WsConn,
    tokio_tungstenite::tungstenite::Message,
>;
type WsStream = futures_util::stream::SplitStream<WsConn>;

// ── Public events & errors ────────────────────────────────────────────────────

/// An incremental transcription event surfaced from the server.
#[derive(Debug, Clone)]
pub enum TranscriptEvent {
    /// A partial transcript token (utterance in progress).
    Delta(String),
    /// Server-detected utterance boundary; the completed full transcript.
    Completed(String),
}

/// Errors specific to the Realtime WebSocket transcription backend.
///
/// Kept separate from `ListenerError` — the WS deps (tokio-tungstenite,
/// futures-util, base64) stay out of the default build, gated to `api`.
#[derive(Debug, thiserror::Error)]
pub enum RealtimeError {
    /// The supplied API key was empty after trimming.
    #[error("API key is empty")]
    MissingKey,
    /// Building the HTTP upgrade request failed (e.g. invalid URI or header value).
    #[error("invalid WebSocket request: {0}")]
    Request(String),
    /// WebSocket connection, TLS handshake, or framing error.
    #[error("WebSocket error: {0}")]
    WebSocket(#[from] tokio_tungstenite::tungstenite::Error),
    /// The server sent an `error` event.
    #[error("server error [{code}]: {message}")]
    Protocol { code: String, message: String },
    /// The WebSocket stream ended (server close or EOF).
    #[error("server closed the connection")]
    Closed,
    /// JSON serialisation or deserialisation failure.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

// ── Session halves ────────────────────────────────────────────────────────────

/// Sending half of a [`RealtimeSession`], obtained via [`RealtimeSession::split`].
///
/// Feed audio chunks with [`feed`][Self::feed]. When done, call
/// [`close`][Self::close] to send a WS close frame before dropping.
pub struct SessionSink {
    ws_sink: WsSink,
}

impl SessionSink {
    /// Resample 16 kHz mono f32 → 24 kHz PCM16, base64-encode, and send as
    /// `input_audio_buffer.append`. Empty slices are silently skipped.
    ///
    /// The OpenAI Realtime API requires PCM16 at 24 kHz; a dependency-free
    /// linear resampler (3:2 ratio) converts from the 16 kHz capture rate.
    pub async fn feed(&mut self, samples: &[f32]) -> Result<(), RealtimeError> {
        if samples.is_empty() {
            return Ok(());
        }
        let resampled = resample_16k_to_24k(samples);
        let pcm_bytes = f32_to_pcm16(&resampled);
        let encoded = B64.encode(&pcm_bytes);

        let msg = serde_json::json!({
            "type": "input_audio_buffer.append",
            "audio": encoded,
        });
        let text = serde_json::to_string(&msg)?;
        self.ws_sink
            .send(tokio_tungstenite::tungstenite::Message::Text(text.into()))
            .await?;
        debug!(samples = samples.len(), resampled = resampled.len(), "fed audio chunk");
        Ok(())
    }

    /// Send `input_audio_buffer.commit` to trigger transcription of buffered audio.
    ///
    /// Required for `gpt-realtime-whisper`, which does not support server-side VAD.
    /// Call after accumulating a meaningful utterance's worth of audio.
    pub async fn commit(&mut self) -> Result<(), RealtimeError> {
        let msg = serde_json::json!({ "type": "input_audio_buffer.commit" });
        let text = serde_json::to_string(&msg)?;
        self.ws_sink
            .send(tokio_tungstenite::tungstenite::Message::Text(text.into()))
            .await?;
        debug!("committed audio buffer");
        Ok(())
    }

    /// Send a WebSocket close frame. Best-effort — errors are logged and swallowed.
    pub async fn close(mut self) {
        if let Err(e) = self
            .ws_sink
            .send(tokio_tungstenite::tungstenite::Message::Close(None))
            .await
        {
            warn!(error = %e, "close frame send failed");
        }
        let _ = self.ws_sink.close().await;
        info!("realtime session sink closed");
    }
}

/// Receiving half of a [`RealtimeSession`], obtained via [`RealtimeSession::split`].
///
/// Poll transcript events with [`next_event`][Self::next_event] until it
/// returns `Err(RealtimeError::Closed)`.
pub struct SessionStream {
    ws_stream: WsStream,
}

impl SessionStream {
    /// Read and parse the next server WebSocket frame.
    ///
    /// Returns:
    /// - `Ok(Some(TranscriptEvent))` — delta or completed transcript text.
    /// - `Ok(None)` — lifecycle event (session created/updated, etc.); skip and
    ///   call again.
    /// - `Err(RealtimeError::Protocol)` — the server sent an `error` event.
    /// - `Err(RealtimeError::Closed)` — the stream ended (connection closed).
    pub async fn next_event(&mut self) -> Result<Option<TranscriptEvent>, RealtimeError> {
        use tokio_tungstenite::tungstenite::Message;
        loop {
            match self.ws_stream.next().await {
                None => return Err(RealtimeError::Closed),
                Some(Err(e)) => return Err(RealtimeError::WebSocket(e)),
                Some(Ok(Message::Close(_))) => return Err(RealtimeError::Closed),
                Some(Ok(Message::Text(text))) => {
                    if let Some(event) = parse_server_event(&text)? {
                        return Ok(Some(event));
                    }
                    // Lifecycle event — loop for the next frame.
                }
                Some(Ok(_)) => {} // ping/pong/binary — ignore
            }
        }
    }
}

// ── RealtimeSession ───────────────────────────────────────────────────────────

/// A connected OpenAI Realtime transcription session (one WS connection = one
/// audio stream).
///
/// Construct via [`connect`][Self::connect], then [`split`][Self::split] into a
/// [`SessionSink`] (for feeding audio) and [`SessionStream`] (for reading
/// transcript events). Run both halves in separate tasks for concurrent operation.
pub struct RealtimeSession {
    sink: WsSink,
    stream: WsStream,
}

impl RealtimeSession {
    /// Connect to the Realtime transcription endpoint and send the caller-provided
    /// `session_update` JSON (a `session.update` message).
    ///
    /// Uses `IntoClientRequest` so tungstenite auto-generates the
    /// `Sec-WebSocket-Key` and other required handshake headers; we then inject
    /// `Authorization` and `OpenAI-Beta` on top.
    ///
    /// Returns `RealtimeError::MissingKey` if `api_key` is blank after trimming.
    pub async fn connect(
        api_key: &str,
        endpoint: &str,
        session_update: &serde_json::Value,
    ) -> Result<Self, RealtimeError> {
        let key = api_key.trim();
        if key.is_empty() {
            return Err(RealtimeError::MissingKey);
        }

        // `into_client_request()` on a &str generates Sec-WebSocket-Key and all
        // other required upgrade headers. Manually using Request::builder()
        // bypasses this and causes the "missing sec-websocket-key" rejection.
        let mut request = endpoint
            .into_client_request()
            .map_err(|e| RealtimeError::Request(e.to_string()))?;
        {
            let headers = request.headers_mut();
            // Only Authorization — sending OpenAI-Beta: realtime=v1 routes the
            // connection to the deprecated beta shape and triggers a rejection.
            headers.insert(
                "Authorization",
                HeaderValue::from_str(&format!("Bearer {key}"))
                    .map_err(|e| RealtimeError::Request(e.to_string()))?,
            );
        }

        info!(endpoint = %endpoint, "connecting to Realtime API");
        let (ws_conn, _) = tokio_tungstenite::connect_async(request).await?;
        let (mut sink, stream) = ws_conn.split();

        // Send the session configuration supplied by the caller verbatim.
        let text = serde_json::to_string(session_update)?;
        sink.send(tokio_tungstenite::tungstenite::Message::Text(text.into()))
            .await?;
        info!("realtime session configured");

        Ok(Self { sink, stream })
    }

    /// Split into independent sink and stream halves that can be moved into
    /// separate tokio tasks for concurrent audio feed and event receive.
    pub fn split(self) -> (SessionSink, SessionStream) {
        (
            SessionSink { ws_sink: self.sink },
            SessionStream { ws_stream: self.stream },
        )
    }
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Upsample 16 kHz mono f32 to 24 kHz using linear interpolation (3:2 ratio).
///
/// For every 2 input samples we produce 3 output samples:
///   out[0] = in[0]
///   out[1] = in[0]*(2/3) + in[1]*(1/3)
///   out[2] = in[0]*(1/3) + in[1]*(2/3)
/// A trailing odd sample is passed through unchanged. No external dep needed —
/// the ratio is a clean 3:2 integer fraction.
fn resample_16k_to_24k(samples: &[f32]) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }
    let pairs = samples.len() / 2;
    let cap = pairs * 3 + (samples.len() & 1);
    let mut out = Vec::with_capacity(cap);

    for i in 0..pairs {
        let a = samples[i * 2];
        let b = samples[i * 2 + 1];
        out.push(a);
        out.push(a * (2.0 / 3.0) + b * (1.0 / 3.0));
        out.push(a * (1.0 / 3.0) + b * (2.0 / 3.0));
    }
    if samples.len() & 1 == 1 {
        out.push(*samples.last().expect("non-empty guaranteed above"));
    }
    out
}

/// Convert f32 samples (clamped to −1.0..1.0) to S16LE PCM bytes.
fn f32_to_pcm16(samples: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        let scaled = (s.clamp(-1.0, 1.0) * f32::from(i16::MAX)) as i16;
        bytes.extend_from_slice(&scaled.to_le_bytes());
    }
    bytes
}

/// Parse one server event frame.
/// - `Ok(Some(event))` for transcript delta/completed.
/// - `Ok(None)` for lifecycle events the caller should skip.
/// - `Err(RealtimeError::Protocol)` for server `error` events.
fn parse_server_event(text: &str) -> Result<Option<TranscriptEvent>, RealtimeError> {
    #[derive(serde::Deserialize)]
    struct Frame {
        #[serde(rename = "type")]
        event_type: String,
        delta: Option<String>,
        transcript: Option<String>,
        error: Option<ServerError>,
    }
    #[derive(serde::Deserialize)]
    struct ServerError {
        #[serde(default = "unknown_str")]
        code: String,
        message: String,
    }
    fn unknown_str() -> String {
        "unknown".to_owned()
    }

    let frame: Frame = serde_json::from_str(text)?;
    match frame.event_type.as_str() {
        // Conversation-item transcript events (transcription intent, GA).
        "conversation.item.input_audio_transcription.delta" => {
            Ok(Some(TranscriptEvent::Delta(frame.delta.unwrap_or_default())))
        }
        "conversation.item.input_audio_transcription.completed" => Ok(Some(
            TranscriptEvent::Completed(frame.transcript.unwrap_or_default()),
        )),
        // Response-level transcript events; also sent by some transcription sessions.
        "response.audio_transcript.delta" => {
            Ok(Some(TranscriptEvent::Delta(frame.delta.unwrap_or_default())))
        }
        "response.audio_transcript.done" => Ok(Some(TranscriptEvent::Completed(
            frame.transcript.unwrap_or_default(),
        ))),
        "error" => {
            let e = frame.error.unwrap_or(ServerError {
                code: "unknown".to_owned(),
                message: "unknown server error".to_owned(),
            });
            Err(RealtimeError::Protocol { code: e.code, message: e.message })
        }
        other => {
            debug!(event_type = other, "server lifecycle event");
            Ok(None)
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── f32_to_pcm16 ─────────────────────────────────────────────────────────

    #[test]
    fn pcm16_clamps_and_scales_known_values() {
        let samples = [0.0f32, 1.0, -1.0, 0.5];
        let bytes = f32_to_pcm16(&samples);
        assert_eq!(bytes.len(), 8);
        let words: Vec<i16> = bytes
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]))
            .collect();
        assert_eq!(words[0], 0);
        assert_eq!(words[1], i16::MAX);
        assert_eq!(words[2], i16::MIN + 1); // -1.0 * 32767 = -32767 (clamped)
        assert_eq!(words[3], (0.5 * f32::from(i16::MAX)) as i16);
    }

    #[test]
    fn pcm16_output_is_two_bytes_per_sample() {
        assert_eq!(f32_to_pcm16(&[0.0f32; 100]).len(), 200);
    }

    // ── resample_16k_to_24k ──────────────────────────────────────────────────

    #[test]
    fn resample_empty_input_is_empty() {
        assert!(resample_16k_to_24k(&[]).is_empty());
    }

    #[test]
    fn resample_single_sample_passthrough() {
        let out = resample_16k_to_24k(&[0.5f32]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], 0.5);
    }

    #[test]
    fn resample_two_samples_produce_three() {
        let out = resample_16k_to_24k(&[0.0f32, 1.0]);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0], 0.0);
        assert!((out[1] - 1.0 / 3.0).abs() < 1e-6, "out[1] = {}", out[1]);
        assert!((out[2] - 2.0 / 3.0).abs() < 1e-6, "out[2] = {}", out[2]);
    }

    #[test]
    fn resample_4096_chunk_produces_6144() {
        let out = resample_16k_to_24k(&[0.0f32; 4096]);
        assert_eq!(out.len(), 6144);
    }

    #[test]
    fn resample_odd_input_length() {
        // 5 samples (2 pairs + 1 trailing) → 7 (2*3 + 1).
        let out = resample_16k_to_24k(&[0.0f32; 5]);
        assert_eq!(out.len(), 7);
    }

    // ── parse_server_event ───────────────────────────────────────────────────

    #[test]
    fn parses_delta_event() {
        let json = r#"{"type":"conversation.item.input_audio_transcription.delta","delta":"Hello"}"#;
        let evt = parse_server_event(json).unwrap().unwrap();
        assert!(matches!(evt, TranscriptEvent::Delta(s) if s == "Hello"));
    }

    #[test]
    fn parses_completed_event() {
        let json = r#"{"type":"conversation.item.input_audio_transcription.completed","transcript":"Hello world."}"#;
        let evt = parse_server_event(json).unwrap().unwrap();
        assert!(matches!(evt, TranscriptEvent::Completed(s) if s == "Hello world."));
    }

    #[test]
    fn lifecycle_events_return_none() {
        let json = r#"{"type":"session.created","session":{}}"#;
        assert!(parse_server_event(json).unwrap().is_none());
    }

    #[test]
    fn unknown_event_type_returns_none() {
        let json = r#"{"type":"session.updated","session":{}}"#;
        assert!(parse_server_event(json).unwrap().is_none());
    }

    #[test]
    fn error_event_returns_err() {
        let json =
            r#"{"type":"error","error":{"code":"invalid_request","message":"bad audio"}}"#;
        let result = parse_server_event(json);
        assert!(
            matches!(result, Err(RealtimeError::Protocol { ref code, ref message })
                if code == "invalid_request" && message == "bad audio")
        );
    }

    // ── RealtimeSession::connect ─────────────────────────────────────────────

    #[tokio::test]
    async fn connect_rejects_blank_key() {
        // Key validation runs before any network I/O.
        let result = RealtimeSession::connect(
            "   ",
            "wss://api.openai.com/v1/realtime?intent=transcription",
            &serde_json::json!({}),
        )
        .await;
        assert!(matches!(result, Err(RealtimeError::MissingKey)));
    }
}
