//! Audio capture and transcription for microphone and system output.
//!
//! `capture` sub-package handles device discovery and blocking PCM recording.
//! `transcribe` sub-package handles model loading and one-shot STT inference.
//! This module owns the async pipe that composes both, and defines the public
//! sink-selection types (`AudioSink`, `InputSink`, `OutputSink`).
//!
//! Deliberately knows nothing about LLM interactions, session state, or UI.
//! All blocking operations run inside `tokio::task::spawn_blocking` so that
//! libpulse objects never cross an `.await` boundary (see Decision Log #11).

mod capture;
mod error;
// The physical directory is `transcribe/` but we name the module `transcriber`
// to avoid a conflict with the public `fn transcribe` below.
#[path = "transcribe/mod.rs"]
mod transcriber;

use std::time::Duration;

pub use capture::{
    CaptureConfig, Devices, Recorder, RecordingHandle, capture_to_wav, discover_default_devices,
};
pub use error::ListenerError;
pub use transcriber::{AudioModel, MoonshineVariant, Transcriber};

// ── Sink selection types ──────────────────────────────────────────────────────

/// Selects which audio stream to transcribe.
#[derive(Debug)]
pub enum AudioSink {
    Input(InputSink),
    Output(OutputSink),
}

/// Selects the microphone (input) source.
#[derive(Debug)]
pub enum InputSink {
    /// Default mic source resolved via PulseAudio introspection.
    Default,
    /// A specific PulseAudio source name.
    Custom(String),
}

/// Selects the system-audio (output) monitor source.
#[derive(Debug)]
pub enum OutputSink {
    /// Default sink monitor resolved via PulseAudio introspection.
    Default,
    /// A specific PulseAudio monitor source name.
    Custom(String),
}

// ── Public async API ──────────────────────────────────────────────────────────

/// Capture audio from `sink` for [`DEFAULT_CAPTURE_SECS`][capture::DEFAULT_CAPTURE_SECS]
/// seconds and transcribe it using `model`.
///
/// If `model` is unavailable (backend feature off or model file missing), falls
/// back to `AudioModel::Default` with a warning log. The full capture + inference
/// pipeline runs inside a blocking task so it never touches the async executor.
pub async fn transcribe(sink: AudioSink, model: AudioModel) -> Result<String, ListenerError> {
    tokio::task::spawn_blocking(move || {
        let device = resolve_device(&sink)?;
        let label = sink_label(&sink);
        let model = model.resolve();
        let samples = capture::capture_samples(
            &device,
            label,
            Duration::from_secs(capture::DEFAULT_CAPTURE_SECS),
        )?;
        let mut loaded = model.load()?;
        loaded.transcribe(&samples, None)
    })
    .await
    .map_err(ListenerError::join)?
}

/// Convenience wrapper: transcribe from a microphone source.
pub async fn transcribe_input(sink: InputSink, model: AudioModel) -> Result<String, ListenerError> {
    transcribe(AudioSink::Input(sink), model).await
}

/// Convenience wrapper: transcribe from a system-audio monitor source.
pub async fn transcribe_output(
    sink: OutputSink,
    model: AudioModel,
) -> Result<String, ListenerError> {
    transcribe(AudioSink::Output(sink), model).await
}

// ── Private helpers ───────────────────────────────────────────────────────────

fn resolve_device(sink: &AudioSink) -> Result<String, ListenerError> {
    match sink {
        AudioSink::Input(InputSink::Default) => Ok(capture::discover_default_devices()?.mic_source),
        AudioSink::Input(InputSink::Custom(name)) => Ok(name.clone()),
        AudioSink::Output(OutputSink::Default) => {
            Ok(capture::discover_default_devices()?.system_monitor)
        }
        AudioSink::Output(OutputSink::Custom(name)) => Ok(name.clone()),
    }
}

fn sink_label(sink: &AudioSink) -> &'static str {
    match sink {
        AudioSink::Input(_) => "kitetsu-transcribe-input",
        AudioSink::Output(_) => "kitetsu-transcribe-output",
    }
}
