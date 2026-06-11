//! Pure synchronous inference: loaded engine + f32 samples → transcript text.
//!
//! Deliberately contains no capture logic and no async code. The rolling-window
//! / live-transcription path will extend this module in a future iteration.

use crate::listener::ListenerError;

use super::model::LoadedModel;

/// Run inference on `samples` (16 kHz mono f32) using the pre-loaded `model`.
///
/// Returns the transcript string on success. The call blocks for the duration
/// of inference; wrap it in `tokio::task::spawn_blocking` at the call site.
pub(crate) fn transcribe_samples(
    model: &mut LoadedModel,
    samples: &[f32],
) -> Result<String, ListenerError> {
    do_transcribe(model, samples)
}

#[cfg(feature = "whisper")]
fn do_transcribe(model: &mut LoadedModel, samples: &[f32]) -> Result<String, ListenerError> {
    use transcribe_rs::whisper_cpp::WhisperInferenceParams;
    match model {
        LoadedModel::Whisper(engine) => engine
            .transcribe_with(samples, &WhisperInferenceParams::default())
            .map(|r| r.text)
            .map_err(|e| ListenerError::Transcription(e.to_string())),
    }
}

#[cfg(not(feature = "whisper"))]
fn do_transcribe(_model: &mut LoadedModel, _samples: &[f32]) -> Result<String, ListenerError> {
    Err(ListenerError::BackendNotCompiled(
        "whisper — enable the 'whisper' feature",
    ))
}
