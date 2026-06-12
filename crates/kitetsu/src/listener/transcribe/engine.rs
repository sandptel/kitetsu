//! Pure synchronous inference: loaded engine + f32 samples → transcript text.
//!
//! Deliberately contains no capture logic and no async code. Progress updates
//! are emitted via an optional `Sender<u8>` (values 0–100). The rolling-window
//! / live-transcription path will extend this module in a future iteration.

use std::sync::mpsc::Sender;

use crate::listener::ListenerError;

use super::model::LoadedModel;

/// Run inference on `samples` (16 kHz mono f32) using the pre-loaded `model`.
///
/// `progress_tx` receives values 0–100 as the decoder advances. The call blocks
/// for the duration of inference; wrap it in `tokio::task::spawn_blocking` at
/// the call site.
pub(crate) fn transcribe_samples(
    model: &mut LoadedModel,
    samples: &[f32],
    progress_tx: Option<Sender<u8>>,
) -> Result<String, ListenerError> {
    do_transcribe(model, samples, progress_tx)
}

#[cfg(feature = "whisper")]
fn do_transcribe(
    model: &mut LoadedModel,
    samples: &[f32],
    progress_tx: Option<Sender<u8>>,
) -> Result<String, ListenerError> {
    use whisper_rs::{FullParams, SamplingStrategy};

    match model {
        LoadedModel::Whisper(m) => {
            let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
            // Pin to English; skips the language-detection pass (~0.5 s per call).
            params.set_language(Some("en"));
            params.set_n_threads(m.n_threads);
            params.set_print_special(false);
            params.set_print_progress(false);
            params.set_print_realtime(false);
            params.set_print_timestamps(false);
            params.set_suppress_blank(true);
            params.set_suppress_nst(true);

            if let Some(tx) = progress_tx {
                params.set_progress_callback_safe(move |p: i32| {
                    let _ = tx.send(p.clamp(0, 100) as u8);
                });
            }

            m.state
                .full(params, samples)
                .map_err(|e| ListenerError::Transcription(e.to_string()))?;

            let n = m.state.full_n_segments();
            let mut parts: Vec<String> = Vec::with_capacity(n as usize);
            for i in 0..n {
                if let Some(seg) = m.state.get_segment(i)
                    && let Ok(txt) = seg.to_str()
                {
                    let trimmed = txt.trim();
                    if !trimmed.is_empty() {
                        parts.push(trimmed.to_owned());
                    }
                }
            }

            Ok(parts.join(" "))
        }
    }
}

#[cfg(not(feature = "whisper"))]
fn do_transcribe(
    _model: &mut LoadedModel,
    _samples: &[f32],
    _progress_tx: Option<Sender<u8>>,
) -> Result<String, ListenerError> {
    Err(ListenerError::BackendNotCompiled(
        "whisper — enable the 'whisper' feature",
    ))
}
