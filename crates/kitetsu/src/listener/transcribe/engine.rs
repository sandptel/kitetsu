//! Pure synchronous inference: loaded engine + f32 samples → transcript text or timed words.
//!
//! Deliberately contains no capture logic and no async code. Progress updates
//! are emitted via an optional `Sender<u8>` (values 0–100).

use std::sync::mpsc::Sender;

use crate::listener::ListenerError;

use super::model::LoadedModel;

// ── Token-level output ────────────────────────────────────────────────────────

/// A single word reconstructed from one or more whisper tokens, carrying the
/// end timestamp of the last token in the word.
///
/// Timestamps are in centiseconds (whisper.cpp convention: 100 units = 1 s).
/// To convert to a sample offset at 16 kHz: `end_cs * 160`.
#[derive(Debug, Clone)]
pub(crate) struct TimedWord {
    /// The word text, with leading/trailing whitespace stripped.
    pub text: String,
    /// End time of this word in centiseconds.
    pub end_cs: i64,
}

/// Run inference on `samples`, returning a `Vec<TimedWord>` with per-word end timestamps.
///
/// Used by the LocalAgreement mode for word-level agreement and precise buffer trimming.
/// `initial_prompt` seeds the decoder with prior context (same semantics as
/// [`transcribe_samples`]). The call blocks for the duration of inference.
pub(crate) fn transcribe_words(
    model: &mut LoadedModel,
    samples: &[f32],
    initial_prompt: &str,
) -> Result<Vec<TimedWord>, ListenerError> {
    do_transcribe_words(model, samples, initial_prompt)
}

/// Run inference on `samples` (16 kHz mono f32) using the pre-loaded `model`.
///
/// `initial_prompt` seeds the decoder with prior context (last ~200 chars of
/// committed text) to improve cross-utterance continuity. Pass `""` to disable.
/// `progress_tx` receives values 0–100 as the decoder advances. The call blocks
/// for the duration of inference; wrap it in `tokio::task::spawn_blocking` at
/// the call site.
pub(crate) fn transcribe_samples(
    model: &mut LoadedModel,
    samples: &[f32],
    initial_prompt: &str,
    progress_tx: Option<Sender<u8>>,
) -> Result<String, ListenerError> {
    do_transcribe(model, samples, initial_prompt, progress_tx)
}

#[cfg(feature = "whisper")]
fn do_transcribe(
    model: &mut LoadedModel,
    samples: &[f32],
    initial_prompt: &str,
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

            if !initial_prompt.is_empty() {
                params.set_initial_prompt(initial_prompt);
            }

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
    _initial_prompt: &str,
    _progress_tx: Option<Sender<u8>>,
) -> Result<String, ListenerError> {
    Err(ListenerError::BackendNotCompiled(
        "whisper — enable the 'whisper' feature",
    ))
}

// ── Token-level inference ─────────────────────────────────────────────────────

#[cfg(feature = "whisper")]
fn do_transcribe_words(
    model: &mut LoadedModel,
    samples: &[f32],
    initial_prompt: &str,
) -> Result<Vec<TimedWord>, ListenerError> {
    use whisper_rs::{FullParams, SamplingStrategy};

    match model {
        LoadedModel::Whisper(m) => {
            let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
            params.set_language(Some("en"));
            params.set_n_threads(m.n_threads);
            params.set_print_special(false);
            params.set_print_progress(false);
            params.set_print_realtime(false);
            // Enable token-level timestamps so t0/t1 are populated on each token.
            params.set_token_timestamps(true);
            params.set_print_timestamps(false);
            params.set_suppress_blank(true);
            params.set_suppress_nst(true);

            if !initial_prompt.is_empty() {
                params.set_initial_prompt(initial_prompt);
            }

            m.state
                .full(params, samples)
                .map_err(|e| ListenerError::Transcription(e.to_string()))?;

            let n_segs = m.state.full_n_segments();
            let mut words: Vec<TimedWord> = Vec::new();

            for seg_i in 0..n_segs {
                let Some(seg) = m.state.get_segment(seg_i) else {
                    continue;
                };
                let n_tok = seg.n_tokens();
                let mut current_word = String::new();
                let mut current_end_cs: i64 = 0;

                for tok_i in 0..n_tok {
                    let Some(tok) = seg.get_token(tok_i) else {
                        continue;
                    };
                    let Ok(text) = tok.to_str() else { continue };
                    // Special tokens look like "[_BEG_]" or "<|endoftext|>"; skip them.
                    if text.is_empty() || text.starts_with('[') || text.starts_with('<') {
                        continue;
                    }
                    let data = tok.token_data();
                    // t1 is the end time of THIS token in centiseconds.
                    let tok_end_cs = data.t1;

                    if text.starts_with(' ') {
                        // Leading space marks a new word boundary. Flush whatever
                        // was accumulating (carrying its own latest t1).
                        if !current_word.is_empty() {
                            words.push(TimedWord {
                                text: std::mem::take(&mut current_word),
                                end_cs: current_end_cs,
                            });
                        }
                        current_word.push_str(text.trim_start());
                    } else {
                        // Sub-word token (e.g. "ing", "tion"): append to in-progress word.
                        current_word.push_str(text);
                    }
                    // Always advance the running end timestamp for this word.
                    current_end_cs = tok_end_cs;
                }

                // Flush the last word in the segment.
                if !current_word.is_empty() {
                    words.push(TimedWord {
                        text: current_word,
                        end_cs: current_end_cs,
                    });
                }
            }

            Ok(words)
        }
    }
}

#[cfg(not(feature = "whisper"))]
fn do_transcribe_words(
    _model: &mut LoadedModel,
    _samples: &[f32],
    _initial_prompt: &str,
) -> Result<Vec<TimedWord>, ListenerError> {
    Err(ListenerError::BackendNotCompiled(
        "whisper — enable the 'whisper' feature",
    ))
}
