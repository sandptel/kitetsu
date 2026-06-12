//! Core streaming abstractions: mode selection, result type, and the
//! top-level `StreamingTranscriber` that dispatches to each mode's implementation.
//!
//! `VadGated` dispatches to `vad::VadSegmenter`. `LocalAgreement` dispatches
//! to `local_agreement::LocalAgreementSegmenter`.
//! Does no audio capture or threading.

use super::Transcriber;
use super::local_agreement::{LaAction, LocalAgreementSegmenter, PassOutcome};
use super::vad::{FlushSignal, VadSegmenter};
use crate::listener::ListenerError;

use tracing::debug;

/// Maximum chars of prior committed text used as the initial-prompt seed.
/// Applies to all streaming modes that use rolling context.
const MAX_PROMPT_CHARS: usize = 200;

// ── Public types ──────────────────────────────────────────────────────────────

/// Selects the live-transcription algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamMode {
    /// Energy-VAD-gated utterance chunking (`vad` module).
    /// Each utterance is transcribed as a unit; silence gaps determine boundaries.
    /// `FeedResult::tentative` is always empty — text appears only once a full
    /// utterance is committed.
    VadGated,
    /// Two-pass LocalAgreement-2 sliding window (`local_agreement` module).
    /// Words are confirmed once two consecutive inference passes agree on them;
    /// the unconfirmed tail is exposed as `FeedResult::tentative`.
    LocalAgreement,
}

/// Result of feeding one block to a [`StreamingTranscriber`].
#[derive(Debug)]
pub struct FeedResult {
    /// A newly committed text, if the active mode fired and inference succeeded.
    /// `None` means the block was consumed but no utterance is ready yet.
    pub committed: Option<String>,
    /// In-progress tentative tail. Always `""` for `VadGated`; populated by
    /// `LocalAgreement` as words stream in before being confirmed.
    pub tentative: String,
}

// ── Per-mode segmenter state ──────────────────────────────────────────────────

enum ModeState {
    Vad(VadSegmenter),
    LocalAgreement(LocalAgreementSegmenter),
}

// ── StreamingTranscriber ──────────────────────────────────────────────────────

/// Dispatches incremental audio blocks to the selected streaming algorithm.
///
/// Feed ~64 ms blocks via [`StreamingTranscriber::feed`]. When the active mode
/// detects an utterance boundary (VAD) or sufficient agreement (LA), inference
/// runs and `FeedResult` carries the new text. The model stays loaded across
/// calls — no per-utterance reload cost.
pub struct StreamingTranscriber {
    transcriber: Transcriber,
    state: ModeState,
    /// Tail of committed text used as the inference prompt seed (≤ MAX_PROMPT_CHARS).
    prev_tail: String,
}

impl StreamingTranscriber {
    pub fn new(transcriber: Transcriber, mode: StreamMode) -> Self {
        let state = match mode {
            StreamMode::VadGated => ModeState::Vad(VadSegmenter::new()),
            StreamMode::LocalAgreement => ModeState::LocalAgreement(LocalAgreementSegmenter::new()),
        };
        Self {
            transcriber,
            state,
            prev_tail: String::new(),
        }
    }

    /// Feed one block of 16 kHz mono f32 samples.
    ///
    /// Returns `Ok(FeedResult)` after every block. Blocks the calling thread
    /// during inference — run the whole loop on a `std::thread`.
    pub fn feed(&mut self, block: &[f32]) -> Result<FeedResult, ListenerError> {
        match &mut self.state {
            ModeState::Vad(_) => self.feed_vad(block),
            ModeState::LocalAgreement(_) => self.feed_la(block),
        }
    }

    // ── VAD path ──────────────────────────────────────────────────────────────

    fn feed_vad(&mut self, block: &[f32]) -> Result<FeedResult, ListenerError> {
        let ModeState::Vad(ref mut seg) = self.state else {
            unreachable!()
        };
        match seg.feed(block) {
            FlushSignal::NoFlush => Ok(FeedResult {
                committed: None,
                tentative: String::new(),
            }),
            FlushSignal::Flush(samples) => {
                debug!(
                    samples = samples.len(),
                    "vad: utterance boundary, running inference"
                );
                let text =
                    self.transcriber
                        .transcribe_with_context(&samples, &self.prev_tail, None)?;
                debug!(
                    chars = text.len(),
                    committed = text.as_str(),
                    "vad: inference complete"
                );
                if !text.is_empty() {
                    self.push_committed(&text);
                    Ok(FeedResult {
                        committed: Some(text),
                        tentative: String::new(),
                    })
                } else {
                    Ok(FeedResult {
                        committed: None,
                        tentative: String::new(),
                    })
                }
            }
        }
    }

    // ── LocalAgreement path ───────────────────────────────────────────────────

    fn feed_la(&mut self, block: &[f32]) -> Result<FeedResult, ListenerError> {
        // Borrow only `state` to call feed (immutable parts follow separately).
        let action = {
            let ModeState::LocalAgreement(ref mut la) = self.state else {
                unreachable!()
            };
            la.feed(block)
        };

        match action {
            LaAction::Wait => Ok(FeedResult {
                committed: None,
                tentative: String::new(),
            }),
            LaAction::RunPass { is_final } => {
                // Clone the buffer for inference (avoids borrow conflict with
                // `self.transcriber` and `self.prev_tail`).
                let samples = {
                    let ModeState::LocalAgreement(ref la) = self.state else {
                        unreachable!()
                    };
                    la.buf.clone()
                };

                debug!(
                    samples = samples.len(),
                    is_final, "la: running inference pass"
                );
                let words = self
                    .transcriber
                    .transcribe_words(&samples, &self.prev_tail)?;
                debug!(words = words.len(), is_final, "la: inference pass complete");

                let outcome: PassOutcome = {
                    let ModeState::LocalAgreement(ref mut la) = self.state else {
                        unreachable!()
                    };
                    la.apply_pass(words, is_final)
                };

                if let Some(ref text) = outcome.committed
                    && !text.is_empty()
                {
                    self.push_committed(text);
                }

                Ok(FeedResult {
                    committed: outcome.committed,
                    tentative: outcome.tentative,
                })
            }
        }
    }

    // ── Shared helpers ────────────────────────────────────────────────────────

    /// Append newly committed text to `prev_tail`, capping at `MAX_PROMPT_CHARS`
    /// on a word boundary so the initial-prompt seed stays compact.
    fn push_committed(&mut self, text: &str) {
        if !self.prev_tail.is_empty() {
            self.prev_tail.push(' ');
        }
        self.prev_tail.push_str(text);

        if self.prev_tail.len() > MAX_PROMPT_CHARS {
            let start = self.prev_tail.len() - MAX_PROMPT_CHARS;
            let trim_at = self.prev_tail[start..]
                .find(' ')
                .map(|i| start + i + 1)
                .unwrap_or(start);
            self.prev_tail = self.prev_tail[trim_at..].to_owned();
        }
    }
}
