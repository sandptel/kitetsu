//! Live VAD-gated streaming transcription.
//!
//! `VadSegmenter` is a pure state machine (no model knowledge) that accumulates
//! audio blocks and signals when an utterance is ready to transcribe. Tests use
//! it directly with synthetic samples. `StreamingTranscriber` composes it with
//! a `Transcriber` and dispatches `StreamMode`. Does no capture or threading.

use super::Transcriber;
use crate::listener::ListenerError;

// ── Constants (all in samples at 16 kHz) ─────────────────────────────────────

/// RMS floor below which a block is considered silence (~-40 dBFS).
const SILENCE_RMS: f32 = 0.01;

/// Minimum consecutive speech samples required before a segment is considered
/// real speech vs. transient noise (250 ms). The noise gate threshold.
const MIN_SPEECH_SAMPLES: usize = 4_000;

/// Minimum consecutive silence samples after speech before flushing (400 ms).
const MIN_SILENCE_SAMPLES: usize = 1_600;

/// Pre-roll prepended when speech starts — catches the attack of the word (100 ms).
const PRE_ROLL_SAMPLES: usize = 1_600;

/// Absolute speech cap: flush regardless of silence if the buffer hits 15 s.
/// Guards against non-stop speech overwhelming accuracy.
const MAX_SPEECH_SAMPLES: usize = 240_000;

/// Minimum viable utterance to bother sending to inference (1.5 s).
/// Shorter clips produce unreliable output from whisper.
const MIN_VIABLE_SAMPLES: usize = 6_400;

/// Maximum chars of prior committed text used as an initial prompt seed.
const MAX_PROMPT_CHARS: usize = 200;

// ── Public types ──────────────────────────────────────────────────────────────

/// Selects the live-transcription algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamMode {
    /// Energy-VAD-gated utterance chunking. Each utterance is transcribed as a
    /// unit; silence gaps determine boundaries. `FeedResult::tentative` is always
    /// empty — text only appears once a full utterance is committed.
    VadGated,
    /// Two-pass LocalAgreement-2 sliding window (reserved — see PLAN Decision #21).
    /// Returns [`ListenerError::NotImplemented`] until wired.
    LocalAgreement,
}

/// Result of feeding one block to a [`StreamingTranscriber`].
#[derive(Debug)]
pub struct FeedResult {
    /// A newly committed utterance, if the VAD fired and inference succeeded.
    /// `None` means the block was consumed but no utterance is ready yet.
    pub committed: Option<String>,
    /// In-progress tentative tail (always `""` for `VadGated`; populated once
    /// `LocalAgreement` is implemented).
    pub tentative: String,
}

// ── VAD state machine (pure, no model dependency) ────────────────────────────

enum VadState {
    Idle,
    /// Speech detected; accumulating until MIN_SPEECH_SAMPLES to reject noise.
    Debouncing {
        speech_run: usize,
    },
    /// Confirmed speech; tracking silence to determine flush point.
    Speaking {
        speech_samples: usize,
        silence_run: usize,
    },
}

/// Result of feeding a block through the VAD state machine.
pub(crate) enum FlushSignal {
    /// Not enough silence or cap reached; continue accumulating.
    NoFlush,
    /// Utterance boundary detected; contains the samples to transcribe.
    Flush(Vec<f32>),
}

/// Energy-based VAD state machine. Pure: no inference, no I/O, no async.
///
/// Feed 16 kHz mono f32 blocks via [`VadSegmenter::feed`]; the machine returns
/// a [`FlushSignal::Flush`] when an utterance boundary is detected.
pub(crate) struct VadSegmenter {
    state: VadState,
    /// Accumulates the current utterance (pre-roll + speech).
    buf: Vec<f32>,
    /// Rolling window of the last PRE_ROLL_SAMPLES before speech starts.
    pre_roll: Vec<f32>,
}

impl VadSegmenter {
    pub(crate) fn new() -> Self {
        Self {
            state: VadState::Idle,
            buf: Vec::new(),
            pre_roll: Vec::with_capacity(PRE_ROLL_SAMPLES * 2),
        }
    }

    /// Feed one block of 16 kHz mono f32 samples.
    pub(crate) fn feed(&mut self, block: &[f32]) -> FlushSignal {
        let is_speech = rms(block) >= SILENCE_RMS;

        match &self.state {
            VadState::Idle => {
                if is_speech {
                    // Prepend pre-roll so the attack of the first word is included.
                    let mut new_buf = std::mem::take(&mut self.pre_roll);
                    new_buf.extend_from_slice(block);
                    self.buf = new_buf;
                    let speech_run = block.len();
                    self.state = if speech_run >= MIN_SPEECH_SAMPLES {
                        VadState::Speaking {
                            speech_samples: speech_run,
                            silence_run: 0,
                        }
                    } else {
                        VadState::Debouncing { speech_run }
                    };
                } else {
                    self.update_pre_roll(block);
                }
                FlushSignal::NoFlush
            }
            VadState::Debouncing { speech_run } => {
                let speech_run = *speech_run;
                if is_speech {
                    self.buf.extend_from_slice(block);
                    let new_run = speech_run + block.len();
                    self.state = if new_run >= MIN_SPEECH_SAMPLES {
                        VadState::Speaking {
                            speech_samples: new_run,
                            silence_run: 0,
                        }
                    } else {
                        VadState::Debouncing {
                            speech_run: new_run,
                        }
                    };
                    FlushSignal::NoFlush
                } else {
                    // Noise gate rejected — too short to be real speech.
                    self.buf.clear();
                    self.pre_roll.clear();
                    self.state = VadState::Idle;
                    FlushSignal::NoFlush
                }
            }
            VadState::Speaking {
                speech_samples,
                silence_run,
            } => {
                let (mut sp, mut sr) = (*speech_samples, *silence_run);
                self.buf.extend_from_slice(block);
                sp += block.len();

                if is_speech {
                    sr = 0;
                    self.state = VadState::Speaking {
                        speech_samples: sp,
                        silence_run: sr,
                    };
                    if sp >= MAX_SPEECH_SAMPLES {
                        return self.flush();
                    }
                } else {
                    sr += block.len();
                    self.state = VadState::Speaking {
                        speech_samples: sp,
                        silence_run: sr,
                    };
                    if sr >= MIN_SILENCE_SAMPLES {
                        return self.flush();
                    }
                }
                FlushSignal::NoFlush
            }
        }
    }

    fn flush(&mut self) -> FlushSignal {
        let samples = std::mem::take(&mut self.buf);
        self.pre_roll.clear();
        self.state = VadState::Idle;
        if samples.len() >= MIN_VIABLE_SAMPLES {
            FlushSignal::Flush(samples)
        } else {
            FlushSignal::NoFlush
        }
    }

    fn update_pre_roll(&mut self, block: &[f32]) {
        self.pre_roll.extend_from_slice(block);
        if self.pre_roll.len() > PRE_ROLL_SAMPLES {
            let excess = self.pre_roll.len() - PRE_ROLL_SAMPLES;
            self.pre_roll.drain(..excess);
        }
    }
}

// ── StreamingTranscriber ──────────────────────────────────────────────────────

/// Wraps a `Transcriber` with a VAD segmenter for live incremental transcription.
///
/// Feed ~64 ms blocks via [`StreamingTranscriber::feed`]. When the VAD detects
/// an utterance boundary, inference runs and `FeedResult::committed` contains
/// the new text. The model stays loaded across calls — no per-utterance reload.
pub struct StreamingTranscriber {
    transcriber: Transcriber,
    segmenter: VadSegmenter,
    mode: StreamMode,
    /// Tail of the last committed utterance used as the prompt seed (≤ MAX_PROMPT_CHARS).
    prev_tail: String,
}

impl StreamingTranscriber {
    pub fn new(transcriber: Transcriber, mode: StreamMode) -> Self {
        Self {
            transcriber,
            segmenter: VadSegmenter::new(),
            mode,
            prev_tail: String::new(),
        }
    }

    /// Feed one block of 16 kHz mono f32 samples.
    ///
    /// Returns `Ok(FeedResult)` after every block; `committed` is `Some(text)`
    /// only when the VAD fired and inference completed. Blocks the calling thread
    /// during inference — run the whole loop on a `std::thread`.
    pub fn feed(&mut self, block: &[f32]) -> Result<FeedResult, ListenerError> {
        match self.mode {
            StreamMode::VadGated => self.feed_vad(block),
            StreamMode::LocalAgreement => {
                // reserved: see PLAN Decision #21
                Err(ListenerError::NotImplemented(
                    "LocalAgreement — reserved: see PLAN Decision #21",
                ))
            }
        }
    }

    fn feed_vad(&mut self, block: &[f32]) -> Result<FeedResult, ListenerError> {
        match self.segmenter.feed(block) {
            FlushSignal::NoFlush => Ok(FeedResult {
                committed: None,
                tentative: String::new(),
            }),
            FlushSignal::Flush(samples) => {
                let text =
                    self.transcriber
                        .transcribe_with_context(&samples, &self.prev_tail, None)?;
                if !text.is_empty() {
                    // Keep the tail of committed text as the seed for the next utterance.
                    let combined = if self.prev_tail.is_empty() {
                        text.clone()
                    } else {
                        format!("{} {}", self.prev_tail, text)
                    };
                    if combined.len() <= MAX_PROMPT_CHARS {
                        self.prev_tail = combined;
                    } else {
                        let start = combined.len() - MAX_PROMPT_CHARS;
                        // Trim to a word boundary if possible.
                        let trim_at = combined[start..]
                            .find(' ')
                            .map(|i| start + i + 1)
                            .unwrap_or(start);
                        self.prev_tail = combined[trim_at..].to_owned();
                    }
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
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn silence(n: usize) -> Vec<f32> {
        vec![0.0f32; n]
    }

    fn tone(n: usize) -> Vec<f32> {
        // 0.5 RMS → well above SILENCE_RMS threshold of 0.01
        vec![0.5f32; n]
    }

    const BLOCK: usize = 1024;

    /// Feed `blocks` of `samples_fn(BLOCK)` into the segmenter; return flush count.
    fn count_flushes(
        seg: &mut VadSegmenter,
        blocks: usize,
        samples_fn: fn(usize) -> Vec<f32>,
    ) -> usize {
        (0..blocks)
            .filter(|_| matches!(seg.feed(&samples_fn(BLOCK)), FlushSignal::Flush(_)))
            .count()
    }

    #[test]
    fn silence_only_never_flushes() {
        let mut seg = VadSegmenter::new();
        // Feed 10 s of silence — no flush.
        assert_eq!(count_flushes(&mut seg, 10 * 16_000 / BLOCK, silence), 0);
    }

    #[test]
    fn short_noise_below_min_speech_does_not_flush() {
        let mut seg = VadSegmenter::new();
        // 2 blocks of tone = 2048 samples < MIN_SPEECH_SAMPLES (4000) → Debouncing
        for _ in 0..2 {
            seg.feed(&tone(BLOCK));
        }
        // Long silence follows → Debouncing → Idle (noise gate rejects, no flush)
        assert_eq!(count_flushes(&mut seg, 20, silence), 0);
    }

    #[test]
    fn speech_then_silence_flushes_one_utterance() {
        let mut seg = VadSegmenter::new();
        // 3 s of speech → Speaking
        count_flushes(&mut seg, 3 * 16_000 / BLOCK, tone);
        // 1 s of silence contains 7 × 1024 = 7168 samples of silence;
        // MIN_SILENCE_SAMPLES = 6400, so the flush fires during this pass.
        let n = count_flushes(&mut seg, 16_000 / BLOCK, silence);
        assert_eq!(n, 1);
    }

    #[test]
    fn max_speech_cap_forces_flush() {
        let mut seg = VadSegmenter::new();
        // 16 s of continuous tone → MAX_SPEECH_SAMPLES (15 s) must fire
        let n = count_flushes(&mut seg, 16 * 16_000 / BLOCK, tone);
        assert!(n >= 1, "expected at least one flush from the 15 s cap");
    }

    #[test]
    fn two_utterances_produce_two_flushes() {
        let mut seg = VadSegmenter::new();
        // First utterance: 3 s speech then 1 s silence
        count_flushes(&mut seg, 3 * 16_000 / BLOCK, tone);
        let n1 = count_flushes(&mut seg, 16_000 / BLOCK, silence);
        // Second utterance: 3 s speech then 1 s silence
        count_flushes(&mut seg, 3 * 16_000 / BLOCK, tone);
        let n2 = count_flushes(&mut seg, 16_000 / BLOCK, silence);
        assert_eq!(
            n1 + n2,
            2,
            "expected exactly two flushes for two utterances"
        );
    }
}
