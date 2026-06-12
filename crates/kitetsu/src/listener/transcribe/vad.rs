//! Energy-based VAD (Voice Activity Detection) segmenter.
//!
//! Pure state machine: no model knowledge, no I/O, no async. Takes 16 kHz mono
//! f32 blocks and signals utterance boundaries via `FlushSignal`. Used by
//! `stream::StreamingTranscriber` for the `StreamMode::VadGated` path. All
//! constants are local to this module; `LocalAgreement` has no dependency here.

// ── Constants (all in samples at 16 kHz) ─────────────────────────────────────

/// RMS floor below which a block is considered silence (~-40 dBFS).
pub(crate) const SILENCE_RMS: f32 = 0.01;

/// Minimum consecutive speech samples before a segment is considered real
/// speech vs. transient noise (250 ms). The noise-gate threshold.
pub(crate) const MIN_SPEECH_SAMPLES: usize = 4_000;

/// Minimum consecutive silence samples after speech before flushing (100 ms).
pub(crate) const MIN_SILENCE_SAMPLES: usize = 1_600;

/// Pre-roll prepended when speech starts — catches the attack of the word (100 ms).
pub(crate) const PRE_ROLL_SAMPLES: usize = 1_600;

/// Absolute speech cap: flush regardless of silence at 15 s.
/// Guards against non-stop speech overwhelming accuracy.
pub(crate) const MAX_SPEECH_SAMPLES: usize = 240_000;

/// Minimum viable utterance to bother sending to inference (400 ms).
/// Shorter clips produce unreliable output from whisper.
pub(crate) const MIN_VIABLE_SAMPLES: usize = 6_400;

// ── Types ─────────────────────────────────────────────────────────────────────

enum VadState {
    Idle,
    /// Speech detected; accumulating until MIN_SPEECH_SAMPLES to reject noise.
    Debouncing {
        speech_run: usize,
    },
    /// Confirmed speech; tracking silence to determine the flush point.
    Speaking {
        speech_samples: usize,
        silence_run: usize,
    },
}

/// Outcome of feeding one block through the VAD state machine.
pub(crate) enum FlushSignal {
    /// No boundary yet; continue accumulating.
    NoFlush,
    /// Utterance boundary detected; contains the samples ready for inference.
    Flush(Vec<f32>),
}

/// Energy-based VAD state machine. Pure: no inference, no I/O, no async.
///
/// Feed 16 kHz mono f32 blocks via [`VadSegmenter::feed`]; the machine returns
/// [`FlushSignal::Flush`] when an utterance boundary is detected.
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
                    // Prepend pre-roll so the attack of the first word is captured.
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

pub(crate) fn rms(samples: &[f32]) -> f32 {
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
        assert_eq!(count_flushes(&mut seg, 10 * 16_000 / BLOCK, silence), 0);
    }

    #[test]
    fn short_noise_below_min_speech_does_not_flush() {
        let mut seg = VadSegmenter::new();
        // 2 blocks = 2048 samples < MIN_SPEECH_SAMPLES (4000) → stays Debouncing
        for _ in 0..2 {
            seg.feed(&tone(BLOCK));
        }
        // Silence forces Debouncing → Idle; no flush
        assert_eq!(count_flushes(&mut seg, 20, silence), 0);
    }

    #[test]
    fn speech_then_silence_flushes_one_utterance() {
        let mut seg = VadSegmenter::new();
        count_flushes(&mut seg, 3 * 16_000 / BLOCK, tone);
        // MIN_SILENCE_SAMPLES = 1600; block = 1024, so block 2 crosses threshold.
        // A 1 s silence window (15 blocks) contains exactly 1 flush.
        let n = count_flushes(&mut seg, 16_000 / BLOCK, silence);
        assert_eq!(n, 1);
    }

    #[test]
    fn max_speech_cap_forces_flush() {
        let mut seg = VadSegmenter::new();
        let n = count_flushes(&mut seg, 16 * 16_000 / BLOCK, tone);
        assert!(n >= 1, "expected at least one flush from the 15 s cap");
    }

    #[test]
    fn two_utterances_produce_two_flushes() {
        let mut seg = VadSegmenter::new();
        count_flushes(&mut seg, 3 * 16_000 / BLOCK, tone);
        let n1 = count_flushes(&mut seg, 16_000 / BLOCK, silence);
        count_flushes(&mut seg, 3 * 16_000 / BLOCK, tone);
        let n2 = count_flushes(&mut seg, 16_000 / BLOCK, silence);
        assert_eq!(
            n1 + n2,
            2,
            "expected exactly two flushes for two utterances"
        );
    }
}
