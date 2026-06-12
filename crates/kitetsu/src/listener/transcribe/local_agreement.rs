//! LocalAgreement-2 sliding-window segmenter.
//!
//! Implements the two-pass LocalAgreement-2 policy from Whisper-Streaming: run
//! inference on a rolling audio window every ~1 s; commit the longest word-prefix
//! that two consecutive passes agree on; display the disagreeing tail as tentative.
//! Energy VAD acts as an execution gate (no inference during silence) and forces a
//! final commit on trailing silence so utterances close promptly.
//!
//! Pure: no I/O, no async, no whisper dependency. The only external surface is
//! `TimedWord` (from `engine`) for the caller to supply inference results.

use super::engine::TimedWord;
use super::vad::{SILENCE_RMS, rms};

// ── Constants (all in samples at 16 kHz) ─────────────────────────────────────

/// New samples between consecutive passes (~3 s step). Lower = lower latency,
/// higher CPU. VAD gate means this only runs during speech.
pub(crate) const STEP_SAMPLES: usize = 48_000;

/// Minimum window size before a pass is worthwhile (~3 s). Guards against
/// running inference on tiny slivers at the start of an utterance.
pub(crate) const MIN_INFERENCE_SAMPLES: usize = 32_000;

/// Trailing silence before forcing a final commit (~0.5 s). Closes the
/// utterance promptly rather than waiting for the next step or the 15 s cap.
pub(crate) const TRAILING_SILENCE_SAMPLES: usize = 16_000;

/// Hard cap on the rolling window (15 s). Forces a final commit for
/// non-stop speech, same guard as `VadGated`.
pub(crate) const MAX_WINDOW_SAMPLES: usize = 240_000;

// ── Types ─────────────────────────────────────────────────────────────────────

/// Outcome of feeding one audio block through the LA segmenter.
/// Does not run inference — only decides *whether* to run a pass.
#[derive(Debug, PartialEq)]
pub(crate) enum LaAction {
    /// Nothing to do; keep buffering.
    Wait,
    /// Time to run inference on `LocalAgreementSegmenter::buf`.
    /// `is_final = true` means this closes the current utterance — commit
    /// everything and reset the window regardless of agreement.
    RunPass { is_final: bool },
}

/// Result of applying an inference pass to the LA segmenter.
#[derive(Debug)]
pub(crate) struct PassOutcome {
    /// Newly committed text (one or more agreed words). `None` if no words
    /// agreed yet (window still accumulating or no overlap between passes).
    pub committed: Option<String>,
    /// In-progress words that haven't been confirmed yet. Empty on a final pass.
    pub tentative: String,
}

/// LocalAgreement-2 sliding-window state machine.
///
/// Call `feed` for each ~64 ms audio block; when it returns `RunPass`, run
/// `transcriber.transcribe_words(self.buf(), prompt)` and pass the result to
/// `apply_pass`. `feed` and `apply_pass` are always called on the same thread.
pub(crate) struct LocalAgreementSegmenter {
    /// Rolling audio window. `buf[0]` is the first non-committed sample.
    pub(crate) buf: Vec<f32>,
    /// Samples appended to `buf` since the last pass (resets after each pass).
    new_since_pass: usize,
    /// The tail of words from the previous pass (not yet agreed upon).
    prev_words: Vec<TimedWord>,
    /// Consecutive silence samples at the end of the current window.
    silence_run: usize,
    /// Whether any speech has been seen since the last utterance reset.
    saw_speech: bool,
}

impl LocalAgreementSegmenter {
    pub(crate) fn new() -> Self {
        Self {
            buf: Vec::new(),
            new_since_pass: 0,
            prev_words: Vec::new(),
            silence_run: 0,
            saw_speech: false,
        }
    }

    /// Feed one ~64 ms block; return whether to run an inference pass and whether
    /// it should be treated as a final (utterance-closing) pass.
    pub(crate) fn feed(&mut self, block: &[f32]) -> LaAction {
        let is_speech = rms(block) >= SILENCE_RMS;

        if !self.saw_speech {
            if !is_speech {
                // Pre-speech silence: drop; don't grow buf.
                return LaAction::Wait;
            }
            // First speech block: start the window, then fall through to the
            // cap/step checks below (handles the edge case where a single huge
            // block already exceeds MAX_WINDOW_SAMPLES in tests).
            self.saw_speech = true;
            self.buf.extend_from_slice(block);
            self.new_since_pass += block.len();
            self.silence_run = 0;
        } else {
            // Already inside an utterance.
            self.buf.extend_from_slice(block);
            self.new_since_pass += block.len();
            if is_speech {
                self.silence_run = 0;
            } else {
                self.silence_run += block.len();
            }
        }

        // Hard cap: force a final commit so the window never exceeds 15 s.
        if self.buf.len() >= MAX_WINDOW_SAMPLES {
            return LaAction::RunPass { is_final: true };
        }

        // Trailing silence: utterance ended — force a final commit.
        if self.silence_run >= TRAILING_SILENCE_SAMPLES {
            return LaAction::RunPass { is_final: true };
        }

        // Regular step: enough new samples and a large enough window.
        if self.new_since_pass >= STEP_SAMPLES && self.buf.len() >= MIN_INFERENCE_SAMPLES {
            return LaAction::RunPass { is_final: false };
        }

        LaAction::Wait
    }

    /// Apply the result of an inference pass.
    ///
    /// On a normal pass: find the longest agreeing word-prefix between the
    /// previous tail and `new_words`, commit those words, and trim `buf` at
    /// the confirmed boundary. Tentative = the disagreeing remainder.
    ///
    /// On a final pass (`is_final = true`, or forced by cap): commit
    /// everything in `new_words`, clear `buf`, and reset for the next utterance.
    pub(crate) fn apply_pass(&mut self, new_words: Vec<TimedWord>, is_final: bool) -> PassOutcome {
        self.new_since_pass = 0;

        if is_final {
            // Close the utterance: commit all words unconditionally.
            self.buf.clear();
            self.prev_words.clear();
            self.saw_speech = false;
            self.silence_run = 0;

            let text = words_to_string(&new_words);
            return PassOutcome {
                committed: if text.is_empty() { None } else { Some(text) },
                tentative: String::new(),
            };
        }

        // Normal pass: find the longest agreeing prefix with the previous tail.
        let k = longest_agreeing_prefix(&self.prev_words, &new_words);

        let committed = if k > 0 {
            // Trim the audio buffer up to the end of the last agreed word.
            // end_cs is in centiseconds; 1 cs = 160 samples at 16 kHz.
            let trim_samples = (new_words[k - 1].end_cs * 160) as usize;
            if trim_samples < self.buf.len() {
                self.buf.drain(..trim_samples);
            } else {
                // Edge case: timestamp exceeds buffer (shouldn't happen in
                // practice, but be defensive).
                self.buf.clear();
            }
            let text = words_to_string(&new_words[..k]);
            Some(text)
        } else {
            None
        };

        // The unconfirmed tail becomes the reference for the next pass.
        self.prev_words = new_words[k..].to_vec();

        let tentative = words_to_string(&self.prev_words);

        PassOutcome {
            committed,
            tentative,
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Find the length of the longest common word-prefix between `prev` and `new`,
/// comparing normalised text (lowercase, ASCII punctuation stripped).
pub(crate) fn longest_agreeing_prefix(prev: &[TimedWord], new: &[TimedWord]) -> usize {
    prev.iter()
        .zip(new.iter())
        .take_while(|(p, n)| norm(&p.text) == norm(&n.text))
        .count()
}

/// Normalise a word for comparison: lowercase and strip leading/trailing ASCII
/// punctuation. Case and punctuation differences shouldn't break agreement.
fn norm(s: &str) -> String {
    s.to_lowercase()
        .trim_matches(|c: char| c.is_ascii_punctuation())
        .to_owned()
}

fn words_to_string(words: &[TimedWord]) -> String {
    words
        .iter()
        .map(|w| w.text.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn w(text: &str, end_cs: i64) -> TimedWord {
        TimedWord {
            text: text.to_owned(),
            end_cs,
        }
    }

    fn tone_block(n: usize) -> Vec<f32> {
        vec![0.5f32; n]
    }

    fn silence_block(n: usize) -> Vec<f32> {
        vec![0.0f32; n]
    }

    const BLOCK: usize = 1024;

    // ── feed() ────────────────────────────────────────────────────────────────

    #[test]
    fn silence_only_never_requests_a_pass() {
        let mut seg = LocalAgreementSegmenter::new();
        for _ in 0..(10 * 16_000 / BLOCK) {
            assert_eq!(seg.feed(&silence_block(BLOCK)), LaAction::Wait);
        }
        assert!(!seg.saw_speech);
        assert!(seg.buf.is_empty());
    }

    #[test]
    fn speech_accumulates_to_runpass_at_step() {
        let mut seg = LocalAgreementSegmenter::new();
        // Feed blocks until we have >= STEP_SAMPLES (16 000) and >= MIN_INFERENCE_SAMPLES
        let target_blocks = STEP_SAMPLES / BLOCK + 1;
        let mut passes = 0usize;
        for _ in 0..target_blocks {
            if matches!(
                seg.feed(&tone_block(BLOCK)),
                LaAction::RunPass { is_final: false }
            ) {
                passes += 1;
            }
        }
        assert_eq!(
            passes, 1,
            "expected exactly one RunPass after reaching step threshold"
        );
    }

    #[test]
    fn trailing_silence_yields_final_pass() {
        let mut seg = LocalAgreementSegmenter::new();
        // 2 s of speech to open the window
        for _ in 0..(2 * 16_000 / BLOCK) {
            seg.feed(&tone_block(BLOCK));
        }
        // Reset new_since_pass so only silence causes the trigger.
        seg.new_since_pass = 0;
        // Feed silence until TRAILING_SILENCE_SAMPLES (8 000) are reached.
        let silence_blocks = TRAILING_SILENCE_SAMPLES / BLOCK + 1;
        let mut got_final = false;
        for _ in 0..silence_blocks {
            if matches!(
                seg.feed(&silence_block(BLOCK)),
                LaAction::RunPass { is_final: true }
            ) {
                got_final = true;
                break;
            }
        }
        assert!(got_final, "expected a final RunPass after trailing silence");
    }

    #[test]
    fn cap_forces_final_pass() {
        let mut seg = LocalAgreementSegmenter::new();
        // Silence-free: simulate exactly MAX_WINDOW_SAMPLES in one go.
        // We build a large block to avoid loop overhead while still exercising feed.
        let big_block = tone_block(MAX_WINDOW_SAMPLES);
        let action = seg.feed(&big_block);
        assert_eq!(
            action,
            LaAction::RunPass { is_final: true },
            "expected final RunPass when window exceeds cap"
        );
    }

    // ── longest_agreeing_prefix ───────────────────────────────────────────────

    #[test]
    fn lcp_empty_inputs_return_zero() {
        assert_eq!(longest_agreeing_prefix(&[], &[]), 0);
        assert_eq!(longest_agreeing_prefix(&[w("hello", 10)], &[]), 0);
        assert_eq!(longest_agreeing_prefix(&[], &[w("hello", 10)]), 0);
    }

    #[test]
    fn lcp_no_overlap() {
        let prev = vec![w("hello", 10)];
        let new = vec![w("world", 20)];
        assert_eq!(longest_agreeing_prefix(&prev, &new), 0);
    }

    #[test]
    fn lcp_full_match() {
        let a = vec![w("the", 10), w("cat", 20), w("sat", 30)];
        let b = vec![w("the", 12), w("cat", 22), w("sat", 32)];
        assert_eq!(longest_agreeing_prefix(&a, &b), 3);
    }

    #[test]
    fn lcp_partial_match() {
        let a = vec![w("the", 10), w("cat", 20), w("sat", 30)];
        let b = vec![w("the", 12), w("cat", 22), w("slept", 35)];
        assert_eq!(longest_agreeing_prefix(&a, &b), 2);
    }

    #[test]
    fn lcp_case_and_punctuation_insensitive() {
        let a = vec![w("Hello,", 10), w("World.", 20)];
        let b = vec![w("hello", 12), w("world", 22)];
        assert_eq!(longest_agreeing_prefix(&a, &b), 2);
    }

    // ── apply_pass ────────────────────────────────────────────────────────────

    #[test]
    fn apply_pass_normal_commits_agreed_prefix_and_trims_buf() {
        let mut seg = LocalAgreementSegmenter::new();
        // Fill buf with 48 000 samples (3 s worth).
        seg.buf = vec![0.5f32; 48_000];
        seg.saw_speech = true;
        seg.prev_words = vec![w("the", 100), w("cat", 200)];

        // new_words agrees on first two, diverges on third.
        let new_words = vec![w("the", 100), w("cat", 200), w("sat", 300)];
        let outcome = seg.apply_pass(new_words, false);

        assert_eq!(outcome.committed.as_deref(), Some("the cat"));
        assert_eq!(outcome.tentative, "sat");
        // buf should be trimmed to start after end_cs=200 → 200*160 = 32 000 samples
        assert_eq!(
            seg.buf.len(),
            48_000 - 32_000,
            "buf should be trimmed by agreed words"
        );
        // prev_words carries the unconfirmed tail
        assert_eq!(seg.prev_words.len(), 1);
        assert_eq!(seg.prev_words[0].text, "sat");
    }

    #[test]
    fn apply_pass_no_agreement_returns_none_committed() {
        let mut seg = LocalAgreementSegmenter::new();
        seg.buf = vec![0.5f32; 16_000];
        seg.saw_speech = true;
        seg.prev_words = vec![w("hello", 100)];

        let new_words = vec![w("world", 100)];
        let outcome = seg.apply_pass(new_words.clone(), false);

        assert!(outcome.committed.is_none());
        // Tentative should be the new tail.
        assert_eq!(outcome.tentative, "world");
    }

    #[test]
    fn apply_pass_final_commits_all_and_resets() {
        let mut seg = LocalAgreementSegmenter::new();
        seg.buf = vec![0.5f32; 32_000];
        seg.saw_speech = true;
        seg.prev_words = vec![w("some", 50)];
        seg.silence_run = 8_000;

        let new_words = vec![w("the", 100), w("cat", 200)];
        let outcome = seg.apply_pass(new_words, true);

        assert_eq!(outcome.committed.as_deref(), Some("the cat"));
        assert_eq!(outcome.tentative, "");
        assert!(seg.buf.is_empty(), "buf must be cleared on final pass");
        assert!(seg.prev_words.is_empty());
        assert!(!seg.saw_speech, "utterance reset after final pass");
    }

    #[test]
    fn apply_pass_final_with_empty_words_commits_none() {
        let mut seg = LocalAgreementSegmenter::new();
        seg.buf = vec![0.5f32; 8_000];
        seg.saw_speech = true;
        let outcome = seg.apply_pass(vec![], true);
        assert!(outcome.committed.is_none());
        assert!(seg.buf.is_empty());
    }
}
