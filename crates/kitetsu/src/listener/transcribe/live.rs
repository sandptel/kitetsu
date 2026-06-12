//! Rolling-window live transcription on top of a load-once [`Transcriber`].
//!
//! Moonshine v2's stateful chunk decoding is private in transcribe-rs, so live
//! transcription here is a rolling window: audio is fed in as it is captured,
//! and the whole current window is re-transcribed on a schedule to produce an
//! updated partial. Inference is fast enough (tens of ms on a few seconds of
//! audio) that re-running on each new chunk stays well ahead of real time.
//!
//! Deliberately not here: silence/VAD-based commit of finished utterances, and
//! audio capture itself (the caller feeds samples). The window is a sliding cap,
//! so text older than `window` scrolls out — widen `window` to keep more.

use std::time::{Duration, Instant};

use crate::listener::ListenerError;

use super::Transcriber;

/// 16 kHz mono — the rate every capture path in `listener` produces.
const SAMPLE_RATE: usize = 16_000;

/// Minimum buffered audio before inference is attempted (0.1 s). The streaming
/// model rejects inputs shorter than this.
const FLOOR_SAMPLES: usize = SAMPLE_RATE / 10;

// ── Tunable parameters ──────────────────────────────────────────────────────────

/// Tunables for [`LiveTranscriber`]. Use [`LiveConfig::default`] for sensible
/// values, then override individual fields.
#[derive(Debug, Clone)]
pub struct LiveConfig {
    /// Maximum audio length re-transcribed each pass. Caps inference latency;
    /// audio older than this slides out of the window. Larger = more context
    /// retained, slower each pass.
    pub window: Duration,
    /// Minimum wall-clock time between inference passes (debounce). Smaller =
    /// snappier partials, more CPU.
    pub interval: Duration,
    /// Minimum newly-fed audio required before another pass runs. Avoids
    /// re-transcribing when almost nothing new has arrived.
    pub min_new: Duration,
}

impl Default for LiveConfig {
    fn default() -> Self {
        Self {
            window: Duration::from_secs(10),
            interval: Duration::from_millis(500),
            min_new: Duration::from_millis(300),
        }
    }
}

// ── Sliding window (model-independent, unit-tested) ─────────────────────────────

/// Holds the rolling sample buffer and the scheduling state. Separated from the
/// model so the feed/trim/debounce logic can be tested without loading weights.
struct Window {
    config: LiveConfig,
    /// Rolling sample buffer, capped to `window` worth of samples.
    buffer: Vec<f32>,
    /// Samples fed since the last inference pass.
    fed_since_run: usize,
    /// When the last inference pass ran; `None` until the first pass.
    last_run: Option<Instant>,
}

impl Window {
    fn new(config: LiveConfig) -> Self {
        Self {
            config,
            buffer: Vec::new(),
            fed_since_run: 0,
            last_run: None,
        }
    }

    /// Append samples and trim the front so the buffer never exceeds `window`.
    fn feed(&mut self, samples: &[f32]) {
        self.buffer.extend_from_slice(samples);
        self.fed_since_run += samples.len();

        let cap = self.window_samples();
        if self.buffer.len() > cap {
            let excess = self.buffer.len() - cap;
            self.buffer.drain(0..excess);
        }
    }

    /// `true` when enough new audio has arrived and the debounce interval has
    /// elapsed for another inference pass to be worthwhile.
    fn should_run(&self) -> bool {
        if self.buffer.len() < FLOOR_SAMPLES {
            return false;
        }
        let new_ok = self.fed_since_run >= self.min_new_samples();
        let time_ok = self
            .last_run
            .is_none_or(|t| t.elapsed() >= self.config.interval);
        new_ok && time_ok
    }

    /// Mark that an inference pass just ran (resets debounce counters).
    fn mark_ran(&mut self) {
        self.last_run = Some(Instant::now());
        self.fed_since_run = 0;
    }

    fn window_samples(&self) -> usize {
        (self.config.window.as_secs_f32() * SAMPLE_RATE as f32) as usize
    }

    fn min_new_samples(&self) -> usize {
        (self.config.min_new.as_secs_f32() * SAMPLE_RATE as f32) as usize
    }
}

// ── LiveTranscriber ──────────────────────────────────────────────────────────────

/// Drives a [`Transcriber`] over a sliding audio window for live partials.
///
/// Feed captured samples with [`feed`][Self::feed]; when
/// [`should_run`][Self::should_run] returns `true`, call
/// [`transcribe`][Self::transcribe] for an updated partial. `LiveTranscriber`
/// is `Send`; inference blocks, so drive it from a dedicated thread (e.g.
/// `tokio::task::spawn_blocking`).
pub struct LiveTranscriber {
    transcriber: Transcriber,
    window: Window,
}

impl LiveTranscriber {
    /// Wrap an already-loaded [`Transcriber`] with rolling-window scheduling.
    pub fn new(transcriber: Transcriber, config: LiveConfig) -> Self {
        Self {
            transcriber,
            window: Window::new(config),
        }
    }

    /// Append newly-captured samples (16 kHz mono f32), trimming the front of
    /// the buffer so it never exceeds the configured window.
    pub fn feed(&mut self, samples: &[f32]) {
        self.window.feed(samples);
    }

    /// `true` when another inference pass is worthwhile (enough new audio and
    /// the debounce interval elapsed).
    pub fn should_run(&self) -> bool {
        self.window.should_run()
    }

    /// Run inference on the current window and return the partial transcript.
    ///
    /// Resets the new-audio and interval counters. Blocks for the inference
    /// duration. Returns an empty string when there is too little audio.
    pub fn transcribe(&mut self) -> Result<String, ListenerError> {
        self.window.mark_ran();
        if self.window.buffer.len() < FLOOR_SAMPLES {
            return Ok(String::new());
        }
        self.transcriber.transcribe(&self.window.buffer, None)
    }

    /// Current buffered audio length in seconds.
    pub fn buffer_secs(&self) -> f32 {
        self.window.buffer.len() as f32 / SAMPLE_RATE as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(window: Duration, interval: Duration, min_new: Duration) -> LiveConfig {
        LiveConfig {
            window,
            interval,
            min_new,
        }
    }

    #[test]
    fn feed_trims_buffer_to_window_cap() {
        // 1 s window = 16_000 samples; feed 25_000 → oldest 9_000 dropped.
        let mut w = Window::new(cfg(
            Duration::from_secs(1),
            Duration::ZERO,
            Duration::from_millis(0),
        ));
        w.feed(&vec![0.0; 25_000]);
        assert_eq!(w.buffer.len(), 16_000);
    }

    #[test]
    fn should_run_false_below_floor() {
        let mut w = Window::new(cfg(Duration::from_secs(10), Duration::ZERO, Duration::ZERO));
        w.feed(&vec![0.0; FLOOR_SAMPLES - 1]);
        assert!(!w.should_run());
    }

    #[test]
    fn should_run_requires_min_new_audio() {
        // min_new = 300 ms = 4_800 samples. Feed enough to clear the floor but
        // less than min_new after a run → should not fire.
        let mut w = Window::new(cfg(
            Duration::from_secs(10),
            Duration::ZERO,
            Duration::from_millis(300),
        ));
        w.feed(&vec![0.0; 5_000]);
        assert!(w.should_run()); // first pass: 5_000 >= 4_800
        w.mark_ran();
        w.feed(&vec![0.0; 1_000]); // only 1_000 new < 4_800
        assert!(!w.should_run());
        w.feed(&vec![0.0; 4_000]); // now 5_000 new >= 4_800
        assert!(w.should_run());
    }

    #[test]
    fn interval_debounces_passes() {
        let mut w = Window::new(cfg(
            Duration::from_secs(10),
            Duration::from_secs(3_600), // effectively never re-fires
            Duration::ZERO,
        ));
        w.feed(&vec![0.0; 5_000]);
        assert!(w.should_run());
        w.mark_ran();
        w.feed(&vec![0.0; 5_000]);
        assert!(!w.should_run()); // interval not elapsed
    }
}
