//! Per-source rolling capture buffers for the current teleprompter window.
//!
//! [`WindowManager`] holds one [`SourceBuffer`] per audio source (mic + system),
//! each accumulating raw 16 kHz mono samples (for the on-trigger pipes) and live
//! transcript text (from the Realtime WS). A trigger calls
//! [`WindowManager::snapshot_and_reset`], which hands the accumulated state to the
//! caller as a [`Window`] and clears the buffers for the next window. A
//! `max_window_secs` cap drops the oldest raw samples so growth is bounded if the
//! operator never presses the button.
//!
//! Not here: capture itself, the WS connection, or the pipes — this is a pure,
//! sync, I/O-free data structure (the daemon owns it behind a lock).

/// Sample rate of the streamed capture chunks (`Recorder::start_streaming`).
pub const SAMPLE_RATE: u32 = 16_000;

/// Which audio source a buffer holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// The microphone — the operator's own voice ("Me").
    Mic,
    /// System audio — the other party ("Them").
    Sys,
}

/// One source's accumulating state for the current window.
#[derive(Debug, Default, Clone, PartialEq)]
struct SourceBuffer {
    /// Raw 16 kHz mono samples since the last snapshot.
    raw: Vec<f32>,
    /// Space-joined completed utterances from the live WS since the last snapshot.
    live_text: String,
}

/// Owns the per-source buffers and the window counter.
#[derive(Debug)]
pub struct WindowManager {
    mic: SourceBuffer,
    sys: SourceBuffer,
    /// Raw-sample cap per source; `0` means unbounded.
    max_samples: usize,
    window_n: u64,
}

/// A frozen window handed to the pipes: raw samples + live text per source.
#[derive(Debug, Clone, PartialEq)]
pub struct Window {
    /// 1-based window number (increments on every snapshot).
    pub n: u64,
    pub mic_raw: Vec<f32>,
    pub sys_raw: Vec<f32>,
    pub mic_text: String,
    pub sys_text: String,
}

impl Window {
    /// Mic audio length in seconds.
    pub fn mic_secs(&self) -> f32 {
        secs(self.mic_raw.len())
    }

    /// System audio length in seconds.
    pub fn sys_secs(&self) -> f32 {
        secs(self.sys_raw.len())
    }
}

impl WindowManager {
    /// Create a manager whose per-source raw buffers are capped at
    /// `max_window_secs` of audio (`0` → unbounded).
    pub fn new(max_window_secs: u64) -> Self {
        Self {
            mic: SourceBuffer::default(),
            sys: SourceBuffer::default(),
            max_samples: max_window_secs as usize * SAMPLE_RATE as usize,
            window_n: 0,
        }
    }

    /// Append a captured chunk to `source`'s raw buffer, dropping the oldest
    /// samples if the cap is exceeded.
    pub fn append_raw(&mut self, source: Source, chunk: &[f32]) {
        let max = self.max_samples;
        let buf = self.buf_mut(source);
        buf.raw.extend_from_slice(chunk);
        if max > 0 && buf.raw.len() > max {
            let overflow = buf.raw.len() - max;
            buf.raw.drain(..overflow);
        }
    }

    /// Append a completed live utterance to `source`'s text (space-joined).
    /// Blank utterances are ignored.
    pub fn append_text(&mut self, source: Source, utterance: &str) {
        let t = utterance.trim();
        if t.is_empty() {
            return;
        }
        let buf = self.buf_mut(source);
        if !buf.live_text.is_empty() {
            buf.live_text.push(' ');
        }
        buf.live_text.push_str(t);
    }

    /// Freeze the current window and clear the buffers for the next one.
    pub fn snapshot_and_reset(&mut self) -> Window {
        self.window_n += 1;
        Window {
            n: self.window_n,
            mic_raw: std::mem::take(&mut self.mic.raw),
            sys_raw: std::mem::take(&mut self.sys.raw),
            mic_text: std::mem::take(&mut self.mic.live_text),
            sys_text: std::mem::take(&mut self.sys.live_text),
        }
    }

    fn buf_mut(&mut self, source: Source) -> &mut SourceBuffer {
        match source {
            Source::Mic => &mut self.mic,
            Source::Sys => &mut self.sys,
        }
    }
}

/// Sample count → seconds at the capture rate.
fn secs(samples: usize) -> f32 {
    samples as f32 / SAMPLE_RATE as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_clears_buffers_and_increments_window() {
        let mut wm = WindowManager::new(300);
        wm.append_raw(Source::Mic, &[0.1, 0.2, 0.3]);
        wm.append_text(Source::Mic, "hello");
        wm.append_text(Source::Sys, "world");

        let w1 = wm.snapshot_and_reset();
        assert_eq!(w1.n, 1);
        assert_eq!(w1.mic_raw, vec![0.1, 0.2, 0.3]);
        assert_eq!(w1.mic_text, "hello");
        assert_eq!(w1.sys_text, "world");

        // Buffers are empty after a snapshot; the counter keeps climbing.
        let w2 = wm.snapshot_and_reset();
        assert_eq!(w2.n, 2);
        assert!(w2.mic_raw.is_empty());
        assert!(w2.mic_text.is_empty());
    }

    #[test]
    fn cap_drops_oldest_raw_samples() {
        // 1 second cap → 16_000 samples.
        let mut wm = WindowManager::new(1);
        let chunk = vec![1.0f32; SAMPLE_RATE as usize];
        wm.append_raw(Source::Sys, &chunk);
        wm.append_raw(Source::Sys, &[2.0, 2.0, 2.0]);

        let w = wm.snapshot_and_reset();
        assert_eq!(w.sys_raw.len(), SAMPLE_RATE as usize);
        // The three newest samples survived; the oldest were dropped.
        assert_eq!(&w.sys_raw[w.sys_raw.len() - 3..], &[2.0, 2.0, 2.0]);
    }

    #[test]
    fn zero_cap_is_unbounded() {
        let mut wm = WindowManager::new(0);
        let chunk = vec![1.0f32; SAMPLE_RATE as usize * 2];
        wm.append_raw(Source::Mic, &chunk);
        let w = wm.snapshot_and_reset();
        assert_eq!(w.mic_raw.len(), SAMPLE_RATE as usize * 2);
    }

    #[test]
    fn text_accumulation_joins_utterances_and_skips_blanks() {
        let mut wm = WindowManager::new(300);
        wm.append_text(Source::Mic, "one");
        wm.append_text(Source::Mic, "   "); // blank → ignored
        wm.append_text(Source::Mic, " two ");
        let w = wm.snapshot_and_reset();
        assert_eq!(w.mic_text, "one two");
    }

    #[test]
    fn window_reports_seconds() {
        let mut wm = WindowManager::new(300);
        wm.append_raw(Source::Mic, &vec![0.0; SAMPLE_RATE as usize]); // 1.0s
        wm.append_raw(Source::Sys, &vec![0.0; SAMPLE_RATE as usize / 2]); // 0.5s
        let w = wm.snapshot_and_reset();
        assert!((w.mic_secs() - 1.0).abs() < 1e-6);
        assert!((w.sys_secs() - 0.5).abs() < 1e-6);
    }
}
