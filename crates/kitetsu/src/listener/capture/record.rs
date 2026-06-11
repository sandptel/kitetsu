//! Blocking audio capture via libpulse-simple-binding and WAV serialisation.
//!
//! Provides two capture entry points: `capture_to_wav` writes a file at 48 kHz
//! for offline inspection; `capture_samples` returns 16 kHz mono f32 samples
//! ready for STT inference. Both are designed to run on a dedicated
//! `std::thread` — never on the tokio runtime.
//!
//! Does no device introspection; the caller supplies a resolved device name
//! (see `listener::capture::discover`).

use std::io::Write as _;
use std::path::Path;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::time::{Duration, Instant};

use libpulse_binding::sample::{Format, Spec};
use libpulse_binding::stream::Direction;
use libpulse_simple_binding::Simple;
use tracing::info;

use crate::listener::ListenerError;

// ── Constants ─────────────────────────────────────────────────────────────────

const SAMPLE_RATE: u32 = 48_000;
const BITS_PER_SAMPLE: u16 = 16;

/// Default one-shot capture window for STT inference.
/// Rolling-window capture (live transcription) is a future iteration.
pub const DEFAULT_CAPTURE_SECS: u64 = 5;

// ── Capture configuration ─────────────────────────────────────────────────────

/// Parameters for one 48 kHz file-dump capture stream. Construct with
/// `CaptureConfig::mic` or `CaptureConfig::system`; the channel count is
/// baked in per stream type.
#[derive(Debug, Clone)]
pub struct CaptureConfig {
    /// Resolved PulseAudio device name (from `discover_default_devices`).
    pub device: String,
    /// Human-readable label shown in PulseAudio clients and in log output.
    pub label: &'static str,
    /// Number of channels: 1 for mic (mono), 2 for system monitor (stereo).
    pub channels: u8,
}

impl CaptureConfig {
    /// Microphone capture: mono, default mic source.
    pub fn mic(device: String) -> Self {
        Self {
            device,
            label: "mic",
            channels: 1,
        }
    }

    /// System-audio capture: stereo, default sink monitor.
    pub fn system(device: String) -> Self {
        Self {
            device,
            label: "system",
            channels: 2,
        }
    }
}

// ── 48 kHz file-dump capture ──────────────────────────────────────────────────

/// Open a PulseAudio record stream on `cfg.device`, capture PCM for
/// `duration`, write a 16-bit WAV to `out_path`, then return.
///
/// Logs the device name and sample spec at `info` level before the first
/// read so the operator can confirm which hardware is being captured.
/// Blocks the calling thread for `duration`.
pub fn capture_to_wav(
    cfg: &CaptureConfig,
    out_path: &Path,
    duration: Duration,
) -> Result<(), ListenerError> {
    let spec = Spec {
        format: Format::S16le,
        rate: SAMPLE_RATE,
        channels: cfg.channels,
    };

    info!(
        label = cfg.label,
        device = %cfg.device,
        rate = SAMPLE_RATE,
        channels = cfg.channels,
        "opening capture stream",
    );

    let simple = Simple::new(
        None, // default PulseAudio server
        "kitetsu",
        Direction::Record,
        Some(cfg.device.as_str()),
        cfg.label, // stream description visible in pavucontrol / pactl
        &spec,
        None, // default channel map
        None, // default buffer attributes
    )
    .map_err(|_| ListenerError::ConnectFailed("could not open PulseAudio record stream"))?;

    info!(label = cfg.label, "stream open — capturing");

    // Each read pulls exactly one chunk of frames. At 48 kHz the chunk below
    // is ~85 ms, giving reasonable granularity without excessive syscall overhead.
    let bytes_per_frame = cfg.channels as usize * (BITS_PER_SAMPLE as usize / 8);
    let chunk_frames = 4096_usize;
    let chunk_bytes = chunk_frames * bytes_per_frame;

    // Pre-allocate the exact number of bytes we expect, then fill via reads.
    let total_frames = SAMPLE_RATE as usize * duration.as_secs() as usize;
    let total_bytes = total_frames * bytes_per_frame;

    let mut pcm: Vec<u8> = Vec::with_capacity(total_bytes);
    let mut buf = vec![0u8; chunk_bytes];
    let deadline = Instant::now() + duration;

    while Instant::now() < deadline && pcm.len() < total_bytes {
        simple
            .read(&mut buf)
            .map_err(|_| ListenerError::CaptureFailed("PulseAudio read returned an error"))?;
        let remaining = total_bytes - pcm.len();
        let to_copy = buf.len().min(remaining);
        pcm.extend_from_slice(&buf[..to_copy]);
    }

    write_wav(out_path, &pcm, cfg.channels, SAMPLE_RATE)?;

    info!(
        label = cfg.label,
        path = %out_path.display(),
        bytes = pcm.len(),
        "capture complete — WAV written",
    );

    Ok(())
}

// ── 16 kHz mono STT capture ───────────────────────────────────────────────────

/// Open a PulseAudio record stream at 16 kHz mono on `device`, capture PCM
/// for `duration`, and return the samples as normalised f32 (range −1.0..1.0).
///
/// PipeWire / PulseAudio resamples and downmixes from the hardware rate; no
/// resampler dependency is needed. The resulting slice is ready to pass
/// directly to a `SpeechModel::transcribe` implementation.
/// Blocks the calling thread for `duration`.
pub fn capture_samples(
    device: &str,
    label: &'static str,
    duration: Duration,
) -> Result<Vec<f32>, ListenerError> {
    const RATE_16K: u32 = 16_000;

    let spec = Spec {
        format: Format::S16le,
        rate: RATE_16K,
        channels: 1,
    };

    info!(
        label,
        device,
        rate = RATE_16K,
        channels = 1,
        "opening 16 kHz mono capture stream for transcription",
    );

    let simple = Simple::new(
        None,
        "kitetsu",
        Direction::Record,
        Some(device),
        label,
        &spec,
        None,
        None,
    )
    .map_err(|_| ListenerError::ConnectFailed("could not open 16 kHz PulseAudio record stream"))?;

    info!(label, "stream open — capturing at 16 kHz mono");

    // S16LE mono: 2 bytes per sample
    let bytes_per_sample = 2_usize;
    let chunk_samples = 4096_usize;
    let chunk_bytes = chunk_samples * bytes_per_sample;
    let total_samples = RATE_16K as usize * duration.as_secs() as usize;
    let total_bytes = total_samples * bytes_per_sample;

    let mut pcm_bytes: Vec<u8> = Vec::with_capacity(total_bytes);
    let mut buf = vec![0u8; chunk_bytes];
    let deadline = Instant::now() + duration;

    while Instant::now() < deadline && pcm_bytes.len() < total_bytes {
        simple
            .read(&mut buf)
            .map_err(|_| ListenerError::CaptureFailed("PulseAudio read returned an error"))?;
        let remaining = total_bytes - pcm_bytes.len();
        let to_copy = buf.len().min(remaining);
        pcm_bytes.extend_from_slice(&buf[..to_copy]);
    }

    // Convert S16LE bytes to normalised f32; whisper expects −1.0..1.0.
    let samples: Vec<f32> = pcm_bytes
        .chunks_exact(2)
        .map(|b| f32::from(i16::from_le_bytes([b[0], b[1]])) / 32_768.0)
        .collect();

    info!(label, sample_count = samples.len(), "STT capture complete");

    Ok(samples)
}

// ── Stop-on-signal recorder ───────────────────────────────────────────────────

/// Captures 16 kHz mono audio on a dedicated thread until the associated
/// [`RecordingHandle`] is stopped. Unlike [`capture_samples`], recording
/// duration is open-ended — the caller decides when to stop.
pub struct Recorder {
    device: String,
    label: &'static str,
}

impl Recorder {
    /// Create a recorder for the given PulseAudio device name.
    pub fn new(device: String, label: &'static str) -> Self {
        Self { device, label }
    }

    /// Spawn the capture thread and return a handle for stopping it.
    pub fn start(self) -> RecordingHandle {
        let stop = Arc::new(AtomicBool::new(false));
        let sample_count = Arc::new(AtomicUsize::new(0));
        let stop_flag = Arc::clone(&stop);
        let count_flag = Arc::clone(&sample_count);

        let thread = std::thread::spawn(move || {
            record_until_stop(&self.device, self.label, &stop_flag, &count_flag)
        });

        RecordingHandle {
            stop,
            sample_count,
            thread,
        }
    }
}

/// Live handle returned by [`Recorder::start`].
///
/// Call [`stop`][RecordingHandle::stop] to signal the capture thread and collect
/// the samples. The PulseAudio read chunk is ~256 ms at 16 kHz, so the thread
/// finishes within one chunk after the stop flag is set.
pub struct RecordingHandle {
    stop: Arc<AtomicBool>,
    sample_count: Arc<AtomicUsize>,
    thread: std::thread::JoinHandle<Result<Vec<f32>, ListenerError>>,
}

impl RecordingHandle {
    /// Signal the capture thread to stop and block until it returns the samples.
    pub fn stop(self) -> Result<Vec<f32>, ListenerError> {
        self.stop.store(true, Ordering::Release);
        self.thread.join().expect("recorder thread panicked")
    }

    /// Approximate number of samples captured so far (lock-free, relaxed read).
    pub fn sample_count(&self) -> usize {
        self.sample_count.load(Ordering::Relaxed)
    }
}

fn record_until_stop(
    device: &str,
    label: &'static str,
    stop: &AtomicBool,
    sample_count: &AtomicUsize,
) -> Result<Vec<f32>, ListenerError> {
    const RATE_16K: u32 = 16_000;

    let spec = Spec {
        format: Format::S16le,
        rate: RATE_16K,
        channels: 1,
    };

    info!(label, device, "opening stop-on-signal capture stream");

    let simple = Simple::new(
        None,
        "kitetsu",
        Direction::Record,
        Some(device),
        label,
        &spec,
        None,
        None,
    )
    .map_err(|_| ListenerError::ConnectFailed("could not open 16 kHz PulseAudio record stream"))?;

    info!(label, "recording started — waiting for stop signal");

    // S16LE mono: 2 bytes per sample. Each read blocks ~256 ms at 16 kHz.
    const CHUNK_SAMPLES: usize = 4096;
    let mut buf = vec![0u8; CHUNK_SAMPLES * 2];
    let mut samples: Vec<f32> = Vec::new();

    while !stop.load(Ordering::Acquire) {
        simple
            .read(&mut buf)
            .map_err(|_| ListenerError::CaptureFailed("PulseAudio read returned an error"))?;
        samples.extend(
            buf.chunks_exact(2)
                .map(|b| f32::from(i16::from_le_bytes([b[0], b[1]])) / 32_768.0),
        );
        sample_count.store(samples.len(), Ordering::Relaxed);
    }

    info!(label, sample_count = samples.len(), "recording stopped");
    Ok(samples)
}

// ── WAV writer ────────────────────────────────────────────────────────────────

/// Write a minimal PCM WAV file (44-byte header + raw S16LE samples).
///
/// The WAV format is a 44-byte little-endian header followed by raw PCM.
/// No external crate is needed — the header fields are stable and well-documented.
fn write_wav(path: &Path, pcm: &[u8], channels: u8, rate: u32) -> Result<(), ListenerError> {
    use std::fs::File;

    let data_len = pcm.len() as u32;
    let byte_rate = rate * channels as u32 * (BITS_PER_SAMPLE as u32 / 8);
    let block_align = channels as u16 * (BITS_PER_SAMPLE / 8);

    let mut f = File::create(path)?;

    // RIFF chunk descriptor (12 bytes)
    f.write_all(b"RIFF")?;
    f.write_all(&(36 + data_len).to_le_bytes())?; // file size − 8
    f.write_all(b"WAVE")?;

    // fmt sub-chunk (24 bytes)
    f.write_all(b"fmt ")?;
    f.write_all(&16u32.to_le_bytes())?; // sub-chunk size (PCM = 16)
    f.write_all(&1u16.to_le_bytes())?; // audio format: 1 = PCM
    f.write_all(&(channels as u16).to_le_bytes())?;
    f.write_all(&rate.to_le_bytes())?;
    f.write_all(&byte_rate.to_le_bytes())?;
    f.write_all(&block_align.to_le_bytes())?;
    f.write_all(&BITS_PER_SAMPLE.to_le_bytes())?;

    // data sub-chunk (8-byte header + PCM body)
    f.write_all(b"data")?;
    f.write_all(&data_len.to_le_bytes())?;
    f.write_all(pcm)?;

    Ok(())
}
