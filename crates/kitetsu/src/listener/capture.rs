//! Blocking audio capture via libpulse-simple-binding and WAV serialisation.
//!
//! `capture_to_wav` opens one PulseAudio record stream, collects PCM for the
//! requested duration, writes a 16-bit WAV file, and returns. Designed to run
//! on a dedicated `std::thread` — never on the tokio runtime.
//!
//! Does no device introspection; the caller supplies a resolved device name
//! (see `listener::discover`).

use std::io::Write as _;
use std::path::Path;
use std::time::{Duration, Instant};

use libpulse_binding::sample::{Format, Spec};
use libpulse_binding::stream::Direction;
use libpulse_simple_binding::Simple;
use tracing::info;

// ── Error type ────────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum ListenerError {
    #[error("PulseAudio introspection failed: {0}")]
    IntrospectionFailed(&'static str),

    #[error("PulseAudio record stream could not be opened: {0}")]
    ConnectFailed(&'static str),

    #[error("PulseAudio read error: {0}")]
    CaptureFailed(&'static str),

    #[error("I/O error writing audio file: {0}")]
    Io(#[from] std::io::Error),
}

// ── Capture configuration ─────────────────────────────────────────────────────

const SAMPLE_RATE: u32 = 48_000;
const BITS_PER_SAMPLE: u16 = 16;

/// Parameters for one capture stream. Construct with `CaptureConfig::mic` or
/// `CaptureConfig::system`; the channel count is baked in per stream type.
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

// ── Public capture entry point ────────────────────────────────────────────────

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
