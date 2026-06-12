//! Chunked streaming capture: push small audio blocks into a bounded channel.
//!
//! Companion to `record.rs`, which accumulates samples into a `Vec<f32>`. This
//! file pushes each read block as `Vec<f32>` into a `SyncSender` so a worker
//! thread can process chunks incrementally. Never touches tokio — always runs
//! on a `std::thread`.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::SyncSender;

use libpulse_binding::{sample, stream};
use libpulse_simple_binding::Simple;

use crate::listener::ListenerError;

/// ~64 ms of audio at 16 kHz — fine enough granularity for the VAD state machine
/// without creating excessive channel traffic.
const STREAM_CHUNK_SAMPLES: usize = 1024;

/// Capture audio from `device` in ~64 ms blocks and push each block into `tx`.
///
/// Runs until `stop` is set OR `tx.send` fails (worker dropped its receiver).
/// Uses the same 16 kHz mono S16le spec as `record.rs`. Backpressure is handled
/// by the bounded `SyncSender`; if the worker falls behind, the capture thread
/// blocks, not the PulseAudio daemon.
pub(crate) fn capture_stream(
    device: &str,
    label: &'static str,
    stop: Arc<AtomicBool>,
    tx: SyncSender<Vec<f32>>,
) -> Result<(), ListenerError> {
    let spec = sample::Spec {
        format: sample::Format::S16le,
        channels: 1,
        rate: 16_000,
    };

    let s = Simple::new(
        None,
        "kitetsu",
        stream::Direction::Record,
        Some(device),
        label,
        &spec,
        None,
        None,
    )
    .map_err(|_| ListenerError::ConnectFailed("capture_stream: could not open record stream"))?;

    let bytes_per_block = STREAM_CHUNK_SAMPLES * 2; // S16le = 2 bytes per sample
    let mut raw = vec![0u8; bytes_per_block];

    while !stop.load(Ordering::Relaxed) {
        s.read(&mut raw)
            .map_err(|_| ListenerError::CaptureFailed("capture_stream: PulseAudio read failed"))?;

        let block: Vec<f32> = raw
            .chunks_exact(2)
            .map(|b| {
                let s = i16::from_le_bytes([b[0], b[1]]);
                f32::from(s) / f32::from(i16::MAX)
            })
            .collect();

        if tx.send(block).is_err() {
            // Worker dropped the receiver — stop cleanly without an error.
            break;
        }
    }

    Ok(())
}
