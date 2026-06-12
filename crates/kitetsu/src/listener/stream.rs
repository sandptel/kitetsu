//! Live streaming transcription session: orchestrates capture + VAD + inference.
//!
//! `StreamSession` spawns two `std::thread`s per source — a capture thread and
//! a worker thread — and fans all `TranscriptUpdate`s into one `Receiver`.
//! Never touches tokio; designed to run entirely on `std::thread`s so that
//! libpulse objects and blocking whisper inference never cross `.await`.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::thread::{self, JoinHandle};

use tracing::{error, info};

use crate::listener::transcriber::stream::{FeedResult, StreamingTranscriber};
use crate::listener::transcriber::{AudioModel, Transcriber};
use crate::listener::{ListenerError, capture};

pub use crate::listener::transcriber::stream::StreamMode;

// ── Public types ──────────────────────────────────────────────────────────────

/// Which audio source a streaming session should capture from.
#[derive(Debug, Clone, Copy)]
pub enum StreamSource {
    /// Microphone (default PulseAudio source).
    Mic,
    /// System audio monitor (default PulseAudio sink monitor).
    System,
}

/// One incremental update from a live streaming session.
#[derive(Debug)]
pub struct TranscriptUpdate {
    /// Which source produced this update.
    pub source: StreamSource,
    /// Newly committed utterance text. Never empty when `committed` is present.
    pub committed: String,
    /// In-progress tentative tail. Always `""` for `VadGated`; populated once
    /// `LocalAgreement` is implemented (see PLAN Decision #21).
    pub tentative: String,
}

/// A running live transcription session.
///
/// Created via [`StreamSession::start`]; stopped via [`StreamSession::stop`].
/// Dropping without calling `stop` will leak threads until the process exits.
pub struct StreamSession {
    stop: Arc<AtomicBool>,
    threads: Vec<JoinHandle<()>>,
}

impl StreamSession {
    /// Start live transcription from `sources`.
    ///
    /// Loads one `Transcriber` per source (each holds its own whisper state),
    /// caps threads at `min(cores/2, 4)` to avoid CPU over-subscription, and
    /// returns a `Receiver` that fans all `TranscriptUpdate`s together.
    ///
    /// `LocalAgreement` mode returns `Err(ListenerError::NotImplemented)` until
    /// it is wired in a future iteration (reserved: see PLAN Decision #21).
    pub fn start(
        sources: &[StreamSource],
        model: AudioModel,
        mode: StreamMode,
    ) -> Result<(StreamSession, Receiver<TranscriptUpdate>), ListenerError> {
        if matches!(mode, StreamMode::LocalAgreement) {
            return Err(ListenerError::NotImplemented(
                "LocalAgreement — reserved: see PLAN Decision #21",
            ));
        }

        // Cap threads so two concurrent whisper instances don't over-subscribe.
        let total_cores = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4);
        let n_threads = if sources.len() > 1 {
            (total_cores / 2).clamp(1, 4)
        } else {
            total_cores.min(8)
        };

        let devices = capture::discover_default_devices()?;
        let stop = Arc::new(AtomicBool::new(false));
        let (update_tx, update_rx) = mpsc::channel::<TranscriptUpdate>();
        let mut threads: Vec<JoinHandle<()>> = Vec::new();

        for source in sources {
            let source = *source;
            let device = match source {
                StreamSource::Mic => devices.mic_source.clone(),
                StreamSource::System => devices.system_monitor.clone(),
            };
            let label: &'static str = match source {
                StreamSource::Mic => "kitetsu-stream-mic",
                StreamSource::System => "kitetsu-stream-system",
            };

            let transcriber = Transcriber::load_with_threads(model.clone(), n_threads)?;
            let mut streaming = StreamingTranscriber::new(transcriber, mode);

            let stop_flag = Arc::clone(&stop);
            let update_tx = update_tx.clone();

            // Bounded channel: 32 blocks × 1024 samples ≈ 2 s of backpressure before
            // the capture thread blocks. Sized so the worker rarely blocks capture.
            let (block_tx, block_rx) = mpsc::sync_channel::<Vec<f32>>(32);

            // Capture thread: pushes ~64 ms blocks until stop is set.
            let capture_stop = Arc::clone(&stop_flag);
            let capture_device = device.clone();
            let capture_thread = thread::Builder::new()
                .name(format!("kitetsu-capture-{label}"))
                .spawn(move || {
                    if let Err(e) =
                        capture::capture_stream(&capture_device, label, capture_stop, block_tx)
                    {
                        error!(source = ?source, error = %e, "capture_stream exited with error");
                    }
                })
                .map_err(ListenerError::Io)?;

            // Worker thread: drains blocks through VAD + inference.
            let worker_thread = thread::Builder::new()
                .name(format!("kitetsu-worker-{label}"))
                .spawn(move || {
                    info!(source = ?source, "streaming worker started");
                    for block in &block_rx {
                        match streaming.feed(&block) {
                            Ok(FeedResult {
                                committed: Some(text),
                                tentative,
                            }) => {
                                let _ = update_tx.send(TranscriptUpdate {
                                    source,
                                    committed: text,
                                    tentative,
                                });
                            }
                            Ok(_) => {}
                            Err(e) => {
                                error!(source = ?source, error = %e, "feed error");
                            }
                        }
                    }
                    info!(source = ?source, "streaming worker exiting");
                })
                .map_err(ListenerError::Io)?;

            threads.push(capture_thread);
            threads.push(worker_thread);
        }

        Ok((StreamSession { stop, threads }, update_rx))
    }

    /// Stop all threads and join them.
    ///
    /// Sets the stop flag (capture threads exit their loop and drop their
    /// `SyncSender`s, which causes the worker threads to exit their `for … in
    /// block_rx` loops), then joins all threads.
    pub fn stop(self) -> Result<(), ListenerError> {
        self.stop.store(true, Ordering::Relaxed);
        for handle in self.threads {
            if let Err(e) = handle.join() {
                error!("streaming thread panicked: {e:?}");
            }
        }
        Ok(())
    }
}
