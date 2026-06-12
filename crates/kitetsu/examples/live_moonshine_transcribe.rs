//! Live (rolling-window) microphone transcription with Moonshine v2.
//!
//! Same model selection as `moonshine_quant_transcribe` (swap `VARIANT` /
//! `QUANTIZATION`), plus live tunables. Audio is captured continuously and the
//! current window is re-transcribed on a schedule, printing an updating partial
//! line. Press Enter to stop and print the final window.
//!
//! Tunables (consts below):
//!   - `VARIANT`      — Tiny | Small | Medium
//!   - `QUANTIZATION` — Fp32 | Fp16 | Int8 | Int4 (falls back to FP32 if absent)
//!   - `WINDOW_SECS`  — max audio length re-transcribed each pass (context cap)
//!   - `INTERVAL_MS`  — min wall time between passes (debounce)
//!   - `MIN_NEW_MS`   — min new audio before another pass runs
//!
//! Usage: `cargo run -p kitetsu --example live_moonshine_transcribe`
//!
//! ── Model files ────────────────────────────────────────────────────────────────
//! Put a Moonshine v2 streaming model directory at
//! `$XDG_DATA_HOME/kitetsu/models/moonshine-tiny-streaming-en/`
//! (override with `$KITETSU_MOONSHINE_MODEL=/path/to/dir`).
//!
//! Required contents (the 5-session streaming pipeline):
//!   - `frontend.onnx`   (and/or `frontend.fp16.onnx`, `frontend.int8.onnx`, ...)
//!   - `encoder.onnx`
//!   - `adapter.onnx`
//!   - `cross_kv.onnx`
//!   - `decoder_kv.onnx`
//!   - `tokenizer.bin`
//!   - `streaming_config.json`
//!
//! Download (tarballs, extract into the models dir):
//!   - tiny:   https://blob.handy.computer/moonshine-tiny-streaming-en.tar.gz
//!   - small:  https://blob.handy.computer/moonshine-small-streaming-en.tar.gz
//!   - medium: https://blob.handy.computer/moonshine-medium-streaming-en.tar.gz
//!
//! Source project: https://github.com/cjpais/transcribe-rs (Moonshine by UsefulSensors).
//!
//! Requires PipeWire-pulse or PulseAudio running.

use std::io::{BufRead as _, Write as _};
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::time::Duration;

use anyhow::Context as _;

use kitetsu::listener::{
    AudioModel, Devices, LiveConfig, LiveTranscriber, MoonshineVariant, Quantization, Recorder,
    Transcriber, discover_default_devices,
};

// ── Model selection (same as moonshine_quant_transcribe) ────────────────────────
const VARIANT: MoonshineVariant = MoonshineVariant::Tiny;
const QUANTIZATION: Quantization = Quantization::Fp16;

// ── Live tunables ───────────────────────────────────────────────────────────────
const WINDOW_SECS: u64 = 10;
const INTERVAL_MS: u64 = 500;
const MIN_NEW_MS: u64 = 300;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("kitetsu=warn")),
        )
        .init();

    // ── Discover the microphone source ─────────────────────────────────────────
    println!("Discovering audio devices...");
    let Devices { mic_source, .. } =
        discover_default_devices().context("PulseAudio device discovery failed")?;
    println!("  mic: {mic_source}\n");

    // ── Load the streaming model ───────────────────────────────────────────────
    let model_dir = std::env::var("KITETSU_MOONSHINE_MODEL")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            let base = std::env::var("XDG_DATA_HOME")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|_| {
                    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
                    std::path::PathBuf::from(home).join(".local/share")
                });
            base.join("kitetsu/models/moonshine-tiny-streaming-en")
        });

    println!("Loading Moonshine v2 ({VARIANT:?}, {QUANTIZATION:?})...");
    let transcriber = Transcriber::load(AudioModel::Moonshine {
        model_dir,
        variant: VARIANT,
        quantization: QUANTIZATION,
    })
    .context("failed to load Moonshine v2 streaming model")?;

    let live = LiveTranscriber::new(
        transcriber,
        LiveConfig {
            window: Duration::from_secs(WINDOW_SECS),
            interval: Duration::from_millis(INTERVAL_MS),
            min_new: Duration::from_millis(MIN_NEW_MS),
        },
    );
    println!("Ready. window={WINDOW_SECS}s interval={INTERVAL_MS}ms min_new={MIN_NEW_MS}ms\n");

    // ── Stream capture into the live transcriber on a blocking thread ──────────
    let (handle, rx) = Recorder::new(mic_source, "kitetsu-live").start_streaming();
    let worker = tokio::task::spawn_blocking(move || run_live(live, rx));

    wait_for_enter("Live transcribing — press Enter to stop.");

    // Stopping the recorder drops its sender, disconnecting `rx` so the worker
    // finishes its final pass and returns.
    let _ = handle.stop();
    let final_text = worker.await?.context("live worker failed")?;

    println!("\n─── Final ───────────────────────────────────────────────────────");
    println!(
        "{}",
        if final_text.is_empty() {
            "(nothing detected)"
        } else {
            &final_text
        }
    );

    Ok(())
}

/// Drains captured chunks, feeds the rolling window, and prints an updating
/// partial line whenever the schedule says a pass is due. Returns the final
/// transcript once the capture channel disconnects.
fn run_live(mut live: LiveTranscriber, rx: Receiver<Vec<f32>>) -> anyhow::Result<String> {
    let mut partial = String::new();

    loop {
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(chunk) => live.feed(&chunk),
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => break,
        }

        if live.should_run() {
            partial = live.transcribe().context("live inference failed")?;
            print!("\r\x1b[K[{:.1}s] {partial}", live.buffer_secs());
            std::io::stdout().flush().ok();
        }
    }

    // Final pass over whatever remains in the window.
    let final_text = live.transcribe().context("final inference failed")?;
    Ok(if final_text.is_empty() {
        partial
    } else {
        final_text
    })
}

fn wait_for_enter(prompt: &str) {
    print!("{prompt} ");
    std::io::stdout().flush().unwrap();
    let _ = std::io::stdin().lock().lines().next();
}
