//! Moonshine v2 (streaming) transcription with a swappable quantization level.
//!
//! Change the single `QUANTIZATION` const below and re-run to compare precision
//! levels (FP32 / FP16 / INT8 / INT4) on the same recording. The backend tries
//! the requested precision's file first and falls back to FP32 if it is absent,
//! so any level is safe to select even if you only downloaded one.
//!
//! Usage: `cargo run -p kitetsu --example quant_transcribe`
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
use std::time::Instant;

use anyhow::Context as _;

use kitetsu::listener::{
    AudioModel, Devices, MoonshineVariant, Quantization, Recorder, Transcriber,
    discover_default_devices,
};

// ── Swap this one line to test each precision ───────────────────────────────────
const QUANTIZATION: Quantization = Quantization::Int8;
// Options: Quantization::Fp32 | Fp16 | Int8 | Int4
const VARIANT: MoonshineVariant = MoonshineVariant::Tiny;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("kitetsu=info")),
        )
        .init();

    // ── Discover the microphone source ─────────────────────────────────────────
    println!("Discovering audio devices...");
    let Devices { mic_source, .. } =
        discover_default_devices().context("PulseAudio device discovery failed")?;
    println!("  mic: {mic_source}\n");

    // ── Load the streaming model at the selected precision ─────────────────────
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
    println!("  dir: {}", model_dir.display());
    let load_start = Instant::now();
    let mut transcriber = Transcriber::load(AudioModel::Moonshine {
        model_dir,
        variant: VARIANT,
        quantization: QUANTIZATION,
    })
    .context("failed to load Moonshine v2 streaming model")?;
    println!(
        "Model ready in {:.2}s.\n",
        load_start.elapsed().as_secs_f32()
    );

    // ── Record ─────────────────────────────────────────────────────────────────
    wait_for_enter("Press Enter to start recording...");
    let handle = Recorder::new(mic_source, "kitetsu-quant").start();
    wait_for_enter("Recording — press Enter to stop.");
    let samples = handle.stop().context("recording failed")?;

    let audio_secs = samples.len() as f32 / 16_000.0;
    println!("\nCaptured {audio_secs:.1}s of audio.\n");

    // ── Transcribe ─────────────────────────────────────────────────────────────
    println!("Transcribing...");
    let t0 = Instant::now();
    let text =
        tokio::task::spawn_blocking(move || transcriber.transcribe(&samples, None)).await??;
    let infer_secs = t0.elapsed().as_secs_f32();

    let speedup = if infer_secs > 0.0 {
        audio_secs / infer_secs
    } else {
        f32::INFINITY
    };

    println!("─────────────────────────────────────────────────────────────────");
    println!("Quantization : {QUANTIZATION:?}");
    println!("Inference    : {infer_secs:.3}s  ({speedup:.1}× real-time)");
    println!("─────────────────────────────────────────────────────────────────");
    println!(
        "{}",
        if text.is_empty() {
            "(nothing detected)"
        } else {
            &text
        }
    );

    Ok(())
}

fn wait_for_enter(prompt: &str) {
    print!("{prompt} ");
    std::io::stdout().flush().unwrap();
    let _ = std::io::stdin().lock().lines().next();
}
