//! Capture mic audio and transcribe it via the OpenAI REST API.
//!
//! Demonstrates the API counterpart to the local Moonshine pipe:
//!   `discover → record → encode WAV → POST → print`
//!
//! Usage:
//!   OPENAI_API_KEY=sk-... \
//!     cargo run -p kitetsu --features api --example api_transcribe
//!
//! Optional:
//!   - `$KITETSU_OPENAI_MODEL` overrides the model (default `gpt-4o-transcribe`).
//!
//! Prerequisites: PipeWire-pulse or PulseAudio running, and network access to the
//! OpenAI API. The key is read from the environment only — never hard-code it.

use std::io::{BufRead as _, Write as _};
use std::time::Instant;

use anyhow::Context as _;

use kitetsu::listener::{ApiConfig, ApiTranscriber, Devices, Recorder, discover_default_devices};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("kitetsu=info")),
        )
        .init();

    // ── Build the API transcriber ──────────────────────────────────────────────
    let api_key = std::env::var("OPENAI_API_KEY")
        .context("set OPENAI_API_KEY in the environment before running")?;

    let mut config = ApiConfig::default();
    if let Ok(model) = std::env::var("KITETSU_OPENAI_MODEL") {
        config.model = model;
    }
    println!("Using model: {}\n", config.model);

    let transcriber =
        ApiTranscriber::new(api_key, config).context("failed to build the API transcriber")?;

    // ── Discover the mic source ────────────────────────────────────────────────
    let Devices { mic_source, .. } =
        discover_default_devices().context("PulseAudio device discovery failed")?;
    println!("mic: {mic_source}\n");

    // ── Record ─────────────────────────────────────────────────────────────────
    wait_for_enter("Press Enter to start recording...");
    let handle = Recorder::new(mic_source, "kitetsu-mic").start();
    wait_for_enter("Recording — press Enter to stop.");
    let samples = handle.stop().context("mic recording failed")?;
    println!(
        "\nCaptured {:.1}s — sending to the API...\n",
        samples.len() as f32 / 16_000.0
    );

    // ── Transcribe ─────────────────────────────────────────────────────────────
    let start = Instant::now();
    let text = transcriber
        .transcribe(&samples)
        .await
        .context("API transcription failed")?;
    println!(
        "Transcription (round-trip {:.1}s):\n{text}",
        start.elapsed().as_secs_f32()
    );

    Ok(())
}

fn wait_for_enter(prompt: &str) {
    print!("{prompt} ");
    std::io::stdout().flush().ok();
    let mut line = String::new();
    std::io::stdin().lock().read_line(&mut line).ok();
}
