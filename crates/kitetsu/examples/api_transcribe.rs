//! Capture mic + system audio simultaneously and transcribe both via the OpenAI REST API.
//!
//! Demonstrates the API counterpart to the local Moonshine parallel-capture pipe:
//!   `discover → record (mic + system) → encode WAV × 2 → POST concurrently → print`
//!
//! Both streams are sent to the API in parallel using `tokio::join!`; total latency
//! is approximately one round trip, not two.
//!
//! Usage:
//!   cargo run -p kitetsu --features api --example api_transcribe
//!
//! Key discovery order (first match wins):
//!   1. `OPENAI_API_KEY` already set in the shell environment.
//!   2. `.env` file at the workspace root (cwd when invoked via `cargo run`).
//!
//! Optional env overrides:
//!   - `KITETSU_OPENAI_MODEL`   — transcription model (default `gpt-4o-transcribe`).
//!   - `KITETSU_OPENAI_LANG`    — ISO-639-1 language hint (e.g. `en`); omit for auto-detect.
//!
//! Prerequisites: PipeWire-pulse or PulseAudio running, and network access to the
//! OpenAI API. Never hard-code the key; use the env or `.env` file.

use std::io::{BufRead as _, Write as _};
use std::time::Instant;

use anyhow::Context as _;

use kitetsu::listener::{
    ApiConfig, ApiTranscriber, Devices, Recorder, OPENAI_API_KEY, discover_default_devices,
    load_dotenv, require,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("kitetsu=info")),
        )
        .init();

    // ── Load .env (workspace root) before any key lookups ─────────────────────
    if let Ok(cwd) = std::env::current_dir() {
        load_dotenv(&cwd.join(".env"));
    }

    // ── Build the API transcriber ──────────────────────────────────────────────
    let api_key = require(OPENAI_API_KEY)
        .context("set OPENAI_API_KEY in the environment or .env file before running")?;

    let mut config = ApiConfig::default();
    if let Ok(model) = std::env::var("KITETSU_OPENAI_MODEL") {
        config.model = model;
    }
    if let Ok(lang) = std::env::var("KITETSU_OPENAI_LANG") {
        config.language = Some(lang);
    }
    println!("model: {}", config.model);
    if let Some(ref lang) = config.language {
        println!("lang:  {lang}");
    }

    let transcriber =
        ApiTranscriber::new(api_key, config).context("failed to build the API transcriber")?;

    // ── Discover both audio sources ────────────────────────────────────────────
    println!("\nDiscovering audio devices...");
    let Devices {
        mic_source,
        system_monitor,
    } = discover_default_devices().context("PulseAudio device discovery failed")?;
    println!("  mic:    {mic_source}");
    println!("  system: {system_monitor}\n");

    // ── Record both streams ────────────────────────────────────────────────────
    wait_for_enter("Press Enter to start recording...");
    let mic_handle = Recorder::new(mic_source, "kitetsu-api-mic").start();
    let sys_handle = Recorder::new(system_monitor, "kitetsu-api-system").start();
    wait_for_enter("Recording mic + system — press Enter to stop.");
    let mic_samples = mic_handle.stop().context("mic recording failed")?;
    let sys_samples = sys_handle.stop().context("system recording failed")?;

    let mic_secs = mic_samples.len() as f32 / 16_000.0;
    let sys_secs = sys_samples.len() as f32 / 16_000.0;
    println!("\nCaptured  mic: {mic_secs:.1}s  |  system: {sys_secs:.1}s");
    println!("Sending both to the API concurrently...\n");

    // ── Transcribe both concurrently (one round-trip worth of latency) ─────────
    let start = Instant::now();
    let (mic_res, sys_res) = tokio::join!(
        transcriber.transcribe(&mic_samples),
        transcriber.transcribe(&sys_samples)
    );
    let elapsed = start.elapsed().as_secs_f32();
    println!("API round-trip: {elapsed:.2}s\n");

    // ── Present ────────────────────────────────────────────────────────────────
    print_stream("Mic", &mic_res.context("mic API transcription failed")?);
    print_stream("System", &sys_res.context("system API transcription failed")?);

    Ok(())
}

fn print_stream(label: &str, text: &str) {
    println!("─── {label} ──────────────────────────────────────────────────────────");
    println!(
        "{}\n",
        if text.trim().is_empty() {
            "(nothing detected)"
        } else {
            text
        }
    );
}

fn wait_for_enter(prompt: &str) {
    print!("{prompt} ");
    std::io::stdout().flush().ok();
    let _ = std::io::stdin().lock().lines().next();
}
