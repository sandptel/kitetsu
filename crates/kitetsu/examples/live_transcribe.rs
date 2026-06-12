//! Capture mic + system audio simultaneously and transcribe both with Moonshine.
//!
//! Demonstrates the full atomic pipe:
//!   `discover → load models (×2) → record → transcribe concurrently → print`
//!
//! Two separate `Transcriber` instances are loaded — one per stream — so both
//! inference calls run in parallel via `spawn_blocking` with no lock contention.
//! Each `Transcriber` holds its own ONNX session; the model files are small
//! enough that the memory cost is negligible compared to the speed gain.
//!
//! Usage: `cargo run -p kitetsu --example live_transcribe`
//!
//! Prerequisites:
//!   - A Moonshine v2 streaming model dir at
//!     `$XDG_DATA_HOME/kitetsu/models/moonshine-tiny-streaming-en/`
//!     (override with `$KITETSU_MOONSHINE_MODEL=/path/to/dir`).
//!     Files needed: `frontend.onnx`, `encoder.onnx`, `adapter.onnx`,
//!     `cross_kv.onnx`, `decoder_kv.onnx`, `tokenizer.bin`, `streaming_config.json`.
//!     Download: https://blob.handy.computer/moonshine-tiny-streaming-en.tar.gz
//!     (see `examples/quant_transcribe.rs` for small/medium + quantization).
//!   - PipeWire-pulse or PulseAudio running.

use std::io::{BufRead as _, Write as _};
use std::time::Instant;

use anyhow::Context as _;

use kitetsu::listener::{AudioModel, Devices, Recorder, Transcriber, discover_default_devices};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("kitetsu=info")),
        )
        .init();

    // ── Discover audio devices ─────────────────────────────────────────────────
    println!("Discovering audio devices...");
    let Devices {
        mic_source,
        system_monitor,
    } = discover_default_devices().context("PulseAudio device discovery failed")?;
    println!("  mic:    {mic_source}");
    println!("  system: {system_monitor}");
    println!();

    // ── Load two Moonshine instances (one per stream, true concurrency) ─────────
    //
    // Each Transcriber owns its own ONNX session. Two instances are loaded here
    // rather than sharing one behind a Mutex so that mic and system inference run
    // on separate blocking threads in parallel with no serialisation overhead.
    println!("Loading Moonshine models (×2)...");
    let load_start = Instant::now();
    let mut mic_t =
        Transcriber::load(AudioModel::Default).context("failed to load Moonshine model (mic)")?;
    let mut sys_t = Transcriber::load(AudioModel::Default)
        .context("failed to load Moonshine model (system)")?;
    println!(
        "Models ready in {:.1}s.\n",
        load_start.elapsed().as_secs_f32()
    );

    // ── Record ─────────────────────────────────────────────────────────────────
    wait_for_enter("Press Enter to start recording...");

    let mic_handle = Recorder::new(mic_source, "kitetsu-mic").start();
    let sys_handle = Recorder::new(system_monitor, "kitetsu-system").start();

    wait_for_enter("Recording mic + system — press Enter to stop.");

    let mic_samples = mic_handle.stop().context("mic recording failed")?;
    let sys_samples = sys_handle.stop().context("system recording failed")?;

    println!(
        "\nCaptured  mic: {:.1}s  |  system: {:.1}s\n",
        mic_samples.len() as f32 / 16_000.0,
        sys_samples.len() as f32 / 16_000.0,
    );

    // ── Transcribe both concurrently ───────────────────────────────────────────
    //
    // Each task moves one Transcriber onto a dedicated blocking thread.
    // Moonshine does not expose incremental progress; the timing line below
    // reports total wall time for both inference calls to complete.
    println!("Transcribing...");
    let transcribe_start = Instant::now();

    let mic_task = tokio::task::spawn_blocking(move || mic_t.transcribe(&mic_samples, None));
    let sys_task = tokio::task::spawn_blocking(move || sys_t.transcribe(&sys_samples, None));

    let mic_text = mic_task.await??;
    let sys_text = sys_task.await??;

    let elapsed = transcribe_start.elapsed().as_secs_f32();
    println!("Transcription done in {elapsed:.2}s.\n");

    // ── Print results ──────────────────────────────────────────────────────────
    println!("─── Mic ─────────────────────────────────────────────────────────");
    println!(
        "{}",
        if mic_text.is_empty() {
            "(nothing detected)"
        } else {
            &mic_text
        }
    );
    println!();
    println!("─── System ──────────────────────────────────────────────────────");
    println!(
        "{}",
        if sys_text.is_empty() {
            "(nothing detected)"
        } else {
            &sys_text
        }
    );

    Ok(())
}

// ── Helpers ────────────────────────────────────────────────────────────────────

fn wait_for_enter(prompt: &str) {
    print!("{prompt} ");
    std::io::stdout().flush().unwrap();
    let _ = std::io::stdin().lock().lines().next();
}
