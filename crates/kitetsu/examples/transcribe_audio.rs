//! Demonstrates the full capture → transcription pipeline with a persistent
//! [`Transcriber`]: models are loaded once before recording starts, so inference
//! begins immediately when recording stops with no model-load latency in the
//! hot path. Both mic and system audio are captured and processed concurrently.
//!
//! Usage: `cargo run -p kitetsu --example transcribe_audio`
//!
//! Prerequisites:
//!   - A GGML Whisper model at `$KITETSU_WHISPER_MODEL` or
//!     `$XDG_DATA_HOME/kitetsu/models/ggml-base.en.bin`.
//!   - PipeWire-pulse or PulseAudio running.

use std::io::{BufRead as _, Write as _};
use std::sync::mpsc;
use std::time::{Duration, Instant};

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

    // ── Discover devices ──────────────────────────────────────────────────────
    println!("Discovering audio devices...");
    let Devices {
        mic_source,
        system_monitor,
    } = discover_default_devices().context("PulseAudio device discovery failed")?;
    println!("  mic:    {mic_source}");
    println!("  system: {system_monitor}");
    println!();

    // ── Load models upfront (once at startup; held alive for reuse) ───────────
    println!("Loading speech models...");
    let load_start = Instant::now();
    let mut mic_t =
        Transcriber::load(AudioModel::Default).context("failed to load whisper model (mic)")?;
    let mut sys_t =
        Transcriber::load(AudioModel::Default).context("failed to load whisper model (system)")?;
    println!(
        "Models ready in {:.1}s.\n",
        load_start.elapsed().as_secs_f32()
    );

    // ── Record ────────────────────────────────────────────────────────────────
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

    // ── Transcribe both concurrently ──────────────────────────────────────────
    println!("Processing...");
    // Reserve two lines for the progress bars.
    println!();
    println!();

    let (mic_prog_tx, mic_prog_rx) = mpsc::channel::<u8>();
    let (sys_prog_tx, sys_prog_rx) = mpsc::channel::<u8>();

    let mic_task =
        tokio::task::spawn_blocking(move || mic_t.transcribe(&mic_samples, Some(mic_prog_tx)));
    let sys_task =
        tokio::task::spawn_blocking(move || sys_t.transcribe(&sys_samples, Some(sys_prog_tx)));

    let poll_start = Instant::now();
    let mut mic_pct = 0u8;
    let mut sys_pct = 0u8;

    loop {
        // Drain latest progress from each channel (keep only the newest value).
        while let Ok(p) = mic_prog_rx.try_recv() {
            mic_pct = p;
        }
        while let Ok(p) = sys_prog_rx.try_recv() {
            sys_pct = p;
        }

        let elapsed = poll_start.elapsed().as_secs_f32();
        let mic_done = mic_task.is_finished();
        let sys_done = sys_task.is_finished();

        // Overwrite the two reserved lines.
        print!("\x1b[2F");
        progress_line("mic   ", if mic_done { 100 } else { mic_pct }, elapsed);
        progress_line("system", if sys_done { 100 } else { sys_pct }, elapsed);
        std::io::stdout().flush()?;

        if mic_done && sys_done {
            break;
        }
        tokio::time::sleep(Duration::from_millis(80)).await;
    }

    println!();

    let mic_text = mic_task.await??;
    let sys_text = sys_task.await??;
    let total = poll_start.elapsed().as_secs_f32();

    println!("Total processing time: {total:.1}s\n");

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

// ── Helpers ───────────────────────────────────────────────────────────────────

fn wait_for_enter(prompt: &str) {
    print!("{prompt} ");
    std::io::stdout().flush().unwrap();
    let _ = std::io::stdin().lock().lines().next();
}

/// Print one progress bar line, overwriting the current terminal line.
///
/// Displays a 24-cell bar, percentage, and elapsed seconds. Call
/// `print!("\x1b[2F")` before two consecutive calls to reposition the cursor
/// to the first line before overwriting.
fn progress_line(label: &str, pct: u8, elapsed_secs: f32) {
    const WIDTH: usize = 24;
    let filled = (pct as usize * WIDTH / 100).min(WIDTH);
    let bar = "█".repeat(filled) + &"░".repeat(WIDTH - filled);
    let marker = if pct >= 100 { "done" } else { "    " };
    println!("\r\x1b[K[{label}] [{bar}] {pct:3}%  {elapsed_secs:.1}s  {marker}");
}
