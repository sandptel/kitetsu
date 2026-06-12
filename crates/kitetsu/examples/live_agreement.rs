//! LocalAgreement-2 live transcription from mic and system audio simultaneously.
//!
//! Prints one log line per update: committed text appears normally, tentative
//! (in-progress words not yet confirmed) is prefixed with `~`.
//!
//! For detailed state (recording / silence / inference pass timing), run with:
//!   RUST_LOG=kitetsu=debug cargo run -p kitetsu --example live_agreement
//!
//! Usage: `cargo run -p kitetsu --example live_agreement`
//!
//! Prerequisites:
//!   - A GGML Whisper model at `$KITETSU_WHISPER_MODEL` or
//!     `$XDG_DATA_HOME/kitetsu/models/ggml-base.en.bin`.
//!     TIP: ggml-tiny.en runs ~4-6x faster — strongly recommended for LA-2.
//!   - PipeWire-pulse or PulseAudio running.

use std::io::BufRead as _;

use anyhow::Context as _;

use kitetsu::listener::{
    AudioModel, StreamMode, StreamSession, StreamSource, discover_default_devices,
};

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("kitetsu=info")),
        )
        .with_target(false)
        .init();

    println!("Discovering audio devices...");
    let devices = discover_default_devices().context("PulseAudio device discovery failed")?;
    println!("  mic:    {}", devices.mic_source);
    println!("  system: {}", devices.system_monitor);
    println!();

    println!("Loading models (one per source)...");
    let sources = [StreamSource::Mic, StreamSource::System];
    let (session, update_rx) =
        StreamSession::start(&sources, AudioModel::Default, StreamMode::LocalAgreement)
            .context("failed to start streaming session")?;
    println!("Models ready. Listening (LocalAgreement-2). Press Enter to stop.");
    println!("Legend: [MIC]/[SYS] = committed text | ~[MIC]/~[SYS] = tentative (in-progress)");
    println!("(RUST_LOG=kitetsu=debug shows each inference pass with sample count and timing)\n");

    let display = std::thread::spawn(move || {
        for update in &update_rx {
            let ts = now_hms();
            let src = match update.source {
                StreamSource::Mic => "MIC",
                StreamSource::System => "SYS",
            };
            // Committed text: finalized, agreed by two consecutive passes.
            if !update.committed.is_empty() {
                println!("[{ts}] [{src}] {}", update.committed);
            }
            // Tentative tail: heard but not yet confirmed by a second pass.
            if !update.tentative.is_empty() {
                println!("[{ts}] ~[{src}] {}", update.tentative);
            }
        }
    });

    let stdin = std::io::stdin();
    stdin.lock().lines().next();

    session.stop().context("error stopping session")?;
    display.join().ok();
    println!("Stopped.");

    Ok(())
}

fn now_hms() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let s = d.as_secs();
    let ms = d.subsec_millis();
    format!(
        "{:02}:{:02}:{:02}.{:03}",
        (s / 3600) % 24,
        (s / 60) % 60,
        s % 60,
        ms
    )
}
