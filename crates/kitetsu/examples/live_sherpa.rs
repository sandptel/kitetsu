//! Sherpa-ONNX native streaming transcription from mic and system audio.
//!
//! Feeds audio block-by-block to a sherpa-onnx online recognizer; partial text
//! streams in real-time as `~[SRC]` until the model's built-in endpoint fires,
//! at which point the utterance is committed as `[SRC]`.
//!
//! Requires the `sherpa` feature and a downloaded Zipformer model directory.
//!
//! ## Setup
//!
//! 1. Download a streaming English Zipformer model from the sherpa-onnx releases, e.g.:
//!    <https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models>
//!    Recommended: `sherpa-onnx-streaming-zipformer-en-2023-06-26` (small, fast)
//!
//! 2. Set the model directory:
//!    ```sh
//!    export KITETSU_SHERPA_MODEL=/path/to/sherpa-onnx-streaming-zipformer-en-2023-06-26
//!    ```
//!    Or place it at `$XDG_DATA_HOME/kitetsu/models/sherpa-onnx-streaming-en/`.
//!
//! 3. Set the library path (NixOS / shared-lib build):
//!    ```sh
//!    export SHERPA_ONNX_LIB_DIR=/path/to/sherpa-onnx/lib
//!    ```
//!
//! 4. Run:
//!    ```sh
//!    cargo run -p kitetsu --features sherpa --example live_sherpa
//!    ```
//!    For inference-level debug output: `RUST_LOG=kitetsu=debug cargo run ...`

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

    let model_dir = AudioModel::default_sherpa_model_dir();
    println!("Using sherpa-onnx model dir: {}", model_dir.display());
    println!("  (override with KITETSU_SHERPA_MODEL=/path/to/model-dir)\n");

    if !model_dir.is_dir() {
        anyhow::bail!(
            "Model directory not found: {}\n\
             Download a Zipformer streaming model and set KITETSU_SHERPA_MODEL.",
            model_dir.display()
        );
    }

    println!("Discovering audio devices...");
    let devices = discover_default_devices().context("PulseAudio device discovery failed")?;
    println!("  mic:    {}", devices.mic_source);
    println!("  system: {}", devices.system_monitor);
    println!();

    println!("Loading sherpa-onnx models (one per source)...");
    let sources = [StreamSource::Mic, StreamSource::System];
    let (session, update_rx) = StreamSession::start(
        &sources,
        AudioModel::SherpaOnnx { model_dir },
        StreamMode::SherpaStreaming,
    )
    .context("failed to start streaming session")?;
    println!("Models ready. Listening (sherpa-onnx streaming). Press Enter to stop.");
    println!("Legend: [MIC]/[SYS] = committed | ~[MIC]/~[SYS] = tentative (in-progress)");
    println!("(RUST_LOG=kitetsu=debug shows per-block endpoint/decode state)\n");

    let display = std::thread::spawn(move || {
        for update in &update_rx {
            let ts = now_hms();
            let src = match update.source {
                StreamSource::Mic => "MIC",
                StreamSource::System => "SYS",
            };
            if !update.committed.is_empty() {
                println!("[{ts}] [{src}] {}", update.committed);
            }
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
