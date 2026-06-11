//! Capture microphone and system audio to WAV files at the workspace root.
//!
//! Usage: `cargo run -p kitetsu --example capture_audio -- [SECONDS]`
//! Default duration: 10 seconds.
//!
//! Produces `mic.wav` (mono, 48 kHz) and `system.wav` (stereo, 48 kHz)
//! at the workspace root. Both are listed in .gitignore.

use std::env;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::Context as _;
use kitetsu::listener::{CaptureConfig, Devices, capture_to_wav, discover_default_devices};
use tracing::{error, info};

fn main() -> anyhow::Result<()> {
    // Respect RUST_LOG; default to info-level output from the kitetsu crate.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("kitetsu=info")),
        )
        .init();

    // — Parse optional duration argument ——————————————————————————————————————

    let duration_secs: u64 = env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let duration = Duration::from_secs(duration_secs);

    // — Introspect PulseAudio / PipeWire-pulse for device names ———————————————

    let Devices {
        mic_source,
        system_monitor,
    } = discover_default_devices().context(
        "failed to introspect PulseAudio — \
             ensure pipewire-pulse or PulseAudio is running",
    )?;

    info!(
        mic    = %mic_source,
        system = %system_monitor,
        "resolved audio devices via PulseAudio introspection",
    );
    info!(
        secs = duration_secs,
        "capturing on two threads — speak and play audio now"
    );

    // — Resolve output paths: workspace root (two levels above crate dir) ——————
    //
    // CARGO_MANIFEST_DIR is set by Cargo to the directory of crates/kitetsu/
    // at build time. Going up twice yields the workspace root (kitetsu/).
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = crate_dir
        .parent() // crates/
        .and_then(|p| p.parent()) // kitetsu/ (workspace root)
        .map(PathBuf::from)
        .context("could not determine workspace root from CARGO_MANIFEST_DIR")?;

    let mic_path = workspace_root.join("mic.wav");
    let system_path = workspace_root.join("system.wav");

    // — Spawn one thread per stream ——————————————————————————————————————————————

    let mic_cfg = CaptureConfig::mic(mic_source);
    let system_cfg = CaptureConfig::system(system_monitor);

    let mic_path_t = mic_path.clone();
    let system_path_t = system_path.clone();

    let mic_thread = std::thread::Builder::new()
        .name("kitetsu-mic-capture".into())
        .spawn(move || capture_to_wav(&mic_cfg, &mic_path_t, duration))
        .context("could not spawn mic capture thread")?;

    let system_thread = std::thread::Builder::new()
        .name("kitetsu-system-capture".into())
        .spawn(move || capture_to_wav(&system_cfg, &system_path_t, duration))
        .context("could not spawn system capture thread")?;

    let mic_result = mic_thread.join().expect("mic capture thread panicked");
    let system_result = system_thread
        .join()
        .expect("system capture thread panicked");

    // — Report results ——————————————————————————————————————————————————————————

    for (label, result, path) in [
        ("mic", mic_result, &mic_path),
        ("system", system_result, &system_path),
    ] {
        match result {
            Ok(()) => {
                let bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
                info!(label, path = %path.display(), bytes, "WAV file written successfully");
            }
            Err(e) => error!(label, error = %e, "capture failed"),
        }
    }

    Ok(())
}
