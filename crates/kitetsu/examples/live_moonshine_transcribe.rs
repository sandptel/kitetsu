//! Live (rolling-window) transcription of **mic + system audio in parallel**,
//! with Moonshine v2 streaming.
//!
//! What happens when you run it:
//!   1. Discover the default mic source and the system-output monitor.
//!   2. Load TWO model instances (one per stream) so both can infer in parallel.
//!   3. Both streams start capturing immediately; two live lines update in place
//!      — the top line is the mic, the bottom line is system audio.
//!   4. Each stream re-transcribes its current rolling window on a schedule and
//!      overwrites its line with the latest partial.
//!   5. Press Enter to stop; the final transcript of each stream is printed.
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
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, TryRecvError};
use std::time::Duration;

use anyhow::Context as _;

use kitetsu::listener::{
    AudioModel, Devices, LiveConfig, LiveTranscriber, LocalTranscriber, MoonshineVariant,
    Quantization, Recorder, discover_default_devices,
};

// ── Moonshine v2 streaming: accuracy vs speed ───────────────────────────────────
//  WER = avg over 8 ASR datasets (AMI, Earnings-22, GigaSpeech, LibriSpeech c/o,
//  SPGISpeech, TED-LIUM, VoxPopuli). Lower WER = more accurate. Speed is shown
//  relative to ONE baseline each way so the variants compare directly: "vs Tiny"
//  (slowdown between the three) and "vs Whisper Large v3" (a single fixed model).
//
//   Variant │ Params │  WER   │ Latency │ vs Tiny (speed) │ vs Whisper Large v3
//   ────────┼────────┼────────┼─────────┼─────────────────┼────────────────────
//   Tiny    │  34 M  │ 12.01% │  ~50 ms │ 1.0× (baseline) │ ~225× faster
//   Small   │ 123 M  │  7.84% │ ~148 ms │ ~3.0× slower    │  ~76× faster
//   Medium  │ 245 M  │  6.65% │ ~258 ms │ ~5.2× slower    │  ~44× faster
//
//  So going Tiny → Small → Medium: accuracy goes UP (WER 12.0 → 7.8 → 6.7%) and
//  speed goes DOWN (each step ~2–3× slower). Latency is single-stream compute;
//  mic+system infer in parallel here, so a pass costs ≈ one stream on a
//  multi-core CPU. Keep INTERVAL_MS above the variant's latency or passes queue.
//  Source: Moonshine v2 paper (arXiv:2602.12241); UsefulSensors benchmarks.
//
//  VARIANT      — ↑ size (Tiny→Medium): accuracy ↑, speed ↓. ↓ size: speed ↑, accuracy ↓.
//  QUANTIZATION — Fp32→Fp16→Int8→Int4: speed ↑ and size ↓ at each step, accuracy ↓
//                 slightly. Int8 ≈ 2× faster than Fp32. Falls back to Fp32 if the
//                 requested precision file is absent, so any setting is safe.
const VARIANT: MoonshineVariant = MoonshineVariant::Medium;
const QUANTIZATION: Quantization = Quantization::Fp32;

// ── Live tunables — how each affects latency vs accuracy ────────────────────────
//  WINDOW_SECS — audio re-transcribed each pass.
//                ↑ : more context on screen, but each pass is slower (more audio
//                    to decode); old audio lingers. ↓ : faster passes, less
//                    context (old words scroll out sooner). For short utterances
//                    a small window is both faster AND no less accurate.
//  INTERVAL_MS — minimum gap between passes (debounce).
//                ↓ : snappier updates / lower latency to first partial, more CPU.
//                ↑ : calmer screen, less CPU, partials lag further behind speech.
//                Does NOT change final accuracy — only how often you see updates.
//  MIN_NEW_MS  — minimum NEW audio before a pass runs.
//                ↓ : a pass fires after fewer words (good for "2–3 words then go"),
//                    more passes. ↑ : waits for more speech first, fewer passes,
//                    higher latency to the first partial. Accuracy unaffected.
//
//  For fast turn-taking (someone stops, you say 2–3 words, hand the text to an
//  LLM): VARIANT=Tiny, QUANTIZATION=Int8, WINDOW_SECS≈4, INTERVAL_MS≈200,
//  MIN_NEW_MS≈150. NOTE: this gives fast PARTIALS but not end-of-turn detection —
//  knowing the speaker stopped (to commit + fire the LLM) needs VAD/endpointing,
//  which is the next iteration (silence-based commit).
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

    // ── Discover both audio sources ────────────────────────────────────────────
    println!("Discovering audio devices...");
    let Devices {
        mic_source,
        system_monitor,
    } = discover_default_devices().context("PulseAudio device discovery failed")?;
    println!("  mic:    {mic_source}");
    println!("  system: {system_monitor}\n");

    // ── Load two model instances (one per stream → parallel inference) ─────────
    let model_dir = moonshine_model_dir();
    println!("Loading Moonshine v2 ×2 ({VARIANT:?}, {QUANTIZATION:?})...");
    let mic_live = make_live(&model_dir).context("failed to load model (mic)")?;
    let sys_live = make_live(&model_dir).context("failed to load model (system)")?;
    println!("Ready. window={WINDOW_SECS}s interval={INTERVAL_MS}ms min_new={MIN_NEW_MS}ms");
    println!("\nSpeak / play audio. Press Enter to stop.\n");

    // ── Start streaming capture on both, drive the live workers ────────────────
    let (mic_handle, mic_rx) = Recorder::new(mic_source, "kitetsu-mic-live").start_streaming();
    let (sys_handle, sys_rx) = Recorder::new(system_monitor, "kitetsu-sys-live").start_streaming();

    let worker =
        tokio::task::spawn_blocking(move || run_dual_live(mic_live, sys_live, mic_rx, sys_rx));

    // Block until Enter (do not print here — the worker owns the two live lines).
    let _ = std::io::stdin().lock().lines().next();

    // Stopping the recorders drops their senders, disconnecting both channels so
    // the worker runs its final pass and returns.
    let _ = mic_handle.stop();
    let _ = sys_handle.stop();
    let (mic_final, sys_final) = worker.await?.context("live worker failed")?;

    println!("\n─── Final ───────────────────────────────────────────────────────");
    println!("mic:    {}", show(&mic_final));
    println!("system: {}", show(&sys_final));

    Ok(())
}

/// Owns both live transcribers and channels. Drains captured chunks into each
/// rolling window, runs the due passes in parallel via scoped threads, and
/// repaints the two live lines. Returns `(mic_final, system_final)` once both
/// capture channels disconnect.
fn run_dual_live(
    mut mic: LiveTranscriber,
    mut sys: LiveTranscriber,
    mic_rx: Receiver<Vec<f32>>,
    sys_rx: Receiver<Vec<f32>>,
) -> anyhow::Result<(String, String)> {
    let mut mic_text = String::new();
    let mut sys_text = String::new();
    let mut mic_done = false;
    let mut sys_done = false;

    // Reserve the two live lines the renderer overwrites.
    println!("  mic     …");
    println!("  system  …");

    while !(mic_done && sys_done) {
        if !mic_done {
            mic_done = drain(&mic_rx, &mut mic);
        }
        if !sys_done {
            sys_done = drain(&sys_rx, &mut sys);
        }

        let run_mic = mic.should_run();
        let run_sys = sys.should_run();
        if run_mic || run_sys {
            // Run the due streams concurrently; each thread gets its own model.
            std::thread::scope(|s| -> anyhow::Result<()> {
                let mh = run_mic.then(|| s.spawn(|| mic.transcribe()));
                let sh = run_sys.then(|| s.spawn(|| sys.transcribe()));
                if let Some(h) = mh {
                    mic_text = h
                        .join()
                        .expect("mic thread panicked")
                        .context("mic infer")?;
                }
                if let Some(h) = sh {
                    sys_text = h
                        .join()
                        .expect("system thread panicked")
                        .context("system infer")?;
                }
                Ok(())
            })?;
            render(&mic_text, &sys_text, mic.buffer_secs(), sys.buffer_secs());
        }

        std::thread::sleep(Duration::from_millis(50));
    }

    // Final pass over whatever audio remains in each window.
    let mf = mic.transcribe().context("final mic infer")?;
    let sf = sys.transcribe().context("final system infer")?;
    if !mf.is_empty() {
        mic_text = mf;
    }
    if !sf.is_empty() {
        sys_text = sf;
    }

    Ok((mic_text, sys_text))
}

/// Drain all currently-available chunks into the window. Returns `true` when the
/// channel has disconnected (recorder stopped).
fn drain(rx: &Receiver<Vec<f32>>, live: &mut LiveTranscriber) -> bool {
    loop {
        match rx.try_recv() {
            Ok(chunk) => live.feed(&chunk),
            Err(TryRecvError::Empty) => return false,
            Err(TryRecvError::Disconnected) => return true,
        }
    }
}

/// Repaint the two reserved live lines in place.
fn render(mic: &str, sys: &str, mic_secs: f32, sys_secs: f32) {
    // Move to the start of the two-line block, then rewrite each line.
    print!("\x1b[2F");
    print!("\r\x1b[K  mic    {mic_secs:5.1}s  {}\n", show(mic));
    print!("\r\x1b[K  system {sys_secs:5.1}s  {}\n", show(sys));
    std::io::stdout().flush().ok();
}

fn make_live(model_dir: &Path) -> anyhow::Result<LiveTranscriber> {
    let transcriber = LocalTranscriber::load(AudioModel::Moonshine {
        model_dir: model_dir.to_path_buf(),
        variant: VARIANT,
        quantization: QUANTIZATION,
    })?;
    Ok(LiveTranscriber::new(
        transcriber,
        LiveConfig {
            window: Duration::from_secs(WINDOW_SECS),
            interval: Duration::from_millis(INTERVAL_MS),
            min_new: Duration::from_millis(MIN_NEW_MS),
        },
    ))
}

fn moonshine_model_dir() -> PathBuf {
    std::env::var("KITETSU_MOONSHINE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let base = std::env::var("XDG_DATA_HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|_| {
                    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
                    PathBuf::from(home).join(".local/share")
                });
            base.join("kitetsu/models/moonshine-tiny-streaming-en")
        })
}

fn show(s: &str) -> &str {
    if s.is_empty() { "…" } else { s }
}
