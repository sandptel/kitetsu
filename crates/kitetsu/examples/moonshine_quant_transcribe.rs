//! Moonshine v2 (streaming) — one-shot transcription of **mic + system audio in
//! parallel**, with a swappable model variant and quantization level.
//!
//! What happens when you run it:
//!   1. Discover the default mic source and the system-output monitor.
//!   2. Load TWO model instances (one per stream) so inference runs truly in
//!      parallel — no shared lock.
//!   3. Press Enter → both streams record at 16 kHz mono until you press Enter
//!      again.
//!   4. Both recordings are transcribed concurrently on blocking threads.
//!   5. Each transcript prints under its own header, with per-stream wall time
//!      and real-time speedup.
//!
//! Tunables (consts below) — change and re-run to compare:
//!   - `VARIANT`: Tiny | Small | Medium. Bigger = more accurate but slower (see the benchmark table by the consts).
//!   - `QUANTIZATION`: Fp32 | Fp16 | Int8 | Int4. Lower precision = smaller + faster at a small WER cost. The backend tries the requested file (e.g. `encoder.int8.onnx`) and falls back to FP32 if absent, so any setting is safe.
//!
//! Usage: `cargo run -p kitetsu --example moonshine_quant_transcribe`
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
use std::time::Instant;

use anyhow::Context as _;

use kitetsu::listener::{
    AudioModel, Devices, LocalTranscriber, MoonshineVariant, Quantization, Recorder,
    discover_default_devices,
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
//  running mic+system in parallel on a multi-core CPU keeps wall-clock ≈ one
//  stream. Source: Moonshine v2 paper (arXiv:2602.12241); UsefulSensors benchmarks.
//
//  VARIANT      — ↑ size (Tiny→Medium): accuracy ↑, speed ↓. ↓ size: speed ↑, accuracy ↓.
//  QUANTIZATION — Fp32→Fp16→Int8→Int4: speed ↑ and size ↓ at each step, accuracy ↓
//                 slightly. Int8 ≈ 2× faster than Fp32, Fp16 in between. Falls back
//                 to Fp32 if the requested precision file is absent, so any is safe.
const VARIANT: MoonshineVariant = MoonshineVariant::Medium;
const QUANTIZATION: Quantization = Quantization::Fp16;

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

    // ── Load two model instances (one per stream → true parallel inference) ─────
    let model_dir = moonshine_model_dir();
    println!("Loading Moonshine v2 ×2 ({VARIANT:?}, {QUANTIZATION:?})...");
    println!("  dir: {}", model_dir.display());
    let load_start = Instant::now();
    let mut mic_t = load_model(&model_dir).context("failed to load model (mic)")?;
    let mut sys_t = load_model(&model_dir).context("failed to load model (system)")?;
    println!(
        "Models ready in {:.2}s.\n",
        load_start.elapsed().as_secs_f32()
    );

    // ── Record both streams ────────────────────────────────────────────────────
    wait_for_enter("Press Enter to start recording...");
    let mic_handle = Recorder::new(mic_source, "kitetsu-mic").start();
    let sys_handle = Recorder::new(system_monitor, "kitetsu-system").start();
    wait_for_enter("Recording mic + system — press Enter to stop.");
    let mic_samples = mic_handle.stop().context("mic recording failed")?;
    let sys_samples = sys_handle.stop().context("system recording failed")?;

    let mic_secs = mic_samples.len() as f32 / 16_000.0;
    let sys_secs = sys_samples.len() as f32 / 16_000.0;
    println!("\nCaptured  mic: {mic_secs:.1}s  |  system: {sys_secs:.1}s\n");

    // ── Transcribe both concurrently ───────────────────────────────────────────
    println!("Transcribing both streams in parallel...");
    let mic_task = tokio::task::spawn_blocking(move || {
        let t0 = Instant::now();
        mic_t
            .transcribe(&mic_samples, None)
            .map(|text| (text, t0.elapsed().as_secs_f32()))
    });
    let sys_task = tokio::task::spawn_blocking(move || {
        let t0 = Instant::now();
        sys_t
            .transcribe(&sys_samples, None)
            .map(|text| (text, t0.elapsed().as_secs_f32()))
    });

    let (mic_text, mic_infer) = mic_task.await?.context("mic inference failed")?;
    let (sys_text, sys_infer) = sys_task.await?.context("system inference failed")?;

    // ── Present ────────────────────────────────────────────────────────────────
    print_stream("Mic", &mic_text, mic_secs, mic_infer);
    print_stream("System", &sys_text, sys_secs, sys_infer);

    Ok(())
}

// ── Helpers ──────────────────────────────────────────────────────────────────────

fn load_model(model_dir: &Path) -> anyhow::Result<LocalTranscriber> {
    LocalTranscriber::load(AudioModel::Moonshine {
        model_dir: model_dir.to_path_buf(),
        variant: VARIANT,
        quantization: QUANTIZATION,
    })
    .map_err(Into::into)
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

fn print_stream(label: &str, text: &str, audio_secs: f32, infer_secs: f32) {
    let speedup = if infer_secs > 0.0 {
        audio_secs / infer_secs
    } else {
        f32::INFINITY
    };
    println!("─── {label} ──  ({infer_secs:.3}s infer, {speedup:.1}× real-time)");
    println!(
        "{}\n",
        if text.is_empty() {
            "(nothing detected)"
        } else {
            text
        }
    );
}

fn wait_for_enter(prompt: &str) {
    print!("{prompt} ");
    std::io::stdout().flush().unwrap();
    let _ = std::io::stdin().lock().lines().next();
}
