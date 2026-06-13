//! Pipes 1 + 2 end-to-end, interactive: continuously capture mic + system audio
//! and live-transcribe both over the Realtime WS while you talk. Each time you
//! press Enter, the current window (everything heard since the last press) fires
//! both pipes in parallel — pipe1 (accumulated live transcript, fast) and pipe2
//! (on-trigger REST re-transcription of the raw audio, slower but more accurate).
//! Each suggestion is printed and appended to its `out/pipe{1,2}.md`, then the
//! window resets — recording never stops.
//!
//! This is the daemon's pipe path driven by the keyboard instead of the control
//! socket, so you can watch the live → LLM → file flow in one terminal.
//!
//! Usage:
//!   cargo run -p kitetsu --features teleprompter --example teleprompter_pipe1
//!
//! Requires prompt.md + context.md next to config.toml, `OPENAI_API_KEY` (Realtime
//! WS + REST transcription + LLM when openai), and `ANTHROPIC_API_KEY` if a pipe's
//! llm_backend is anthropic.

use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::Context as _;

use kitetsu::listener::{
    ANTHROPIC_API_KEY, ApiConfig, ApiTranscriber, OPENAI_API_KEY, RealtimeError, RealtimeSession,
    Recorder, RecordingHandle, TranscriptEvent, discover_default_devices, load_dotenv, require,
};
use kitetsu::teleprompter::config::LlmBackendKind;
use kitetsu::teleprompter::{
    Backend, Config, History, Labels, PipeContext, Source, WindowManager, run_pipe1, run_pipe2,
};

const REALTIME_ENDPOINT: &str = "wss://api.openai.com/v1/realtime?intent=transcription";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("kitetsu=warn")),
        )
        .init();

    // ── Config + composed system prompt (prompt.md + context.md) ──────────────
    let config_path = Path::new("config.toml");
    let config =
        Config::load(config_path).with_context(|| format!("loading {}", config_path.display()))?;
    let base_dir = config_path.parent().unwrap_or_else(|| Path::new("."));
    let system_prompt = config
        .load_system_prompt(base_dir)
        .context("loading prompt.md / context.md (copy the .example files)")?;

    // ── Keys + backend ────────────────────────────────────────────────────────
    if let Ok(cwd) = std::env::current_dir() {
        load_dotenv(&cwd.join(".env"));
    }
    let openai_key =
        require(OPENAI_API_KEY).context("OPENAI_API_KEY is needed for the Realtime WS + REST")?;
    let make_backend = |kind: LlmBackendKind| -> anyhow::Result<Backend> {
        let key = match kind {
            LlmBackendKind::Openai => openai_key.clone(),
            LlmBackendKind::Anthropic => {
                require(ANTHROPIC_API_KEY).context("llm_backend = anthropic needs the key")?
            }
        };
        Backend::new(kind, key).context("building LLM backend")
    };

    // pipe1 (live) + pipe2 (REST re-transcribe) backends, models, histories.
    let p1_backend = make_backend(config.pipe1.llm_backend)?;
    let p1_model = config.pipe1.llm_model.clone();
    let p1_history: History = Arc::new(Mutex::new(Vec::new()));

    let p2_backend = make_backend(config.pipe2.llm_backend)?;
    let p2_model = config.pipe2.llm_model.clone();
    let p2_history: History = Arc::new(Mutex::new(Vec::new()));
    let transcriber = ApiTranscriber::new(
        openai_key.clone(),
        ApiConfig {
            model: config.pipe2.stt_model.clone(),
            language: config.pipe2.stt_language.clone(),
            ..ApiConfig::default()
        },
    )
    .context("building pipe2 REST transcriber")?;

    let labels = Labels {
        mic: config.audio.mic_label.clone(),
        system: config.audio.system_label.clone(),
    };

    // ── Capture + live WS per source → shared window ──────────────────────────
    let windows = Arc::new(Mutex::new(WindowManager::new(config.audio.max_window_secs)));
    let devices = discover_default_devices().context("PulseAudio device discovery failed")?;

    let mut handles: Vec<RecordingHandle> = Vec::new();
    if config.audio.mic {
        let device = config
            .audio
            .mic_source
            .clone()
            .unwrap_or(devices.mic_source.clone());
        eprintln!("mic:    {device}");
        handles.push(spawn_source(Source::Mic, device, "telep-mic", &windows, &openai_key, &config).await?);
    }
    if config.audio.system {
        let device = config
            .audio
            .system_source
            .clone()
            .unwrap_or(devices.system_monitor.clone());
        eprintln!("system: {device}");
        handles.push(spawn_source(Source::Sys, device, "telep-sys", &windows, &openai_key, &config).await?);
    }

    println!("\nListening. Talk, then press Enter for a suggestion. Ctrl-D to quit.");

    // ── Event loop: Enter → snapshot → pipe1 → print, recording continues ─────
    loop {
        let line = tokio::task::spawn_blocking(|| {
            use std::io::BufRead as _;
            let mut s = String::new();
            match std::io::stdin().lock().read_line(&mut s) {
                Ok(0) => None, // EOF (Ctrl-D)
                Ok(_) => Some(s),
                Err(_) => None,
            }
        })
        .await?;
        if line.is_none() {
            break;
        }

        let window = windows
            .lock()
            .expect("window lock poisoned")
            .snapshot_and_reset();
        println!(
            "── window {} — mic {:.1}s/{}ch · sys {:.1}s/{}ch ──",
            window.n,
            window.mic_secs(),
            window.mic_text.len(),
            window.sys_secs(),
            window.sys_text.len(),
        );

        let p1_ctx = PipeContext {
            backend: &p1_backend,
            system_prompt: &system_prompt,
            labels: &labels,
            out_dir: config.output.dir.as_path(),
            mode: config.output.mode,
        };
        let p2_ctx = PipeContext {
            backend: &p2_backend,
            system_prompt: &system_prompt,
            labels: &labels,
            out_dir: config.output.dir.as_path(),
            mode: config.output.mode,
        };

        // Fire both pipes in parallel; pipe1 (live) usually returns first.
        let (r1, r2) = tokio::join!(
            run_pipe1(&window, &p1_history, &p1_model, &p1_ctx),
            run_pipe2(&window, &p2_history, &transcriber, &p2_model, &p2_ctx),
        );
        match r1 {
            Ok(reply) => println!("\n[pipe1 live ] >>> {reply}"),
            Err(e) => eprintln!("\n[pipe1 live ] error: {e}"),
        }
        match r2 {
            Ok(reply) => println!("[pipe2 chunk] >>> {reply}\n"),
            Err(e) => eprintln!("[pipe2 chunk] error: {e}\n"),
        }
    }

    println!("\nStopping...");
    for handle in handles {
        let _ = tokio::task::spawn_blocking(move || handle.stop()).await;
    }
    Ok(())
}

/// Start streaming capture for one source: append every chunk to the shared
/// window's raw buffer and feed a Realtime session whose completed utterances
/// accumulate as the window's live text.
async fn spawn_source(
    source: Source,
    device: String,
    label: &'static str,
    windows: &Arc<Mutex<WindowManager>>,
    api_key: &str,
    config: &Config,
) -> anyhow::Result<RecordingHandle> {
    let (handle, std_rx) = Recorder::new(device, label).start_streaming();

    let session_update = serde_json::json!({
        "type": "session.update",
        "session": {
            "type": "transcription",
            "audio": { "input": {
                "format": { "type": "audio/pcm", "rate": 24000 },
                "transcription": {
                    "model": config.pipe1.stt_model,
                    "language": config.pipe1.stt_language,
                    "delay": "high",
                }
            }}
        }
    });
    let session = RealtimeSession::connect(api_key, REALTIME_ENDPOINT, &session_update)
        .await
        .with_context(|| format!("connecting {label} realtime session"))?;
    let (mut sink, mut events) = session.split();

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Vec<f32>>();
    let commit_every = config.pipe1.commit_every_chunks.max(1);

    // Feed task: bridged chunks → WS sink, commit periodically.
    tokio::spawn(async move {
        let mut n = 0usize;
        while let Some(chunk) = rx.recv().await {
            if sink.feed(&chunk).await.is_err() {
                break;
            }
            n += 1;
            if n % commit_every == 0 && sink.commit().await.is_err() {
                break;
            }
        }
        sink.close().await;
    });

    // Event task: completed utterances → window live text.
    let win = Arc::clone(windows);
    tokio::spawn(async move {
        loop {
            match events.next_event().await {
                Ok(Some(TranscriptEvent::Completed(text))) => {
                    win.lock()
                        .expect("window lock poisoned")
                        .append_text(source, &text);
                }
                Ok(_) => {}
                Err(RealtimeError::Closed) => break,
                Err(_) => break,
            }
        }
    });

    // Bridge: blocking capture channel → raw buffer (+ forward to WS feed).
    let win = Arc::clone(windows);
    tokio::task::spawn_blocking(move || {
        while let Ok(chunk) = std_rx.recv() {
            win.lock()
                .expect("window lock poisoned")
                .append_raw(source, &chunk);
            let _ = tx.send(chunk);
        }
    });

    Ok(handle)
}
