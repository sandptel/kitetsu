//! Pipe 1 end-to-end, interactive: continuously capture mic + system audio and
//! live-transcribe both over the Realtime WS while you talk. Each time you press
//! Enter, the current window (everything heard since the last press) is sent to
//! the LLM with prompt.md + context.md, the suggestion is printed and appended to
//! `out/pipe1.md`, and the window resets — recording never stops.
//!
//! This is the daemon's pipe-1 path driven by the keyboard instead of the control
//! socket, so you can watch the live → LLM → file flow in one terminal.
//!
//! Usage:
//!   cargo run -p kitetsu --features teleprompter --example teleprompter_pipe1
//!
//! Requires prompt.md + context.md next to config.toml, `OPENAI_API_KEY` (Realtime
//! WS + pipe1 LLM when openai), and `ANTHROPIC_API_KEY` if pipe1 uses anthropic.

use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::Context as _;

use kitetsu::listener::{
    ANTHROPIC_API_KEY, OPENAI_API_KEY, RealtimeError, RealtimeSession, Recorder, RecordingHandle,
    TranscriptEvent, discover_default_devices, load_dotenv, require,
};
use kitetsu::teleprompter::config::LlmBackendKind;
use kitetsu::teleprompter::{
    Backend, Config, History, Labels, PipeContext, Source, WindowManager, run_pipe1,
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
        require(OPENAI_API_KEY).context("OPENAI_API_KEY is needed for the Realtime WS")?;
    let llm_key = match config.pipe1.llm_backend {
        LlmBackendKind::Openai => openai_key.clone(),
        LlmBackendKind::Anthropic => {
            require(ANTHROPIC_API_KEY).context("pipe1.llm_backend = anthropic needs the key")?
        }
    };
    let backend = Backend::new(config.pipe1.llm_backend, llm_key).context("building LLM backend")?;
    let model = config.pipe1.llm_model.clone();
    let labels = Labels {
        mic: config.audio.mic_label.clone(),
        system: config.audio.system_label.clone(),
    };
    let history: History = Arc::new(Mutex::new(Vec::new()));

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

        let ctx = PipeContext {
            backend: &backend,
            system_prompt: &system_prompt,
            labels: &labels,
            out_dir: config.output.dir.as_path(),
            mode: config.output.mode,
        };
        match run_pipe1(&window, &history, &model, &ctx).await {
            Ok(reply) => println!("\n>>> {reply}\n"),
            Err(e) => eprintln!("pipe1 error: {e}\n"),
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
