//! The long-lived teleprompter daemon: audio capture, the rolling window, and
//! the control-socket accept loop.
//!
//! Each enabled source (mic, system) runs a streaming `Recorder`; every captured
//! chunk is appended to the shared [`WindowManager`]'s raw buffer (for the
//! on-trigger pipes) and, when pipe1 is enabled, fed to a `RealtimeSession` whose
//! completed utterances accumulate as live text. On `Next`, the window is
//! snapshotted and reset; the pipes that consume it are wired in later iterations
//! — for now the daemon logs the window's size. Not here: the pipes, the LLM
//! client, or output writing.

use std::sync::{Arc, Mutex};

use tracing::{info, warn};

use crate::listener::{
    KeyError, ListenerError, OPENAI_API_KEY, RealtimeError, RealtimeSession, Recorder,
    RecordingHandle, TranscriptEvent, discover_default_devices, load_dotenv, require,
};

use super::config::{Config, Pipe1Config};
use super::ipc::{self, Command, IpcError};
use super::window::{Source, WindowManager};

/// OpenAI Realtime transcription endpoint (pipe1 live path).
const REALTIME_ENDPOINT: &str = "wss://api.openai.com/v1/realtime?intent=transcription";

/// Failures that terminate the daemon.
#[derive(Debug, thiserror::Error)]
pub enum DaemonError {
    /// Control-plane transport failure (bind or accept).
    #[error(transparent)]
    Ipc(#[from] IpcError),
    /// Audio device discovery or capture setup failed.
    #[error("audio capture: {0}")]
    Capture(#[source] ListenerError),
    /// The OpenAI key needed for live (pipe1) transcription was missing.
    #[error("missing API key for live transcription: {0}")]
    Key(#[source] KeyError),
    /// A live-transcription WebSocket failed to connect at startup.
    #[error("realtime session connect: {0}")]
    Realtime(#[source] RealtimeError),
}

/// Bind the control socket, start capture, and serve commands until `Stop`.
///
/// `config` and `system_prompt` are loaded once by the caller and held for the
/// daemon's lifetime. Commands are handled one connection at a time — the control
/// plane is low-traffic (human key presses), so sequential handling keeps
/// ordering obvious and needs no shared locking on the socket.
pub async fn run(config: Config, system_prompt: String) -> Result<(), DaemonError> {
    log_summary(&config, &system_prompt);

    // The live (pipe1) WS path authenticates with the OpenAI key; load `.env`
    // from the working directory first so a key file there is picked up.
    if let Ok(cwd) = std::env::current_dir() {
        load_dotenv(&cwd.join(".env"));
    }
    let api_key = if config.pipe1.enabled {
        Some(require(OPENAI_API_KEY).map_err(DaemonError::Key)?)
    } else {
        None
    };

    let windows = Arc::new(Mutex::new(WindowManager::new(config.audio.max_window_secs)));

    // Resolve devices: an explicit config source wins; otherwise auto-discover.
    let need_discovery = (config.audio.mic && config.audio.mic_source.is_none())
        || (config.audio.system && config.audio.system_source.is_none());
    let discovered = if need_discovery {
        Some(discover_default_devices().map_err(DaemonError::Capture)?)
    } else {
        None
    };

    let mut handles: Vec<RecordingHandle> = Vec::new();
    let mut tasks: Vec<tokio::task::JoinHandle<()>> = Vec::new();

    if config.audio.mic {
        let device = config
            .audio
            .mic_source
            .clone()
            .or_else(|| discovered.as_ref().map(|d| d.mic_source.clone()))
            .expect("mic device resolved when audio.mic is enabled");
        info!(source = "mic", %device, "starting capture");
        let handle = spawn_source(
            Source::Mic,
            device,
            "telep-mic",
            Arc::clone(&windows),
            api_key.as_deref(),
            &config.pipe1,
            &mut tasks,
        )
        .await?;
        handles.push(handle);
    }
    if config.audio.system {
        let device = config
            .audio
            .system_source
            .clone()
            .or_else(|| discovered.as_ref().map(|d| d.system_monitor.clone()))
            .expect("system device resolved when audio.system is enabled");
        info!(source = "system", %device, "starting capture");
        let handle = spawn_source(
            Source::Sys,
            device,
            "telep-sys",
            Arc::clone(&windows),
            api_key.as_deref(),
            &config.pipe1,
            &mut tasks,
        )
        .await?;
        handles.push(handle);
    }

    let listener = ipc::bind()?;
    info!(socket = %ipc::socket_path().display(), "teleprompter daemon listening");

    loop {
        let (mut stream, _) = listener.accept().await.map_err(IpcError::Io)?;

        let cmd = match ipc::read_command(&mut stream).await {
            Ok(Some(cmd)) => cmd,
            Ok(None) => continue,
            Err(e) => {
                warn!(error = %e, "ignoring malformed control message");
                continue;
            }
        };

        match cmd {
            Command::Ping => {
                let _ = ipc::write_ack(&mut stream, "pong").await;
            }
            Command::Next => {
                let window = windows
                    .lock()
                    .expect("window lock poisoned")
                    .snapshot_and_reset();
                info!(
                    window = window.n,
                    mic_secs = format!("{:.1}", window.mic_secs()),
                    mic_chars = window.mic_text.len(),
                    sys_secs = format!("{:.1}", window.sys_secs()),
                    sys_chars = window.sys_text.len(),
                    "window snapshot (pipes not wired yet)",
                );
                let ack = format!("queued window {}", window.n);
                let _ = ipc::write_ack(&mut stream, &ack).await;
            }
            Command::Stop => {
                info!("stop received — shutting down");
                let _ = ipc::write_ack(&mut stream, "stopping").await;
                break;
            }
        }
    }

    // Stop capture: each `stop` joins its libpulse thread (~256 ms), which closes
    // the std channel, ending the bridge task, which drops the WS feed sender and
    // unwinds the feed/event tasks. Run the blocking joins off the runtime.
    for handle in handles {
        let _ = tokio::task::spawn_blocking(move || handle.stop()).await;
    }
    for task in tasks {
        task.abort();
    }

    ipc::cleanup();
    Ok(())
}

/// Wire one audio source: start streaming capture, append every chunk to the
/// shared window's raw buffer, and (when pipe1 is enabled) feed a Realtime
/// session whose completed utterances accumulate as live text.
///
/// Returns the capture handle so the caller can stop it on shutdown. Spawned
/// tasks are pushed into `tasks`.
async fn spawn_source(
    source: Source,
    device: String,
    label: &'static str,
    windows: Arc<Mutex<WindowManager>>,
    api_key: Option<&str>,
    pipe1: &Pipe1Config,
    tasks: &mut Vec<tokio::task::JoinHandle<()>>,
) -> Result<RecordingHandle, DaemonError> {
    let (handle, std_rx) = Recorder::new(device, label).start_streaming();

    // Optional live WS path: a sender that the capture bridge forwards chunks to.
    let ws_tx = if let Some(key) = api_key {
        let session_update = build_session_update(pipe1);
        let session = RealtimeSession::connect(key, REALTIME_ENDPOINT, &session_update)
            .await
            .map_err(DaemonError::Realtime)?;
        let (mut sink, mut event_stream) = session.split();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Vec<f32>>();
        let commit_every = pipe1.commit_every_chunks.max(1);

        // Feed task: pull bridged chunks → WS sink, committing periodically
        // (gpt-realtime-whisper has no server VAD).
        tasks.push(tokio::spawn(async move {
            let mut n = 0usize;
            while let Some(chunk) = rx.recv().await {
                if let Err(e) = sink.feed(&chunk).await {
                    warn!(?source, error = %e, "ws feed error");
                    break;
                }
                n += 1;
                if n % commit_every == 0 {
                    if let Err(e) = sink.commit().await {
                        warn!(?source, error = %e, "ws commit error");
                        break;
                    }
                }
            }
            sink.close().await;
        }));

        // Event task: completed utterances accumulate into the window's text.
        let win = Arc::clone(&windows);
        tasks.push(tokio::spawn(async move {
            loop {
                match event_stream.next_event().await {
                    Ok(Some(TranscriptEvent::Completed(text))) => {
                        win.lock()
                            .expect("window lock poisoned")
                            .append_text(source, &text);
                    }
                    Ok(_) => {}
                    Err(RealtimeError::Closed) => break,
                    Err(e) => {
                        warn!(?source, error = %e, "ws stream error");
                        break;
                    }
                }
            }
        }));
        Some(tx)
    } else {
        None
    };

    // Bridge: drain the blocking capture channel → append raw (+ forward to WS).
    // libpulse runs on a std::thread, so this consumer is a blocking task.
    let win = Arc::clone(&windows);
    tasks.push(tokio::task::spawn_blocking(move || {
        while let Ok(chunk) = std_rx.recv() {
            win.lock()
                .expect("window lock poisoned")
                .append_raw(source, &chunk);
            if let Some(ref tx) = ws_tx {
                // If the feed task is gone we still keep buffering raw for pipe2/3.
                let _ = tx.send(chunk);
            }
        }
    }));

    Ok(handle)
}

/// Build the Realtime `session.update` JSON for pipe1 from its config.
///
/// `gpt-realtime-whisper` requires PCM input and manual commits (no server VAD),
/// so `turn_detection` is omitted; the sink resamples 16 kHz capture to the 24
/// kHz the API expects.
fn build_session_update(pipe1: &Pipe1Config) -> serde_json::Value {
    let mut transcription = serde_json::json!({
        "model": pipe1.stt_model,
        "delay": "high",
    });
    if let Some(lang) = &pipe1.stt_language {
        transcription["language"] = serde_json::Value::String(lang.clone());
    }
    serde_json::json!({
        "type": "session.update",
        "session": {
            "type": "transcription",
            "audio": {
                "input": {
                    "format": { "type": "audio/pcm", "rate": 24000 },
                    "transcription": transcription,
                }
            }
        }
    })
}

/// Log the resolved pipes, models, and prompt size at startup so the operator
/// can confirm what the daemon will do before speaking a word.
fn log_summary(config: &Config, system_prompt: &str) {
    info!(
        mic = config.audio.mic,
        system = config.audio.system,
        max_window_secs = config.audio.max_window_secs,
        out_dir = %config.output.dir.display(),
        out_mode = ?config.output.mode,
        system_prompt_chars = system_prompt.len(),
        "config loaded",
    );
    if config.pipe1.enabled {
        info!(
            stt = %config.pipe1.stt_model,
            llm_backend = ?config.pipe1.llm_backend,
            llm = %config.pipe1.llm_model,
            "pipe1 (live) enabled",
        );
    }
    if config.pipe2.enabled {
        info!(
            stt = %config.pipe2.stt_model,
            llm_backend = ?config.pipe2.llm_backend,
            llm = %config.pipe2.llm_model,
            "pipe2 (chunk) enabled",
        );
    }
    if config.pipe3.enabled {
        info!(
            llm_backend = ?config.pipe3.llm_backend,
            llm = %config.pipe3.llm_model,
            "pipe3 (audio) enabled",
        );
    }
}
