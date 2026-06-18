//! The long-lived teleprompter daemon: audio capture, the rolling window, and
//! the control-socket accept loop.
//!
//! Each enabled source (mic, system) runs a streaming `Recorder`; every captured
//! chunk is appended to the shared [`WindowManager`]'s raw buffer (for the
//! on-trigger pipes) and, when pipe1 is enabled, fed to a `RealtimeSession` whose
//! completed utterances accumulate as live text. On `Next`, the window is
//! snapshotted and reset, then the enabled pipes run in the background: pipe1
//! (live transcript) and pipe2 (REST re-transcription). Not here: the pipe bodies
//! themselves (`pipes`), the LLM client (`llm`), or output writing (`output`).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use tracing::{info, warn};

use crate::listener::{
    ANTHROPIC_API_KEY, ApiConfig, ApiTranscriber, KeyError, ListenerError, OPENAI_API_KEY,
    RealtimeError, RealtimeSession, Recorder, RecordingHandle, TranscriptEvent,
    discover_default_devices, load_dotenv, require,
};

use tokio::sync::mpsc::UnboundedSender;

use super::config::{Config, LiveConfig, LlmBackendKind, PipeKind};
use super::ipc::{self, Command, GlobalAction, IpcError, TeleprompterAction};
use super::llm::{Backend, LlmError};
use super::pipes::{History, Labels, PipeContext, run_pipe1, run_pipe2};
use super::ui::{Stage, UiEvent};
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
    /// The LLM backend could not be constructed (e.g. missing key).
    #[error("LLM backend: {0}")]
    Llm(#[source] LlmError),
}

/// Owned, cheaply-cloneable pipe-1 runtime shared into each spawned trigger task.
#[derive(Clone)]
struct Pipe1Runtime {
    backend: Arc<Backend>,
    model: String,
    system_prompt: Arc<str>,
    labels: Arc<Labels>,
    history: History,
}

/// Owned, cheaply-cloneable pipe-2 runtime (adds the REST transcriber over the
/// pipe-1 shape).
#[derive(Clone)]
struct Pipe2Runtime {
    transcriber: Arc<ApiTranscriber>,
    backend: Arc<Backend>,
    model: String,
    system_prompt: Arc<str>,
    labels: Arc<Labels>,
    history: History,
}

/// Bind the control socket, start capture, and serve commands until `Stop`.
///
/// `config` and `system_prompt` are loaded once by the caller and held for the
/// daemon's lifetime. Commands are handled one connection at a time — the control
/// plane is low-traffic (human key presses), so sequential handling keeps
/// ordering obvious and needs no shared locking on the socket.
///
/// `ui_tx` carries pipe replies to the overlay; the pipe-dispatch sites start
/// sending on it in the next iteration (it is the structural seam for now).
pub async fn run(
    config: Config,
    system_prompt: String,
    ui_tx: UnboundedSender<UiEvent>,
) -> Result<(), DaemonError> {
    log_summary(&config, &system_prompt);

    // OpenAI authenticates both the pipe1 WS and the pipe2 REST transcription;
    // load `.env` from the working directory first so a key file there is picked up.
    if let Ok(cwd) = std::env::current_dir() {
        load_dotenv(&cwd.join(".env"));
    }
    // Either pipe needs the OpenAI key for transcription (live WS / chunk REST).
    let openai_key = Some(require(OPENAI_API_KEY).map_err(DaemonError::Key)?);
    // Only the live pipe opens the realtime WS; chunk only needs the key for REST.
    let ws_key = if config.pipe == PipeKind::Live {
        openai_key.as_deref()
    } else {
        None
    };

    let windows = Arc::new(Mutex::new(WindowManager::new(config.audio.max_window_secs)));
    // Recording gate: while paused, capture chunks are dropped (the already-
    // accumulated window is kept). Shared with each source's capture bridge.
    let paused = Arc::new(AtomicBool::new(false));

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
            Arc::clone(&paused),
            ws_key,
            &config.live,
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
            Arc::clone(&paused),
            ws_key,
            &config.live,
            &mut tasks,
        )
        .await?;
        handles.push(handle);
    }

    // Shared inputs for the LLM pipes (built once).
    let system_prompt: Arc<str> = Arc::from(system_prompt.as_str());
    let labels = Arc::new(Labels {
        mic: config.audio.mic_label.clone(),
        system: config.audio.system_label.clone(),
    });

    // Build the pipe-1 runtime once (its LLM key may differ from the OpenAI key
    // when the configured llm_backend is anthropic).
    let pipe1_rt = if config.pipe == PipeKind::Live {
        let backend = build_backend(
            config.live.llm_backend,
            openai_key.as_deref(),
        )?;
        Some(Pipe1Runtime {
            backend: Arc::new(backend),
            model: config.live.llm_model.clone(),
            system_prompt: Arc::clone(&system_prompt),
            labels: Arc::clone(&labels),
            history: Arc::new(Mutex::new(Vec::new())),
        })
    } else {
        None
    };

    // Build the pipe-2 runtime: a REST transcriber (OpenAI) plus its LLM backend.
    let pipe2_rt = if config.pipe == PipeKind::Chunk {
        let key = openai_key
            .clone()
            .expect("OpenAI key always loaded");
        let transcriber = ApiTranscriber::new(
            key,
            ApiConfig {
                model: config.chunk.stt_model.clone(),
                language: config.chunk.stt_language.clone(),
                ..ApiConfig::default()
            },
        )
        .map_err(|_| DaemonError::Key(KeyError::Missing(OPENAI_API_KEY)))?;
        let backend = build_backend(config.chunk.llm_backend, openai_key.as_deref())?;
        Some(Pipe2Runtime {
            transcriber: Arc::new(transcriber),
            backend: Arc::new(backend),
            model: config.chunk.llm_model.clone(),
            system_prompt: Arc::clone(&system_prompt),
            labels: Arc::clone(&labels),
            history: Arc::new(Mutex::new(Vec::new())),
        })
    } else {
        None
    };

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
            Command::Kitetsu(GlobalAction::Ping) => {
                let _ = ipc::write_ack(&mut stream, "pong").await;
            }
            Command::Teleprompter(TeleprompterAction::Process) => {
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
                    "window snapshot",
                );
                let ack = format!("queued window {}", window.n);
                let _ = ipc::write_ack(&mut stream, &ack).await;

                // Fire the selected pipe in the background; the ack is already sent
                // so the client returns immediately while the suggestion lands on
                // the overlay card.
                let window = Arc::new(window);

                if let Some(rt) = &pipe1_rt {
                    let rt = rt.clone();
                    let window = Arc::clone(&window);
                    let ui_tx = ui_tx.clone();
                    tokio::spawn(async move {
                        let ctx = PipeContext {
                            backend: rt.backend.as_ref(),
                            system_prompt: rt.system_prompt.as_ref(),
                            labels: rt.labels.as_ref(),
                        };
                        // pipe1's transcript is already live, so it goes straight
                        // to the LLM: show "Fetching response…".
                        let _ = ui_tx.send(UiEvent::Status {
                            stage: Some(Stage::Fetching),
                        });
                        match run_pipe1(&window, &rt.history, &rt.model, &ctx).await {
                            Ok(outcome) => {
                                let chars = outcome.reply.len();
                                // Best-effort: if the overlay is gone the send just fails.
                                let _ = ui_tx.send(UiEvent::Reply {
                                    text: outcome.reply,
                                });
                                info!(window = window.n, chars, "pipe1 suggestion written");
                            }
                            Err(e) => {
                                // Clear the status so the header returns to its baseline.
                                let _ = ui_tx.send(UiEvent::Status { stage: None });
                                warn!(window = window.n, error = %e, "pipe1 failed");
                            }
                        }
                    });
                }

                if let Some(rt) = &pipe2_rt {
                    let rt = rt.clone();
                    let window = Arc::clone(&window);
                    let ui_tx = ui_tx.clone();
                    tokio::spawn(async move {
                        let ctx = PipeContext {
                            backend: rt.backend.as_ref(),
                            system_prompt: rt.system_prompt.as_ref(),
                            labels: rt.labels.as_ref(),
                        };
                        // pipe2 re-transcribes first, then calls the LLM: show
                        // "Transcribing…" now, "Fetching response…" once that's done.
                        let _ = ui_tx.send(UiEvent::Status {
                            stage: Some(Stage::Transcribing),
                        });
                        let on_transcribed = {
                            let ui_tx = ui_tx.clone();
                            move || {
                                let _ = ui_tx.send(UiEvent::Status {
                                    stage: Some(Stage::Fetching),
                                });
                            }
                        };
                        match run_pipe2(
                            &window,
                            &rt.history,
                            &rt.transcriber,
                            &rt.model,
                            &ctx,
                            on_transcribed,
                        )
                        .await
                        {
                            Ok(outcome) => {
                                let chars = outcome.reply.len();
                                let _ = ui_tx.send(UiEvent::Reply {
                                    text: outcome.reply,
                                });
                                info!(window = window.n, chars, "pipe2 suggestion written");
                            }
                            Err(e) => {
                                let _ = ui_tx.send(UiEvent::Status { stage: None });
                                warn!(window = window.n, error = %e, "pipe2 failed");
                            }
                        }
                    });
                }
            }
            Command::Teleprompter(TeleprompterAction::Trash) => {
                windows
                    .lock()
                    .expect("window lock poisoned")
                    .clear();
                info!("conversation trashed — window cleared, recording afresh");
                let _ = ui_tx.send(UiEvent::Trashed);
                // End the title flash after 3s (rides this runtime — no UI timer).
                let ui_tx = ui_tx.clone();
                tokio::spawn(async move {
                    tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                    let _ = ui_tx.send(UiEvent::Untrash);
                });
                let _ = ipc::write_ack(&mut stream, "trashed").await;
            }
            Command::Teleprompter(TeleprompterAction::Pause) => {
                // fetch_xor flips the flag and returns its previous value, so after
                // the flip we are recording iff we were paused before.
                let recording = paused.fetch_xor(true, Ordering::Relaxed);
                let _ = ui_tx.send(UiEvent::Recording(recording));
                let ack = if recording { "resumed" } else { "paused" };
                info!(recording, "pause toggled");
                let _ = ipc::write_ack(&mut stream, ack).await;
            }
            Command::Teleprompter(TeleprompterAction::Toggle) => {
                let _ = ui_tx.send(UiEvent::Toggle);
                let _ = ipc::write_ack(&mut stream, "toggled").await;
            }
            // History nav is purely presentational: the overlay owns the reply
            // list, so the daemon just relays the keypress.
            Command::Teleprompter(TeleprompterAction::Forward) => {
                let _ = ui_tx.send(UiEvent::Forward);
                let _ = ipc::write_ack(&mut stream, "forward").await;
            }
            Command::Teleprompter(TeleprompterAction::Backward) => {
                let _ = ui_tx.send(UiEvent::Backward);
                let _ = ipc::write_ack(&mut stream, "backward").await;
            }
            Command::Kitetsu(GlobalAction::Stop) => {
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
    paused: Arc<AtomicBool>,
    api_key: Option<&str>,
    live: &LiveConfig,
    tasks: &mut Vec<tokio::task::JoinHandle<()>>,
) -> Result<RecordingHandle, DaemonError> {
    let (handle, std_rx) = Recorder::new(device, label).start_streaming();

    // Optional live WS path: a sender that the capture bridge forwards chunks to.
    let ws_tx = if let Some(key) = api_key {
        let session_update = build_session_update(live);
        let session = RealtimeSession::connect(key, REALTIME_ENDPOINT, &session_update)
            .await
            .map_err(DaemonError::Realtime)?;
        let (mut sink, mut event_stream) = session.split();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Vec<f32>>();
        let commit_every = live.commit_every_chunks.max(1);

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
            // Paused: drop the chunk so nothing is appended, but keep the window
            // intact (resume continues accumulating; only `trash` clears it).
            if paused.load(Ordering::Relaxed) {
                continue;
            }
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

/// Build the Realtime `session.update` JSON for the live pipe from its config.
///
/// `gpt-realtime-whisper` requires PCM input and manual commits (no server VAD),
/// so `turn_detection` is omitted; the sink resamples 16 kHz capture to the 24
/// kHz the API expects.
fn build_session_update(live: &LiveConfig) -> serde_json::Value {
    let mut transcription = serde_json::json!({
        "model": live.stt_model,
        "delay": "high",
    });
    if let Some(lang) = &live.stt_language {
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

/// Build an LLM backend for the given kind, reusing the already-loaded OpenAI key
/// for the OpenAI backend and requiring the Anthropic key otherwise.
fn build_backend(kind: LlmBackendKind, openai_key: Option<&str>) -> Result<Backend, DaemonError> {
    let key = match kind {
        LlmBackendKind::Openai => openai_key
            .map(str::to_owned)
            .ok_or(DaemonError::Key(KeyError::Missing(OPENAI_API_KEY)))?,
        LlmBackendKind::Anthropic => require(ANTHROPIC_API_KEY).map_err(DaemonError::Key)?,
    };
    Backend::new(kind, key).map_err(DaemonError::Llm)
}

/// Log the resolved pipes, models, and prompt size at startup so the operator
/// can confirm what the daemon will do before speaking a word.
fn log_summary(config: &Config, system_prompt: &str) {
    info!(
        mic = config.audio.mic,
        system = config.audio.system,
        max_window_secs = config.audio.max_window_secs,
        system_prompt_chars = system_prompt.len(),
        "config loaded",
    );
    match config.pipe {
        PipeKind::Live => info!(
            stt = %config.live.stt_model,
            llm_backend = ?config.live.llm_backend,
            llm = %config.live.llm_model,
            "live pipe selected",
        ),
        PipeKind::Chunk => info!(
            stt = %config.chunk.stt_model,
            llm_backend = ?config.chunk.llm_backend,
            llm = %config.chunk.llm_model,
            "chunk pipe selected",
        ),
    }
}
