//! Pipe 1 (live) end-to-end, interactive — the daemon's pipe1 path driven by the
//! keyboard instead of the control socket.
//!
//! Continuously captures mic + system audio and live-transcribes both over the
//! Realtime WS while you talk; a recording-seconds counter ticks the whole time.
//! Each time you press Enter, the current window (everything heard since the last
//! press) is snapshotted: the accumulated live transcript — mic and system shown
//! separately — is printed as exactly what gets sent to the LLM, then the LLM's
//! suggestion is printed and appended to `out/pipe1.md`. The window then resets;
//! recording never stops. Ctrl-D quits.
//!
//! This mirrors `daemon::run`'s pipe1 wiring 1:1 (same `spawn_source`, same
//! `run_pipe1`), so what you see here is what the daemon does on `kitetsu next`.
//!
//! Usage:
//!   cargo run -p kitetsu --features teleprompter --example teleprompter_pipe1
//!
//! Requires `config.toml` with `prompt.md` + `context.md` beside it, and
//! `OPENAI_API_KEY` (Realtime WS + LLM when openai). If pipe1's `llm_backend`
//! is anthropic, `ANTHROPIC_API_KEY` too.

use std::io::Write as _;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Context as _;

use kitetsu::listener::{
    ANTHROPIC_API_KEY, OPENAI_API_KEY, RealtimeError, RealtimeSession, Recorder, RecordingHandle,
    TranscriptEvent, discover_default_devices, load_dotenv, require,
};
use kitetsu::teleprompter::config::LlmBackendKind;
use kitetsu::teleprompter::{
    Backend, Config, History, Labels, PipeContext, PipeKind, Source, WindowManager, run_pipe1,
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
    let config_path = Path::new("teleprompter/teleprompter.toml");
    let config =
        Config::load(config_path).with_context(|| format!("loading {}", config_path.display()))?;
    let base_dir = config_path.parent().unwrap_or_else(|| Path::new("."));
    let system_prompt = config
        .load_system_prompt(base_dir)
        .context("loading prompt.md / context.md (copy the .example files)")?;

    if config.pipe != PipeKind::Live {
        anyhow::bail!("set `pipe = \"live\"` in teleprompter.toml to run this example");
    }

    // ── Keys + backend (same selection the daemon makes) ──────────────────────
    if let Ok(cwd) = std::env::current_dir() {
        load_dotenv(&cwd.join(".env"));
    }
    let openai_key =
        require(OPENAI_API_KEY).context("OPENAI_API_KEY is needed for the Realtime WS")?;
    let backend = match config.live.llm_backend {
        LlmBackendKind::Openai => Backend::new(LlmBackendKind::Openai, openai_key.clone()),
        LlmBackendKind::Anthropic => Backend::new(
            LlmBackendKind::Anthropic,
            require(ANTHROPIC_API_KEY).context("live.llm_backend = anthropic needs the key")?,
        ),
    }
    .context("building pipe1 LLM backend")?;
    let model = config.live.llm_model.clone();
    let history: History = Arc::new(Mutex::new(Vec::new()));

    let labels = Labels {
        mic: config.audio.mic_label.clone(),
        system: config.audio.system_label.clone(),
    };

    // ── Capture + live WS per source → shared window ──────────────────────────
    let windows = Arc::new(Mutex::new(WindowManager::new(config.audio.max_window_secs)));
    let devices = discover_default_devices().context("PulseAudio device discovery failed")?;

    let mut handles: Vec<RecordingHandle> = Vec::new();
    let mut mic_dev = String::from("(off)");
    let mut sys_dev = String::from("(off)");
    if config.audio.mic {
        mic_dev = config
            .audio
            .mic_source
            .clone()
            .unwrap_or(devices.mic_source.clone());
        handles.push(spawn_source(Source::Mic, mic_dev.clone(), "telep-mic", &windows, &openai_key, &config).await?);
    }
    if config.audio.system {
        sys_dev = config
            .audio
            .system_source
            .clone()
            .unwrap_or(devices.system_monitor.clone());
        handles.push(spawn_source(Source::Sys, sys_dev.clone(), "telep-sys", &windows, &openai_key, &config).await?);
    }

    print_banner(&config, &system_prompt, &mic_dev, &sys_dev);

    // ── Event loop: counter ticks; Enter → show transcript → wait for the
    // response (recording keeps running) → show response → resume counter. ─────
    loop {
        let start = Instant::now();
        if !wait_for_enter(start).await? {
            break;
        }

        let window = windows
            .lock()
            .expect("window lock poisoned")
            .snapshot_and_reset();
        let recorded = start.elapsed().as_secs();

        // pipe1's transcript is the accumulated live text — known immediately, so
        // show what is being sent the moment Enter is pressed.
        print_transcript(window.n, recorded, &labels, window.mic_text.trim(), window.sys_text.trim());

        let ctx = PipeContext {
            backend: &backend,
            system_prompt: &system_prompt,
            labels: &labels,
        };

        // Await the LLM while the capture tasks keep recording in the background;
        // the spinner shows the wait without resuming the recording counter.
        let call = Instant::now();
        let res = await_with_spinner("waiting for response", &model, run_pipe1(&window, &history, &model, &ctx)).await;
        match res {
            Ok(outcome) => print_response("pipe1 live", &model, call.elapsed(), &outcome.reply),
            Err(e) => eprintln!("  pipe1 error: {e}\n"),
        }
    }

    println!("\nStopping...");
    for handle in handles {
        let _ = tokio::task::spawn_blocking(move || handle.stop()).await;
    }
    Ok(())
}

/// Print the model/config summary so you can confirm what will run before talking.
fn print_banner(config: &Config, system_prompt: &str, mic_dev: &str, sys_dev: &str) {
    println!("\n┌─ teleprompter · pipe1 (live WS transcript → LLM) ───────────");
    println!("│ config   {}", "teleprompter/teleprompter.toml");
    println!(
        "│ prompt   {} chars (prompt.md + context.md)",
        system_prompt.len()
    );
    println!(
        "│ audio    mic[{}]={}  system[{}]={}",
        config.audio.mic_label, mic_dev, config.audio.system_label, sys_dev
    );
    println!(
        "│ stt      {} (lang {})",
        config.live.stt_model,
        config.live.stt_language.as_deref().unwrap_or("auto")
    );
    println!(
        "│ llm      {:?} / {}",
        config.live.llm_backend, config.live.llm_model
    );
    println!("└─────────────────────────────────────────────────────────────");
    println!("Talk; the counter shows seconds recorded. Enter = suggestion, Ctrl-D = quit.");
}

/// Block on stdin while showing a live recording-seconds counter. Returns
/// `false` on EOF (Ctrl-D). `start` marks the beginning of the current window.
async fn wait_for_enter(start: Instant) -> anyhow::Result<bool> {
    let mut read = tokio::task::spawn_blocking(|| {
        use std::io::BufRead as _;
        let mut s = String::new();
        matches!(std::io::stdin().lock().read_line(&mut s), Ok(n) if n > 0)
    });
    let mut ticker = tokio::time::interval(Duration::from_secs(1));
    let pressed = loop {
        tokio::select! {
            r = &mut read => break r?,
            _ = ticker.tick() => {
                print!("\r🔴 recording {:>4}s — Enter = suggestion, Ctrl-D = quit ", start.elapsed().as_secs());
                let _ = std::io::stdout().flush();
            }
        }
    };
    print!("\r\x1b[K"); // clear the counter line
    let _ = std::io::stdout().flush();
    Ok(pressed)
}

/// Print the window header and the transcript being sent (mic + system separate).
fn print_transcript(window_n: u64, recorded_secs: u64, labels: &Labels, mic: &str, sys: &str) {
    println!("══ window {window_n} · recorded {recorded_secs}s ═══════════════════════════");
    println!("  ▶ SENT TO LLM (transcript)");
    println!("    [{}] {}", labels.system, show(sys));
    println!("    [{}] {}", labels.mic, show(mic));
}

/// Print the LLM response.
fn print_response(tag: &str, model: &str, latency: Duration, reply: &str) {
    println!("  ◀ RESPONSE ({tag} · {model} · +{:.1}s)", latency.as_secs_f64());
    for line in reply.lines() {
        println!("    {line}");
    }
    println!();
}

/// Await `fut` while showing a live waiting-seconds spinner that makes clear the
/// recording keeps running in the background. Clears the spinner line on finish.
async fn await_with_spinner<F: std::future::Future>(label: &str, model: &str, fut: F) -> F::Output {
    tokio::pin!(fut);
    let start = Instant::now();
    let mut ticker = tokio::time::interval(Duration::from_secs(1));
    loop {
        tokio::select! {
            out = &mut fut => {
                print!("\r\x1b[K");
                let _ = std::io::stdout().flush();
                return out;
            }
            _ = ticker.tick() => {
                print!("\r  ⏳ {label} from {model} {}s — still recording… ", start.elapsed().as_secs());
                let _ = std::io::stdout().flush();
            }
        }
    }
}

/// Render a transcript side, marking an empty one rather than printing blank.
fn show(text: &str) -> &str {
    if text.is_empty() {
        "(silence)"
    } else {
        text
    }
}

/// Start streaming capture for one source: append every chunk to the shared
/// window's raw buffer and feed a Realtime session whose completed utterances
/// accumulate as the window's live text. Mirrors `daemon::spawn_source`.
async fn spawn_source(
    source: Source,
    device: String,
    label: &'static str,
    windows: &Arc<Mutex<WindowManager>>,
    api_key: &str,
    config: &Config,
) -> anyhow::Result<RecordingHandle> {
    let (handle, std_rx) = Recorder::new(device, label).start_streaming();

    let mut transcription = serde_json::json!({
        "model": config.live.stt_model,
        "delay": "high",
    });
    if let Some(lang) = &config.live.stt_language {
        transcription["language"] = serde_json::Value::String(lang.clone());
    }
    let session_update = serde_json::json!({
        "type": "session.update",
        "session": {
            "type": "transcription",
            "audio": { "input": {
                "format": { "type": "audio/pcm", "rate": 24000 },
                "transcription": transcription,
            }}
        }
    });
    let session = RealtimeSession::connect(api_key, REALTIME_ENDPOINT, &session_update)
        .await
        .with_context(|| format!("connecting {label} realtime session"))?;
    let (mut sink, mut events) = session.split();

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Vec<f32>>();
    let commit_every = config.live.commit_every_chunks.max(1);

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
