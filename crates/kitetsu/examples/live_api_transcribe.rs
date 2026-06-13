//! Stream mic + system audio to the OpenAI Realtime transcription WebSocket
//! and print each completed utterance as it arrives from both sources.
//!
//! The session configuration JSON is defined at the top of `main` — edit it
//! directly to change the model, language, VAD behaviour, etc. without digging
//! into library code. Reference:
//!   https://developers.openai.com/api/docs/guides/realtime-transcription
//!
//! Usage:
//!   cargo run -p kitetsu --features api --example live_api_transcribe
//!
//! Key discovery (first match wins):
//!   1. `OPENAI_API_KEY` in the shell environment.
//!   2. `.env` at the workspace root (cwd when invoked via `cargo run`).

use std::io::{BufRead as _, Write as _};

use anyhow::Context as _;

use kitetsu::listener::{
    Devices, RealtimeError, RealtimeSession, Recorder, TranscriptEvent,
    OPENAI_API_KEY, discover_default_devices, load_dotenv, require,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("kitetsu=info")),
        )
        .init();

    // ── Load .env before any key lookups ──────────────────────────────────────
    if let Ok(cwd) = std::env::current_dir() {
        load_dotenv(&cwd.join(".env"));
    }

    let api_key = require(OPENAI_API_KEY)
        .context("set OPENAI_API_KEY in the environment or .env file")?;

    // ── Realtime API endpoint ─────────────────────────────────────────────────
    let endpoint = "wss://api.openai.com/v1/realtime?intent=transcription";

    // ── Session configuration — edit here ────────────────────────────────────
    // Full field reference: https://developers.openai.com/api/docs/guides/realtime-transcription
    //
    // gpt-realtime-whisper does NOT support server_vad — audio must be committed
    // manually (see COMMIT_EVERY_CHUNKS below). Omit turn_detection or set to null.
    let session_update = serde_json::json!({
        "type": "session.update",
        "session": {
            "type": "transcription",
            "audio": {
                "input": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": 24000
                    },
                    "transcription": {
                        "model": "gpt-realtime-whisper",
                        "language": "en"
                        // "delay": "low"  // minimal | low | medium | high | xhigh
                    }
                }
            }
        }
    });

    // Commit the audio buffer every N chunks (~256 ms each → 2 s per commit at 8).
    // gpt-realtime-whisper requires manual commits; adjust for latency vs accuracy.
    const COMMIT_EVERY_CHUNKS: usize = 8;

    // ── Discover devices ──────────────────────────────────────────────────────
    println!("Discovering audio devices...");
    let Devices { mic_source, system_monitor } =
        discover_default_devices().context("PulseAudio device discovery failed")?;
    println!("  mic:    {mic_source}");
    println!("  system: {system_monitor}");

    // ── Connect both Realtime sessions ────────────────────────────────────────
    println!("\nConnecting to the Realtime API...");
    let mic_session = RealtimeSession::connect(&api_key, endpoint, &session_update)
        .await
        .context("failed to connect mic session")?;
    let sys_session = RealtimeSession::connect(&api_key, endpoint, &session_update)
        .await
        .context("failed to connect system session")?;
    println!("  sessions ready.\n");

    // ── Start streaming capture ───────────────────────────────────────────────
    let (mic_handle, mic_std_rx) =
        Recorder::new(mic_source, "live-api-mic").start_streaming();
    let (sys_handle, sys_std_rx) =
        Recorder::new(system_monitor, "live-api-system").start_streaming();

    // ── Split sessions into independent sink / stream halves ──────────────────
    let (mut mic_sink, mut mic_stream) = mic_session.split();
    let (mut sys_sink, mut sys_stream) = sys_session.split();

    // ── Bridge: std::sync::mpsc → tokio::sync::mpsc ──────────────────────────
    //
    // libpulse runs on a std::thread; the WS sink is async. A spawn_blocking
    // loop drains the std channel and forwards each chunk into a tokio channel.
    let (mic_tx, mut mic_chunk_rx) = tokio::sync::mpsc::unbounded_channel::<Vec<f32>>();
    let (sys_tx, mut sys_chunk_rx) = tokio::sync::mpsc::unbounded_channel::<Vec<f32>>();

    tokio::task::spawn_blocking(move || {
        while let Ok(chunk) = mic_std_rx.recv() {
            if mic_tx.send(chunk).is_err() {
                break;
            }
        }
    });
    tokio::task::spawn_blocking(move || {
        while let Ok(chunk) = sys_std_rx.recv() {
            if sys_tx.send(chunk).is_err() {
                break;
            }
        }
    });

    // ── Feed tasks: pull chunks → WS sink ────────────────────────────────────
    let mic_feed = tokio::spawn(async move {
        let mut n = 0usize;
        while let Some(chunk) = mic_chunk_rx.recv().await {
            if let Err(e) = mic_sink.feed(&chunk).await {
                eprintln!("mic feed error: {e}");
                break;
            }
            n += 1;
            if n % COMMIT_EVERY_CHUNKS == 0 {
                if let Err(e) = mic_sink.commit().await {
                    eprintln!("mic commit error: {e}");
                    break;
                }
            }
        }
        mic_sink.close().await;
    });
    let sys_feed = tokio::spawn(async move {
        let mut n = 0usize;
        while let Some(chunk) = sys_chunk_rx.recv().await {
            if let Err(e) = sys_sink.feed(&chunk).await {
                eprintln!("sys feed error: {e}");
                break;
            }
            n += 1;
            if n % COMMIT_EVERY_CHUNKS == 0 {
                if let Err(e) = sys_sink.commit().await {
                    eprintln!("sys commit error: {e}");
                    break;
                }
            }
        }
        sys_sink.close().await;
    });

    // ── Shared display channel ────────────────────────────────────────────────
    enum Source { Mic, Sys }
    let (display_tx, mut display_rx) =
        tokio::sync::mpsc::unbounded_channel::<(Source, String)>();
    let display_tx_sys = display_tx.clone();

    // ── Event tasks: forward completed utterances to the display task ─────────
    let mic_events = tokio::spawn(async move {
        loop {
            match mic_stream.next_event().await {
                Ok(Some(TranscriptEvent::Completed(t))) => {
                    let t = t.trim().to_owned();
                    if !t.is_empty() { let _ = display_tx.send((Source::Mic, t)); }
                }
                Ok(_) => {}
                Err(RealtimeError::Closed) => break,
                Err(e) => { eprintln!("[mic] {e}"); break; }
            }
        }
    });
    let sys_events = tokio::spawn(async move {
        loop {
            match sys_stream.next_event().await {
                Ok(Some(TranscriptEvent::Completed(t))) => {
                    let t = t.trim().to_owned();
                    if !t.is_empty() { let _ = display_tx_sys.send((Source::Sys, t)); }
                }
                Ok(_) => {}
                Err(RealtimeError::Closed) => break,
                Err(e) => { eprintln!("[sys] {e}"); break; }
            }
        }
    });

    // ── Display task: accumulate full transcript and redraw on each utterance ──
    let display = tokio::spawn(async move {
        fn render(mic: &str, sys: &str) {
            // Clear screen, cursor to home, then draw both panels.
            print!("\x1B[2J\x1B[H");
            println!("━━━ Microphone ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            if mic.is_empty() { println!("(listening...)"); } else { println!("{mic}"); }
            println!();
            println!("━━━ System Audio ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            if sys.is_empty() { println!("(listening...)"); } else { println!("{sys}"); }
            println!();
            print!("Press Enter to stop.");
            let _ = std::io::stdout().flush();
        }

        let mut mic_text = String::new();
        let mut sys_text = String::new();
        render(&mic_text, &sys_text);

        while let Some((source, chunk)) = display_rx.recv().await {
            match source {
                Source::Mic => {
                    if !mic_text.is_empty() { mic_text.push(' '); }
                    mic_text.push_str(&chunk);
                }
                Source::Sys => {
                    if !sys_text.is_empty() { sys_text.push(' '); }
                    sys_text.push_str(&chunk);
                }
            }
            render(&mic_text, &sys_text);
        }
    });

    // ── Run until Enter ───────────────────────────────────────────────────────
    tokio::task::spawn_blocking(|| {
        let _ = std::io::stdin().lock().lines().next();
    })
    .await
    .ok();

    println!("\nStopping...");

    // `stop()` joins the capture thread (~256 ms); run off-runtime to avoid blocking.
    tokio::task::spawn_blocking(|| { let _ = mic_handle.stop(); }).await.ok();
    tokio::task::spawn_blocking(|| { let _ = sys_handle.stop(); }).await.ok();

    // Capture stop → std channels close → bridge tasks exit → tokio channels
    // close → feed tasks drain + close WS sinks → server closes → stream tasks exit.
    let _ = tokio::join!(mic_feed, sys_feed, mic_events, sys_events, display);

    println!("Done.");
    Ok(())
}
