//! The long-lived teleprompter daemon: control-socket accept loop and command
//! handling.
//!
//! Iteration 1 is a skeleton — it binds the control socket, acks commands, and
//! shuts down on `Stop`. Audio capture, the window buffer, and the three pipes
//! are wired in later iterations. Not here: transport framing (that is `ipc`).

use tracing::{info, warn};

use super::config::Config;
use super::ipc::{self, Command, IpcError};

/// Failures that terminate the daemon.
#[derive(Debug, thiserror::Error)]
pub enum DaemonError {
    /// Control-plane transport failure (bind or accept).
    #[error(transparent)]
    Ipc(#[from] IpcError),
}

/// Bind the control socket and serve commands until `Stop` is received.
///
/// `config` and `system_prompt` are loaded once by the caller and held for the
/// daemon's lifetime; iteration 2 only logs a summary of them — capture and the
/// pipes consume them in later iterations.
///
/// Commands are handled one connection at a time — the control plane is
/// low-traffic (human key presses), so sequential handling keeps ordering
/// obvious and needs no shared locking.
pub async fn run(config: Config, system_prompt: String) -> Result<(), DaemonError> {
    log_summary(&config, &system_prompt);

    let listener = ipc::bind()?;
    info!(socket = %ipc::socket_path().display(), "teleprompter daemon listening");

    let mut window: u64 = 0;
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
                window += 1;
                info!(window, "trigger received (pipes not wired yet)");
                let ack = format!("queued window {window}");
                let _ = ipc::write_ack(&mut stream, &ack).await;
            }
            Command::Stop => {
                info!("stop received — shutting down");
                let _ = ipc::write_ack(&mut stream, "stopping").await;
                break;
            }
        }
    }

    ipc::cleanup();
    Ok(())
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
