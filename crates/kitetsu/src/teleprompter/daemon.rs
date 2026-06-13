//! The long-lived teleprompter daemon: control-socket accept loop and command
//! handling.
//!
//! Iteration 1 is a skeleton — it binds the control socket, acks commands, and
//! shuts down on `Stop`. Audio capture, the window buffer, and the three pipes
//! are wired in later iterations. Not here: transport framing (that is `ipc`).

use tracing::{info, warn};

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
/// Commands are handled one connection at a time — the control plane is
/// low-traffic (human key presses), so sequential handling keeps ordering
/// obvious and needs no shared locking.
pub async fn run() -> Result<(), DaemonError> {
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
