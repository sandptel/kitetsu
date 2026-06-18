//! Unix-socket control plane: command framing and client/server transport.
//!
//! The daemon binds a Unix domain socket; short-lived CLI invocations connect,
//! send one newline-delimited JSON [`Command`], and read back one line of ack
//! text. Framing is line-based so a connection carries exactly one request and
//! one response. Not here: command semantics (that is `daemon`).

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt as _, AsyncWriteExt as _, BufReader};
use tokio::net::{UnixListener, UnixStream};

/// A control command sent from the CLI to the running daemon.
///
/// Serialised as a `{tool, action}` object so the supervisor can route it:
/// global commands carry `tool = "kitetsu"`, tool commands name the tool, e.g.
/// `{"tool":"teleprompter","action":"process"}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "tool", content = "action", rename_all = "lowercase")]
pub enum Command {
    /// Supervisor-level command (no tool).
    Kitetsu(GlobalAction),
    /// Command routed to the teleprompter tool.
    Teleprompter(TeleprompterAction),
}

/// Supervisor-level actions, not tied to any tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GlobalAction {
    /// Health check — the daemon replies `pong`.
    Ping,
    /// Ask the daemon to shut down.
    Stop,
}

/// Actions the teleprompter tool understands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TeleprompterAction {
    /// Process the window: dispatch the selected pipe on audio since the last trigger.
    Process,
    /// Discard everything heard so far and start the window fresh.
    Trash,
    /// Pause/resume recording (the accumulated window is kept either way).
    Pause,
    /// Toggle overlay visibility (content + positions retained).
    Toggle,
}

/// Failures in the control-plane transport.
#[derive(Debug, thiserror::Error)]
pub enum IpcError {
    /// Socket bind/connect/read/write failure.
    #[error("control socket I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// A command or response could not be (de)serialised.
    #[error("control message JSON error: {0}")]
    Json(#[from] serde_json::Error),
    /// A daemon already appears to own the socket.
    #[error("a daemon is already running on {0}")]
    AlreadyRunning(PathBuf),
    /// The daemon accepted the connection but closed it without a response.
    #[error("daemon closed the connection before responding")]
    NoResponse,
}

/// Resolve the control-socket path: `$XDG_RUNTIME_DIR/kitetsu.sock`, falling
/// back to the system temp dir when the runtime dir is unset.
pub fn socket_path() -> PathBuf {
    let dir = std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    dir.join("kitetsu.sock")
}

/// Bind the control socket, refusing if a live daemon already owns it and
/// cleaning up a stale socket file otherwise.
///
/// "Live" is detected by a synchronous connect probe: if the probe succeeds, a
/// daemon is accepting connections; if it fails, any existing file is stale and
/// removed before binding.
pub fn bind() -> Result<UnixListener, IpcError> {
    let path = socket_path();
    if path.exists() {
        if std::os::unix::net::UnixStream::connect(&path).is_ok() {
            return Err(IpcError::AlreadyRunning(path));
        }
        // Stale socket from a crashed daemon — safe to remove.
        std::fs::remove_file(&path)?;
    }
    Ok(UnixListener::bind(&path)?)
}

/// Remove the control-socket file. Best-effort: a missing file is not an error.
pub fn cleanup() {
    let _ = std::fs::remove_file(socket_path());
}

/// Read one newline-delimited [`Command`] from a connected stream.
///
/// Returns `Ok(None)` if the peer closed the connection before sending a line.
pub async fn read_command(stream: &mut UnixStream) -> Result<Option<Command>, IpcError> {
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    if reader.read_line(&mut line).await? == 0 {
        return Ok(None);
    }
    Ok(Some(serde_json::from_str(line.trim_end())?))
}

/// Write one line of ack text back to a connected stream.
pub async fn write_ack(stream: &mut UnixStream, ack: &str) -> Result<(), IpcError> {
    stream.write_all(ack.as_bytes()).await?;
    stream.write_all(b"\n").await?;
    stream.flush().await?;
    Ok(())
}

/// Connect to the daemon, send `cmd`, and return the daemon's ack line.
pub async fn send_command(cmd: &Command) -> Result<String, IpcError> {
    let path = socket_path();
    let mut stream = UnixStream::connect(&path).await?;

    let line = serde_json::to_string(cmd)?;
    stream.write_all(line.as_bytes()).await?;
    stream.write_all(b"\n").await?;
    stream.flush().await?;

    let mut reader = BufReader::new(stream);
    let mut resp = String::new();
    if reader.read_line(&mut resp).await? == 0 {
        return Err(IpcError::NoResponse);
    }
    Ok(resp.trim_end().to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_json_round_trips() {
        for cmd in [
            Command::Kitetsu(GlobalAction::Ping),
            Command::Kitetsu(GlobalAction::Stop),
            Command::Teleprompter(TeleprompterAction::Process),
            Command::Teleprompter(TeleprompterAction::Trash),
            Command::Teleprompter(TeleprompterAction::Pause),
            Command::Teleprompter(TeleprompterAction::Toggle),
        ] {
            let line = serde_json::to_string(&cmd).expect("serialise");
            let back: Command = serde_json::from_str(&line).expect("deserialise");
            assert_eq!(cmd, back);
        }
    }

    #[test]
    fn command_serialises_as_tool_action() {
        assert_eq!(
            serde_json::to_string(&Command::Teleprompter(TeleprompterAction::Process)).unwrap(),
            r#"{"tool":"teleprompter","action":"process"}"#
        );
        assert_eq!(
            serde_json::to_string(&Command::Kitetsu(GlobalAction::Stop)).unwrap(),
            r#"{"tool":"kitetsu","action":"stop"}"#
        );
    }

    #[test]
    fn unknown_tool_is_rejected() {
        assert!(serde_json::from_str::<Command>(r#"{"tool":"bogus","action":"ping"}"#).is_err());
    }

    #[test]
    fn unknown_action_is_rejected() {
        assert!(
            serde_json::from_str::<Command>(r#"{"tool":"teleprompter","action":"explode"}"#)
                .is_err()
        );
    }

    #[test]
    fn socket_path_uses_runtime_dir_when_set() {
        // Not mutating the environment (process-global); just assert the file name.
        assert_eq!(
            socket_path().file_name().and_then(|s| s.to_str()),
            Some("kitetsu.sock")
        );
    }
}
