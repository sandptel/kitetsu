//! kitetsu teleprompter — thin binary entry point.
//!
//! Parses the CLI and dispatches: `--daemon` runs the long-lived listener;
//! `next`/`stop`/`ping` are short-lived clients that send one command to a
//! running daemon over the control socket. All real logic lives in
//! `kitetsu::teleprompter` (the lib) so it stays unit-testable.

use anyhow::Context as _;
use clap::{Parser, Subcommand};

use kitetsu::teleprompter::{Command, daemon, send_command};

/// Live teleprompter: listen to mic + system audio and suggest what to say.
#[derive(Parser)]
#[command(name = "kitetsu", about = "live teleprompter daemon + control CLI")]
struct Cli {
    /// Run the long-lived daemon (audio capture + pipes + control socket).
    #[arg(long)]
    daemon: bool,

    #[command(subcommand)]
    command: Option<ClientCommand>,
}

/// Control commands sent to a running daemon.
#[derive(Subcommand)]
enum ClientCommand {
    /// Trigger a window: dispatch the pipes on audio since the last trigger.
    Next,
    /// Ask the running daemon to shut down.
    Stop,
    /// Health-check the running daemon.
    Ping,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("kitetsu=info")),
        )
        .init();

    let cli = Cli::parse();

    if cli.daemon {
        daemon::run().await.context("teleprompter daemon failed")?;
        return Ok(());
    }

    let command = match cli.command {
        Some(ClientCommand::Next) => Command::Next,
        Some(ClientCommand::Stop) => Command::Stop,
        Some(ClientCommand::Ping) => Command::Ping,
        None => {
            anyhow::bail!("nothing to do: pass --daemon, or a command (next | stop | ping)");
        }
    };

    let ack = send_command(&command)
        .await
        .context("could not reach the daemon — is `kitetsu --daemon` running?")?;
    println!("{ack}");
    Ok(())
}
