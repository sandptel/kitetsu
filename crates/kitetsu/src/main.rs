//! kitetsu teleprompter — thin binary entry point.
//!
//! `--daemon` runs the overlay on the main thread (iced owns it) and the
//! audio/capture/pipe daemon on a background tokio runtime, bridged by an
//! unbounded channel. `next`/`stop`/`ping` are short-lived clients that send one
//! command to a running daemon over the control socket. All real logic lives in
//! `kitetsu::teleprompter` (the lib) so it stays unit-testable.

use std::path::PathBuf;

use anyhow::Context as _;
use clap::{Parser, Subcommand};
use tokio::sync::mpsc;

use kitetsu::teleprompter::{CardInit, Command, Config, PipeId, daemon, send_command, ui};

/// Live teleprompter: listen to mic + system audio and suggest what to say.
#[derive(Parser)]
#[command(name = "kitetsu", about = "live teleprompter daemon + control CLI")]
struct Cli {
    /// Run the long-lived daemon (audio capture + pipes + control socket) plus overlay.
    #[arg(long)]
    daemon: bool,

    /// Path to the TOML config (daemon only).
    #[arg(long, default_value = "config.toml")]
    config: PathBuf,

    #[command(subcommand)]
    command: Option<ClientCommand>,
}

/// Control commands sent to a running daemon.
#[derive(Subcommand)]
enum ClientCommand {
    /// Trigger a window: dispatch the pipes on audio since the last trigger.
    Next,
    /// Toggle overlay visibility (content + positions retained).
    Toggle,
    /// Ask the running daemon to shut down.
    Stop,
    /// Health-check the running daemon.
    Ping,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("kitetsu=info")),
        )
        .init();

    let cli = Cli::parse();

    if cli.daemon {
        return run_daemon(&cli.config);
    }

    let command = match cli.command {
        Some(ClientCommand::Next) => Command::Next,
        Some(ClientCommand::Toggle) => Command::Toggle,
        Some(ClientCommand::Stop) => Command::Stop,
        Some(ClientCommand::Ping) => Command::Ping,
        None => {
            anyhow::bail!(
                "nothing to do: pass --daemon, or a command (next | toggle | stop | ping)"
            );
        }
    };

    // Client commands need a runtime only to talk to the daemon over the socket.
    let rt = tokio::runtime::Runtime::new().context("building client runtime")?;
    let ack = rt
        .block_on(send_command(&command))
        .context("could not reach the daemon — is `kitetsu --daemon` running?")?;
    println!("{ack}");
    Ok(())
}

/// Launch the daemon (background tokio runtime) and the overlay (this thread).
fn run_daemon(config_path: &std::path::Path) -> anyhow::Result<()> {
    let config = Config::load(config_path)
        .with_context(|| format!("loading config from {}", config_path.display()))?;
    let base_dir = config_path.parent().unwrap_or_else(|| std::path::Path::new("."));
    let system_prompt = config
        .load_system_prompt(base_dir)
        .context("loading prompt.md / context.md")?;

    // Build the overlay card specs from config (only enabled pipes get a card).
    let cards = card_specs(&config);

    // Bridge: daemon (tokio) → overlay (iced). Park the receiver for the
    // subscription worker before the iced loop starts.
    let (ui_tx, ui_rx) = mpsc::unbounded_channel();
    ui::install_receiver(ui_rx);

    // The daemon owns its own multi-thread runtime on a dedicated thread; iced
    // must own the main thread (Wayland event loop).
    let daemon_thread = std::thread::Builder::new()
        .name("telep-daemon".into())
        .spawn(move || -> anyhow::Result<()> {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .context("building daemon runtime")?;
            rt.block_on(daemon::run(config, system_prompt, ui_tx))
                .context("teleprompter daemon failed")?;
            Ok(())
        })
        .context("spawning daemon thread")?;

    // Run the overlay on the main thread; blocks until the surface closes.
    ui::run(cards).map_err(|e| anyhow::anyhow!("overlay failed: {e}"))?;

    // Overlay closed → the daemon thread winds down with the process.
    drop(daemon_thread);
    Ok(())
}

/// Build overlay card specs for the enabled pipes (pipe1 / pipe2 only).
fn card_specs(config: &Config) -> Vec<CardInit> {
    let mut cards = Vec::new();
    if config.pipe1.enabled {
        let p = &config.pipe1;
        cards.push(CardInit {
            id: PipeId::Pipe1,
            header: "Pipe 1 · live".to_owned(),
            model: format!("@ {}", p.llm_model),
            font_size: p.font_size,
            opacity: p.opacity,
            bg_opacity: p.bg_opacity,
            text_opacity: p.text_opacity,
            width: p.width,
            pos_x: p.pos_x,
            pos_y: p.pos_y,
        });
    }
    if config.pipe2.enabled {
        let p = &config.pipe2;
        cards.push(CardInit {
            id: PipeId::Pipe2,
            header: "Pipe 2 · chunk".to_owned(),
            model: format!("@ {}", p.llm_model),
            font_size: p.font_size,
            opacity: p.opacity,
            bg_opacity: p.bg_opacity,
            text_opacity: p.text_opacity,
            width: p.width,
            pos_x: p.pos_x,
            pos_y: p.pos_y,
        });
    }
    cards
}
