//! kitetsu — thin binary entry point.
//!
//! `kitetsu daemon` runs the overlay on the main thread (iced owns it) and the
//! audio/capture/pipe daemon on a background tokio runtime, bridged by an
//! unbounded channel. `kitetsu ping`/`stop` and `kitetsu teleprompter <action>`
//! are short-lived clients that send one command to a running daemon over the
//! control socket. All real logic lives in `kitetsu::teleprompter` (the lib) so
//! it stays unit-testable.

use std::path::{Path, PathBuf};

use anyhow::Context as _;
use clap::{Parser, Subcommand};
use tokio::sync::mpsc;
use tracing::info;

use kitetsu::settings::{self, Settings};
use kitetsu::teleprompter::{
    CardInit, Command, Config, GlobalAction, PipeId, PipeKind, TeleprompterAction, daemon, layout,
    send_command, ui,
};

/// Live teleprompter: listen to mic + system audio and suggest what to say.
#[derive(Parser)]
#[command(name = "kitetsu", about = "kitetsu supervisor + tool control CLI")]
struct Cli {
    #[command(subcommand)]
    command: TopCommand,
}

/// Top-level commands: run the daemon, supervisor globals, or a tool namespace.
#[derive(Subcommand)]
enum TopCommand {
    /// Run the long-lived daemon (supervisor + overlay) until stopped.
    Daemon {
        /// Config directory. Defaults to $XDG_CONFIG_HOME/kitetsu, else
        /// ~/.config/kitetsu. Holds kitetsu.toml and teleprompter/.
        #[arg(long)]
        config_dir: Option<PathBuf>,
    },
    /// Health-check the running daemon.
    Ping,
    /// Ask the running daemon to shut down.
    Stop,
    /// Control the teleprompter tool.
    Teleprompter {
        #[command(subcommand)]
        action: TeleprompterCli,
    },
}

/// Teleprompter tool actions (routed to a running daemon).
#[derive(Subcommand)]
enum TeleprompterCli {
    /// Process the window: dispatch the selected pipe on audio since the last trigger.
    Process,
    /// Discard everything heard so far and start recording fresh.
    Trash,
    /// Pause/resume recording (the accumulated window is kept either way).
    Pause,
    /// Toggle overlay visibility (content + positions retained).
    Toggle,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("kitetsu=info")),
        )
        .init();

    let cli = Cli::parse();

    let command = match cli.command {
        TopCommand::Daemon { config_dir } => return run_daemon(config_dir.as_deref()),
        TopCommand::Ping => Command::Kitetsu(GlobalAction::Ping),
        TopCommand::Stop => Command::Kitetsu(GlobalAction::Stop),
        TopCommand::Teleprompter { action } => Command::Teleprompter(match action {
            TeleprompterCli::Process => TeleprompterAction::Process,
            TeleprompterCli::Trash => TeleprompterAction::Trash,
            TeleprompterCli::Pause => TeleprompterAction::Pause,
            TeleprompterCli::Toggle => TeleprompterAction::Toggle,
        }),
    };

    // Client commands need a runtime only to talk to the daemon over the socket.
    let rt = tokio::runtime::Runtime::new().context("building client runtime")?;
    let ack = rt
        .block_on(send_command(&command))
        .context("could not reach the daemon — is `kitetsu daemon` running?")?;
    println!("{ack}");
    Ok(())
}

/// Launch the daemon (background tokio runtime) and the overlay (this thread).
///
/// Resolves the config dir, reads the central `kitetsu.toml` registry, and only
/// starts the teleprompter when it is enabled there. The tool's own config and
/// prompt/context files live in `<config_dir>/teleprompter/`.
fn run_daemon(override_dir: Option<&Path>) -> anyhow::Result<()> {
    let config_dir = settings::config_dir(override_dir);
    let settings = Settings::load(&config_dir)
        .with_context(|| format!("loading central config from {}", config_dir.display()))?;

    if !settings.teleprompter.enabled {
        info!(
            config_dir = %config_dir.display(),
            "teleprompter disabled in kitetsu.toml — nothing to run",
        );
        return Ok(());
    }

    let tool_dir = config_dir.join("teleprompter");
    let config_path = tool_dir.join("teleprompter.toml");
    let config = Config::load(&config_path)
        .with_context(|| format!("loading config from {}", config_path.display()))?;
    let system_prompt = config
        .load_system_prompt(&tool_dir)
        .context("loading prompt.md / context.md")?;

    // Build the overlay card specs from config (only enabled pipes get a card),
    // then let any saved layout override the geometry so cards reopen where the
    // user last left them.
    let mut cards = card_specs(&config);
    let layout_path = layout::layout_path();
    let saved = layout::load(&layout_path);
    for card in &mut cards {
        if let Some(g) = saved.get(card.id) {
            card.pos_x = g.pos_x;
            card.pos_y = g.pos_y;
            card.width = g.width;
            card.height = g.height;
        }
    }

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
    ui::run(cards, layout_path).map_err(|e| anyhow::anyhow!("overlay failed: {e}"))?;

    // Overlay closed → the daemon thread winds down with the process.
    drop(daemon_thread);
    Ok(())
}

/// Build the overlay card spec for the selected pipe (exactly one).
fn card_specs(config: &Config) -> Vec<CardInit> {
    let (id, label, p): (PipeId, &str, &_) = match config.pipe {
        PipeKind::Live => (PipeId::Pipe1, "live", &config.live),
        // chunk's style fields live on a different type; handle it in its own arm.
        PipeKind::Chunk => {
            let p = &config.chunk;
            return vec![CardInit {
                id: PipeId::Pipe2,
                model: format!("@ chunk · {}", p.llm_model),
                font_size: p.font_size,
                opacity: p.opacity,
                bg_opacity: p.bg_opacity,
                text_opacity: p.text_opacity,
                width: p.width,
                height: p.height,
                pos_x: p.pos_x,
                pos_y: p.pos_y,
            }];
        }
    };
    vec![CardInit {
        id,
        model: format!("@ {label} · {}", p.llm_model),
        font_size: p.font_size,
        opacity: p.opacity,
        bg_opacity: p.bg_opacity,
        text_opacity: p.text_opacity,
        width: p.width,
        height: p.height,
        pos_x: p.pos_x,
        pos_y: p.pos_y,
    }]
}
