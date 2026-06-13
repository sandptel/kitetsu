//! Prove the teleprompter LLM client end-to-end: load `config.toml` +
//! `prompt.md` + `context.md`, feed them as the system prompt, send a simulated
//! transcript turn, and print the model's "what to say" suggestion.
//!
//! This exercises the same `Backend`/`Turn` path the pipes will use — only the
//! transcript is faked (typed on the CLI) instead of captured from audio.
//!
//! Usage (uses pipe1's backend + model from config.toml):
//!   cargo run -p kitetsu --features teleprompter --example teleprompter_llm
//!   cargo run -p kitetsu --features teleprompter --example teleprompter_llm -- "So, tell me about yourself."
//!
//! Requires `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY` if pipe1 uses anthropic)
//! in the shell or `.env`, plus prompt.md + context.md next to config.toml.

use std::path::Path;

use anyhow::Context as _;

use kitetsu::listener::{ANTHROPIC_API_KEY, OPENAI_API_KEY, load_dotenv, require};
use kitetsu::teleprompter::config::LlmBackendKind;
use kitetsu::teleprompter::{Backend, Config, Turn};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("kitetsu=info")),
        )
        .init();

    // ── Load config + the composed system prompt (prompt.md + context.md) ─────
    let config_path = Path::new("config.toml");
    let config = Config::load(config_path)
        .with_context(|| format!("loading {}", config_path.display()))?;
    let base_dir = config_path.parent().unwrap_or_else(|| Path::new("."));
    let system_prompt = config
        .load_system_prompt(base_dir)
        .context("loading prompt.md / context.md (copy the .example files)")?;

    println!("System prompt ({} chars) — prompt.md + context.md:", system_prompt.len());
    println!("────────────────────────────────────────────────────");
    println!("{system_prompt}");
    println!("────────────────────────────────────────────────────\n");

    // ── Build the backend from pipe1's config ─────────────────────────────────
    if let Ok(cwd) = std::env::current_dir() {
        load_dotenv(&cwd.join(".env"));
    }
    let backend_kind = config.pipe1.llm_backend;
    let key_name = match backend_kind {
        LlmBackendKind::Openai => OPENAI_API_KEY,
        LlmBackendKind::Anthropic => ANTHROPIC_API_KEY,
    };
    let api_key = require(key_name)
        .with_context(|| format!("set {key_name} in the environment or .env"))?;
    let backend = Backend::new(backend_kind, api_key).context("building LLM backend")?;
    let model = &config.pipe1.llm_model;

    // ── Build a simulated transcript turn (the "them" side) ───────────────────
    let them = {
        let args: Vec<String> = std::env::args().skip(1).collect();
        if args.is_empty() {
            "So, tell me a bit about yourself and what you're working on.".to_owned()
        } else {
            args.join(" ")
        }
    };
    let user_turn = format!("{}: {them}", config.audio.system_label);
    let turns = [Turn::user(&user_turn)];

    println!("Transcript window:\n  {user_turn}\n");
    println!("Asking {backend_kind:?} model `{model}`...\n");

    // ── Call the LLM with the system prompt + transcript ──────────────────────
    let reply = backend
        .chat(model, &system_prompt, &turns)
        .await
        .context("LLM chat request failed")?;

    println!("Suggested reply as \"{}\" ({} chars):", config.audio.mic_label, reply.len());
    println!("════════════════════════════════════════════════════");
    println!("{reply}");
    println!("════════════════════════════════════════════════════");

    Ok(())
}
