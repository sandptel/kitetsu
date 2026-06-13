//! kitetsu — Wayland-native AI overlay assistant.
//!
//! Structured into five modules: `presenter` (iced_layershell UI),
//! `router` (LLM back-ends, IPC, agent loop), `configurator` (config +
//! hot-reload), `prompter` (memory, context, prompt assembly), and
//! `listener` (audio capture via PulseAudio/PipeWire-pulse). See
//! `KITETSU_PLAN.md` for the full architecture and module dependency rules.

pub mod listener;

// Experimental live-teleprompter binary support: CLI/daemon control plane, TOML
// config, LLM client, and pipe orchestration. Lives in the lib (not the bin) so
// it is unit-testable; `src/main.rs` is a thin entry point over it. Gated to the
// `teleprompter` feature so the default library build is unaffected.
#[cfg(feature = "teleprompter")]
pub mod teleprompter;
