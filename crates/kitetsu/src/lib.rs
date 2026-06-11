//! kitetsu — Wayland-native AI overlay assistant.
//!
//! Structured into five modules: `presenter` (iced_layershell UI),
//! `router` (LLM back-ends, IPC, agent loop), `configurator` (config +
//! hot-reload), `prompter` (memory, context, prompt assembly), and
//! `listener` (audio capture via PulseAudio/PipeWire-pulse). See
//! `KITETSU_PLAN.md` for the full architecture and module dependency rules.

pub mod listener;
