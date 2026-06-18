//! Experimental live teleprompter: a daemon that listens to mic + system audio
//! and, on each CLI trigger, suggests what to say via an LLM.
//!
//! `ipc` owns the Unix-socket control plane (command framing + client/server).
//! `daemon` owns the long-lived process: capture wiring, window state, and
//! trigger handling. `config` parses the TOML; `window` holds the rolling
//! per-source capture buffers; `llm` is the chat-LLM client; `pipes` turns a
//! window into a suggestion delivered to the overlay card.
//!
//! Deliberately separate from the planned `presenter`/`router`/… architecture —
//! this is throwaway, branch-local experimentation (see the plan). It reuses the
//! `listener` audio stack but adds no obligations to the main build.

pub mod config;
pub mod daemon;
pub mod ipc;
pub mod layout;
pub mod llm;
pub mod pipes;
pub mod ui;
pub mod window;

pub use config::{Config, ConfigError};
pub use ipc::{Command, IpcError, send_command, socket_path};
pub use layout::{Geometry, Layout};
pub use llm::{AudioPart, Backend, LlmError, Role, Turn};
pub use pipes::{History, Labels, PipeContext, PipeError, PipeOutcome, run_pipe1, run_pipe2};
pub use ui::{CardInit, PipeId, Stage, UiEvent};
pub use window::{Source, Window, WindowManager};
