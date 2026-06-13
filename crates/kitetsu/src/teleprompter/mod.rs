//! Experimental live teleprompter: a daemon that listens to mic + system audio
//! and, on each CLI trigger, suggests what to say via an LLM.
//!
//! `ipc` owns the Unix-socket control plane (command framing + client/server).
//! `daemon` owns the long-lived process: capture wiring, window state, and
//! trigger handling. Later iterations add `config`, `window`, `llm`, `pipes`,
//! and `output`.
//!
//! Deliberately separate from the planned `presenter`/`router`/… architecture —
//! this is throwaway, branch-local experimentation (see the plan). It reuses the
//! `listener` audio stack but adds no obligations to the main build.

pub mod daemon;
pub mod ipc;

pub use ipc::{Command, IpcError, send_command, socket_path};
