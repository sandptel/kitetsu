//! Pipe orchestration: turn a snapshotted [`Window`] into a suggestion file.
//!
//! Each pipe builds a labeled user turn, calls the LLM with the stable system
//! prompt plus the pipe's accumulating history, and writes a timestamped block to
//! the pipe's output file. A failed call writes an error block (and the failed
//! user turn is *not* committed to history, so continuity survives). Pipe 1 (live
//! WS text) lives here; pipes 2 and 3 land in later iterations.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use tracing::warn;

use super::config::OutputMode;
use super::llm::{Backend, LlmError, Turn};
use super::output::write_block;
use super::window::Window;

/// Shared, accumulating conversation history for one pipe.
pub type History = Arc<Mutex<Vec<Turn>>>;

/// Speaker labels (mic = "Me", system = "Them") for the transcript turn.
#[derive(Debug, Clone)]
pub struct Labels {
    pub mic: String,
    pub system: String,
}

/// The per-run inputs a pipe shares across every window (built once by the
/// daemon/example and borrowed for each trigger).
pub struct PipeContext<'a> {
    pub backend: &'a Backend,
    pub system_prompt: &'a str,
    pub labels: &'a Labels,
    pub out_dir: &'a Path,
    pub mode: OutputMode,
}

impl PipeContext<'_> {
    /// Output file path for a given pipe file name (e.g. `pipe1.md`).
    fn out_path(&self, file: &str) -> PathBuf {
        self.out_dir.join(file)
    }
}

/// Run pipe 1 on a snapshotted window: send the accumulated live transcript to
/// the LLM and write the suggestion to `<out_dir>/pipe1.md`.
///
/// On success the user + assistant turns are appended to `history` and the reply
/// is returned. On failure an error block is written, a warning logged, and the
/// error returned (history is left unchanged so a bad window doesn't poison it).
pub async fn run_pipe1(
    window: &Window,
    history: &History,
    model: &str,
    ctx: &PipeContext<'_>,
) -> Result<String, LlmError> {
    let started = Instant::now();
    let out_path = ctx.out_path("pipe1.md");
    let user_turn = Turn::user(transcript_content(window, ctx.labels));

    // Snapshot history + the new turn for the request without holding the lock
    // across the await (the lock is touched only briefly, never during I/O).
    let request_turns = {
        let guard = history.lock().expect("pipe1 history lock poisoned");
        let mut turns = guard.clone();
        turns.push(user_turn.clone());
        turns
    };

    match ctx.backend.chat(model, ctx.system_prompt, &request_turns).await {
        Ok(reply) => {
            {
                let mut guard = history.lock().expect("pipe1 history lock poisoned");
                guard.push(user_turn);
                guard.push(Turn::assistant(reply.clone()));
            }
            if let Err(e) = write_block(&out_path, window.n, started.elapsed(), &reply, ctx.mode) {
                warn!(window = window.n, error = %e, "pipe1 failed to write output file");
            }
            Ok(reply)
        }
        Err(e) => {
            let body = format!("**pipe1 error:** {e}");
            if let Err(io) = write_block(&out_path, window.n, started.elapsed(), &body, ctx.mode) {
                warn!(window = window.n, error = %io, "pipe1 failed to write error block");
            }
            warn!(window = window.n, error = %e, "pipe1 LLM call failed");
            Err(e)
        }
    }
}

/// Build the labeled transcript the LLM sees, system side first. Blank sides are
/// skipped; an empty window yields an explicit "no speech" note.
pub fn transcript_content(window: &Window, labels: &Labels) -> String {
    let mut lines = Vec::new();
    let sys = window.sys_text.trim();
    if !sys.is_empty() {
        lines.push(format!("{}: {sys}", labels.system));
    }
    let mic = window.mic_text.trim();
    if !mic.is_empty() {
        lines.push(format!("{}: {mic}", labels.mic));
    }
    if lines.is_empty() {
        "(no speech detected in this window)".to_owned()
    } else {
        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels() -> Labels {
        Labels {
            mic: "Me".into(),
            system: "Them".into(),
        }
    }

    fn window(mic: &str, sys: &str) -> Window {
        Window {
            n: 1,
            mic_raw: Vec::new(),
            sys_raw: Vec::new(),
            mic_text: mic.to_owned(),
            sys_text: sys.to_owned(),
        }
    }

    #[test]
    fn transcript_puts_system_first_then_mic() {
        let c = transcript_content(&window("I'm well", "How are you?"), &labels());
        assert_eq!(c, "Them: How are you?\nMe: I'm well");
    }

    #[test]
    fn transcript_skips_blank_sides() {
        let c = transcript_content(&window("", "  Hello  "), &labels());
        assert_eq!(c, "Them: Hello");
    }

    #[test]
    fn empty_window_yields_no_speech_note() {
        let c = transcript_content(&window("", ""), &labels());
        assert_eq!(c, "(no speech detected in this window)");
    }
}
