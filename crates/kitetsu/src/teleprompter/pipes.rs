//! Pipe orchestration: turn a snapshotted [`Window`] into a suggestion file.
//!
//! Each pipe builds a labeled user turn, calls the LLM with the stable system
//! prompt plus the pipe's accumulating history, and writes a timestamped block to
//! the pipe's output file. A failed call writes an error block (and the failed
//! user turn is *not* committed to history, so continuity survives).
//! - Pipe 1: accumulated live WS transcript.
//! - Pipe 2: on-trigger REST re-transcription of the raw window (more accurate).
//! Pipe 3 (audio-direct) lands in the next iteration.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use tracing::warn;

use crate::listener::{ApiError, ApiTranscriber};

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

/// A pipe run failure: either transcription (pipe 2) or the LLM call.
#[derive(Debug, thiserror::Error)]
pub enum PipeError {
    /// The LLM chat call failed.
    #[error(transparent)]
    Llm(#[from] LlmError),
    /// REST transcription failed (pipe 2 only).
    #[error("transcription failed: {0}")]
    Transcribe(#[from] ApiError),
}

/// Run pipe 1 on a snapshotted window: send the accumulated live transcript to
/// the LLM and write the suggestion to `<out_dir>/pipe1.md`.
pub async fn run_pipe1(
    window: &Window,
    history: &History,
    model: &str,
    ctx: &PipeContext<'_>,
) -> Result<String, PipeError> {
    let started = Instant::now();
    let content = label_transcript(window.mic_text.trim(), window.sys_text.trim(), ctx.labels);
    chat_and_write(window.n, content, history, model, ctx, "pipe1", started).await
}

/// Run pipe 2 on a snapshotted window: REST-transcribe the raw mic + system
/// audio (in parallel) with the configured STT model, then send that transcript
/// to the LLM and write the suggestion to `<out_dir>/pipe2.md`.
pub async fn run_pipe2(
    window: &Window,
    history: &History,
    transcriber: &ApiTranscriber,
    model: &str,
    ctx: &PipeContext<'_>,
) -> Result<String, PipeError> {
    let started = Instant::now();
    let out_path = ctx.out_path("pipe2.md");

    let (mic, sys) = tokio::join!(
        transcribe_side(transcriber, &window.mic_raw),
        transcribe_side(transcriber, &window.sys_raw),
    );
    let (mic_text, sys_text) = match (mic, sys) {
        (Ok(m), Ok(s)) => (m, s),
        (Err(e), _) | (_, Err(e)) => {
            let body = format!("**pipe2 error:** {e}");
            if let Err(io) = write_block(&out_path, window.n, started.elapsed(), &body, ctx.mode) {
                warn!(window = window.n, error = %io, "pipe2 failed to write error block");
            }
            warn!(window = window.n, error = %e, "pipe2 transcription failed");
            return Err(PipeError::Transcribe(e));
        }
    };

    let content = label_transcript(mic_text.trim(), sys_text.trim(), ctx.labels);
    chat_and_write(window.n, content, history, model, ctx, "pipe2", started).await
}

/// Shared tail of every LLM pipe: append the user turn to history (without
/// holding the lock across the await), call the LLM, commit the turns + write the
/// block on success, or write an error block and leave history untouched.
///
/// `tag` names the pipe (`"pipe1"`); the output file is `<tag>.md`.
async fn chat_and_write(
    window_n: u64,
    user_content: String,
    history: &History,
    model: &str,
    ctx: &PipeContext<'_>,
    tag: &str,
    started: Instant,
) -> Result<String, PipeError> {
    let out_path = ctx.out_path(&format!("{tag}.md"));
    let user_turn = Turn::user(user_content);

    let request_turns = {
        let guard = history.lock().expect("pipe history lock poisoned");
        let mut turns = guard.clone();
        turns.push(user_turn.clone());
        turns
    };

    match ctx.backend.chat(model, ctx.system_prompt, &request_turns).await {
        Ok(reply) => {
            {
                let mut guard = history.lock().expect("pipe history lock poisoned");
                guard.push(user_turn);
                guard.push(Turn::assistant(reply.clone()));
            }
            if let Err(e) = write_block(&out_path, window_n, started.elapsed(), &reply, ctx.mode) {
                warn!(window = window_n, %tag, error = %e, "pipe failed to write output file");
            }
            Ok(reply)
        }
        Err(e) => {
            let body = format!("**{tag} error:** {e}");
            if let Err(io) = write_block(&out_path, window_n, started.elapsed(), &body, ctx.mode) {
                warn!(window = window_n, %tag, error = %io, "pipe failed to write error block");
            }
            warn!(window = window_n, %tag, error = %e, "pipe LLM call failed");
            Err(PipeError::Llm(e))
        }
    }
}

/// Transcribe one side's raw samples; empty audio yields empty text (not an
/// error), so a single-source window still works.
async fn transcribe_side(transcriber: &ApiTranscriber, samples: &[f32]) -> Result<String, ApiError> {
    if samples.is_empty() {
        return Ok(String::new());
    }
    transcriber.transcribe(samples).await
}

/// Build the labeled transcript the LLM sees, system side first. Blank sides are
/// skipped; an empty window yields an explicit "no speech" note.
fn label_transcript(mic: &str, sys: &str, labels: &Labels) -> String {
    let mut lines = Vec::new();
    if !sys.is_empty() {
        lines.push(format!("{}: {sys}", labels.system));
    }
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

    #[test]
    fn transcript_puts_system_first_then_mic() {
        let c = label_transcript("I'm well", "How are you?", &labels());
        assert_eq!(c, "Them: How are you?\nMe: I'm well");
    }

    #[test]
    fn transcript_skips_blank_sides() {
        let c = label_transcript("", "Hello", &labels());
        assert_eq!(c, "Them: Hello");
    }

    #[test]
    fn empty_window_yields_no_speech_note() {
        let c = label_transcript("", "", &labels());
        assert_eq!(c, "(no speech detected in this window)");
    }
}
