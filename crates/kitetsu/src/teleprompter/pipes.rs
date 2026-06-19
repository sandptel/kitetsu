//! Pipe orchestration: turn a snapshotted [`Window`] into a suggestion.
//!
//! Each pipe builds a labeled user turn, calls the LLM with the stable system
//! prompt plus the pipe's accumulating history, and returns the reply (the daemon
//! routes it to the overlay card). A failed call leaves the failed user turn
//! *uncommitted* to history, so continuity survives.
//! - Live pipe: accumulated live WS transcript.
//! - Chunk pipe: on-trigger REST re-transcription of the raw window (more accurate).

use std::sync::{Arc, Mutex};

use tracing::warn;

use crate::listener::{ApiError, ApiTranscriber};

use super::llm::{Backend, LlmError, Turn};
use super::window::Window;

/// Shared, accumulating conversation history for one pipe.
pub type History = Arc<Mutex<Vec<Turn>>>;

/// Speaker labels (mic = "Me", system = "Them") for the transcript turn.
#[derive(Debug, Clone)]
pub struct Labels {
    pub mic: String,
    pub system: String,
}

/// What a pipe produced for one window: the transcript the LLM saw (mic and
/// system kept separate so a caller can show each side) and the suggestion it
/// returned. The joined, labeled form of `mic_text`/`sys_text` is exactly what
/// was sent as the user turn.
#[derive(Debug, Clone)]
pub struct PipeOutcome {
    pub mic_text: String,
    pub sys_text: String,
    pub reply: String,
}

/// The per-run inputs a pipe shares across every window (built once by the
/// daemon/example and borrowed for each trigger).
pub struct PipeContext<'a> {
    pub backend: &'a Backend,
    pub system_prompt: &'a str,
    pub labels: &'a Labels,
}

/// A pipe run failure: either transcription (chunk pipe) or the LLM call.
#[derive(Debug, thiserror::Error)]
pub enum PipeError {
    /// The LLM chat call failed.
    #[error(transparent)]
    Llm(#[from] LlmError),
    /// REST transcription failed (chunk pipe only).
    #[error("transcription failed: {0}")]
    Transcribe(#[from] ApiError),
}

/// Run the live pipe on a snapshotted window: send the accumulated live
/// transcript to the LLM and return the suggestion.
pub async fn run_live(
    window: &Window,
    history: &History,
    model: &str,
    ctx: &PipeContext<'_>,
) -> Result<PipeOutcome, PipeError> {
    let mic_text = window.mic_text.trim().to_owned();
    let sys_text = window.sys_text.trim().to_owned();
    let content = label_transcript(&mic_text, &sys_text, ctx.labels);
    let reply = chat_and_commit(window.n, content, history, model, ctx, "live").await?;
    Ok(PipeOutcome {
        mic_text,
        sys_text,
        reply,
    })
}

/// Run the chunk pipe on a snapshotted window: REST-transcribe the raw mic + system
/// audio (in parallel) with the configured STT model, then send that transcript
/// to the LLM and return the suggestion.
/// `on_transcribed` runs once the REST transcript is ready, just before the LLM
/// call — the daemon uses it to flip the card's status from "Transcribing" to
/// "Fetching response". It does not fire if transcription fails.
pub async fn run_chunk(
    window: &Window,
    history: &History,
    transcriber: &ApiTranscriber,
    model: &str,
    ctx: &PipeContext<'_>,
    on_transcribed: impl FnOnce(),
) -> Result<PipeOutcome, PipeError> {
    let (mic, sys) = tokio::join!(
        transcribe_side(transcriber, &window.mic_raw),
        transcribe_side(transcriber, &window.sys_raw),
    );
    let (mic_text, sys_text) = match (mic, sys) {
        (Ok(m), Ok(s)) => (m.trim().to_owned(), s.trim().to_owned()),
        (Err(e), _) | (_, Err(e)) => {
            warn!(window = window.n, error = %e, "chunk transcription failed");
            return Err(PipeError::Transcribe(e));
        }
    };

    on_transcribed();
    let content = label_transcript(&mic_text, &sys_text, ctx.labels);
    let reply = chat_and_commit(window.n, content, history, model, ctx, "chunk").await?;
    Ok(PipeOutcome {
        mic_text,
        sys_text,
        reply,
    })
}

/// Shared tail of every LLM pipe: append the user turn to history (without
/// holding the lock across the await), call the LLM, and on success commit both
/// turns and return the reply; on failure leave history untouched.
///
/// `tag` names the pipe (`"live"` / `"chunk"`) for log lines only.
async fn chat_and_commit(
    window_n: u64,
    user_content: String,
    history: &History,
    model: &str,
    ctx: &PipeContext<'_>,
    tag: &str,
) -> Result<String, PipeError> {
    let user_turn = Turn::user(user_content);

    let request_turns = {
        let guard = history.lock().expect("pipe history lock poisoned");
        let mut turns = guard.clone();
        turns.push(user_turn.clone());
        turns
    };

    match ctx.backend.chat(model, ctx.system_prompt, &request_turns).await {
        Ok(reply) => {
            let mut guard = history.lock().expect("pipe history lock poisoned");
            guard.push(user_turn);
            guard.push(Turn::assistant(reply.clone()));
            Ok(reply)
        }
        Err(e) => {
            // `?e` (Debug) keeps the source chain — e.g. the inner reqwest kind
            // (timeout / connection-closed / decode) that `%e` would discard.
            warn!(window = window_n, %tag, error = ?e, "pipe LLM call failed");
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
