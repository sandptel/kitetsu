//! Live rolling-window transcription from mic and system audio simultaneously.
//!
//! Renders two stacked ANSI panels — one per source — redrawn in place as
//! utterances arrive. Committed text is shown in normal colour; the tentative
//! tail (empty for VadGated, populated once LocalAgreement lands) is dimmed.
//!
//! Usage: `cargo run -p kitetsu --example live_transcribe`
//!
//! Prerequisites:
//!   - A GGML Whisper model at `$KITETSU_WHISPER_MODEL` or
//!     `$XDG_DATA_HOME/kitetsu/models/ggml-base.en.bin`.
//!   - PipeWire-pulse or PulseAudio running.

use std::io::{BufRead as _, Write as _};
use std::sync::{Arc, Mutex};

use anyhow::Context as _;

use kitetsu::listener::{
    AudioModel, StreamMode, StreamSession, StreamSource, discover_default_devices,
};

// ── Display constants ─────────────────────────────────────────────────────────

/// Characters of text content per panel line (between the │ borders).
const PANEL_INNER_WIDTH: usize = 66;
/// Number of text lines inside each panel box.
const PANEL_CONTENT_LINES: usize = 3;
/// Total lines per panel (content + top border + bottom border).
const PANEL_HEIGHT: usize = PANEL_CONTENT_LINES + 2;
/// Total lines drawn: two panels + one prompt line.
const TOTAL_DRAW_LINES: usize = PANEL_HEIGHT * 2 + 1;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("kitetsu=info")),
        )
        .init();

    // ── Discover devices ──────────────────────────────────────────────────────
    println!("Discovering audio devices...");
    let devices = discover_default_devices().context("PulseAudio device discovery failed")?;
    println!("  mic:    {}", devices.mic_source);
    println!("  system: {}", devices.system_monitor);
    println!();

    // ── Load models + start session ───────────────────────────────────────────
    println!("Loading models (one per source)...");
    let sources = [StreamSource::Mic, StreamSource::System];
    let (session, update_rx) =
        StreamSession::start(&sources, AudioModel::Default, StreamMode::VadGated)
            .context("failed to start streaming session")?;
    println!("Models ready. Starting live transcription.\n");

    // ── Initial empty panel draw ──────────────────────────────────────────────
    let state = Arc::new(Mutex::new(DisplayState::default()));
    {
        let s = state.lock().unwrap();
        print_panels(&s);
        std::io::stdout().flush().ok();
    }

    // ── Display thread: receives updates and redraws panels ───────────────────
    let state_for_display = Arc::clone(&state);
    let display = std::thread::spawn(move || {
        for update in &update_rx {
            let mut s = state_for_display.lock().unwrap();
            match update.source {
                StreamSource::Mic => {
                    if !s.mic.is_empty() {
                        s.mic.push(' ');
                    }
                    s.mic.push_str(&update.committed);
                    s.mic_tentative = update.tentative;
                }
                StreamSource::System => {
                    if !s.sys.is_empty() {
                        s.sys.push(' ');
                    }
                    s.sys.push_str(&update.committed);
                    s.sys_tentative = update.tentative;
                }
            }
            print_panels(&s);
            std::io::stdout().flush().ok();
        }
    });

    // ── Main thread: wait for Enter ───────────────────────────────────────────
    // Read from stdin while the display thread owns stdout redraws.
    let stdin = std::io::stdin();
    stdin.lock().lines().next();

    session.stop().context("error stopping session")?;
    display.join().ok();

    // Move cursor below the panels for clean final output.
    println!("\n\nStopped.");
    let s = state.lock().unwrap();
    if !s.mic.is_empty() {
        println!("\n[Mic]\n{}", s.mic);
    }
    if !s.sys.is_empty() {
        println!("\n[System]\n{}", s.sys);
    }

    Ok(())
}

// ── Display ───────────────────────────────────────────────────────────────────

#[derive(Default)]
struct DisplayState {
    mic: String,
    mic_tentative: String,
    sys: String,
    sys_tentative: String,
}

/// Overwrite the panel area and reprint the prompt line.
///
/// Uses `\x1b[{n}F` (CPL — Cursor Previous Line) to jump back to the top of the
/// panel area. On the first call there is nothing above to overwrite; subsequent
/// calls redraw in place.
fn print_panels(s: &DisplayState) {
    // On the very first call the cursor is already below the last `println!`
    // from setup. Subsequent calls cursor is just below the prompt line, so
    // we need to go up TOTAL_DRAW_LINES lines to reach the top of the Mic panel.
    static FIRST: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    let is_first = FIRST.set(()).is_ok();
    if !is_first {
        print!("\x1b[{TOTAL_DRAW_LINES}F");
    }

    draw_panel("Mic", &s.mic, &s.mic_tentative);
    draw_panel("System", &s.sys, &s.sys_tentative);
    print!("Listening... press Enter to stop.    ");
    println!(); // advance past the prompt so next CPL lands here
}

fn draw_panel(title: &str, committed: &str, tentative: &str) {
    // Top border: ┌─ Title ──...──┐
    let title_seg = format!("─ {title} ");
    let fill_len = PANEL_INNER_WIDTH + 2 - title_seg.len().min(PANEL_INNER_WIDTH + 2);
    println!("┌{}{}┐", title_seg, "─".repeat(fill_len));

    let lines = wrap_text(committed, PANEL_INNER_WIDTH);
    let n = lines.len();

    // Decide how many lines to give committed vs tentative.
    let tentative_lines = if tentative.is_empty() { 0 } else { 1 };
    let committed_lines = PANEL_CONTENT_LINES - tentative_lines;

    for i in 0..committed_lines {
        let idx = n.saturating_sub(committed_lines) + i;
        let text = lines.get(idx).map(String::as_str).unwrap_or("");
        println!("│ {text:<PANEL_INNER_WIDTH$} │");
    }

    if tentative_lines > 0 {
        let t = truncate_right(tentative, PANEL_INNER_WIDTH);
        println!("│ \x1b[2m{t:<PANEL_INNER_WIDTH$}\x1b[0m │");
    }

    // Bottom border
    println!("└{}┘", "─".repeat(PANEL_INNER_WIDTH + 2));
}

/// Word-wrap `text` into lines of at most `width` characters.
fn wrap_text(text: &str, width: usize) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        if current.is_empty() {
            current.push_str(word);
        } else if current.len() + 1 + word.len() <= width {
            current.push(' ');
            current.push_str(word);
        } else {
            lines.push(std::mem::take(&mut current));
            current.push_str(word);
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    lines
}

/// Return the last `width` chars of `s` (right-aligned truncation).
fn truncate_right(s: &str, width: usize) -> &str {
    if s.len() <= width {
        return s;
    }
    // Trim to a char boundary.
    let start = s.len() - width;
    let trim = s[start..].find(|_: char| true).unwrap_or(0);
    &s[start + trim..]
}
