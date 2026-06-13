//! Writes pipe results as timestamped Markdown blocks to `out/pipe{1,2,3}.md`.
//!
//! Each window appends (or overwrites, per [`OutputMode`]) a block headed
//! `## window N — HH:MM:SS UTC (+Xs)` so a `tail -f` of the file reads as a log
//! of suggestions. Not here: the LLM call or window state — callers hand this a
//! finished body string.

use std::fs::{self, OpenOptions};
use std::io::Write as _;
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use super::config::OutputMode;

/// Append or overwrite a window's Markdown block at `path`.
///
/// Creates the parent directory if needed. `latency` is the trigger-to-result
/// duration; `body` is the suggestion (or an error message) — it is trimmed and
/// wrapped in a block with a header and a trailing `---` rule.
pub fn write_block(
    path: &Path,
    window_n: u64,
    latency: Duration,
    body: &str,
    mode: OutputMode,
) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let block = format!(
        "## window {window_n} — {} UTC (+{:.1}s)\n\n{}\n\n---\n\n",
        utc_hms(SystemTime::now()),
        latency.as_secs_f32(),
        body.trim(),
    );

    match mode {
        OutputMode::Append => {
            let mut file = OpenOptions::new().create(true).append(true).open(path)?;
            file.write_all(block.as_bytes())
        }
        OutputMode::Overwrite => fs::write(path, block.as_bytes()),
    }
}

/// Format the UTC wall-clock time of `now` as `HH:MM:SS` (no date, no chrono dep).
fn utc_hms(now: SystemTime) -> String {
    let secs = now
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let day = secs % 86_400;
    format!("{:02}:{:02}:{:02}", day / 3600, (day % 3600) / 60, day % 60)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    fn temp_path() -> std::path::PathBuf {
        static N: AtomicU32 = AtomicU32::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "kitetsu_out_{}_{}_{n}.md",
            std::process::id(),
            "test"
        ))
    }

    #[test]
    fn append_keeps_every_block() {
        let path = temp_path();
        let _ = fs::remove_file(&path);
        write_block(&path, 1, Duration::from_millis(900), "first", OutputMode::Append).unwrap();
        write_block(&path, 2, Duration::from_secs(2), "second", OutputMode::Append).unwrap();

        let text = fs::read_to_string(&path).unwrap();
        assert!(text.contains("## window 1"));
        assert!(text.contains("## window 2"));
        assert!(text.contains("first"));
        assert!(text.contains("second"));
        assert!(text.contains("(+0.9s)"));
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn overwrite_keeps_only_latest() {
        let path = temp_path();
        let _ = fs::remove_file(&path);
        write_block(&path, 1, Duration::ZERO, "old", OutputMode::Overwrite).unwrap();
        write_block(&path, 2, Duration::ZERO, "new", OutputMode::Overwrite).unwrap();

        let text = fs::read_to_string(&path).unwrap();
        assert!(!text.contains("old"));
        assert!(text.contains("new"));
        assert!(text.contains("## window 2"));
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn creates_missing_parent_dir() {
        let dir = std::env::temp_dir().join(format!("kitetsu_outdir_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let path = dir.join("pipe1.md");
        write_block(&path, 1, Duration::ZERO, "hi", OutputMode::Append).unwrap();
        assert!(path.exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn utc_hms_is_zero_padded_and_in_range() {
        let s = utc_hms(UNIX_EPOCH + Duration::from_secs(3661)); // 01:01:01
        assert_eq!(s, "01:01:01");
    }
}
