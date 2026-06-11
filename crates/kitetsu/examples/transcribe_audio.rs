//! Transcribe microphone or system audio to text via the default Whisper model.
//!
//! Usage: `cargo run -p kitetsu --example transcribe_audio -- [input|output]`
//! Default: `input` (microphone). Records for DEFAULT_CAPTURE_SECS seconds.
//!
//! Prerequisites:
//!   - A GGML Whisper model at `$KITETSU_WHISPER_MODEL` or
//!     `$XDG_DATA_HOME/kitetsu/models/ggml-base.en.bin`.
//!   - PipeWire-pulse or PulseAudio running.

use kitetsu::listener::{AudioModel, InputSink, OutputSink, transcribe_input, transcribe_output};
use tracing::info;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("kitetsu=info")),
        )
        .init();

    let stream = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "input".to_owned());
    let model = AudioModel::Default;

    info!(
        model_available = model.available(),
        "using AudioModel::Default",
    );

    let text = match stream.as_str() {
        "output" => {
            info!("transcribing system audio output");
            transcribe_output(OutputSink::Default, model).await?
        }
        _ => {
            info!("transcribing microphone input");
            transcribe_input(InputSink::Default, model).await?
        }
    };

    println!("{text}");
    Ok(())
}