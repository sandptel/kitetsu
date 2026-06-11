//! Audio capture sub-package: device discovery and blocking PCM recording.
//!
//! `discover` resolves PulseAudio device names via server introspection.
//! `record` provides blocking capture loops (48 kHz WAV dump + 16 kHz STT).
//! Neither sub-module touches tokio; both are designed for `std::thread`.

mod discover;
mod record;

pub use discover::{Devices, discover_default_devices};
pub use record::{CaptureConfig, DEFAULT_CAPTURE_SECS, capture_samples, capture_to_wav};
