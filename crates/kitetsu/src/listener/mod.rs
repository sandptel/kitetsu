//! Audio capture from the microphone and system output (sink monitor) via
//! PulseAudio / PipeWire-pulse. This module owns device discovery and all
//! blocking audio I/O.
//!
//! Deliberately knows nothing about LLM interactions, session state, UI, or
//! transcription — those are the concern of `prompter` and `router`.
//! Each capture stream runs on a dedicated `std::thread`; libpulse objects
//! are thread-local and must never be placed on the tokio runtime.

mod capture;
mod discover;

pub use capture::{CaptureConfig, ListenerError, capture_to_wav};
pub use discover::{Devices, discover_default_devices};
