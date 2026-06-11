//! Speech-recognition sub-package: model loading and one-shot inference.
//!
//! `model` defines `AudioModel` (the public backend selector) and the internal
//! `LoadedModel` handle. `engine` runs synchronous inference on f32 samples.
//! Neither module performs audio capture; rolling-window / live transcription
//! is a future iteration.

mod engine;
pub(crate) mod model;

pub(crate) use engine::transcribe_samples;
pub use model::AudioModel;
