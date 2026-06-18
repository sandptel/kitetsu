//! kitetsu primitives — the iced render layer.
//!
//! `elements` holds shared, message-polymorphic building blocks (buttons,
//! typography, window chrome); `presets` composes them into ready-made card UIs
//! for kitetsu tools (currently `teleprompter`). Deliberately knows nothing about
//! IPC, sessions, audio, the agent loop, or the filesystem — callers build a
//! preset's data struct and render it; positioning and app wiring live in the
//! binary that hosts the iced loop.

pub mod elements;
pub mod presets;

pub use elements::chrome::Edge;
pub use elements::typography::{FONT_BOLD, FONT_NAME, FONT_REGULAR};
