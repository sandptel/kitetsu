//! kitetsu primitives — the iced render layer.
//!
//! `elements` holds shared, message-polymorphic building blocks (buttons,
//! typography, window chrome); `presets` composes them into ready-made card UIs
//! for kitetsu tools (currently `teleprompter`); `theme` is the base16 colour
//! palette presets paint from. Deliberately knows nothing about IPC, sessions,
//! audio, the agent loop, or the filesystem — callers build a preset's data
//! struct + palette and render it; loading and app wiring live in the binary
//! that hosts the iced loop.

pub mod elements;
pub mod presets;
pub mod theme;

pub use elements::chrome::Edge;
pub use elements::typography::{FONT_BOLD, FONT_NAME, FONT_REGULAR};
pub use theme::Base16;
