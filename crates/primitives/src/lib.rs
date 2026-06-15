//! kitetsu primitives — the iced render layer.
//!
//! Holds the typed UI primitives and their iced rendering. Currently the single
//! `DescriptionCard` primitive (`card`). Deliberately knows nothing about IPC,
//! sessions, audio, or the agent loop — callers build a [`card::Card`] and render
//! it; positioning and app wiring live in the binary that hosts the iced loop.

pub mod card;

pub use card::{Card, Message, view};
