//! Shared, message-polymorphic building blocks composed by presets: buttons,
//! typography, and window chrome (drag handle + resize grips). Each is generic
//! over the host's message type and free of any single preset's logic.

pub mod button;
pub mod chrome;
pub mod typography;
