//! Shared window chrome: the drag handle and resize grips that make a floating
//! card movable + resizable. Each builder is generic over the host's message
//! type — the preset passes the message it wants on grab. Geometry + the faint
//! grip fill live here; what the gestures *do* is the host's business.

use iced::widget::{container, mouse_area, space, svg};
use iced::{Background, Color, Element, Length};

/// Which grip the user grabbed to resize a card.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Edge {
    /// Right edge — adjusts width only.
    Right,
    /// Bottom edge — adjusts height only.
    Bottom,
    /// Bottom-right corner — adjusts both.
    Corner,
}

const DRAG_SVG: &[u8] = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/drag.svg"));

/// Resize-grip thickness and corner size (px).
pub const GRIP: f32 = 8.0;
pub const CORNER: f32 = 16.0;

/// The drag handle: an SVG at `icon_alpha` that emits `on_press` on press-down so
/// the host can start moving the card.
pub fn drag_handle<'a, M: Clone + 'a>(icon_alpha: f32, on_press: M) -> Element<'a, M> {
    mouse_area(
        container(
            svg(svg::Handle::from_memory(DRAG_SVG))
                .width(22)
                .height(22)
                .opacity(icon_alpha),
        )
        .padding(8),
    )
    .on_press(on_press)
    .into()
}

/// Right-edge resize grip (full height, [`GRIP`] wide).
pub fn right_grip<'a, M: Clone + 'a>(msg: M) -> Element<'a, M> {
    grip(Length::Fixed(GRIP), Length::Fill, msg)
}

/// Bottom-edge resize grip (full width, [`GRIP`] tall).
pub fn bottom_grip<'a, M: Clone + 'a>(msg: M) -> Element<'a, M> {
    grip(Length::Fill, Length::Fixed(GRIP), msg)
}

/// Bottom-right corner resize grip ([`CORNER`] square).
pub fn corner_grip<'a, M: Clone + 'a>(msg: M) -> Element<'a, M> {
    grip(Length::Fixed(CORNER), Length::Fixed(CORNER), msg)
}

/// A press-to-resize grip of the given size, faintly filled so it is discoverable.
fn grip<'a, M: Clone + 'a>(width: Length, height: Length, msg: M) -> Element<'a, M> {
    mouse_area(
        container(space())
            .width(width)
            .height(height)
            .style(grip_style),
    )
    .on_press(msg)
    .into()
}

/// Faint, semi-transparent fill so the resize grips are discoverable.
fn grip_style(_theme: &iced::Theme) -> container::Style {
    container::Style {
        background: Some(Background::Color(Color::from_rgba(0.0, 0.0, 0.0, 0.10))),
        ..container::Style::default()
    }
}
