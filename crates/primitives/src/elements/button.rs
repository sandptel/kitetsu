//! Shared button elements. Currently a small press-to-activate SVG icon,
//! generic over the message a preset wants to emit on press. No styling opinions
//! beyond size + alpha; the SVG carries its own look.

use iced::Element;
use iced::widget::{container, mouse_area, svg};

/// A small press-to-activate icon: an SVG at `alpha` that emits `msg` on press.
///
/// `mouse_area` (not `button`) so the press registers immediately on press-down.
pub fn icon_button<'a, M: Clone + 'a>(bytes: &'static [u8], alpha: f32, msg: M) -> Element<'a, M> {
    mouse_area(
        container(
            svg(svg::Handle::from_memory(bytes))
                .width(20)
                .height(20)
                .opacity(alpha),
        )
        .padding(6),
    )
    .on_press(msg)
    .into()
}
