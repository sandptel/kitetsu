//! Renders the DescriptionCard primitive — layout and styling only, no business logic.
//! Deliberately does not know about IPC, sessions, or the agent loop.
//! The Card struct holds the data; `view()` turns it into an iced Element.

use iced::widget::text::Span;
use iced::widget::{
    column, container, mouse_area, rich_text, row, scrollable, space, span, svg, text,
};
use iced::{Alignment, Background, Border, Color, Element, Font, Length, Shadow, Vector, font};

/// Which grip the user grabbed to resize the card.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Edge {
    /// Right edge — adjusts width only.
    Right,
    /// Bottom edge — adjusts height only.
    Bottom,
    /// Bottom-right corner — adjusts both.
    Corner,
}

/// Card pointer interactions emitted to the host app.
#[derive(Debug, Clone)]
pub enum Message {
    /// Drag handle pressed — start moving the card.
    Drag,
    /// A resize grip pressed — start resizing from that edge.
    Resize(Edge),
}

// Assets embedded at compile time — avoids runtime working-directory dependency.
const LOGO_SVG: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/logo-gemini.svg"
));
const DRAG_SVG: &[u8] = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/drag.svg"));

/// Base body font size used when a card does not override it (matches the mockup).
pub const DEFAULT_FONT_SIZE: f32 = 19.0;

/// All data needed to render one description card.
pub struct Card {
    pub header: String,
    pub model: String,
    /// Body text broken into runs; each run is (text, is_bold).
    pub body: Vec<(String, bool)>,
    /// Body text size; header and model derive from it (≈1.15× / ≈0.9×).
    pub font_size: f32,
    /// Master alpha (0.0–1.0). Icons use this; bg/text inherit it unless overridden.
    pub opacity: f32,
    /// Optional background-alpha override; `None` → inherit `opacity`.
    pub bg_opacity: Option<f32>,
    /// Optional text-alpha override; `None` → inherit `opacity`.
    pub text_opacity: Option<f32>,
    /// Fixed card width in px (content wraps to this).
    pub width: f32,
    /// Maximum card height in px; the body scrolls beyond it. `f32::INFINITY` = no cap.
    /// Ignored once `manual_height` is set.
    pub max_height: f32,
    /// Explicit height set by manual resize; `None` → auto-grow (capped by `max_height`).
    pub manual_height: Option<f32>,
}

impl Card {
    /// Filler card that matches the design mockup.
    pub fn dummy() -> Self {
        Self {
            header: "<Header from the llm>".into(),
            model: "@ gemini-3.5-flash".into(),
            body: vec![
                (
                    "Hello Sir, I can see that you are currently reviewing an article on ".into(),
                    false,
                ),
                ("Supervised Fine Tuning".into(), true),
                (
                    " (SFT) which is the process of taking a pre-trained machine learning \
                     model and training it further on a smaller, labeled dataset to adapt \
                     it for a specific task.\n\nPlease mention any thing I can assist you with?"
                        .into(),
                    false,
                ),
            ],
            font_size: DEFAULT_FONT_SIZE,
            opacity: 1.0,
            bg_opacity: None,
            text_opacity: None,
            width: 600.0,
            max_height: f32::INFINITY,
            manual_height: None,
        }
    }

    /// Background alpha after applying the override-then-inherit rule.
    fn bg_alpha(&self) -> f32 {
        self.bg_opacity.unwrap_or(self.opacity)
    }

    /// Text alpha after applying the override-then-inherit rule.
    fn text_alpha(&self) -> f32 {
        self.text_opacity.unwrap_or(self.opacity)
    }
}

/// Render a DescriptionCard as an iced Element.
///
/// Sizes scale off `card.font_size`; background, text, and the drag icon use the
/// card's resolved alphas so the whole card can be made see-through over the desktop.
pub fn view(card: &Card) -> Element<'_, Message> {
    let text_alpha = card.text_alpha();
    let bg_alpha = card.bg_alpha();
    let icon_alpha = card.opacity;

    // Header/model derive from the body size for proportional scaling.
    let body_size = card.font_size;
    let header_size = body_size * 1.15;
    let model_size = body_size * 0.9;

    // Captured before the param is shadowed by the card container below.
    let manual_height = card.manual_height;
    let max_height = card.max_height;

    // Black/muted-gray text, faded by the resolved text alpha.
    let text_color = Color::from_rgba(0.0, 0.0, 0.0, text_alpha);
    let muted_color = Color::from_rgba(0.37, 0.39, 0.41, text_alpha); // #5f6368

    // ── Left cluster: logo + header column ───────────────────────────────────
    let logo = svg(svg::Handle::from_memory(LOGO_SVG)).width(52).height(52);

    let header_col = column![
        text(&card.header)
            .size(header_size)
            .color(text_color)
            .font(Font {
                weight: font::Weight::Bold,
                ..Font::default()
            }),
        text(&card.model).size(model_size).color(muted_color),
    ]
    .spacing(2);

    let left_cluster = row![logo, header_col]
        .spacing(14)
        .align_y(Alignment::Center);

    // ── Right cluster: the drag handle only ──────────────────────────────────
    // A mouse_area (not a button) so the grab starts on press-down — buttons only
    // fire on release, which can't drive hold-to-drag. The host app handles `Drag`.
    let drag_handle = mouse_area(
        container(
            svg(svg::Handle::from_memory(DRAG_SVG))
                .width(22)
                .height(22)
                .opacity(icon_alpha),
        )
        .padding(8),
    )
    .on_press(Message::Drag);

    // ── Top bar: left + flexible spacer + drag handle ─────────────────────────
    let top_bar = row![left_cluster, space().width(Length::Fill), drag_handle]
        .width(Length::Fill)
        .align_y(Alignment::Center);

    // ── Caption: body text with inline-bold spans ─────────────────────────────
    // rich_text wraps at the container width, growing the card vertically.
    // Link = () because these are plain text spans with no hyperlinks.
    let spans: Vec<Span<'_, ()>> = card
        .body
        .iter()
        .map(|(txt, bold)| {
            let s = span(txt.as_str()).color(text_color);
            if *bold {
                s.font(Font {
                    weight: font::Weight::Bold,
                    ..Font::default()
                })
            } else {
                s
            }
        })
        .collect();

    let caption = rich_text(spans).size(body_size).width(Length::Fill);

    // ── Body: header bar + scrollable text, padded for breathing room ──────────
    // Width is fixed; the body grows the card down to `max_height`, then scrolls.
    let body = scrollable(caption).height(Length::Fill).width(Length::Fill);
    let inner = container(column![top_bar, body].spacing(20))
        .padding(28)
        .width(Length::Fill)
        .height(Length::Fill);

    // ── Resize grips at the true edges (outside the inner padding) ─────────────
    let right_grip = mouse_area(
        container(space())
            .width(Length::Fixed(GRIP))
            .height(Length::Fill)
            .style(grip_style),
    )
    .on_press(Message::Resize(Edge::Right));
    let bottom_grip = mouse_area(
        container(space())
            .width(Length::Fill)
            .height(Length::Fixed(GRIP))
            .style(grip_style),
    )
    .on_press(Message::Resize(Edge::Bottom));
    let corner_grip = mouse_area(
        container(space())
            .width(Length::Fixed(CORNER))
            .height(Length::Fixed(CORNER))
            .style(grip_style),
    )
    .on_press(Message::Resize(Edge::Corner));

    // inner + right grip, then a bottom row with the bottom grip + corner.
    let upper = row![inner, right_grip];
    let lower = row![bottom_grip, corner_grip];
    let card_col = column![upper, lower];

    // ── Card: fixed width; height is manual (fixed) or auto (shrink + cap) ─────
    let card = container(card_col)
        .width(Length::Fixed(card.width))
        .style(move |_theme| container::Style {
            background: Some(Background::Color(Color::from_rgba(1.0, 1.0, 1.0, bg_alpha))),
            border: Border {
                radius: 20.0.into(),
                ..Border::default()
            },
            shadow: Shadow {
                color: Color::from_rgba(0.0, 0.0, 0.0, 0.18 * bg_alpha),
                offset: Vector::new(0.0, 4.0),
                blur_radius: 24.0,
            },
            ..container::Style::default()
        });

    match manual_height {
        Some(h) => card.height(Length::Fixed(h)),
        None => card.height(Length::Shrink).max_height(max_height),
    }
    .into()
}

/// Resize-grip thickness and corner size (px).
const GRIP: f32 = 8.0;
const CORNER: f32 = 16.0;

/// Faint, semi-transparent fill so the resize grips are discoverable.
fn grip_style(_theme: &iced::Theme) -> container::Style {
    container::Style {
        background: Some(Background::Color(Color::from_rgba(0.0, 0.0, 0.0, 0.10))),
        ..container::Style::default()
    }
}
