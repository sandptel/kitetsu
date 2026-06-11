//! Renders the DescriptionCard primitive — layout and styling only, no business logic.
//! Deliberately does not know about IPC, sessions, or the agent loop.
//! The Card struct holds the data; `view()` turns it into an iced Element.

use iced::widget::text::Span;
use iced::widget::{button, column, container, rich_text, row, space, span, svg, text};
use iced::{
    Alignment, Background, Border, Color, Element, Font, Length, Padding, Shadow, Vector, font,
};

use crate::Message;

// Assets embedded at compile time — avoids runtime working-directory dependency.
const LOGO_SVG: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/logo-gemini.svg"
));
const BOOKMARK_SVG: &[u8] =
    include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/bookmark.svg"));
const SHARE_SVG: &[u8] = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/share.svg"));
const MENU_SVG: &[u8] = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/menu.svg"));
const DRAG_SVG: &[u8] = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/drag.svg"));

/// All data needed to render one description card.
pub struct Card {
    pub header: String,
    pub model: String,
    /// Body text broken into runs; each run is (text, is_bold).
    pub body: Vec<(String, bool)>,
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
        }
    }
}

/// Render a DescriptionCard as an iced Element.
pub fn view(card: &Card) -> Element<'_, Message> {
    // ── Left cluster: logo + header column ───────────────────────────────────
    let logo = svg(svg::Handle::from_memory(LOGO_SVG)).width(52).height(52);

    let header_col = column![
        text(&card.header).size(22).font(Font {
            weight: font::Weight::Bold,
            ..Font::default()
        }),
        text(&card.model)
            .size(17)
            .color(Color::from_rgb(0.37, 0.39, 0.41)), // #5f6368 muted gray
    ]
    .spacing(2);

    let left_cluster = row![logo, header_col]
        .spacing(14)
        .align_y(Alignment::Center);

    // ── Right cluster: icon action buttons (no-op for now) ───────────────────
    // Transparent button style: no background, no border.
    // A hover highlight is added so buttons are still discoverable.
    let icon_btn = |bytes: &'static [u8], msg: Message| {
        let handle = svg::Handle::from_memory(bytes);
        button(svg(handle).width(22).height(22))
            .on_press(msg)
            .style(|_theme, status| {
                let bg = match status {
                    button::Status::Hovered | button::Status::Pressed => {
                        Some(Background::Color(Color::from_rgba(0.0, 0.0, 0.0, 0.06)))
                    }
                    _ => None,
                };
                button::Style {
                    background: bg,
                    border: Border {
                        radius: 8.0.into(),
                        ..Border::default()
                    },
                    ..Default::default()
                }
            })
            .padding(8)
    };

    let right_cluster = row![
        icon_btn(DRAG_SVG, Message::Drag),
        icon_btn(BOOKMARK_SVG, Message::Bookmark),
        icon_btn(SHARE_SVG, Message::Share),
        icon_btn(MENU_SVG, Message::Menu),
    ]
    .spacing(4)
    .align_y(Alignment::Center);

    // ── Top bar: left + flexible spacer + right ───────────────────────────────
    let top_bar = row![left_cluster, space().width(Length::Fill), right_cluster]
        .width(Length::Fill)
        .align_y(Alignment::Center);

    // ── Caption: body text with inline-bold spans ─────────────────────────────
    // rich_text wraps at the container width, growing the card vertically.
    // Link = () because these are plain text spans with no hyperlinks.
    let spans: Vec<Span<'_, ()>> = card
        .body
        .iter()
        .map(|(txt, bold)| {
            if *bold {
                span(txt.as_str()).font(Font {
                    weight: font::Weight::Bold,
                    ..Font::default()
                })
            } else {
                span(txt.as_str())
            }
        })
        .collect();

    let caption = rich_text(spans).size(19).width(Length::Fill);

    // ── Card: column in a white rounded-corner container with drop shadow ─────
    // max_width caps horizontal growth; vertical growth is unbounded (content-driven).
    let card_col = column![top_bar, caption].spacing(20).padding(Padding::ZERO);

    container(card_col)
        .padding(28)
        .max_width(860)
        .style(|_theme| container::Style {
            background: Some(Background::Color(Color::WHITE)),
            border: Border {
                radius: 20.0.into(),
                ..Border::default()
            },
            shadow: Shadow {
                color: Color::from_rgba(0.0, 0.0, 0.0, 0.18),
                offset: Vector::new(0.0, 4.0),
                blur_radius: 24.0,
            },
            ..container::Style::default()
        })
        .into()
}
