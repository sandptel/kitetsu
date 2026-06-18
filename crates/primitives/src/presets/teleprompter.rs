//! The teleprompter preset: a description card (logo + header + suggestion body)
//! with overlay controls and resize chrome. Layout + styling only — no IPC,
//! sessions, or agent loop. The [`Card`] struct holds the data; [`view`] turns it
//! into an iced Element. Composes the shared [`crate::elements`].

use iced::widget::text::Span;
use iced::widget::{column, container, rich_text, row, scrollable, space, span, svg, text};
use iced::{Alignment, Background, Border, Color, Element, Length, Shadow, Vector, font};

use crate::elements::button::icon_button;
use crate::elements::chrome::{self, Edge};
use crate::elements::typography::{DEFAULT_FONT_SIZE, jb};
use crate::theme::Base16;

/// Card pointer interactions emitted to the host app. Drag/Resize drive in-app
/// gestures; Toggle/Pause/Trash mirror the daemon control commands one-for-one
/// (the host fires the matching IPC command).
#[derive(Debug, Clone)]
pub enum Message {
    /// Drag handle pressed — start moving the card.
    Drag,
    /// A resize grip pressed — start resizing from that edge.
    Resize(Edge),
    /// Hide/show the overlay (same as the `toggle` command).
    Toggle,
    /// Pause/resume recording (same as the `pause` command).
    Pause,
    /// Discard the conversation so far (same as the `trash` command).
    Trash,
}

// Teleprompter-specific assets, embedded at compile time (no runtime cwd dependency).
const LOGO_SVG: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/logo-gemini.svg"
));
const TOGGLE_SVG: &[u8] = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/toggle.svg"));
const RECORD_SVG: &[u8] = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/record.svg"));
const RESUME_SVG: &[u8] = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/resume.svg"));
const TRASH_SVG: &[u8] = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/trash.svg"));

/// All data needed to render one teleprompter card.
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
    /// Whether audio is being recorded. `true` → flashing red dot; `false` → resume (play) icon.
    pub recording: bool,
    /// Flash phase for the recording dot, toggled by the host on a timer.
    pub blink_on: bool,
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
            manual_height: Some(400.0),
            recording: true,
            blink_on: true,
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

/// A palette colour with its alpha replaced by `a` (for the card's opacity rules).
fn with_alpha(c: Color, a: f32) -> Color {
    Color { a, ..c }
}

/// Render the teleprompter card as an iced Element.
///
/// Colours come from `palette` (a base16 scheme); sizes scale off
/// `card.font_size`; background, text, and icons use the card's resolved alphas
/// so the whole card can be made see-through over the desktop.
pub fn view<'a>(card: &'a Card, palette: &Base16) -> Element<'a, Message> {
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

    // Body text in the palette's foreground; the model line is its muted tone.
    let text_color = with_alpha(palette.text(), text_alpha);
    let muted_color = with_alpha(palette.muted(), text_alpha);
    let bg_color = with_alpha(palette.background(), bg_alpha);

    // ── Left cluster: logo + header column ───────────────────────────────────
    let logo = svg(svg::Handle::from_memory(LOGO_SVG)).width(52).height(52);

    let header_col = column![
        text(&card.header)
            .size(header_size)
            .color(text_color)
            .font(jb(font::Weight::Bold)),
        text(&card.model)
            .size(model_size)
            .color(muted_color)
            .font(jb(font::Weight::Normal)),
    ]
    .spacing(2);

    let left_cluster = row![logo, header_col]
        .spacing(14)
        .align_y(Alignment::Center);

    // ── Right cluster: control icons + the drag handle ────────────────────────
    let toggle_btn = icon_button(TOGGLE_SVG, icon_alpha, Message::Toggle);
    let trash_btn = icon_button(TRASH_SVG, icon_alpha, Message::Trash);

    // Pause control: a red dot while recording (flashing via `blink_on`), or the
    // resume (play) icon while paused.
    let (pause_icon, pause_alpha) = if card.recording {
        let a = if card.blink_on { icon_alpha } else { icon_alpha * 0.2 };
        (RECORD_SVG, a)
    } else {
        (RESUME_SVG, icon_alpha)
    };
    let pause_btn = icon_button(pause_icon, pause_alpha, Message::Pause);

    let drag_handle = chrome::drag_handle(icon_alpha, Message::Drag);

    // ── Top bar: left + flexible spacer + controls + drag handle ──────────────
    let controls = row![toggle_btn, pause_btn, trash_btn, drag_handle]
        .spacing(4)
        .align_y(Alignment::Center);
    let top_bar = row![left_cluster, space().width(Length::Fill), controls]
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
                s.font(jb(font::Weight::Bold))
            } else {
                s.font(jb(font::Weight::Normal))
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
    let right_grip = chrome::right_grip(Message::Resize(Edge::Right));
    let bottom_grip = chrome::bottom_grip(Message::Resize(Edge::Bottom));
    let corner_grip = chrome::corner_grip(Message::Resize(Edge::Corner));

    // inner + right grip, then a bottom row with the bottom grip + corner.
    // `upper`/`card_col` fill so the bottom grips pin to the card's true bottom
    // when it has a fixed height (initial or manually resized).
    let upper = row![inner, right_grip].height(Length::Fill);
    let lower = row![bottom_grip, corner_grip];
    let card_col = column![upper, lower].height(Length::Fill);

    // ── Card: fixed width; height is manual (fixed) or auto (shrink + cap) ─────
    let card = container(card_col)
        .width(Length::Fixed(card.width))
        .style(move |_theme| container::Style {
            // Card background from the palette.
            background: Some(Background::Color(bg_color)),
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
