//! Prototype launcher for the kitetsu DescriptionCard primitive.
//! Spawns a single iced_layershell surface anchored top-left, renders one
//! dummy card, and exits cleanly. No IPC, no LLM, no daemon — layout only.

use iced::{Color, Element, Task, theme};
use iced_layershell::application;
use iced_layershell::reexport::{Anchor, KeyboardInteractivity};
use iced_layershell::settings::{LayerShellSettings, Settings};
use iced_layershell::to_layer_message;

use kitetsu_primitives::card::{self, Card};

fn main() -> iced_layershell::Result {
    application(init, namespace, update, view)
        .settings(Settings {
            layer_settings: LayerShellSettings {
                // Generous surface; the card shrinks to its content within this.
                size: Some((900, 700)),
                // Bottom-left anchor with a 20 px margin on both edges.
                anchor: Anchor::Bottom | Anchor::Left,
                margin: (20, 0, 0, 20),
                // Non-interactive: clicks that miss the card fall through to the desktop.
                keyboard_interactivity: KeyboardInteractivity::None,
                ..Default::default()
            },
            ..Default::default()
        })
        .style(|_state, _theme| theme::Style {
            // Transparent surface so only the white card is visible over the desktop.
            background_color: Color::TRANSPARENT,
            text_color: Color::BLACK,
        })
        .run()
}

// ── Application state ─────────────────────────────────────────────────────────

struct App {
    card: Card,
}

// ── Messages ──────────────────────────────────────────────────────────────────

/// App messages. The card's own [`card::Message`] is mapped into `Card` here.
/// #[to_layer_message] adds the LayerShell action variants required by the
/// iced_layershell TryInto<LayerShellCustomActionWithId> bound. See PLAN §4.1.
#[to_layer_message]
#[derive(Debug, Clone)]
pub enum Message {
    /// A button on the card was pressed (drag handle only, for now).
    Card(card::Message),
}

// ── iced_layershell program functions ─────────────────────────────────────────

/// State factory — called once at startup.
fn init() -> (App, Task<Message>) {
    (
        App {
            card: Card::dummy(),
        },
        Task::none(),
    )
}

/// Wayland namespace string (used by the compositor to identify this surface).
fn namespace() -> String {
    "kitetsu".into()
}

/// Message handler — buttons are no-ops for now.
fn update(_app: &mut App, _msg: Message) -> Task<Message> {
    Task::none()
}

/// View — delegates to the card module, tagging its messages as `Message::Card`.
fn view(app: &App) -> Element<'_, Message> {
    card::view(&app.card).map(Message::Card)
}
