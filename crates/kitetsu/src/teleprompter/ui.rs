//! The teleprompter overlay — the only place in the binary that touches iced.
//!
//! A single transparent, fullscreen Wayland layer surface. Each enabled pipe is
//! drawn as a [`kitetsu_primitives::card::Card`] positioned at an `(x, y)` offset
//! inside that one surface (anchored top-left). Pipe replies arrive from the
//! tokio daemon over an unbounded channel and are pumped into the iced loop via a
//! [`Subscription`]; because `Subscription::run` takes a bare `fn` pointer, the
//! receiver is parked in a process static and taken once when the loop starts.
//!
//! Not here (yet): dragging (iteration 5), hide/show (iteration 6), and config-
//! driven appearance (iteration 4). This iteration stands up the surface, the two
//! placeholder cards, and the channel bridge.

use std::sync::Mutex;

use futures_util::{Stream, stream};
use iced::widget::{Space, container, stack};
use iced::{
    Color, Element, Event, Length, Padding, Point, Size, Task, Vector, event, mouse, theme, window,
};
use iced_layershell::actions::ActionCallback;
use iced_layershell::application;
use iced_layershell::reexport::{Anchor, KeyboardInteractivity, Layer, WlRegion};
use iced_layershell::settings::{LayerShellSettings, Settings};
use iced_layershell::to_layer_message;
use tokio::sync::mpsc::UnboundedReceiver;

use kitetsu_primitives::card::{self, Card, Edge};

/// Which pipe a card represents. Plain data — carried across the channel and used
/// to route a reply to the right card.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipeId {
    Pipe1,
    Pipe2,
}

/// An event from the daemon (tokio side) to the overlay (iced side).
#[derive(Debug, Clone)]
pub enum UiEvent {
    /// A pipe produced a new suggestion to display.
    Reply { pipe: PipeId, text: String },
    /// Toggle the overlay's visibility (content + positions retained).
    Toggle,
}

/// Process-wide parking spot for the daemon→overlay receiver.
///
/// `Subscription::run` accepts only a capture-free `fn`, so the externally-created
/// receiver cannot be closed over; it is installed here before the loop starts and
/// taken exactly once by the subscription worker.
static UI_RX: Mutex<Option<UnboundedReceiver<UiEvent>>> = Mutex::new(None);

/// Park the receiver for the subscription worker. Call once, before [`run`].
pub fn install_receiver(rx: UnboundedReceiver<UiEvent>) {
    *UI_RX.lock().expect("ui receiver lock poisoned") = Some(rx);
}

/// Everything the overlay needs to build one card, derived from config by the
/// caller. Kept iced-free (plain `f32` position) so `main` need not touch iced.
pub struct CardInit {
    pub id: PipeId,
    pub header: String,
    pub model: String,
    pub font_size: f32,
    pub opacity: f32,
    pub bg_opacity: Option<f32>,
    pub text_opacity: Option<f32>,
    pub width: f32,
    pub pos_x: f32,
    pub pos_y: f32,
}

/// Gap left below a card so it never touches the screen edge.
const HEIGHT_MARGIN: f32 = 40.0;
/// Floor for the computed card height cap.
const MIN_CARD_HEIGHT: f32 = 120.0;

/// Max card height given the surface height and the card's top y. Infinite (no
/// cap) until the surface size is known.
fn capped_height(surface: Size, pos_y: f32) -> f32 {
    if surface.height.is_finite() {
        (surface.height - pos_y - HEIGHT_MARGIN).max(MIN_CARD_HEIGHT)
    } else {
        f32::INFINITY
    }
}

/// Process-wide parking spot for the card specs (same `fn`-pointer constraint as
/// the receiver: `init` cannot capture).
static CARD_INIT: Mutex<Option<Vec<CardInit>>> = Mutex::new(None);

/// One card plus where it sits in the surface.
struct Slot {
    id: PipeId,
    card: Card,
    pos: Point,
}

impl Slot {
    fn from_init(c: CardInit) -> Self {
        Slot {
            id: c.id,
            card: Card {
                header: c.header,
                model: c.model,
                body: vec![("Waiting for the first suggestion…".to_owned(), false)],
                font_size: c.font_size,
                opacity: c.opacity,
                bg_opacity: c.bg_opacity,
                text_opacity: c.text_opacity,
                width: c.width,
                // Recomputed once the surface size is known.
                max_height: f32::INFINITY,
                manual_height: None,
            },
            pos: Point::new(c.pos_x, c.pos_y),
        }
    }
}

/// What an in-progress pointer grab is doing to a card.
#[derive(Clone, Copy)]
enum GrabKind {
    /// Moving the card; `offset` is the cursor→top-left delta captured at grab time.
    Move { offset: Vector },
    /// Resizing from an edge/corner (top-left stays fixed).
    Resize { edge: Edge },
}

/// An in-progress grab: which card and what it's doing.
#[derive(Clone, Copy)]
struct Grab {
    pipe: PipeId,
    kind: GrabKind,
}

/// Minimum card dimensions when resizing.
const MIN_WIDTH: f32 = 200.0;
const MIN_HEIGHT: f32 = 120.0;

/// Overlay state: the cards, their positions, and any in-progress drag.
struct App {
    slots: Vec<Slot>,
    /// Last known cursor position (surface coords; fullscreen ⇒ screen coords).
    cursor: Point,
    grab: Option<Grab>,
    /// Current surface size; drives the per-card height cap.
    surface: Size,
    /// Whether the cards are shown. Hidden ⇒ empty input region (click-through).
    visible: bool,
}

#[to_layer_message]
#[derive(Debug, Clone)]
pub enum Message {
    /// A button press inside a card (drag handle only, for now).
    Card(PipeId, card::Message),
    /// A new suggestion arrived for a pipe.
    Reply { pipe: PipeId, text: String },
    /// The cursor moved (tracked globally so a grabbed card follows it).
    CursorMoved(Point),
    /// The left mouse button was released anywhere — ends any drag.
    Released,
    /// The surface was (re)sized — recompute height caps.
    SurfaceResized(Size),
    /// Toggle overlay visibility.
    Toggle,
}

/// Split a reply into (text, is_bold) runs on `**…**` markers.
///
/// Segments alternate non-bold/bold starting non-bold; empty segments are
/// dropped. An unterminated `**` leaves its trailing segment bold (a reasonable
/// best effort for streamed/clipped LLM output).
fn parse_bold(text: &str) -> Vec<(String, bool)> {
    let mut runs: Vec<(String, bool)> = text
        .split("**")
        .enumerate()
        .filter(|(_, seg)| !seg.is_empty())
        .map(|(i, seg)| (seg.to_owned(), i % 2 == 1))
        .collect();
    if runs.is_empty() {
        runs.push((String::new(), false));
    }
    runs
}

fn init() -> (App, Task<Message>) {
    let slots = CARD_INIT
        .lock()
        .expect("card init lock poisoned")
        .take()
        .unwrap_or_default()
        .into_iter()
        .map(Slot::from_init)
        .collect();
    (
        App {
            slots,
            cursor: Point::ORIGIN,
            grab: None,
            surface: Size::new(f32::INFINITY, f32::INFINITY),
            visible: true,
        },
        Task::none(),
    )
}

/// Build the input-region action for the current visibility. The compositor side
/// clears the region first, so hidden = add nothing (click-through) and visible =
/// add a full-surface rect.
fn input_region_task(surface: Size, visible: bool) -> Task<Message> {
    let callback = if visible {
        let w = if surface.width.is_finite() {
            surface.width as i32
        } else {
            100_000
        };
        let h = if surface.height.is_finite() {
            surface.height as i32
        } else {
            100_000
        };
        ActionCallback::new(move |region: &WlRegion| region.add(0, 0, w.max(1), h.max(1)))
    } else {
        ActionCallback::new(|_region: &WlRegion| {})
    };
    Task::done(Message::SetInputRegion(callback))
}

fn namespace() -> String {
    "kitetsu-teleprompter".into()
}

fn update(app: &mut App, message: Message) -> Task<Message> {
    match message {
        // A card grip was pressed: start moving or resizing.
        Message::Card(pipe, msg) => {
            let pos = app.slots.iter().find(|s| s.id == pipe).map(|s| s.pos);
            if let Some(pos) = pos {
                let kind = match msg {
                    // Capture the cursor→top-left offset so the card doesn't jump.
                    card::Message::Drag => GrabKind::Move {
                        offset: app.cursor - pos,
                    },
                    card::Message::Resize(edge) => GrabKind::Resize { edge },
                };
                app.grab = Some(Grab { pipe, kind });
            }
        }
        Message::CursorMoved(position) => {
            app.cursor = position;
            if let Some(grab) = app.grab {
                let surface = app.surface;
                if let Some(slot) = app.slots.iter_mut().find(|s| s.id == grab.pipe) {
                    match grab.kind {
                        GrabKind::Move { offset } => {
                            slot.pos = position - offset;
                            // Moving down shrinks the height cap so it stays on-screen.
                            slot.card.max_height = capped_height(surface, slot.pos.y);
                        }
                        GrabKind::Resize { edge } => {
                            if matches!(edge, Edge::Right | Edge::Corner) {
                                slot.card.width = (position.x - slot.pos.x).max(MIN_WIDTH);
                            }
                            if matches!(edge, Edge::Bottom | Edge::Corner) {
                                slot.card.manual_height =
                                    Some((position.y - slot.pos.y).max(MIN_HEIGHT));
                            }
                        }
                    }
                }
            }
        }
        Message::Released => app.grab = None,
        Message::SurfaceResized(size) => {
            app.surface = size;
            for slot in &mut app.slots {
                slot.card.max_height = capped_height(size, slot.pos.y);
            }
            // Keep the input region matched to the (now known) surface size.
            if app.visible {
                return input_region_task(size, true);
            }
        }
        Message::Toggle => {
            app.visible = !app.visible;
            return input_region_task(app.surface, app.visible);
        }
        Message::Reply { pipe, text } => {
            if let Some(slot) = app.slots.iter_mut().find(|s| s.id == pipe) {
                slot.card.body = parse_bold(&text);
            }
        }
        // #[to_layer_message] adds LayerShell action variants we don't emit.
        _ => {}
    }
    Task::none()
}

fn view(app: &App) -> Element<'_, Message> {
    // Hidden: draw nothing (content + positions retained in state).
    if !app.visible {
        return Space::new().into();
    }

    let layers: Vec<Element<'_, Message>> = app
        .slots
        .iter()
        .map(|slot| {
            let id = slot.id;
            let positioned = card::view(&slot.card).map(move |m| Message::Card(id, m));
            container(positioned)
                .width(Length::Fill)
                .height(Length::Fill)
                .padding(Padding {
                    top: slot.pos.y,
                    left: slot.pos.x,
                    right: 0.0,
                    bottom: 0.0,
                })
                .into()
        })
        .collect();

    stack(layers)
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
}

fn subscription(_app: &App) -> iced::Subscription<Message> {
    iced::Subscription::batch([
        iced::Subscription::run(ui_event_stream),
        event::listen_with(on_event),
    ])
}

/// Map raw window events to drag messages. Must be a bare `fn` (listen_with bound).
fn on_event(event: Event, _status: event::Status, _id: window::Id) -> Option<Message> {
    match event {
        Event::Mouse(mouse::Event::CursorMoved { position }) => {
            Some(Message::CursorMoved(position))
        }
        Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left)) => Some(Message::Released),
        Event::Window(window::Event::Opened { size, .. })
        | Event::Window(window::Event::Resized(size)) => Some(Message::SurfaceResized(size)),
        _ => None,
    }
}

/// Stream the daemon→overlay events as iced messages. Takes the parked receiver
/// once; panics only if called before [`install_receiver`] (a programmer error).
fn ui_event_stream() -> impl Stream<Item = Message> {
    let rx = UI_RX
        .lock()
        .expect("ui receiver lock poisoned")
        .take()
        .expect("ui receiver installed before the iced loop starts");
    stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|event| {
            let message = match event {
                UiEvent::Reply { pipe, text } => Message::Reply { pipe, text },
                UiEvent::Toggle => Message::Toggle,
            };
            (message, rx)
        })
    })
}

/// Run the overlay on the current (main) thread. Blocks until the surface closes.
///
/// `cards` are the per-pipe specs (built from config); they are parked for `init`.
pub fn run(cards: Vec<CardInit>) -> iced_layershell::Result {
    *CARD_INIT.lock().expect("card init lock poisoned") = Some(cards);
    application(init, namespace, update, view)
        .subscription(subscription)
        .settings(Settings {
            layer_settings: LayerShellSettings {
                // Fullscreen: anchor all four edges, no explicit size.
                anchor: Anchor::Top | Anchor::Bottom | Anchor::Left | Anchor::Right,
                layer: Layer::Overlay,
                size: None,
                margin: (0, 0, 0, 0),
                // Pointer still reaches the cards (needed for drag); keyboard falls through.
                keyboard_interactivity: KeyboardInteractivity::None,
                ..Default::default()
            },
            ..Default::default()
        })
        .style(|_state, _theme| theme::Style {
            // Transparent surface so only the cards show over the desktop.
            background_color: Color::TRANSPARENT,
            text_color: Color::BLACK,
        })
        .run()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_text_is_one_non_bold_run() {
        assert_eq!(parse_bold("hello world"), vec![("hello world".into(), false)]);
    }

    #[test]
    fn splits_inline_bold() {
        assert_eq!(
            parse_bold("say **this** now"),
            vec![
                ("say ".into(), false),
                ("this".into(), true),
                (" now".into(), false),
            ]
        );
    }

    #[test]
    fn unterminated_marker_leaves_tail_bold() {
        assert_eq!(
            parse_bold("a **b"),
            vec![("a ".into(), false), ("b".into(), true)]
        );
    }

    #[test]
    fn empty_text_yields_single_empty_run() {
        assert_eq!(parse_bold(""), vec![(String::new(), false)]);
    }
}
