//! The teleprompter overlay — the only place in the binary that touches iced.
//!
//! A single transparent, fullscreen Wayland layer surface holding one
//! [`kitetsu_primitives::presets::teleprompter::Card`] positioned at an `(x, y)`
//! offset inside it (anchored top-left). Pipe replies arrive from the tokio
//! daemon over an unbounded channel and are pumped into the iced loop via a
//! [`Subscription`]; because `Subscription::run` takes a bare `fn` pointer, the
//! receiver is parked in a process static and taken once when the loop starts.
//!
//! The card keeps every reply it has shown; `forward`/`backward` walk that
//! history, and a fresh reply snaps back to the newest. Colours come from a
//! base16 [`Base16`] palette (loaded from `colors.toml`, baked default otherwise).

use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Duration;

use futures_util::{Stream, stream};
use iced::widget::{Space, container};
use iced::{
    Color, Element, Event, Font, Length, Padding, Point, Size, Task, Vector, event, mouse, theme,
    window,
};
use iced_layershell::actions::ActionCallback;
use iced_layershell::application;
use iced_layershell::reexport::{Anchor, KeyboardInteractivity, Layer, WlRegion};
use iced_layershell::settings::{LayerShellSettings, Settings};
use iced_layershell::to_layer_message;
use tokio::sync::mpsc::UnboundedReceiver;

use kitetsu_primitives::presets::teleprompter::{self as card, Card};
use kitetsu_primitives::{Base16, Edge, FONT_BOLD, FONT_NAME, FONT_REGULAR};

use super::ipc::{Command, TeleprompterAction, send_command};
use super::layout::{self, Geometry, Layout};

/// A processing stage the pipe passes through on a trigger, shown in the header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// The chunk pipe is REST-transcribing the window.
    Transcribing,
    /// The LLM call is in flight.
    Fetching,
}

impl Stage {
    fn header(self) -> &'static str {
        match self {
            Stage::Transcribing => "Transcribing…",
            Stage::Fetching => "Fetching response…",
        }
    }
}

/// An event from the daemon (tokio side) to the overlay (iced side).
#[derive(Debug, Clone)]
pub enum UiEvent {
    /// The pipe produced a new suggestion to display.
    Reply { text: String },
    /// The pipe changed processing stage; `None` clears it back to the baseline.
    Status { stage: Option<Stage> },
    /// Recording state changed: `true` = recording, `false` = paused.
    Recording(bool),
    /// Toggle the overlay's visibility (content + position retained).
    Toggle,
    /// Conversation trashed — flash the card title.
    Trashed,
    /// End the trash flash — restore the card title.
    Untrash,
    /// Step to the next (newer) suggestion in history.
    Forward,
    /// Step to the previous (older) suggestion in history.
    Backward,
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

/// Everything the overlay needs to build the card, derived from config by the
/// caller. Kept iced-free (plain `f32` position) so `main` need not touch iced.
pub struct CardInit {
    pub model: String,
    pub font_size: f32,
    pub opacity: f32,
    pub bg_opacity: Option<f32>,
    pub text_opacity: Option<f32>,
    pub width: f32,
    pub height: f32,
    pub pos_x: f32,
    pub pos_y: f32,
}

/// Gap left below the card so it never touches the screen edge.
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

/// Process-wide parking spot for the card spec (same `fn`-pointer constraint as
/// the receiver: `init` cannot capture).
static CARD_INIT: Mutex<Option<CardInit>> = Mutex::new(None);

/// Process-wide parking spot for the layout file path (same `fn`-pointer
/// constraint: `init` cannot capture it).
static LAYOUT_PATH: Mutex<Option<PathBuf>> = Mutex::new(None);

/// Process-wide parking spot for the colour palette (same `fn`-pointer constraint).
static PALETTE: Mutex<Option<Base16>> = Mutex::new(None);

/// What an in-progress pointer grab is doing to the card.
#[derive(Clone, Copy)]
enum GrabKind {
    /// Moving the card; `offset` is the cursor→top-left delta captured at grab time.
    Move { offset: Vector },
    /// Resizing from an edge/corner (top-left stays fixed).
    Resize { edge: Edge },
}

/// Minimum card dimensions when resizing.
const MIN_WIDTH: f32 = 200.0;
const MIN_HEIGHT: f32 = 120.0;

/// Overlay state: the card, its position, history, and any in-progress drag.
struct App {
    card: Card,
    pos: Point,
    /// Whether audio is being recorded (drives the idle header + dot).
    recording: bool,
    /// The transient per-trigger stage, if any (overrides the idle header).
    stage: Option<Stage>,
    /// Whether the trash flash is showing (overrides everything for 3s).
    trashed: bool,
    /// Every suggestion shown so far; `idx` selects which is on the card.
    history: Vec<String>,
    /// Index into `history` of the currently shown suggestion.
    idx: usize,
    /// Last known cursor position (surface coords; fullscreen ⇒ screen coords).
    cursor: Point,
    grab: Option<GrabKind>,
    /// Current surface size; drives the card height cap.
    surface: Size,
    /// Whether the card is shown. Hidden ⇒ empty input region (click-through).
    visible: bool,
    /// Where to persist card geometry; saved after each drag/resize.
    layout_path: PathBuf,
    /// The base16 colour palette the card paints from.
    palette: Base16,
}

impl App {
    /// Recompute the header from current mode: trash flash wins, then the active
    /// stage, else the recording/paused baseline. Also mirrors `recording` onto
    /// the card so its dot/resume icon matches.
    fn refresh_header(&mut self) {
        self.card.header = if self.trashed {
            "Trashed".to_owned()
        } else if let Some(stage) = self.stage {
            stage.header().to_owned()
        } else if self.recording {
            "Listening…".to_owned()
        } else {
            "Paused".to_owned()
        };
        self.card.recording = self.recording;
    }

    /// Render the suggestion at `idx` into the card body. No-op if history empty.
    fn show_current(&mut self) {
        if let Some(text) = self.history.get(self.idx) {
            self.card.body = parse_bold(text);
        }
    }
}

/// Snapshot the card's current geometry for persistence.
fn current_layout(app: &App) -> Layout {
    Layout {
        card: Some(Geometry {
            pos_x: app.pos.x,
            pos_y: app.pos.y,
            width: app.card.width,
            // manual_height is always Some here (seeded on init, updated on resize).
            height: app.card.manual_height.unwrap_or(MIN_CARD_HEIGHT),
        }),
    }
}

#[to_layer_message]
#[derive(Debug, Clone)]
pub enum Message {
    /// A button press inside the card.
    Card(card::Message),
    /// A new suggestion arrived.
    Reply { text: String },
    /// The pipe changed processing stage (`None` clears it).
    Status { stage: Option<Stage> },
    /// Recording state changed.
    Recording(bool),
    /// Recording-dot flash tick.
    Blink,
    /// A button-fired IPC command finished (result ignored — the daemon echoes
    /// the effect back as its own event).
    CmdDone,
    /// The cursor moved (tracked globally so a grabbed card follows it).
    CursorMoved(Point),
    /// The left mouse button was released anywhere — ends any drag.
    Released,
    /// The surface was (re)sized — recompute the height cap.
    SurfaceResized(Size),
    /// Toggle overlay visibility.
    Toggle,
    /// Flash the card title to "Trashed".
    Trashed,
    /// Restore the card title after the flash.
    Untrash,
    /// Step to the next (newer) suggestion.
    Forward,
    /// Step to the previous (older) suggestion.
    Backward,
}

/// Fire a control command over the socket — the exact path the CLI uses. The
/// daemon's reply is ignored; its side effect comes back as a `UiEvent`.
fn fire(cmd: Command) -> Task<Message> {
    Task::perform(async move { send_command(&cmd).await }, |_| Message::CmdDone)
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
    let init = CARD_INIT
        .lock()
        .expect("card init lock poisoned")
        .take()
        .expect("card spec installed before the iced loop starts");
    let layout_path = LAYOUT_PATH
        .lock()
        .expect("layout path lock poisoned")
        .take()
        .unwrap_or_else(layout::layout_path);
    let palette = PALETTE
        .lock()
        .expect("palette lock poisoned")
        .take()
        .unwrap_or(kitetsu_primitives::theme::DEFAULT);

    let card = Card {
        header: String::new(),
        model: init.model,
        body: vec![("Waiting for the first suggestion…".to_owned(), false)],
        font_size: init.font_size,
        opacity: init.opacity,
        bg_opacity: init.bg_opacity,
        text_opacity: init.text_opacity,
        width: init.width,
        // Recomputed once the surface size is known.
        max_height: f32::INFINITY,
        // Seed a definite starting height; manual resize updates it.
        manual_height: Some(init.height),
        recording: true,
        blink_on: true,
    };
    let mut app = App {
        card,
        pos: Point::new(init.pos_x, init.pos_y),
        recording: true,
        stage: None,
        trashed: false,
        history: Vec::new(),
        idx: 0,
        cursor: Point::ORIGIN,
        grab: None,
        surface: Size::new(f32::INFINITY, f32::INFINITY),
        visible: true,
        layout_path,
        palette,
    };
    app.refresh_header();
    (app, Task::none())
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
        // A card control was pressed. Drag/Resize start an in-app gesture;
        // Toggle/Pause/Trash forward to the daemon exactly like the CLI.
        Message::Card(msg) => {
            let kind = match msg {
                card::Message::Toggle => {
                    return fire(Command::Teleprompter(TeleprompterAction::Toggle));
                }
                card::Message::Pause => {
                    return fire(Command::Teleprompter(TeleprompterAction::Pause));
                }
                card::Message::Trash => {
                    return fire(Command::Teleprompter(TeleprompterAction::Trash));
                }
                card::Message::Drag => GrabKind::Move {
                    // Capture the cursor→top-left offset so the card doesn't jump.
                    offset: app.cursor - app.pos,
                },
                card::Message::Resize(edge) => GrabKind::Resize { edge },
            };
            app.grab = Some(kind);
        }
        Message::CursorMoved(position) => {
            app.cursor = position;
            if let Some(grab) = app.grab {
                match grab {
                    GrabKind::Move { offset } => {
                        app.pos = position - offset;
                        // Moving down shrinks the height cap so it stays on-screen.
                        app.card.max_height = capped_height(app.surface, app.pos.y);
                    }
                    GrabKind::Resize { edge } => {
                        if matches!(edge, Edge::Right | Edge::Corner) {
                            app.card.width = (position.x - app.pos.x).max(MIN_WIDTH);
                        }
                        if matches!(edge, Edge::Bottom | Edge::Corner) {
                            app.card.manual_height = Some((position.y - app.pos.y).max(MIN_HEIGHT));
                        }
                    }
                }
            }
        }
        Message::Released => {
            // A grab just ended (move or resize) ⇒ persist the new geometry. File
            // I/O runs in a Task, never inline in update().
            if app.grab.take().is_some() {
                let layout = current_layout(app);
                let path = app.layout_path.clone();
                return Task::perform(async move { layout::save(&path, &layout) }, |_| {
                    Message::CmdDone
                });
            }
        }
        Message::SurfaceResized(size) => {
            app.surface = size;
            app.card.max_height = capped_height(size, app.pos.y);
            // Keep the input region matched to the (now known) surface size.
            if app.visible {
                return input_region_task(size, true);
            }
        }
        Message::Toggle => {
            app.visible = !app.visible;
            return input_region_task(app.surface, app.visible);
        }
        Message::Reply { text } => {
            // Record it and snap to the newest; the header returns to baseline.
            app.history.push(text);
            app.idx = app.history.len() - 1;
            app.show_current();
            app.stage = None;
            app.refresh_header();
        }
        Message::Forward => {
            if app.idx + 1 < app.history.len() {
                app.idx += 1;
                app.show_current();
            }
        }
        Message::Backward => {
            if app.idx > 0 {
                app.idx -= 1;
                app.show_current();
            }
        }
        Message::Status { stage } => {
            app.stage = stage;
            app.refresh_header();
        }
        Message::Recording(recording) => {
            app.recording = recording;
            app.refresh_header();
        }
        Message::Blink => {
            app.card.blink_on = !app.card.blink_on;
        }
        Message::CmdDone => {}
        Message::Trashed => {
            app.trashed = true;
            app.refresh_header();
        }
        Message::Untrash => {
            app.trashed = false;
            app.refresh_header();
        }
        // #[to_layer_message] adds LayerShell action variants we don't emit.
        _ => {}
    }
    Task::none()
}

fn view(app: &App) -> Element<'_, Message> {
    // Hidden: draw nothing (content + position retained in state).
    if !app.visible {
        return Space::new().into();
    }

    let positioned = card::view(&app.card, &app.palette).map(Message::Card);
    container(positioned)
        .width(Length::Fill)
        .height(Length::Fill)
        .padding(Padding {
            top: app.pos.y,
            left: app.pos.x,
            right: 0.0,
            bottom: 0.0,
        })
        .into()
}

fn subscription(_app: &App) -> iced::Subscription<Message> {
    iced::Subscription::batch([
        iced::Subscription::run(ui_event_stream),
        event::listen_with(on_event),
        // Drives the recording-dot flash. Always ticks (cheap); the view only
        // animates when the card is actually recording.
        iced::time::every(Duration::from_millis(600)).map(|_| Message::Blink),
    ])
}

/// Map raw window events to drag messages. Must be a bare `fn` (listen_with bound).
fn on_event(event: Event, _status: event::Status, _id: window::Id) -> Option<Message> {
    match event {
        Event::Mouse(mouse::Event::CursorMoved { position }) => Some(Message::CursorMoved(position)),
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
                UiEvent::Reply { text } => Message::Reply { text },
                UiEvent::Status { stage } => Message::Status { stage },
                UiEvent::Recording(on) => Message::Recording(on),
                UiEvent::Toggle => Message::Toggle,
                UiEvent::Trashed => Message::Trashed,
                UiEvent::Untrash => Message::Untrash,
                UiEvent::Forward => Message::Forward,
                UiEvent::Backward => Message::Backward,
            };
            (message, rx)
        })
    })
}

/// Run the overlay on the current (main) thread. Blocks until the surface closes.
///
/// `card` is the spec (built from config); it and `palette` are parked for `init`.
pub fn run(card: CardInit, palette: Base16, layout_path: PathBuf) -> iced_layershell::Result {
    *CARD_INIT.lock().expect("card init lock poisoned") = Some(card);
    *LAYOUT_PATH.lock().expect("layout path lock poisoned") = Some(layout_path);
    *PALETTE.lock().expect("palette lock poisoned") = Some(palette);
    application(init, namespace, update, view)
        .subscription(subscription)
        // Register JetBrains Mono (regular + bold) and make it the default so
        // every glyph — including widgets that don't set a font — uses it.
        .font(FONT_REGULAR)
        .font(FONT_BOLD)
        .default_font(Font::with_name(FONT_NAME))
        .settings(Settings {
            layer_settings: LayerShellSettings {
                // Fullscreen: anchor all four edges, no explicit size.
                anchor: Anchor::Top | Anchor::Bottom | Anchor::Left | Anchor::Right,
                layer: Layer::Overlay,
                size: None,
                margin: (0, 0, 0, 0),
                // Pointer still reaches the card (needed for drag); keyboard falls through.
                keyboard_interactivity: KeyboardInteractivity::None,
                ..Default::default()
            },
            ..Default::default()
        })
        .style(|_state, _theme| theme::Style {
            // Transparent surface so only the card shows over the desktop.
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
