//! Resolves the real PulseAudio/PipeWire-pulse device names for the two
//! capture streams via server introspection.
//!
//! Drives a short-lived standard Mainloop synchronously (typically < 20 ms),
//! disconnects, and returns the names. Does no audio I/O.
//! Must be called before spawning capture threads; Rc types inside are
//! intentionally not Send.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use libpulse_binding::callbacks::ListResult;
use libpulse_binding::context::{Context, FlagSet, State};
use libpulse_binding::mainloop::standard::{IterateResult, Mainloop};

use crate::listener::ListenerError;

// ── Public types ──────────────────────────────────────────────────────────────

/// Resolved PulseAudio device names for the two capture streams.
#[derive(Debug, Clone)]
pub struct Devices {
    /// Default source name (microphone input).
    pub mic_source: String,
    /// Monitor source name for the default sink (system audio output tap).
    pub system_monitor: String,
}

// ── Entry point ───────────────────────────────────────────────────────────────

/// Query the running PulseAudio / PipeWire-pulse server and return the
/// default mic source name and the default sink's monitor source name.
///
/// Fails fast with [`ListenerError::IntrospectionFailed`] if the server is
/// unreachable, the context fails to connect, or the default sink has no
/// monitor source exposed.
pub fn discover_default_devices() -> Result<Devices, ListenerError> {
    let mut mainloop = Mainloop::new().ok_or(ListenerError::IntrospectionFailed(
        "could not allocate PulseAudio mainloop",
    ))?;

    let mut ctx = Context::new(&mainloop, "kitetsu-discover").ok_or(
        ListenerError::IntrospectionFailed("could not create PulseAudio context"),
    )?;

    // FlagSet::empty() → fail immediately if no server is reachable, rather
    // than retrying forever (PA_CONTEXT_NOFAIL would retry indefinitely).
    ctx.connect(None, FlagSet::empty(), None)
        .map_err(|_| ListenerError::IntrospectionFailed("context connect rejected by server"))?;

    // — Wait for context to reach Ready ————————————————————————————————————

    loop {
        match mainloop.iterate(true) {
            IterateResult::Err(_) => {
                return Err(ListenerError::IntrospectionFailed(
                    "mainloop iterate error while waiting for context ready",
                ));
            }
            IterateResult::Quit(_) => {
                return Err(ListenerError::IntrospectionFailed(
                    "mainloop quit unexpectedly before context was ready",
                ));
            }
            IterateResult::Success(_) => {}
        }
        match ctx.get_state() {
            State::Ready => break,
            State::Failed => {
                return Err(ListenerError::IntrospectionFailed(
                    "PulseAudio context failed — is pipewire-pulse or PulseAudio running?",
                ));
            }
            State::Terminated => {
                return Err(ListenerError::IntrospectionFailed(
                    "PulseAudio context terminated unexpectedly",
                ));
            }
            _ => {}
        }
    }

    // — Step 1: get default source + sink names ——————————————————————————————

    // The server_info callback fires exactly once; we use Option to detect it.
    let server_pair: Rc<RefCell<Option<(String, String)>>> = Rc::new(RefCell::new(None));
    let server_pair_cb = Rc::clone(&server_pair);

    // Kept alive so the Operation is not cancelled before the callback fires.
    let _server_op = ctx.introspect().get_server_info(move |info| {
        let mic = info
            .default_source_name
            .as_deref()
            .unwrap_or("unknown")
            .to_owned();
        let sink = info
            .default_sink_name
            .as_deref()
            .unwrap_or("unknown")
            .to_owned();
        *server_pair_cb.borrow_mut() = Some((mic, sink));
    });

    loop {
        match mainloop.iterate(true) {
            IterateResult::Err(_) | IterateResult::Quit(_) => {
                return Err(ListenerError::IntrospectionFailed(
                    "mainloop error while querying server info",
                ));
            }
            IterateResult::Success(_) => {}
        }
        if server_pair.borrow().is_some() {
            break;
        }
    }

    let (mic_source, default_sink) = server_pair
        .borrow()
        .clone()
        .expect("server_pair is Some — we only break when it is");

    // — Step 2: get the default sink's monitor source name ————————————————————

    // get_sink_info_by_name fires ListResult::Item (one or more) then
    // ListResult::End. We capture the monitor name on Item and break on End/Error.
    let monitor_done: Rc<Cell<bool>> = Rc::new(Cell::new(false));
    let monitor_name: Rc<RefCell<Option<String>>> = Rc::new(RefCell::new(None));
    let done_cb = Rc::clone(&monitor_done);
    let name_cb = Rc::clone(&monitor_name);

    let _sink_op =
        ctx.introspect()
            .get_sink_info_by_name(&default_sink, move |result| match result {
                ListResult::Item(info) => {
                    if let Some(mon) = &info.monitor_source_name {
                        *name_cb.borrow_mut() = Some(mon.as_ref().to_owned());
                    }
                }
                ListResult::End | ListResult::Error => {
                    done_cb.set(true);
                }
            });

    loop {
        match mainloop.iterate(true) {
            IterateResult::Err(_) | IterateResult::Quit(_) => {
                return Err(ListenerError::IntrospectionFailed(
                    "mainloop error while querying sink monitor source",
                ));
            }
            IterateResult::Success(_) => {}
        }
        if monitor_done.get() {
            break;
        }
    }

    let system_monitor =
        monitor_name
            .borrow()
            .clone()
            .ok_or(ListenerError::IntrospectionFailed(
                "default sink has no monitor source \
             — is pipewire-pulse or PulseAudio exposing one?",
            ))?;

    ctx.disconnect();

    Ok(Devices {
        mic_source,
        system_monitor,
    })
}
