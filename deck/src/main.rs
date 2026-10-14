slint::include_modules!();

use crate::app_state::{AppState, ConnState};
use crate::commands::{Command, ParamKind};
use crate::config::{load_config, save_config, Config};
use crate::diagnostics::Diagnostics;
use crate::osc::client::OscClient;
use crate::osc::protocol::{self, OscIn};
use crate::osc::receiver::OscReceiver;
use chrono::Utc;
use slint::{ComponentHandle, Timer, TimerMode};
use std::cell::RefCell;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::net::UdpSocket;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant};

mod app_state;
mod commands;
mod config;
mod diagnostics;
mod logging;
mod osc;

/// How long without a heartbeat before the connection is shown as stale.
const STALE_AFTER_SECS: i64 = 3;

/// Parameter changes are coalesced for this long before an OSC message is sent,
/// so dragging a slider does not flood the server.
const PARAM_DEBOUNCE: Duration = Duration::from_millis(300);

/// Default slider positions, chosen to match the server's config defaults.
const DEFAULT_GUIDANCE: f32 = 1.0;
const DEFAULT_DELTA: f32 = 1.0;
const DEFAULT_MAXFPS: f32 = 4.0;
const DEFAULT_STEP_INDEX: f32 = 17.0;

fn main() -> anyhow::Result<()> {
    let _log_guard = crate::logging::init()?;
    let cfg = load_config();
    tracing::info!(
        host = %cfg.server_host,
        server_port = cfg.server_port,
        local_port = cfg.local_port,
        fullscreen = cfg.fullscreen,
        "prism-deck starting"
    );
    let state = Rc::new(RefCell::new(AppState::default()));
    let diag = Rc::new(RefCell::new(Diagnostics::default()));
    let fullscreen = Arc::new(AtomicBool::new(cfg.fullscreen));
    let pending: Rc<RefCell<HashMap<ParamKind, (f64, Instant)>>> =
        Rc::new(RefCell::new(HashMap::new()));

    let (cmd_tx, cmd_rx) = mpsc::channel::<Command>();
    let (osc_tx, osc_rx) = mpsc::channel::<OscIn>();
    let (log_tx, log_rx) = mpsc::channel::<String>();

    let window = MainWindow::new()?;
    init_window(&window, &cfg, &state.borrow());

    // Bind the telemetry socket exactly once. Sender and receiver share it so
    // the server sees a single, stable return address.
    let bind_addr: SocketAddr = format!("0.0.0.0:{}", cfg.local_port).parse()?;
    let sock = match UdpSocket::bind(bind_addr) {
        Ok(s) => s,
        Err(e) => {
            let msg = format!(
                "Cannot bind local telemetry port {}: {}. \
                 Another process may be using it. Change local_port in \
                 {} and restart.",
                cfg.local_port,
                e,
                crate::config::config_path().display()
            );
            tracing::error!(port = cfg.local_port, error = %e, "failed to bind telemetry port");
            diag.borrow_mut().add_error("startup".into(), msg.clone());
            window.set_errors(diag.borrow().errors_text().into());
            window.set_status("Port unavailable".into());
            window.set_last_heartbeat("Never".into());
            window.run()?;
            return Ok(());
        }
    };

    let target_addr: SocketAddr = format!("{}:{}", cfg.server_host, cfg.server_port).parse()?;
    let send_sock = sock.try_clone()?;
    let client = OscClient::new(send_sock, target_addr, log_tx.clone());
    let _receiver = OscReceiver::new(sock, osc_tx);

    {
        let mut s = state.borrow_mut();
        s.conn_state = ConnState::WaitingForHeartbeat;
    }

    // Establish our return address with the server. No NDI discovery endpoints
    // exist in this release, so we send a harmless probe using the existing
    // API. `osc_send_ip: "auto"` replies to the last peer that sent a command.
    client.send(protocol::encode_heartbeat_probe());

    setup_callbacks(&window, cmd_tx.clone(), fullscreen.clone());

    // Drive channel draining + UI updates from the Slint event loop.
    let weak = window.as_weak();
    let state_t = state.clone();
    let diag_t = diag.clone();
    let fs_t = fullscreen.clone();
    let pending_t = pending.clone();
    let client = Rc::new(client);
    let client_t = client.clone();

    let timer = Timer::default();
    timer.start(TimerMode::Repeated, Duration::from_millis(16), move || {
        let Some(window) = weak.upgrade() else {
            return;
        };

        while let Ok(cmd) = cmd_rx.try_recv() {
            handle_command(&client_t, &state_t, &pending_t, cmd);
        }
        while let Ok(ev) = osc_rx.try_recv() {
            handle_event(&state_t, &diag_t, ev);
        }
        while let Ok(log) = log_rx.try_recv() {
            diag_t.borrow_mut().add_sent(log);
        }

        flush_pending_params(&client_t, &state_t, &pending_t);

        update_ui(&window, &state_t, &diag_t, &fs_t);
    });

    window.run()?;

    // Persist final state on exit.
    let fs = fullscreen.load(Ordering::Relaxed);
    let cfg = Config {
        fullscreen: fs,
        ..cfg
    };
    let _ = save_config(&cfg);

    Ok(())
}

fn init_window(window: &MainWindow, cfg: &Config, state: &AppState) {
    window.set_fullscreen(cfg.fullscreen);
    window.set_host(cfg.server_host.clone().into());
    window.set_server_port(cfg.server_port as i32);
    window.set_local_port(cfg.local_port as i32);
    window.set_status("Disconnected".into());
    window.set_prompt(state.prompt.clone().into());
    window.set_negative_prompt(state.negative_prompt.clone().into());
    window.set_seed_str(state.seed.to_string().into());
    window.set_guidance(DEFAULT_GUIDANCE);
    window.set_delta(DEFAULT_DELTA);
    window.set_maxfps(DEFAULT_MAXFPS);
    window.set_steps(DEFAULT_STEP_INDEX);
    window.set_input_text(state.input_text.clone().into());
    window.set_errors(String::new().into());
    window.set_sent_log(String::new().into());
    window.set_fps(state.fps);
    window.set_frames(state.frames);
    window.set_output_name(state.output_name.clone().into());
    window.set_confirmed_input(state.confirmed_input.clone().into());
    window.set_last_heartbeat("Never".into());
}

fn handle_command(
    client: &OscClient,
    state: &Rc<RefCell<AppState>>,
    pending: &Rc<RefCell<HashMap<ParamKind, (f64, Instant)>>>,
    cmd: Command,
) {
    tracing::debug!(?cmd, "dispatching command");
    match cmd {
        Command::Play => client.send(protocol::encode_play()),
        Command::Pause => client.send(protocol::encode_pause()),
        Command::Stop => client.send(protocol::encode_stop()),
        Command::Unload => client.send(protocol::encode_unload()),
        Command::ApplyPrompts { prompt, negative } => {
            let mut s = state.borrow_mut();
            s.prompt = prompt.clone();
            s.negative_prompt = negative.clone();
            client.send(protocol::encode_prompt(&prompt));
            client.send(protocol::encode_negative(&negative));
        }
        Command::ApplyInput { name } => {
            state.borrow_mut().input_text = name.clone();
            client.send(protocol::encode_input(&name));
        }
        Command::SetParam { kind, value } => {
            // Only remember the latest value; the timer sends it once the user
            // stops moving the control.
            pending.borrow_mut().insert(kind, (value, Instant::now()));
        }
    }
}

/// Send any parameter whose debounce window has elapsed.
fn flush_pending_params(
    client: &OscClient,
    state: &Rc<RefCell<AppState>>,
    pending: &Rc<RefCell<HashMap<ParamKind, (f64, Instant)>>>,
) {
    for (kind, value) in take_due(pending, Instant::now(), PARAM_DEBOUNCE) {
        send_param(client, state, kind, value);
    }
}

/// Remove and return the parameters whose last change is older than `debounce`.
fn take_due(
    pending: &Rc<RefCell<HashMap<ParamKind, (f64, Instant)>>>,
    now: Instant,
    debounce: Duration,
) -> Vec<(ParamKind, f64)> {
    let mut pending = pending.borrow_mut();
    let due: Vec<(ParamKind, f64)> = pending
        .iter()
        .filter(|(_, (_, t))| now.duration_since(*t) >= debounce)
        .map(|(k, (v, _))| (*k, *v))
        .collect();
    for (kind, _) in &due {
        pending.remove(kind);
    }
    due
}

fn send_param(client: &OscClient, state: &Rc<RefCell<AppState>>, kind: ParamKind, value: f64) {
    match kind {
        ParamKind::Seed => {
            let v = value.round() as i32;
            state.borrow_mut().seed = v;
            client.send(protocol::encode_seed(v));
        }
        ParamKind::Guidance => {
            let v = value as f32;
            state.borrow_mut().guidance = v;
            client.send(protocol::encode_guidance(v));
        }
        ParamKind::Delta => {
            let v = value as f32;
            state.borrow_mut().delta = v;
            client.send(protocol::encode_delta(v));
        }
        ParamKind::Maxfps => {
            let v = value.round() as i32;
            state.borrow_mut().maxfps = v;
            client.send(protocol::encode_maxfps(v));
        }
        ParamKind::Steps => {
            let v = value.round() as i32;
            state.borrow_mut().steps = vec![v];
            client.send(protocol::encode_steps(&[v]));
        }
    }
}

fn handle_event(state: &Rc<RefCell<AppState>>, diag: &Rc<RefCell<Diagnostics>>, ev: OscIn) {
    tracing::debug!(?ev, "received event");
    match ev {
        OscIn::Active(_) => {
            let mut s = state.borrow_mut();
            s.conn_state = ConnState::Connected;
            s.last_heartbeat = Some(Utc::now());
        }
        OscIn::Fps(f) => state.borrow_mut().fps = f,
        OscIn::OutputName(n) => state.borrow_mut().output_name = n,
        OscIn::InputName(n) => {
            let mut s = state.borrow_mut();
            s.confirmed_input = n.clone();
            if s.input_text.is_empty() {
                s.input_text = n;
            }
        }
        OscIn::Frames(f) => state.borrow_mut().frames = f,
        OscIn::Error { address, reason } => {
            tracing::warn!(address = %address, reason = %reason, "server rejected a command");
            diag.borrow_mut().add_error(address, reason);
        }
        OscIn::Unknown => {}
    }
}

fn update_ui(
    window: &MainWindow,
    state: &Rc<RefCell<AppState>>,
    diag: &Rc<RefCell<Diagnostics>>,
    fullscreen: &Arc<AtomicBool>,
) {
    let errors = diag.borrow().errors_text();
    if window.get_errors().as_str() != errors {
        window.set_errors(errors.into());
    }
    let sent = diag.borrow().sent_text();
    if window.get_sent_log().as_str() != sent {
        window.set_sent_log(sent.into());
    }

    {
        let mut s = state.borrow_mut();
        if let Some(hb) = s.last_heartbeat {
            if s.conn_state == ConnState::Connected
                && (Utc::now() - hb).num_seconds() > STALE_AFTER_SECS
            {
                tracing::info!("heartbeat stopped; marking connection stale");
                s.conn_state = ConnState::Stale;
            }
        }
    }

    let s = state.borrow();

    let status_str = match s.conn_state {
        ConnState::Disconnected => "Disconnected",
        ConnState::WaitingForHeartbeat => "Waiting for heartbeat",
        ConnState::Connected => "Connected",
        ConnState::Stale => "Stale",
    };
    window.set_status(status_str.into());

    match s.last_heartbeat {
        Some(hb) => window.set_last_heartbeat(hb.format("%H:%M:%S").to_string().into()),
        None => window.set_last_heartbeat("Never".into()),
    }

    window.set_fps(s.fps);
    window.set_frames(s.frames);
    window.set_output_name(s.output_name.clone().into());
    window.set_confirmed_input(s.confirmed_input.clone().into());

    // Fullscreen: apply the shared flag if it changed outside the callback.
    let fs = fullscreen.load(Ordering::Relaxed);
    if window.get_fullscreen() != fs {
        window.set_fullscreen(fs);
        window.window().set_fullscreen(fs);
    }
}

fn setup_callbacks(window: &MainWindow, tx: mpsc::Sender<Command>, fullscreen: Arc<AtomicBool>) {
    let tx_c = tx.clone();
    window.on_play(move || {
        let _ = tx_c.send(Command::Play);
    });
    let tx_c = tx.clone();
    window.on_pause(move || {
        let _ = tx_c.send(Command::Pause);
    });
    let tx_c = tx.clone();
    window.on_stop(move || {
        let _ = tx_c.send(Command::Stop);
    });
    let tx_c = tx.clone();
    window.on_unload(move || {
        let _ = tx_c.send(Command::Unload);
    });
    let tx_c = tx.clone();
    window.on_apply_prompts(move |p, n| {
        let _ = tx_c.send(Command::ApplyPrompts {
            prompt: p.to_string(),
            negative: n.to_string(),
        });
    });
    let tx_c = tx.clone();
    window.on_param_changed(move |kind, value| {
        let kind = match kind.as_str() {
            "seed" => ParamKind::Seed,
            "guidance" => ParamKind::Guidance,
            "delta" => ParamKind::Delta,
            "maxfps" => ParamKind::Maxfps,
            "steps" => ParamKind::Steps,
            other => {
                tracing::warn!(kind = %other, "unknown parameter from UI");
                return;
            }
        };
        let _ = tx_c.send(Command::SetParam {
            kind,
            value: value as f64,
        });
    });
    let tx_c = tx.clone();
    window.on_apply_input(move |n| {
        let _ = tx_c.send(Command::ApplyInput {
            name: n.to_string(),
        });
    });

    let w = window.as_weak();
    window.on_toggle_fullscreen(move || {
        let next = !fullscreen.load(Ordering::Relaxed);
        fullscreen.store(next, Ordering::Relaxed);
        if let Some(w) = w.upgrade() {
            w.set_fullscreen(next);
            w.window().set_fullscreen(next);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stale_before_heartbeat_secs_is_sane() {
        assert!(STALE_AFTER_SECS >= 2);
    }

    #[test]
    fn take_due_only_returns_debounced_entries() {
        let now = Instant::now();
        let pending = Rc::new(RefCell::new(HashMap::new()));
        pending
            .borrow_mut()
            .insert(ParamKind::Guidance, (2.0, now - Duration::from_millis(500)));
        pending.borrow_mut().insert(ParamKind::Delta, (1.5, now));

        let due = take_due(&pending, now, Duration::from_millis(300));
        assert_eq!(due, vec![(ParamKind::Guidance, 2.0)]);
        // Delta is still pending; guidance was consumed.
        assert!(!pending.borrow().contains_key(&ParamKind::Guidance));
        assert!(pending.borrow().contains_key(&ParamKind::Delta));
    }

    #[test]
    fn take_due_keeps_latest_value() {
        let now = Instant::now();
        let pending = Rc::new(RefCell::new(HashMap::new()));
        pending
            .borrow_mut()
            .insert(ParamKind::Maxfps, (10.0, now - Duration::from_secs(1)));
        pending.borrow_mut().insert(ParamKind::Maxfps, (20.0, now));

        // The most recent value replaces the earlier one and resets the timer.
        let due = take_due(&pending, now, Duration::from_millis(300));
        assert!(due.is_empty());
        assert_eq!(pending.borrow().get(&ParamKind::Maxfps).unwrap().0, 20.0);
    }

    #[test]
    fn flush_sends_debounced_param_over_udp() {
        let server = UdpSocket::bind("127.0.0.1:0").unwrap();
        server
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let target = server.local_addr().unwrap();

        let client_sock = UdpSocket::bind("127.0.0.1:0").unwrap();
        let (log_tx, _log_rx) = mpsc::channel();
        let client = OscClient::new(client_sock, target, log_tx);

        let state = Rc::new(RefCell::new(AppState::default()));
        let pending = Rc::new(RefCell::new(HashMap::new()));
        pending.borrow_mut().insert(
            ParamKind::Guidance,
            (2.5, Instant::now() - Duration::from_millis(400)),
        );

        flush_pending_params(&client, &state, &pending);

        let mut buf = [0u8; 1024];
        let (n, _) = server.recv_from(&mut buf).unwrap();
        let (_, packet) = rosc::decoder::decode_udp(&buf[..n]).unwrap();
        match packet {
            rosc::OscPacket::Message(m) => {
                assert_eq!(m.addr, "/guidance");
                assert_eq!(m.args, vec![rosc::OscType::Float(2.5)]);
            }
            other => panic!("expected a message, got {other:?}"),
        }

        // State reflects the (desired) value that was just sent.
        assert!((state.borrow().guidance - 2.5).abs() < 1e-6);
        // Nothing left pending.
        assert!(pending.borrow().is_empty());
    }
}
