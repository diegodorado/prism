# Prism Deck

`prism deck` is the native desktop controller for **Prism Server**. It is a
[Rust](https://www.rust-lang.org/) + [Slint](https://slint.dev) application that
speaks the server's OSC API over UDP.

The initial release is a **control surface only**. It does not embed or receive
the generated NDI video; monitor that in an external NDI-capable application.

## Status / scope

Implemented:

- OSC control of prompts, seed, guidance, delta, maxfps, and timestep index.
- Generation parameters are live sliders (no Apply button): moving one sends
  the value automatically after a 300 ms debounce, so dragging does not flood
  the server. Ranges: Guidance `0.0–5.0` (step `0.1`), Delta `0.0–2.0`
  (step `0.1`), Max FPS `1–30` (step `1`), Step index `0–49` (step `1`;
  single value, matching the server's current one-entry `steps` config).
  Seed stays a text field (`-1` = random each frame) and is also debounced.
- Transport: Play, Pause, Stop, Unload.
- Input selection by exact NDI source name via the existing `/input` message.
- Telemetry display: heartbeat, FPS, frame count, output name, confirmed input.
- Timestamped `/error` diagnostics and an optional sent-message log.
- Conventional resizable window and a screen-sized fullscreen mode.
- Local config in TOML (`~/Library/Application Support/prism-deck/config.toml`
  on macOS, or the platform equivalent from `dirs::config_dir()`).

Deferred (per the implementation plan):

- NDI source discovery over OSC (`/ndi/refresh` → `/ndi/sources`). Until the
  server ships those endpoints, the input control is a plain text box that sends
  the server's currently supported selection message.
- Presets (save / load / apply).
- Embedded NDI video preview.

## Protocol

Existing OSC messages used by the deck (see `server/main.py`):

| Address | Args | Direction |
|---|---|---|
| `/prompt` | `s` | Deck → Server |
| `/negative` | `s` | Deck → Server |
| `/seed` | `i` (`-1` = random per frame) | Deck → Server |
| `/guidance` | `f` | Deck → Server |
| `/delta` | `f` | Deck → Server |
| `/steps` | 1+ `i` | Deck → Server |
| `/maxfps` | `i` | Deck → Server |
| `/input` | `s` | Deck → Server |
| `/play` `/pause` `/stop` `/unload` | none | Deck → Server |
| `/active` | `i` | Server → Deck |
| `/fps` | `f` | Server → Deck |
| `/output_name` | `s` | Server → Deck |
| `/input_name` | `s` | Server → Deck |
| `/frames` | `i` | Server → Deck |
| `/error` | `s`, `s` | Server → Deck |

### Connection behavior

1. Bind the local telemetry socket (default `6503`) before sending anything.
2. Send an initial control message so the server learns our return address.
   `osc_send_ip: "auto"` replies to the last peer that sent a control message,
   so this step is required.
3. Treat the server as reachable only after a `/active` heartbeat arrives.
4. Mark the connection **Stale** when heartbeats stop (this does not imply the
   server process exited).
5. Only one controller is supported at a time; if another controller sends a
   message, telemetry routing follows it.

## Configuration

Defaults:

```toml
server_host = "127.0.0.1"
server_port = 7187
local_port = 6503
fullscreen = false
```

The socket layer binds `0.0.0.0:<local_port>` once and shares that socket
between the sender and receiver, so the server always sees a single, stable
return address. If the port is already in use, the app opens with the error
shown in **Diagnostics**; change `local_port` in the config file and restart.

Logs are written to a daily-rolling file under `prism-deck/logs/` inside the
platform state directory (on macOS: `~/Library/Application Support/prism-deck/logs`).
The path is printed to stderr on startup. Set `RUST_LOG=debug` for verbose
output including every sent command and received telemetry message.

## Build and run

Requires a Rust toolchain and, on Linux, the usual Slint system dependencies
(`libfontconfig`, `libxkbcommon`, OpenGL). On macOS no extra setup is needed.

```sh
cd deck
cargo run --release
```

Run the unit tests with:

```sh
cargo test
```

## Architecture

```
deck/
├── build.rs              # Slint codegen
├── ui/main.slint         # declarative UI (layout + event forwarding only)
└── src/
    ├── main.rs           # wiring: window <-> channels <-> OSC
    ├── app_state.rs      # UI-facing state + connection state machine
    ├── commands.rs       # user intentions (messages to the OSC thread)
    ├── config.rs         # TOML persistence
    ├── diagnostics.rs    # timestamped errors + sent log
    ├── logging.rs        # tracing to a rolling log file
    └── osc/
        ├── client.rs     # encode + send on a dedicated thread
        ├── receiver.rs   # decode on a background thread
        └── protocol.rs   # typed addresses/arguments + unit tests
```

Networking never blocks the UI thread: Slint callbacks push `Command` values
onto a channel; dedicated send/receive threads share one bound UDP socket; and
incoming OSC events come back over another channel. A Slint `Timer` on the UI
thread drains those channels and is the only place that mutates `AppState`, so
commands and telemetry can never race on shared UI state.
