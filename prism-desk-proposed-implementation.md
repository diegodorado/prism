# Prism Desk — Proposed Implementation

## 1. Purpose

`desk/` is the native desktop controller for Prism, a real-time AI video-generation system. It runs on macOS initially and controls `prism/server` over OSC.

Prism Desk is written in Rust and uses **Slint** for its UI. The initial release is a control surface and does **not** embed or receive the generated NDI video. Video monitoring remains in an external NDI-capable application.

The application should be designed so embedded NDI preview can be added later without redesigning the OSC layer or moving application logic into UI callbacks.

## 2. Repository context

The project is a monorepo:

```text
prism/
├── desk/       # Rust + Slint desktop controller
├── server/     # Existing generation server
└── README.md
```

Prism Desk and Prism Server are independently runnable applications. The OSC API specification is the source of truth for the protocol. Do not introduce a shared Rust crate solely to share protocol definitions with the server.

## 3. Technology choices

- **Language:** Rust.
- **UI:** Slint, using its declarative `.slint` UI language and Rust integration.
- **OSC:** `rosc` or another maintained Rust OSC library selected during implementation.
- **Networking:** UDP sockets for OSC.
- **Concurrency:** Keep network I/O off the UI thread. Use a background receiver and message passing to update application state.
- **Settings and presets:** `serde` with TOML or JSON for local persistence; select one format and document it.
- **Logging:** `tracing` with a user-accessible diagnostics view or log file.

Keep networking, protocol handling, application state, and UI presentation in separate modules. Slint callbacks should issue application commands, not perform blocking network operations.

## 4. Goals

- Control the existing server through OSC.
- Edit positive and negative prompts.
- Set seed, guidance, delta, timestep indices, and maximum FPS.
- Start, pause, stop, and unload the generation model.
- Discover NDI inputs through an OSC request/response exchange and show them in a dropdown.
- Show the server's selected input, output name, activity, measured FPS, frame count, and errors.
- Save and load local parameter/prompt presets.
- Provide a conventional resizable control window and a fullscreen performance mode with a togglable translucent control overlay.
- Keep the initial application usable without embedded video preview.
- Distinguish commands sent from state confirmed by the server.

## 5. Non-goals for the initial release

- Receiving or rendering the generated NDI output inside Prism Desk.
- Rewriting the existing generation server in Rust.
- Replacing OSC with HTTP, WebSockets, or another application-level message transport.
- Implementing multi-client coordination; the server is currently single-client.
- Treating OSC/UDP as authenticated or reliable transport.

## 6. OSC transport and connection behavior

### Existing server contract

- Control input: server UDP port `7187` by default.
- Telemetry/error destination: client's UDP port `6503` by default.
- The server's `osc_send_ip: "auto"` behavior replies to the last peer that sent a control message.
- Send at least one control message before expecting replies.
- The server reports telemetry and rejected commands, not per-parameter acknowledgements.
- Only one client is supported at a time.

Ports should be configurable in Prism Desk. The default client receive port is `6503`, but the UI should explain if the port cannot be bound because another process is using it.

### Initial connection

1. Bind the local OSC receive socket before sending any requests.
2. Send an initial message so the server learns the client's return address. Prefer the NDI discovery request described below if supported by the server; otherwise use a documented harmless message.
3. Wait for `/active` heartbeat messages.
4. Mark the server as reachable only after a heartbeat arrives within the configured timeout.
5. Request the NDI source list and current state.
6. Display a disconnected/stale status if heartbeats stop; do not infer that a stopped heartbeat necessarily means the server process has exited.

Because the server is single-client and replies to the last peer that sent a control message, the UI should document that another controller can take over telemetry routing.

### Existing control messages

Prism Desk should preserve these addresses and argument types:

| Address | OSC arguments | UI behavior |
|---|---|---|
| `/prompt` | `s` | Set positive prompt |
| `/negative` | `s` | Set negative prompt |
| `/seed` | `i` | Set seed; `-1` means randomize every frame |
| `/guidance` | `f` | Set CFG scale; server documentation says SD-Turbo should use `1.0` |
| `/delta` | `f` | Set residual-noise multiplier |
| `/steps` | One or more `i` arguments | Set timestep indices; argument count must match server configuration |
| `/maxfps` | `i` | Cap output rate |
| `/input` | `s` | Existing substring-based NDI input selection; retain for compatibility during migration |
| `/play` | No arguments | Start/resume generation |
| `/pause` | No arguments | Pause generation |
| `/stop` | No arguments | Stop the server |
| `/unload` | No arguments | Unload model/free VRAM |

The server's `/error` message contains two strings: the rejected OSC address and a reason. Show these as timestamped diagnostic entries.

### Telemetry messages

| Address | Arguments | UI behavior |
|---|---|---|
| `/active` | `i` | Heartbeat; expected approximately every 0.75 seconds |
| `/fps` | `f` | Show measured output FPS |
| `/output_name` | `s` | Show configured NDI output sender name |
| `/input_name` | `s` | Show currently selected input |
| `/frames` | `i` | Show frame counter |
| `/error` | `s`, `s` | Add an error to diagnostics and surface it visibly |

Telemetry may arrive independently and in any order. Treat each message as an update to application state rather than expecting a fixed sequence.

## 7. NDI source discovery over bidirectional OSC

NDI source discovery is owned by **Prism Server**. Prism Desk does not independently enumerate NDI sources. The client requests the list, and the server responds over the existing bidirectional OSC connection. OSC remains the only application messaging transport.

### Proposed request/response

Use these proposed addresses:

| Direction | Address | Arguments | Meaning |
|---|---|---|---|
| Desk → Server | `/ndi/refresh` | none | Request an updated source list |
| Server → Desk | `/ndi/sources` | Multiple `s` arguments | Return the currently discovered source names |
| Server → Desk | `/ndi/selected` | `s` | Report the actual selected input name |
| Desk → Server | `/ndi/select` | `s` | Request selection by exact source name |

`/ndi/refresh`, `/ndi/sources`, `/ndi/selected`, and `/ndi/select` are proposed protocol additions and must be implemented/documented on the server side before the UI relies on them.

**Use one OSC message with multiple string arguments for `/ndi/sources`, not a comma-separated string.** Example:

```text
/ndi/sources "Swift NDI Camera" "OBS Virtual Camera" "Prism Output"
```

This preserves source names containing commas and avoids inventing an escaping convention. An empty list should be represented as a valid `/ndi/sources` message with zero source arguments. Test that the chosen OSC library and server implementation support empty-argument messages correctly.

If implementation constraints require a single string argument, define and test an escaping/serialization format before implementation; do not silently assume commas cannot occur in source names.

### Discovery flow

1. Prism Desk sends `/ndi/refresh` after establishing its OSC return path and when the user presses Refresh.
2. Prism Server refreshes or reads its authoritative NDI discovery results.
3. Prism Server replies to the requesting peer with `/ndi/sources`.
4. Prism Desk replaces the displayed source list with the returned list.
5. When the user selects a source, Prism Desk sends `/ndi/select` with the selected exact name.
6. Prism Server validates and attempts the selection.
7. Prism Server sends `/ndi/selected` only when it can report the actual selected source. Rejected requests should use `/error` with the relevant address and reason.
8. Existing `/input_name` telemetry may continue to report the current input for compatibility. Define whether it remains authoritative or is deprecated in favor of `/ndi/selected`; do not allow contradictory UI state.

The server should also push `/ndi/sources` updates when the discovery list changes, if practical. The explicit request/response must still work and should not depend on waiting for an unsolicited update.

### Selector behavior

- Show source names in a dropdown, with a Refresh action.
- Show a loading state while a refresh is outstanding.
- Keep the last known list visible if a refresh fails, but mark it stale.
- Distinguish the requested source from the server-confirmed selected source.
- If the selected source disappears, show it as unavailable; do not silently switch to another source.
- Handle duplicate display names or ambiguous identities explicitly. Exact names are better than substring matching, but if the NDI SDK exposes a stable source identifier, consider using it in a future protocol revision.
- Retain `/input` only for compatibility with existing tools; new Prism Desk selection should use `/ndi/select`.

## 8. UI structure

### Conventional control window

Organize the window into compact, clearly labeled sections:

1. **Connection/status**
   - Server host and control port.
   - Local telemetry port.
   - Connect/reconnect or connection configuration.
   - Heartbeat/reachability indicator.
   - Output name, selected input, FPS, and frame count.

2. **Transport**
   - Play/resume.
   - Pause.
   - Stop server, with confirmation.
   - Unload model, with confirmation and clear explanation that it frees VRAM.
   - Do not present Pause, Stop, and Unload as equivalent actions.

3. **Prompts**
   - Positive prompt multiline editor.
   - Negative prompt multiline editor.
   - Explicit Apply buttons by default; optionally add configurable live updates.
   - Avoid sending a new prompt on every keystroke unless the user deliberately enables that behavior.

4. **Generation parameters**
   - Seed, including a clear indication that `-1` randomizes every frame.
   - Guidance.
   - Delta.
   - Maximum FPS.
   - Timestep index editor supporting the required number of integer values.
   - Validate local input before sending, but treat server `/error` messages as the final indication that a request was rejected.

5. **NDI input**
   - Source dropdown from `/ndi/sources`.
   - Refresh button.
   - Current confirmed selection.
   - Empty-list, loading, stale-list, and unavailable-source states.

6. **Presets**
   - Save/load/delete local presets.
   - A preset stores positive/negative prompts and generation parameters.
   - Presets should not implicitly start/stop the server or change NDI routing unless a future explicit option is added.
   - Make it clear that loading a preset sends multiple OSC commands and is not an atomic server-side transaction.

7. **Diagnostics**
   - Timestamped `/error` messages.
   - Last heartbeat time and connection status.
   - Optional sent-message log for troubleshooting.
   - Avoid flooding the UI with every telemetry update; update status fields in place.

### Fullscreen performance mode

Provide two modes sharing the same application state:

- **Conventional mode:** resizable window for configuration, prompts, presets, and diagnostics.
- **Fullscreen performance mode:** fullscreen presentation with a toggleable translucent control overlay and keyboard shortcuts.

The initial release does not embed or receive the NDI output. Therefore, a transparent fullscreen Prism Deck window above an external NDI viewer depends on macOS window stacking, focus, fullscreen behavior, and click handling; it is not guaranteed by Slint alone. Treat this as a constrained mode with the conventional window as a reliable fallback.

When embedded NDI preview is added later, Prism Deck can render the video and overlay in one window. Keep UI state and OSC logic independent of the rendering implementation to support that future work.

## 9. Application architecture

Keep UI presentation separate from networking and state management.

Suggested conceptual modules:

```text
desk/
├── Cargo.toml
├── build.rs
├── ui/
│   ├── main.slint
│   ├── components/
│   └── views/
└── src/
    ├── main.rs
    ├── app_state.rs
    ├── commands.rs
    ├── config.rs
    ├── osc/
    │   ├── mod.rs
    │   ├── client.rs
    │   ├── receiver.rs
    │   └── protocol.rs
    ├── presets.rs
    └── diagnostics.rs
```

This is a proposed starting layout, not a requirement to create a module for every file if the implementation remains small.

### Responsibilities

- `app_state`: UI-facing state, connection state, confirmed server state, source list, and pending operations.
- `commands`: user intentions such as apply prompt, change parameter, refresh sources, select source, play, or pause.
- `osc/client`: encode and send OSC messages.
- `osc/receiver`: receive and decode messages on a background thread.
- `osc/protocol`: typed representations of known OSC addresses and argument validation.
- `config`: persisted connection and UI preferences.
- `presets`: local preset serialization.
- `diagnostics`: timestamped errors and optional message logging.
- Slint files: layout, styling, bindings, and event forwarding only.

Use a message-passing boundary between background networking and the UI/application state. Never block the UI thread waiting for UDP responses.

## 10. State semantics

Maintain separate representations for:

- **Desired values:** values the user currently intends to send.
- **Last-sent values:** values transmitted to the server.
- **Confirmed/reported values:** values the server has actually reported, where the protocol exposes them.
- **Connection freshness:** when the last heartbeat was received.
- **Pending requests:** especially NDI discovery and source selection.

Because existing parameter messages have no acknowledgements, do not label a parameter “confirmed” merely because its OSC message was sent. The UI can show the local control value and, separately, whether a request was rejected.

For source discovery, consider a request generation counter or request timestamp so delayed responses can be handled sensibly. OSC has no built-in request IDs in this proposed protocol, so avoid claiming that a response can be correlated perfectly if multiple refreshes are outstanding. Initially, permit only one outstanding refresh at a time.

## 11. Configuration and security

Persist locally:

- Last-used server host.
- Server control port (default `7187`).
- Local telemetry port (default `6503`).
- UI mode and window preferences.
- Local presets.

Do not store secrets unless authentication is added in a future protocol revision.

OSC over UDP does not authenticate senders or encrypt traffic. Keep the server and its control port on a trusted network, and do not expose it directly to the public internet. Tailscale can provide a private network path, but it does not change the semantics of OSC itself.

## 12. Testing strategy

### Unit tests

- OSC encoding for every control message.
- OSC decoding and validation for telemetry and errors.
- `/ndi/sources` with zero, one, and multiple arguments, including source names containing commas.
- Handling malformed or unexpected OSC messages.
- Preset serialization and validation.
- Parameter bounds and timestep-list validation.

### Integration tests

Use a small fake OSC server to verify:

- Initial connection and heartbeat timeout.
- Requests are sent to the configured host and port.
- Source refresh populates the dropdown.
- Selecting a source updates only after server confirmation.
- `/error` messages appear in diagnostics.
- Out-of-order telemetry does not corrupt state.
- Server silence is reflected as stale/disconnected status.
- Another peer taking over the server's reply destination is explained in diagnostics where detectable.

### Manual acceptance tests

- Run Prism Desk on macOS and Prism Server on Linux.
- Verify source discovery across the LAN and under the actual firewall/VLAN setup.
- Verify prompt and parameter control while generation is running.
- Verify that Pause, Stop, and Unload have distinct effects.
- Exercise conventional and fullscreen modes with the external NDI viewer.
- Test server restart, NDI source disappearance, and reconnect behavior.

## 13. Implementation milestones

1. **Protocol contract:** document existing OSC messages and agree on the new NDI discovery request/response messages with the server implementation.
2. **Rust application skeleton:** create the Slint window, settings model, and application-state boundary.
3. **OSC transport:** send controls, receive telemetry, display heartbeat and errors.
4. **NDI discovery and selection:** implement `/ndi/refresh` → `/ndi/sources`, the dropdown, `/ndi/select`, and confirmed selection.
5. **Core controls:** prompts, parameters, play/pause/stop/unload, and validation.
6. **Presets and diagnostics:** local persistence and troubleshooting UI.
7. **Fullscreen mode:** overlay, keyboard shortcuts, and fallback behavior without embedded video.
8. **Integration and packaging:** test against the real Linux server, document setup, and produce a macOS build.

## 14. Definition of done for the initial release

Prism Desk is ready for initial use when it can:

- Connect to the existing server over OSC without blocking the UI.
- Display heartbeat, FPS, frame count, output name, and errors.
- Control all existing documented OSC parameters and lifecycle commands.
- Request and display the server's NDI source list over OSC.
- Select an NDI input and reflect the server-confirmed result.
- Save and restore local presets and UI preferences.
- Switch between conventional and fullscreen control modes without losing application state.
- Operate without embedding NDI video.
- Recover gracefully from server restart, missing sources, malformed messages, and network interruption.

## 15. Open decisions before implementation

- Confirm the exact argument convention for `/ndi/sources` (recommended: multiple OSC string arguments, not comma-separated text).
- Decide whether `/ndi/selected` replaces `/input_name` or supplements it during migration.
- Confirm the behavior and argument types for `/ndi/refresh` and `/ndi/select` in the server.
- Determine the server's configured number of `/steps` indices and how Prism Desk discovers or configures that count.
- Decide whether fullscreen overlay over an external viewer is a best-effort feature or whether fullscreen mode initially means a fullscreen control surface only.
