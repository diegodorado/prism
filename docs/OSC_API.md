# OSC control API

The server is a **single-client** real-time device: you send setpoints over OSC, it
never asks for confirmation, and everything it says back is telemetry or a single
error channel.

## Transport

| direction | address | port (default) | config keys |
| --- | --- | --- | --- |
| control in (you → server) | `0.0.0.0:7187` | `osc_port` | defaults keep the TouchDesigner contract |
| status out (server → you) | your machine's `6503` | `osc_reply_port` | you receive on 6503 |
| error out | same as status | — | `/error` only |

With `osc_send_ip: "auto"` (default) the server replies to the **last peer that
sent a control message**. Send at least one control message before expecting
status. A fixed reply host can be set with `osc_send_ip`.

## Control — you send to port 7187

| address | type | meaning |
| --- | --- | --- |
| `/prompt` | string | replace the positive prompt (live) |
| `/negative` | string | replace the negative prompt (live) |
| `/seed` | int | set the noise seed; `-1` randomizes **every frame** |
| `/guidance` | float | CFG scale. `> 1.0` enables classifier-free guidance (dual embedding path). For sd-turbo keep `1.0` |
| `/delta` | float | residual-noise multiplier: higher = more drift/motion, lower = sticks closer to the input |
| `/steps` | int… | StreamDiffusion **timestep indices**, e.g. `/steps i 17`. The list length must match the configured `steps` list and every index must be a valid scheduler timestep index (`0..49` for this setup); otherwise `/error` is sent |
| `/maxfps` | int | cap the output rate (e.g. `4`) |
| `/input` | string | select the NDI input source (substring of the sender name, e.g. `"Swift NDI Camera"`). Empty string clears the selection |
| `/play` | — | start / resume generation |
| `/pause` | — | freeze output (idles until `/play`) |
| `/stop` | — | stop the server |
| `/unload` | — | unload the model (frees VRAM) |

## Status — server sends to your port 6503

| address | type | meaning |
| --- | --- | --- |
| `/active` | int | heartbeat `1` every 0.75 s |
| `/fps` | float | measured output rate (updated when a frame was made) |
| `/output_name` | string | NDI sender name (`config.json` → `output`) |
| `/input_name` | string | currently selected NDI input |
| `/frames` | int | frame counter |

## Error — the single feedback channel

| address | args | meaning |
| --- | --- | --- |
| `/error` | string, string | `<address> <reason>` — sent only when a control message is rejected (e.g. `/steps` with the wrong length). Silence otherwise |

## Example (`oscsend` from the controller via liblo)

```bash
oscsend 192.168.0.151 7187 /prompt s "a cinematic invasion by surreal Martians"
oscsend 192.168.0.151 7187 /negative s "distorted"
oscsend 192.168.0.151 7187 /seed i 42
oscsend 192.168.0.151 7187 /seed i -1
oscsend 192.168.0.151 7187 /guidance f 1.0
oscsend 192.168.0.151 7187 /delta f 0.9
oscsend 192.168.0.151 7187 /steps i 17
oscsend 192.168.0.151 7187 /maxfps i 8
oscsend 192.168.0.151 7187 /input s "Swift NDI Camera"
oscsend 192.168.0.151 7187 /pause
oscsend 192.168.0.151 7187 /play
```

## Notes

* No per-parameter acknowledgements; the client knows what it sent. Only
  malformed/rejected values produce `/error`.
* The server starts **empty**: no prompt and no NDI input selected. Send
  `/prompt` (and `/input` to pick the camera) before use; until the selected
  input source feeds frames the server idles without generating.
* `/prompt` is a single text string; the engine's internal prompt blending runs
  with one `[text, 1.0]` entry.
* Frames keep their natural streaming continuity; prompt changes are applied
  live, with no explicit crossfade.