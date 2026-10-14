# StreamDiffusion server — headless Linux setup (NDI + OSC)

Runs the StreamDiffusion pipeline headless on a Linux/NVIDIA box, with
**video over NDI** and **control/status over OSC**. No TouchDesigner required.

```
 Mac (controller, NDI camera pub)                 Linux GPU box (headless)
 ┌───────────────────────────┐                    ┌──────────────────────────────┐
 │ camera ──► NDI source ────┼─── NDI (in) ──────►│ ndi receive → sd-turbo       │
 │                           │                    │      (img2img)               │
 │ NDI receiver ◄────────────┼─── NDI (out) ◄─────│ ndi send                     │
 │ OSC sender ─── control ───┼─── OSC ───────────►│ OSC server (0.0.0.0)         │
 │ OSC receiver ◄─ status ───┼─── OSC ────────────│ OSC heartbeat/fps/name       │
 └───────────────────────────┘                    └──────────────────────────────┘
```

## 1. System dependencies

NDI discovery on Linux uses mDNS, which requires Avahi:

```bash
sudo apt update && sudo apt install -y avahi-daemon libnss-mdns
sudo systemctl enable --now avahi-daemon
```

`ndi-python` bundles `libndi.so`, so no separate NDI SDK install is needed.

## 2. Python environment (native venv, CPython 3.11)

```bash
cd StreamDiffusion
uv venv --python 3.11
uv pip install -r ../requirements.txt
uv pip install -e . --no-deps          # the streamdiffusion package itself
```

The exact proven pins live in `requirements.txt` at the repo root
(torch 2.1.0+cu121, xformers 0.0.22.post7, diffusers 0.24.0, transformers 4.48.3,
numpy 1.26.4, ndi-python 6.3.2.5).

TensorRT/ONNX are intentionally omitted; the engine runs with `acceleration: "none"`
(xformers still ships as a dependency).

## 3. Configuration (`configs/config.json`)

Minimal launch config — only deployment-specific values live here; the server
fills in engine defaults. The full OSC surface is documented in
[`docs/OSC_API.md`](OSC_API.md).

| key | type | meaning |
| --- | --- | --- |
| `model` | string | HF model id or local path (default setup: `stabilityai/sd-turbo`) |
| `width`, `height` | int | pipeline resolution (defaults 512) |
| `seed` | int | starting seed (default `-1` = randomize every frame) |
| `steps` | int[] | StreamDiffusion timestep indices (default `[17]`) |
| `guidance` | float | CFG scale (default `1.0`; `> 1` enables CFG) |
| `delta` | float | residual-noise multiplier (default `1.0`) |
| `maxfps` | int | output rate cap (default `4`) |
| `output` | string | name this process publishes on NDI (stable; e.g. `"StreamDiffusion"`) |

The prompt (positive/negative) and the NDI input source are **not** stored in
the config — the server starts empty. Set them at runtime over OSC:
`/prompt`, `/negative` and `/input` (see [`docs/OSC_API.md`](OSC_API.md)).

Optional keys with sane defaults: `osc_port` (**7187**), `osc_reply_port`
(**6503**), `osc_bind_ip` (`0.0.0.0`), `osc_send_ip` (`"auto"`), `gpu_id` (`0`),
`ndi_control_name`, `ndi_extra_ips` (comma-separated unicast discovery peers for
when mDNS/Avahi is unavailable).

NDI is the only frame transport; engine behavior (img2img + single prompt +
denoising batch) is frozen in `server/main.py`.

Ports (TouchDesigner convention):

* **Control in:** the box listens on `osc_port` (default **7187**). Send
  `/prompt`, `/seed`, `/guidance`, `/delta`, `/maxfps`, `/input`, `/pause`,
  `/play`, ... here.
* **Status out:** the box sends `/active`, `/fps`, `/output_name`,
  `/input_name`, `/frames` to the controller's `osc_reply_port` (default **6503**).
* With `osc_send_ip: "auto"`, the controller must send at least one OSC message
  before status starts flowing back to it.

## 4. Run

From the repo root:

```bash
./run.sh
# or, equivalently:
cd StreamDiffusion && .venv/bin/python ../server/main.py -c ../configs/config.json
```

The server resolves `models/` relative to the current working directory, so it
must be started with `StreamDiffusion/` as the cwd (what `run.sh` does).

First run downloads `stabilityai/sd-turbo` + `madebyollin/taesd` (~4 GB) into the HF cache.

## 5. NDI notes

* On the network a sender appears as `HOSTNAME (sd_name)`, e.g. `AI-STACK (StreamDiffusion)`.
  Match on the part in parentheses.
* Without Avahi, mDNS discovery returns nothing. As a fallback set `ndi_extra_ips`
  to the peer's IP (a unicast query, which also yields the peer's `ip:port`).
* Input frames of any size are resized to `width`x`height` (512x512 by default).
  Output is published as RGBA at the pipeline resolution.

### LAN advertisement ("why does my Mac not see the output?")

`libndi` automatically advertises each sender on the one network interface it
picks. On hosts that run Docker, the library can pick the `docker0` bridge
instead of the physical NIC — the source then *does not show up* for remote
machines on the LAN. This is a library NIC-selection quirk, **not** a problem
with your network.

You do **not** need to touch Docker, remove bridge IPs, or hardcode this
machine's IP anywhere. `server/main.py` sidesteps the quirk using NDI's
`SendAdvertiser` API: after the sender is up it publishes an additional
`_ndi._tcp` record at

```
<hostname>.local:<port>        e.g. ai-stack.local:5962
```

Avahi serves that record on the LAN interface and resolves the hostname to
the physical NIC's IP automatically (per-interface mDNS), so remote NDI
clients discover `HOSTNAME (StreamDiffusion)` and connect without any
configuration. If you ever see the output source advertised at a docker-bridge
address (e.g. `172.17.0.1`), ignore it — the `ai-stack.local` entry is the
reachable one.
