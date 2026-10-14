# ndi.py
"""NDI transport for the StreamDiffusion server.

Uses the ``ndi-python`` bindings (import name ``NDIlib``; the wheel bundles
``libndi.so``, so no separate NDI SDK install is required).

* Receives one input NDI source (optionally a second "control" source) as
  RGB ``uint8`` HxWx3 numpy arrays, resized to the pipeline resolution.
* Publishes generated frames as an NDI sender.

Discovery uses mDNS (needs ``avahi-daemon`` on Linux).  When ``extra_ips`` is
provided it is used as a unicast fallback so peers can be found without mDNS.
"""
import multiprocessing
import os
import signal
import socket
import threading
import time
from queue import Empty, Queue
from threading import Thread

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - cv2 is a project dependency
    cv2 = None

try:
    import NDIlib as ndi
except ImportError:
    print("NDIlib not found. NDI functionality will not be available.")
    ndi = None


def _as_str(value):
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", "replace")
    return str(value)


# -------------------------------------------------- dedicated NDI sender process
# The sender (plus the finder used to detect its port, and the LAN advertiser)
# runs in its own process so nothing on the send path can be disturbed by the
# receive path.  Note: the libndi 6.x core can abort (double-free) when a
# *remote* source stalls / goes static; a live source streams fine.


def _set_pdeathsig():
    """Linux: die when the parent process dies.

    The NDI sender is spawned inside the image-generation worker. When that
    worker is killed (e.g. the server is taken down), this sender must not
    survive as an orphan still publishing an NDI source, so arm
    PR_SET_PDEATHSIG to inherit SIGTERM from the kernel."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM, 0, 0, 0)  # PR_SET_PDEATHSIG = 1
    except Exception:
        pass
    # Parent already gone before prctl() armed the signal: we are orphaned.
    if os.getppid() == 1:
        os.kill(os.getpid(), signal.SIGTERM)


def _send_frame(send, frame):
    if send is None:
        return
    if frame.ndim == 3 and frame.shape[2] == 4:
        rgba = frame
    else:
        height, width = frame.shape[:2]
        rgba = np.empty((height, width, 4), dtype=np.uint8)
        rgba[:, :, :3] = frame[:, :, :3]
        rgba[:, :, 3] = 255
    rgba = np.ascontiguousarray(rgba, dtype=np.uint8)
    height, width = rgba.shape[:2]
    video = ndi.VideoFrameV2()
    video.data = rgba
    video.FourCC = ndi.FOURCC_VIDEO_TYPE_RGBA
    video.xres = width
    video.yres = height
    video.line_stride_in_bytes = width * 4
    video.frame_rate_N = 30
    video.frame_rate_D = 1
    ndi.send_send_video_v2(send, video)


def _detect_own_port(sender_name, extra_ips=None):
    """Return the UDP port our NDI sender advertises (same port it listens on)."""
    try:
        settings = ndi.FindCreate()
        settings.show_local_sources = True
        if extra_ips:
            settings.extra_ips = extra_ips
        finder = ndi.find_create_v2(settings)
    except Exception:
        return None
    if finder is None:
        return None
    try:
        deadline = time.time() + 2.0
        while time.time() < deadline:
            needle = sender_name.lower()
            for source in ndi.find_get_current_sources(finder):
                if needle in _as_str(source.ndi_name).lower() and source.url_address:
                    url = _as_str(source.url_address)
                    if ":" in url:
                        try:
                            return int(url.rsplit(":", 1)[1])
                        except ValueError:
                            pass
            time.sleep(0.1)
    finally:
        try:
            ndi.find_destroy(finder)
        except Exception:
            pass
    return None


def _sender_process_main(sender_name, frame_queue, extra_ips=None, host=None):
    """Entry point of the spawned NDI sender process."""
    _set_pdeathsig()
    try:
        ndi.initialize()
    except Exception:
        pass
    settings = ndi.SendCreate()
    settings.ndi_name = sender_name
    settings.clock_video = False
    settings.clock_audio = False
    send = ndi.send_create(settings)
    if send is None:
        print(f"[NDI-out] failed to create sender '{sender_name}'")
        return
    print(f"[NDI-out] publishing sender '{sender_name}'")

    adv_instance = None
    port = _detect_own_port(sender_name, extra_ips)
    if port:
        try:
            adv = ndi.SendAdvertiserCreateV2()
            hostname = (host or socket.gethostname()).split(".")[0].lower()
            adv.url_address = f"{hostname}.local:{port}"
            adv_instance = ndi.send_advertiser_create(adv)
            print(f"[NDI-out] LAN advertiser up at {hostname}.local:{port}")
        except Exception as exc:  # noqa: BLE001
            print(f"[NDI-out] LAN advertiser error: {exc}")
    try:
        while True:
            frame = frame_queue.get()
            if frame is None:
                break
            try:
                _send_frame(send, frame)
            except Exception as exc:  # noqa: BLE001
                print(f"[NDI-out] send error: {exc}")
    finally:
        if adv_instance is not None:
            try:
                ndi.send_advertiser_destroy(adv_instance)
            except Exception:
                pass
        try:
            ndi.send_destroy(send)
        except Exception:
            pass
    print("[NDI-out] sender process exiting")


class NDIUtils:
    def __init__(
        self,
        sender_name="StreamDiffusion_NDI_out",
        input_name=None,
        control_name=None,
        width=None,
        height=None,
        extra_ips=None,
        find_timeout_ms=5000,
        debug=False,
    ):
        self.sender_name = sender_name
        self.input_name = input_name
        self.control_name = control_name
        self.width = width
        self.height = height
        self.extra_ips = extra_ips
        self.find_timeout_ms = find_timeout_ms
        self.debug = debug

        self.sender_queue = None
        self.sender_proc = None
        self._spawn_ctx = multiprocessing.get_context("spawn")
        self._sender_lock = threading.Lock()
        self._last_spawn = 0.0
        self._respawn_failures = 0
        self.input_queue = Queue(maxsize=1)
        self.control_queue = Queue(maxsize=1) if control_name else None
        self.transmit_queue = Queue(maxsize=1)

        self.running = False
        self.input_thread = None
        self.control_thread = None
        self.feeder_thread = None
        self.monitor_thread = None

    # ------------------------------------------------------------------ lifecycle
    def start(self):
        if self.running:
            return
        if ndi is None:
            print("NDI requested but NDIlib is unavailable; NDI transport disabled.")
            return
        try:
            ndi.initialize()
        except Exception:
            pass
        self.running = True
        # Always run the input capture loop: the source can be selected at
        # runtime via set_input_name() (e.g. from an /input OSC message).
        self.input_thread = Thread(
            target=self._capture_loop,
            args=(lambda: self.input_name, self.input_queue),
            daemon=True,
        )
        self.input_thread.start()
        if self.control_name:
            self.control_thread = Thread(
                target=self._capture_loop,
                args=(lambda: self.control_name, self.control_queue),
                daemon=True,
            )
            self.control_thread.start()
        self._start_sender_process()
        self.feeder_thread = Thread(target=self._sender_feeder_loop, daemon=True)
        self.feeder_thread.start()
        self.monitor_thread = Thread(target=self._sender_monitor_loop, daemon=True)
        self.monitor_thread.start()

    def _start_sender_process(self):
        with self._sender_lock:
            self.sender_queue = self._spawn_ctx.Queue(maxsize=2)
            self.sender_proc = self._spawn_ctx.Process(
                target=_sender_process_main,
                args=(self.sender_name, self.sender_queue),
                kwargs={
                    "extra_ips": self.extra_ips,
                    "host": socket.gethostname().split(".")[0].lower(),
                },
                name="ndi-sender",
                daemon=True,
            )
            self.sender_proc.start()
            self._last_spawn = time.time()

    def _sender_queue_ref(self):
        with self._sender_lock:
            return self.sender_queue

    def _sender_feeder_loop(self):
        while self.running:
            try:
                frame = self.transmit_queue.get(timeout=0.2)
            except Empty:
                continue
            queue = self._sender_queue_ref()
            if queue is None:
                continue
            try:
                queue.put(frame, timeout=0.5)
            except Exception as exc:  # noqa: BLE001
                if self.debug:
                    print(f"[NDI-out] feed dropped frame: {exc}")

    def _sender_monitor_loop(self):
        while self.running:
            proc = self.sender_proc
            if proc is None:
                self._start_sender_process()
            elif not proc.is_alive():
                self._respawn_failures += 1
                if time.time() - self._last_spawn > 3.0 * self._respawn_failures:
                    print(f"[NDI-out] sender process died (attempt {self._respawn_failures}); respawning")
                    self._start_sender_process()
            else:
                self._respawn_failures = 0
            time.sleep(1.0)

    def stop(self):
        self.running = False
        for thread in (self.input_thread, self.control_thread, self.feeder_thread, self.monitor_thread):
            if thread is not None:
                thread.join(timeout=2.0)
        queue = self._sender_queue_ref()
        if queue is not None:
            try:
                queue.put(None, timeout=1.0)
            except Exception:
                pass
        proc = self.sender_proc
        if proc is not None:
            proc.join(timeout=2.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2.0)
        self.sender_proc = None
        self.sender_queue = None

    # ------------------------------------------------------------- source helpers
    def _make_finder(self):
        settings = ndi.FindCreate()
        settings.show_local_sources = True
        if self.extra_ips:
            settings.extra_ips = self.extra_ips
        return ndi.find_create_v2(settings)

    def _find_source(self, source_name, finder):
        needle = source_name.lower()
        for source in ndi.find_get_current_sources(finder):
            if needle in _as_str(source.ndi_name).lower():
                return source
        return None

    def _resize(self, frame):
        if self.width is None or self.height is None:
            return frame
        if frame.shape[0] == self.height and frame.shape[1] == self.width:
            return frame
        if cv2 is None:
            return frame
        return cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

    # --------------------------------------------------------------- capture side
    def _capture_loop(self, name_provider, queue):
        while self.running:
            source_name = name_provider()
            if not source_name:
                # No source selected yet (e.g. waiting for /input over OSC).
                time.sleep(0.2)
                continue
            finder = self._make_finder()
            if finder is None:
                print(f"[NDI] failed to create finder for '{source_name}'")
                time.sleep(1.0)
                continue
            try:
                source = None
                deadline = time.time() + self.find_timeout_ms / 1000.0
                while self.running and source is None and time.time() < deadline:
                    if name_provider() != source_name:
                        break  # selection changed while searching
                    source = self._find_source(source_name, finder)
                    if source is None:
                        time.sleep(0.25)
                if name_provider() != source_name:
                    continue  # selection changed; re-discover with the new name
                if source is None:
                    if self.debug:
                        print(f"[NDI] waiting for source matching '{source_name}'")
                    time.sleep(0.5)
                    continue
                if self.debug:
                    print(f"[NDI] connected to source '{_as_str(source.ndi_name)}'")
                self._receive_loop(source, queue, source_name, name_provider)
            except Exception as exc:  # noqa: BLE001
                print(f"[NDI] capture error for '{source_name}': {exc}")
                time.sleep(0.5)
            finally:
                try:
                    ndi.find_destroy(finder)
                except Exception:
                    pass
            if self.running:
                time.sleep(0.25)  # source vanished; retry discovery

    def _receive_loop(self, source, queue, source_name, name_provider):
        settings = ndi.RecvCreateV3()
        settings.color_format = ndi.RECV_COLOR_FORMAT_RGBX_RGBA
        receiver = ndi.recv_create_v3(settings)
        if receiver is None:
            print("[NDI] failed to create receiver")
            return
        try:
            ndi.recv_connect(receiver, source)
            while self.running:
                if name_provider() != source_name:
                    break  # /input changed; drop this source and re-discover
                frame_type, video, _audio, _meta = ndi.recv_capture_v2(receiver, 200, True, False, False)
                if frame_type == ndi.FRAME_TYPE_VIDEO:
                    try:
                        self._enqueue(queue, self._to_rgb(video))
                    finally:
                        ndi.recv_free_video_v2(receiver, video)
                elif frame_type == ndi.FRAME_TYPE_ERROR:
                    if self.debug:
                        print("[NDI] receiver error; reconnecting")
                    break
        finally:
            try:
                ndi.recv_destroy(receiver)
            except Exception:
                pass

    def _to_rgb(self, video):
        data = video.data
        if not isinstance(data, np.ndarray):
            data = np.frombuffer(data, dtype=np.uint8)
        height, width = video.yres, video.xres
        if data.ndim == 1:
            stride_px = max(video.line_stride_in_bytes // 4, width)
            data = data.reshape(height, stride_px, 4)[:, :width, :]
        rgb = np.ascontiguousarray(data[:, :, :3])
        return self._resize(rgb)

    @staticmethod
    def _enqueue(queue, frame):
        if queue.full():
            try:
                queue.get_nowait()
            except Empty:
                pass
        queue.put(frame)

    # ---------------------------------------------------------------- output side
    def transmit_frame(self, frame):
        """Queue a frame for the dedicated NDI sender process."""
        self._enqueue(self.transmit_queue, frame)

    # -------------------------------------------------------------------- public API
    def set_input_name(self, source_name):
        """Select the input NDI source at runtime (e.g. from /input OSC).

        ``None``/empty clears the selection; the receive loop then waits until a
        new name is set. The capture loop reconnects on its next iteration."""
        source_name = source_name or None
        if source_name == self.input_name:
            return
        self.input_name = source_name
        if source_name:
            print(f"\033[36m[NDI] switching input to '{source_name}'\033[0m")
        else:
            print("\033[36m[NDI] input cleared; waiting for a source name\033[0m")

    def capture_input_frame(self):
        try:
            return self.input_queue.get_nowait()
        except Empty:
            return None

    def capture_control_frame(self):
        if self.control_queue is None:
            return None
        try:
            return self.control_queue.get_nowait()
        except Empty:
            return None

    def transmit_frame(self, frame):
        self._enqueue(self.transmit_queue, frame)
