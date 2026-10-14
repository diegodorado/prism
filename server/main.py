# server/main.py: OSC-driven StreamDiffusion server (NDI transport)
import os
import sys
import json
import argparse
import signal
from multiprocessing import Process, Queue, Manager
from typing import Dict, Literal, Optional, List, Union  # Required for type hints
import time
import threading
import subprocess
import re
from collections import deque
from contextlib import nullcontext
import cProfile
import pstats

# The OSC server and generation worker run as separate processes. CUDA cannot
# be re-initialized in a forked child once the parent has touched the GPU, so
# children are always spawned (Linux-only setup).
import multiprocessing as _mp

try:
    _mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

def set_pdeathsig():
    """Linux: receive SIGTERM when the parent (server) process dies.

    PR_SET_PDEATHSIG makes the kernel deliver SIGTERM to this worker as soon
    as its parent exits — including parent death by SIGKILL (kill -9), which
    no signal handler in the parent could catch. Without it spawned workers
    survive as orphans after the server is killed, keeping control port 7187
    and the NDI sender alive."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM, 0, 0, 0)  # PR_SET_PDEATHSIG = 1
    except Exception:
        pass
    # If the parent died before prctl() armed the signal, the kernel will not
    # retroactively notify us — we have been reparented to init. Exit now.
    if os.getppid() == 1:
        os.kill(os.getpid(), signal.SIGTERM)

# Every spawned worker — including the SyncManager server process, which
# multiprocessing launches without running any of our entry points — imports
# this module at startup, so arm PR_SET_PDEATHSIG here for all of them. The
# main process itself is skipped (its parent is the shell/systemd, not us).
if _mp.current_process().name != "MainProcess":
    set_pdeathsig()


def read_initial_config(config_path):
    """Minimal config reading just to get GPU ID"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config.get("gpu_id", 0)

def init_gpu_process(gpu_id):
    """Set the CUDA device before any torch-related imports."""
    # Import torch here just for GPU detection
    import torch
    
    # Get number of available GPUs
    num_gpus = torch.cuda.device_count()
    
    if num_gpus == 0:
        print("\033[33mWarning: No CUDA devices available. Defaulting to CPU.\033[0m")
        return
        
    # Validate requested GPU ID
    if gpu_id >= num_gpus:
        print(f"\033[33mWarning: Requested GPU {gpu_id} not available. Found {num_gpus} GPU(s).")
        print(f"Available devices:\033[0m")
        for i in range(num_gpus):
            print(f"\033[33m  GPU {i}: {torch.cuda.get_device_name(i)}\033[0m")
        print(f"\033[33mDefaulting to GPU 0.\033[0m")
        gpu_id = 0
        
    # Set CUDA device and environment variable
    torch.cuda.set_device(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Print selected device info
    print(f"\033[32mUsing GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}\033[0m")

# 3. Set up paths and GPU before any CUDA/torch imports
current_script_dir = os.path.dirname(os.path.abspath(__file__))
default_config_path = os.path.join(current_script_dir, '..', 'configs', 'config.json')
gpu_id = read_initial_config(default_config_path)
init_gpu_process(gpu_id)

# Additional imports after GPU setup
import numpy as np
import cv2
import fire
from queue import Empty
import torch
from torchvision.transforms.functional import to_tensor
import torch.nn.functional as F
import uuid
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
import array
from itertools import repeat
from pythonosc import udp_client


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from wrapper import StreamDiffusionWrapper
from pipeline import StreamDiffusion

# from streamdiffusion.image_utils import postprocess_image
from attention import CachedSTXFormersAttnProcessor, CachedSTAttnProcessor2_0  #V2V


class OSCClientFactory:
    _clients = {}
    host = "127.0.0.1"

    @staticmethod
    def set_host(host):
        """Set the destination host for outgoing OSC (the controller machine)."""
        if host and host != OSCClientFactory.host:
            OSCClientFactory.host = host
            OSCClientFactory._clients.clear()

    @staticmethod
    def get_client(osc_out_port):
        key = (OSCClientFactory.host, osc_out_port)
        if key not in OSCClientFactory._clients:
            OSCClientFactory._clients[key] = udp_client.SimpleUDPClient(OSCClientFactory.host, osc_out_port)
        return OSCClientFactory._clients[key]

def send_osc_message(address, value, osc_out_port):
    client = OSCClientFactory.get_client(osc_out_port)
    client.send_message(address, value)


class _AutoTargetOSCUDPServer(BlockingOSCUDPServer):
    """OSC server that remembers the last client, so status replies go back to
    whichever machine is controlling us (used when osc_send_ip == 'auto')."""

    shared_data = None
    auto_target = False

    def verify_request(self, request, client_address):
        valid = super().verify_request(request, client_address)
        if valid and self.auto_target:
            try:
                peer_ip = client_address[0]
            except Exception:
                peer_ip = None
            if peer_ip:
                OSCClientFactory.set_host(peer_ip)
                if self.shared_data is not None:
                    self.shared_data["osc_peer_ip"] = peer_ip
        return valid


def calculate_fps_and_send_osc(start_time, transmit_count, osc_out_port, sender_name, frame_created, use_controlnet, input_name=None):
    if frame_created:
        transmit_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time >= 1.0:
        if frame_created:
            fps = round(transmit_count / elapsed_time, 3)
            send_osc_message('/fps', fps, osc_out_port)
            controlnet_status = "ControlNet is Active | " if use_controlnet else ""
            print(f"Streaming... Active | {controlnet_status}Sender: {sender_name} | FPS: {fps}\r", end='', flush=True)
        send_osc_message('/output_name', sender_name, osc_out_port)
        if input_name is not None:
            send_osc_message('/input_name', input_name, osc_out_port)
        start_time = time.time()
        if frame_created:
            transmit_count = 0
    return start_time, transmit_count

def osc_server(shared_data, ip='0.0.0.0', port=8247, osc_transmit_port=8248, osc_send_ip='127.0.0.1'):
    set_pdeathsig()  # never outlive the server process
    auto_target = osc_send_ip is None or str(osc_send_ip).strip().lower() in ("auto", "", "none")
    if not auto_target:
        OSCClientFactory.set_host(osc_send_ip)
    def send_error(address, reason):
        """Single feedback channel: `/error <address> <reason>`."""
        try:
            send_osc_message("/error", (address, str(reason)), osc_transmit_port)
        except Exception:
            pass

    # --- live control ---
    def set_prompt_handler(address, *args):
        try:
            prompt = args[0]
            if not isinstance(prompt, str):
                raise ValueError("expected a string")
        except Exception as e:
            send_error(address, e)
            return
        shared_data["prompt_list"] = [[prompt, 1.0]]  # internal [[text, weight]] form

    def set_negative_prompt_handler(address, *args):
        try:
            if not isinstance(args[0], str):
                raise ValueError("expected a string")
        except Exception as e:
            send_error(address, e)
            return
        shared_data["negative_prompt"] = args[0]

    def set_seed_handler(address, *args):
        try:
            seed = int(round(float(args[0])))  # -1 randomizes each frame
        except Exception as e:
            send_error(address, e)
            return
        shared_data["seed_list"] = [[seed, 1.0]]  # internal [[seed, weight]] form

    def set_guidance_handler(address, *args):
        try:
            guidance = float(args[0])
        except Exception as e:
            send_error(address, e)
            return
        shared_data["guidance_scale"] = guidance

    def set_delta_handler(address, *args):
        try:
            delta = float(args[0])
        except Exception as e:
            send_error(address, e)
            return
        shared_data["delta"] = delta

    def set_steps_handler(address, *args):
        try:
            steps = [int(round(float(a))) for a in args]
            if not steps:
                raise ValueError("expected at least one timestep")
        except Exception as e:
            send_error(address, e)
            return
        shared_data["t_list"] = steps

    def set_maxfps_handler(address, *args):
        try:
            maxfps = int(round(float(args[0])))
            if maxfps <= 0:
                raise ValueError("must be a positive number")
        except Exception as e:
            send_error(address, e)
            return
        shared_data["max_fps"] = maxfps

    def set_input_handler(address, *args):
        """Select which NDI source feeds the pipeline. Empty string clears it."""
        try:
            if not args or not isinstance(args[0], str):
                raise ValueError("expected a source name string")
        except Exception as e:
            send_error(address, e)
            return
        shared_data["ndi_input_name"] = args[0]

    # --- transport ---
    def stop_stream_handler(address, *args):
        shared_data["stop_stream"] = True
    def pause_stream_handler(address, *args):
        shared_data["pause_stream"] = True
    def play_stream_handler(address, *args):
        shared_data["play_stream"] = True
    def unload_stream_handler(address, *args):
        shared_data["unload_stream"] = True

    dispatcher = Dispatcher()
    dispatcher.map("/prompt", set_prompt_handler)
    dispatcher.map("/negative", set_negative_prompt_handler)
    dispatcher.map("/seed", set_seed_handler)
    dispatcher.map("/guidance", set_guidance_handler)
    dispatcher.map("/delta", set_delta_handler)
    dispatcher.map("/steps", set_steps_handler)
    dispatcher.map("/maxfps", set_maxfps_handler)
    dispatcher.map("/input", set_input_handler)
    dispatcher.map("/stop", stop_stream_handler)
    dispatcher.map("/pause", pause_stream_handler)
    dispatcher.map("/play", play_stream_handler)
    dispatcher.map("/unload", unload_stream_handler)
    server = _AutoTargetOSCUDPServer((ip, port), dispatcher)
    server.shared_data = shared_data
    server.auto_target = auto_target
    print(f"\033[35m=======================================\nOSC ready on {ip}:{port}\n=======================================\033[0m\n")
    def send_heartbeat():
        while True:
            send_osc_message("/active", 1, osc_transmit_port)
            time.sleep(.75)
    heartbeat_thread = threading.Thread(target=send_heartbeat)
    heartbeat_thread.daemon = True
    heartbeat_thread.start()
    server.serve_forever()

def print_banner():
    version = "0.2.99"
    banner = f"""
\033[33m=======================================\033[0m
\033[33mStreamDiffusion Server v{version}\033[0m
\033[33m=======================================\033[0m
    """
    print(banner)

def print_gpu_info(gpu_id=0):
    query_attributes = ['name', 'memory.total', 'memory.used']
    command = f'nvidia-smi --id={gpu_id} --query-gpu={",".join(query_attributes)} --format=csv,nounits,noheader'
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True, check=True)
        nvidia_smi_info = dict(zip(query_attributes, [value.strip() for value in result.stdout.strip().split(',')]))
        gpu_id_str = f" {gpu_id}" if gpu_id != 0 else ""
        print(f"\033[33mGPU{gpu_id_str} Info: \033[37m{nvidia_smi_info['name']}\033[33m [ {round(int(nvidia_smi_info['memory.total']) / 1024, 1)} GiB total, {round(int(nvidia_smi_info['memory.used']) / 1024, 1)} GiB used ]\033[0m\n")
    except subprocess.CalledProcessError:
        print(f"Failed to execute nvidia-smi for GPU {gpu_id}")

def terminate_processes(processes):
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join()

def image_generation_process(
    model_id_or_path: str,
    lora_dict: Optional[Dict[str, float]],
    prompt: str,
    negative_prompt: str,
    frame_buffer_size: int,
    width: int,
    height: int,
    acceleration: Literal["none", "xformers", "tensorrt"],
    use_denoising_batch: bool,
    seed: int,
    cfg_type: Literal["none", "full", "self", "initialize"],
    guidance_scale: float,
    delta: float,
    do_add_noise: bool,
    enable_similar_image_filter: bool,
    similar_image_filter_threshold: float,
    similar_image_filter_max_skip_frame: float,
    shared_data,
    t_index_list: List[int] ,
    mode:str,
    lcm_lora_id: Optional[str] = None,
    vae_id: Optional[str] = None,
    osc_transmit_port: Optional[int] = None,
    scheduler_name: str = "EulerAncestral",
    use_karras_sigmas: bool = False,
    device: Literal["cpu", "cuda"] = "cuda",
    gpu_id: int = 0,  # New parameter for GPU ID
    use_controlnet: bool = False,
    controlnet_model: Optional[str] = None,
    controlnet_weight: float = 0.5,
    use_cached_attn: bool = True,
    sd_model_type: Optional[Literal["sd15", "sd21", "sdxl"]] = None,
    max_fps: int = 60,
    textual_inversion_dict: Optional[Dict[str, Optional[str]]] = None
) -> None:
    """
    Process for generating images based on a prompt using a specified model.
    Parameters
    ----------
    queue : Queue
        The queue to put the generated images in.
    fps_queue : Queue
        The queue to put the calculated fps.
    model_id_or_path : str
        The name of the model to use for image generation.
    lora_dict : Optional[Dict[str, float]], optional
        The lora_dict to load, by default None.
        Keys are the LoRA names and values are the LoRA scales.
        Example: {"LoRA_1" : 0.5 , "LoRA_2" : 0.7 ,...}
    prompt : str
        The prompt to generate images from.
    negative_prompt : str, optional
        The negative prompt to use.
    frame_buffer_size : int, optional
        The frame buffer size for denoising batch, by default 1.
    width : int, optional
        The width of the image, by default 512.
    height : int, optional
        The height of the image, by default 512.
    acceleration : Literal["none", "xformers", "tensorrt"], optional
        The acceleration method, by default "tensorrt".
    use_denoising_batch : bool, optional
        Whether to use denoising batch or not, by default True.
    seed : int, optional
        The seed, by default 2. if -1, use random seed.
    cfg_type : Literal["none", "full", "self", "initialize"],
    optional
        The cfg_type for img2img mode, by default "self".
        You cannot use anything other than "none" for txt2img mode.
    guidance_scale : float, optional
        The CFG scale, by default 1.2.
    delta : float, optional
        The delta multiplier of virtual residual noise,
        by default 1.0.
    do_add_noise : bool, optional
        Whether to add noise for following denoising steps or not,
        by default True.
    enable_similar_image_filter : bool, optional
        Whether to enable similar image filter or not,
        by default False.
    similar_image_filter_threshold : float, optional
        The threshold for similar image filter, by default 0.98.
    similar_image_filter_max_skip_frame : int, optional
        The max skip frame for similar image filter, by default 10.
    """
    set_pdeathsig()  # never outlive the server process
    shared_data["loading_status"] = True    
    _osc_target = shared_data.get("osc_send_ip", "127.0.0.1")
    auto_osc_target = _osc_target is None or str(_osc_target).strip().lower() in ("auto", "", "none")
    if not auto_osc_target:
        OSCClientFactory.set_host(_osc_target)
    use_lcm_lora = True if lcm_lora_id != 'skip' else False
    stream = StreamDiffusionWrapper(
        model_id_or_path=model_id_or_path,
        lora_dict=lora_dict,
        t_index_list=t_index_list,
        frame_buffer_size=frame_buffer_size,
        width=width,
        height=height,
        warmup=10,
        acceleration=acceleration,
        do_add_noise=do_add_noise,
        use_lcm_lora = use_lcm_lora,
        enable_similar_image_filter=enable_similar_image_filter,
        similar_image_filter_threshold=similar_image_filter_threshold,
        similar_image_filter_max_skip_frame=similar_image_filter_max_skip_frame,
        use_denoising_batch=use_denoising_batch,
        cfg_type=cfg_type,
        seed=seed,
        lcm_lora_id=lcm_lora_id,
        vae_id=vae_id,
        gpu_id=gpu_id,
        scheduler_name=scheduler_name,
        use_karras_sigmas=use_karras_sigmas,
        device=device,
        use_controlnet=use_controlnet,
        controlnet_model=controlnet_model,
        use_cached_attn=use_cached_attn,
        sd_model_type=sd_model_type,
        textual_inversion_dict=textual_inversion_dict
    )
    if lora_dict is not None:
        stream.update_lora_weights(lora_dict, print_info=True)
    current_prompt = prompt
    current_prompt_list = shared_data.get("prompt_list", [[prompt, 1.0]])
    current_seed_list = shared_data.get("seed_list", [[seed, 1.0]])
    # Publish the current values so the loop sees a stable baseline and the
    # single-value /prompt and /seed OSC messages hook into the same machinery.
    shared_data["prompt_list"] = current_prompt_list
    shared_data["seed_list"] = current_seed_list
    shared_data['t_list'] = t_index_list 
    o_index = t_index_list
    last_list = t_index_list
    reported_t_list = t_index_list
    noise_bank = {}
    prompt_cache = {}
    start_time = time.time()
    prompt_changed = False
    frame_count = 0
    transmit_count = 0
    loaded_lora_models = {}
    v2v_state = False
    interpolation_method = "average"
    print('Preparing Stream...')
    stream.prepare(
        prompt=current_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=50,
        guidance_scale=guidance_scale,
        delta=delta,
    )
    time.sleep(.5)
    if device == "cuda":
        print_gpu_info(gpu_id=gpu_id)

    # CUDA is the only accelerated path; the fallback branch is plain CPU.
    use_cuda_accel = (device == "cuda")

    TIME_DEBUG = False
    if TIME_DEBUG:
        def timer(func):
            def wrapper(*args, **kwargs):
                start = time.time()
                result = func(*args, **kwargs)
                end = time.time()
                print(f"{func.__name__} took {end - start:.4f} seconds")
                return result
            return wrapper
        timing_stats = {
            'parameter_update': [],
            'input_processing': [],
            'image_generation': [],
            'postprocessing': [],
            'osc_messages': []
        }
        loop_times = []
    previous_frame = None
    frame_changed = False
    input_tensor = torch.empty((3, height, width), dtype=torch.float32, device=device)
    control_input_tensor = torch.empty((3, height, width), dtype=torch.float32, device=device)
    
    # Pre-allocate tensors for optimization
    if use_cuda_accel:
        # GPU-PERSISTENT BUFFER OPTIMIZATION: Pre-allocate all tensors on GPU with pinned memory staging
        # Staging buffers (pinned memory for faster CPU->GPU transfers)
        staging_input_buffer_hwc = torch.empty((height, width, 3), dtype=torch.float32, pin_memory=True)
        staging_control_buffer_hwc = torch.empty((height, width, 3), dtype=torch.float32, pin_memory=True)
        staging_input_buffer_uint8 = torch.empty((height, width, 3), dtype=torch.uint8, pin_memory=True)
        staging_control_buffer_uint8 = torch.empty((height, width, 3), dtype=torch.uint8, pin_memory=True)
        
        # FIXED ASYNC POSTPROCESSING OPTIMIZATION: Correct buffer dimensions for inference output
        # The inference output is (1, 3, H, W) so we need buffers that match this exactly
        # GPU buffer for processing (matches inference output dimensions)
        async_output_buffer_gpu = torch.empty((1, 3, height, width), dtype=torch.float32, device=device)
        # Pinned CPU buffer for final result (HWC format for numpy output)
        async_output_buffer_pinned = torch.empty((height, width, 3), dtype=torch.uint8, pin_memory=True)
        # Intermediate GPU buffer for dimension conversion
        async_output_buffer_hwc_gpu = torch.empty((height, width, 3), dtype=torch.uint8, device=device)
        
        # GPU buffers for input processing
        gpu_input_buffer_uint8 = torch.empty((height, width, 3), dtype=torch.uint8, device=device)
        gpu_input_buffer_hwc = torch.empty((height, width, 3), dtype=torch.float32, device=device)
        gpu_control_buffer_uint8 = torch.empty((height, width, 3), dtype=torch.uint8, device=device)
        gpu_control_buffer_hwc = torch.empty((height, width, 3), dtype=torch.float32, device=device)
        
        # Create dedicated CUDA streams for async operations
        input_stream = torch.cuda.Stream()
        control_stream = torch.cuda.Stream()
        postprocess_stream = torch.cuda.Stream()
        

        memory_pools = create_optimized_memory_pools(height, width, torch.device(device))
        # Pre-allocate frequently used buffers for maximum efficiency
        main_sync_event = torch.cuda.Event()
        postprocess_sync_event = torch.cuda.Event()

    else:
        # CPU-only fallback - no async processing
        staging_input_buffer_hwc = torch.empty((height, width, 3), dtype=torch.float32)
        staging_control_buffer_hwc = torch.empty((height, width, 3), dtype=torch.float32)
        staging_input_buffer_uint8 = torch.empty((height, width, 3), dtype=torch.uint8)
        staging_control_buffer_uint8 = torch.empty((height, width, 3), dtype=torch.uint8)
        async_output_buffer_pinned = torch.empty((height, width, 3), dtype=torch.uint8)
        input_stream = None
        control_stream = None
        postprocess_stream = None
    
    # NDI is the only supported frame transport.
    stream_method = "ndi"
    ndi_handler = None

    from ndi import NDIUtils
    ndi_sender_name = shared_data.get("ndi_sender_name") or "StreamDiffusion"
    output_mem_name = ndi_sender_name
    ndi_input_name = shared_data.get("ndi_input_name")
    ndi_control_name = shared_data.get("ndi_control_name")
    print(
        f"\033[36mUsing NDI for frame transfer "
        f"(sender='{ndi_sender_name}', input='{ndi_input_name}')\033[0m"
    )
    ndi_handler = NDIUtils(
        sender_name=ndi_sender_name,
        input_name=ndi_input_name,
        control_name=ndi_control_name,
        width=width,
        height=height,
        extra_ips=shared_data.get("ndi_extra_ips"),
    )
    ndi_handler.start()

    shared_data["loading_status"] = False
    
    performance_baseline_fps = 0  # Track performance baseline
    peak_fps_tracker = deque(maxlen=10)  # Track recent peak performance
    
    # profiler = cProfile.Profile()
    # profiler.enable()  
    while True:
        try:
            if auto_osc_target:
                osc_peer_ip = shared_data.get("osc_peer_ip")
                if osc_peer_ip:
                    OSCClientFactory.set_host(osc_peer_ip)
            if TIME_DEBUG:
                param_update_start = time.time()#timer stats
            loop_start_time = time.time()#timer stats
            paused = shared_data.get("paused", False)
            if paused:
                time.sleep(.001)
                continue

            # OPTIMIZATION: Batch all shared_data.get() calls to reduce overhead
            # feedback_safe = shared_data.get("feedback_safe", False)  # Removed - handled in TouchDesigner
            new_lora_weights = shared_data.get("lora_weights", lora_dict)
            new_guidance_scale = float(shared_data.get("guidance_scale", guidance_scale))
            new_delta = float(shared_data.get("delta", delta))
            new_prompt_list = shared_data.get("prompt_list", {})
            new_negative_prompt = shared_data.get("negative_prompt", negative_prompt)
            new_interpolation_method = shared_data.get("interpolation_method", "average")
            new_ti_dict = shared_data.get("textual_inversion_dict", None)
            new_seed_list = shared_data.get("seed_list", current_seed_list)
            new_t_list = shared_data.get('t_list', t_index_list)
            new_controlnet_conditioning_scale = shared_data.get("controlnet_conditioning_scale", stream.stream.controlnet_conditioning_scale)
            new_use_controlnet = shared_data.get("use_controlnet", stream.stream.use_controlnet)
            # V2V parameters
            disable_cached_attn = shared_data.get("disable_cached_attn", True)
            new_use_cached_attn = shared_data.get("use_cached_attn", False)
            new_use_cached_attn_settings = shared_data.get("use_cached_attn_settings", {})
            
            # NDI input source can be re-pointed at runtime via /input, so the
            # operator picks the camera on the controller and the server never
            # hard-codes a source name.
            if stream_method == "ndi" and ndi_handler is not None:
                requested_input = shared_data.get("ndi_input_name")
                if requested_input != ndi_input_name:
                    ndi_input_name = requested_input
                    ndi_handler.set_input_name(ndi_input_name)

            # OPTIMIZATION: Only check and update parameters that actually changed
            
            # LoRA weights update
            if new_lora_weights != lora_dict:
                lora_dict = new_lora_weights
                update_lora_weights(stream.stream, lora_dict, use_lcm_lora, lcm_lora_id, stream)
            
            # PROMPT DICT + GUIDANCE SCALE + DELTA - Combined check for efficiency
            prompt_params_changed = (
                new_prompt_list != current_prompt_list or 
                new_guidance_scale != guidance_scale or 
                new_delta != delta or 
                new_negative_prompt != negative_prompt or
                new_interpolation_method != interpolation_method
                or frame_count < 10
            )
            
            if prompt_params_changed:
                current_prompt_list = new_prompt_list
                guidance_scale = new_guidance_scale
                delta = new_delta
                negative_prompt = new_negative_prompt
                interpolation_method = new_interpolation_method
                update_combined_prompts_and_parameters(
                    stream.stream, 
                    current_prompt_list, 
                    guidance_scale, 
                    delta, 
                    negative_prompt,
                    prompt_cache,
                    interpolation_method,
                    shared_data
                )
            
            # TEXTUAL INVERSIONS DICT - Optimized comparison
            if new_ti_dict is not None:
                current_ti_dict = {
                    info["path"]: info["token"]
                    for info in getattr(stream, 'loaded_textual_inversions', {}).values()
                }
                
                if new_ti_dict != current_ti_dict:
                    stream.update_textual_inversions(new_ti_dict, print_info=True)
            
            # SEED DICT - Only update if changed OR if any seed is -1 (randomize each frame)
            seed_randomization_needed = any(seed == -1 for seed, _ in new_seed_list)
            
            if new_seed_list != current_seed_list or seed_randomization_needed:
                # If randomization is needed, create a new seed list with -1 seeds replaced by random values
                if seed_randomization_needed:
                    randomized_seed_list = []
                    for seed_val, weight in new_seed_list:
                        if seed_val == -1:
                            # Generate random seed between 0 and 999999
                            random_seed = np.random.randint(0, 1000000)
                            randomized_seed_list.append([random_seed, weight])
                        else:
                            randomized_seed_list.append([seed_val, weight])
                    current_seed_list = randomized_seed_list
                else:
                    current_seed_list = new_seed_list
                    
                if any(weight > 0 for _, weight in current_seed_list):
                    blended_noise = blend_noise_tensors(current_seed_list, noise_bank, stream.stream)
                    stream.stream.init_noise = blended_noise
            
            # T_LIST - Optimized validation: the list length must match the
            # configured list and every index must exist in the scheduler's
            # timesteps (an out-of-range index would otherwise raise inside
            # update_t_list_attributes and kill the generation process).
            timestep_count = len(stream.stream.timesteps)
            if len(new_t_list) != len(o_index):
                reason = f"timestep list must have {len(o_index)} entries"
            elif any(t < 0 or t >= timestep_count for t in new_t_list):
                reason = f"timestep indices must be in 0..{timestep_count - 1}"
            else:
                reason = None
            if reason:
                if new_t_list != reported_t_list:
                    reported_t_list = new_t_list
                    send_osc_message("/error", ("/steps", reason), osc_transmit_port)
            elif new_t_list != last_list:
                last_list = new_t_list
                try:
                    update_t_list_attributes(stream.stream, new_t_list)
                except Exception as e:
                    send_osc_message("/error", ("/steps", f"failed to apply: {e}"), osc_transmit_port)
            
            # V2V - Optimized state management
            if disable_cached_attn and not v2v_state:
                update_attn_processors(stream, {}, disable_cached_attn=True, device=device)
                stream.stream.use_cached_attn = False
                stream.use_cached_attn_settings = {}
                v2v_state = True
            elif not disable_cached_attn and v2v_state:
                v2v_state = False
                if new_use_cached_attn:
                    stream.stream.use_cached_attn = new_use_cached_attn
                    stream.use_cached_attn_settings = new_use_cached_attn_settings.copy()
                    update_attn_processors(stream.stream, new_use_cached_attn_settings, disable_cached_attn=False, device=device)
            elif not disable_cached_attn:
                # Only update if values actually changed
                if new_use_cached_attn != stream.stream.use_cached_attn:
                    stream.stream.use_cached_attn = new_use_cached_attn
                if new_use_cached_attn_settings != stream.use_cached_attn_settings:
                    stream.use_cached_attn_settings = new_use_cached_attn_settings.copy()
                    update_attn_processors(stream.stream, new_use_cached_attn_settings, disable_cached_attn=False, device=device)
            
            # ControlNet updates - Only if changed
            if new_controlnet_conditioning_scale != stream.stream.controlnet_conditioning_scale:
                update_controlnet_conditioning_scale(stream.stream, shared_data)
            if new_use_controlnet != stream.stream.use_controlnet:
                stream.stream.use_controlnet = new_use_controlnet
            
            if TIME_DEBUG:
                timing_stats['parameter_update'].append(time.time() - param_update_start) #timer stats
                input_processing_start = time.time() #timer stats
            
            # Controlnet frame processing - OPTIMIZED with GPU-persistent buffers
            if use_controlnet:
                control_frame = ndi_handler.capture_control_frame()
                if control_frame is not None:
                    if device == "cpu":
                        control_input_tensor.copy_(torch.from_numpy(control_frame).permute(2, 0, 1).float().div(255))
                    else:
                        # GPU-PERSISTENT OPTIMIZATION: Use staging buffer and async transfer
                        with torch.cuda.stream(control_stream) if control_stream else nullcontext():
                            staging_control_buffer_uint8.copy_(torch.from_numpy(control_frame))
                            gpu_control_buffer_uint8.copy_(staging_control_buffer_uint8, non_blocking=True)
                            gpu_control_buffer_hwc.copy_(gpu_control_buffer_uint8.float().div(255), non_blocking=True)
                            control_input_tensor.copy_(gpu_control_buffer_hwc.permute(2, 0, 1), non_blocking=True)
            
            # img2img mode - OPTIMIZED with GPU-persistent buffers and async transfers
            if mode == "img2img":
                frame = ndi_handler.capture_input_frame()
                if frame is None:
                    # No input source selected yet (or a brief NDI stall): do not
                    # run the model on stale buffers. Idle until the next frame.
                    time.sleep(0.05)
                    continue
                try:
                    if device == "cpu":
                        input_tensor.copy_(torch.from_numpy(frame).permute(2, 0, 1).float().div(255))
                    else:
                        # GPU-PERSISTENT OPTIMIZATION: Use staging buffer and async transfer
                        with torch.cuda.stream(input_stream) if input_stream else nullcontext():
                            staging_input_buffer_uint8.copy_(torch.from_numpy(frame))
                            gpu_input_buffer_uint8.copy_(staging_input_buffer_uint8, non_blocking=True)
                            # Convert to target dtype (e.g., float16) and normalize on GPU
                            gpu_input_buffer_hwc.copy_(gpu_input_buffer_uint8.to(gpu_input_buffer_hwc.dtype).div_(255), non_blocking=True)
                            input_tensor.copy_(gpu_input_buffer_hwc.permute(2, 0, 1), non_blocking=True)
                except Exception as e:
                    print(f"\033[91mError processing NDI input frame: {e}\033[0m")
                if TIME_DEBUG:
                    timing_stats['input_processing'].append(time.time() - input_processing_start) #timer stats
                    image_generation_start = time.time() #timer stats
                
                # Use events instead of synchronize() for minimal blocking
                if device != "cpu":
                    if input_stream:
                        # Record event instead of full synchronization for better performance
                        main_sync_event.record(input_stream)
                        # PAUSE MODE: Force immediate synchronization to prevent frame delays
                        if paused:
                            main_sync_event.synchronize()  # Immediate sync in pause mode
                        else:
                            main_sync_event.synchronize()  # Minimal sync only when needed
                    if use_controlnet and control_stream:
                        postprocess_sync_event.record(control_stream)
                        # PAUSE MODE: Force immediate synchronization to prevent frame delays
                        if paused:
                            postprocess_sync_event.synchronize()  # Immediate sync in pause mode
                        else:
                            postprocess_sync_event.synchronize()  # Minimal sync only when needed
                
                if use_controlnet:
                    processed_tensor = stream.stream(input_tensor, control_input_tensor)
                else:
                    processed_tensor = stream.stream(input_tensor)
            # txt2img mode
            elif mode == "txt2img":
                if TIME_DEBUG:
                    timing_stats['input_processing'].append(0.0)  # No input processing for txt2img
                    image_generation_start = time.time() #timer stats
                
                # OPTIMIZED TXT2IMG: Use enhanced function with memory pools support
                # PAUSE MODE: Bypass memory pool optimizations for immediate processing
                if paused:
                    # Direct processing in pause mode - no memory pool buffering
                    processed_np = custom_txt2img_using_prepared_noise(
                        stream_diffusion=stream.stream, 
                        expected_batch_size=1, 
                        output_type='np', 
                        control_image=control_input_tensor if use_controlnet else None
                    )
                    # print(f"\033[36mPAUSE MODE: Direct txt2img generation (no memory pools)\033[0m")
                else:
                    # Normal mode: Use optimized memory pools for better performance
                    if 'memory_pools' in locals():
                        processed_np = optimized_txt2img_using_prepared_noise(
                            stream_diffusion=stream.stream, 
                            expected_batch_size=1, 
                            control_image=control_input_tensor if use_controlnet else None,
                            memory_pools=memory_pools,
                            paused=False
                        )
                    else:
                        # Fallback if memory pools are unavailable (CPU-only path)
                        processed_np = optimized_txt2img_using_prepared_noise(
                            stream_diffusion=stream.stream, 
                            expected_batch_size=1, 
                            control_image=control_input_tensor if use_controlnet else None,
                            memory_pools=None,
                            paused=True  # Force pause mode for fallback
                        )

            if TIME_DEBUG:
                timing_stats['image_generation'].append(time.time() - image_generation_start)
                postprocessing_start = time.time()

            if mode == "img2img":
                # Use unified postprocessing function that handles both pause and normal modes
                if 'memory_pools' in locals():
                    processed_np = optimized_postprocess_image(
                        processed_tensor, 
                        memory_pools=memory_pools, 
                        output_type="np",
                        paused=paused
                    )
                else:
                    # Fallback if memory pools are unavailable (CPU-only path)
                    processed_np = optimized_postprocess_image(
                        processed_tensor, 
                        memory_pools=None, 
                        output_type="np",
                        paused=True  # Force pause mode for fallback
                    )
                
                # if paused:
                    # print(f"\033[36mPAUSE MODE: Direct postprocessing (no memory pools)\033[0m")
            elif mode == "txt2img":
                # Check if we need postprocessing (for fallback case)
                if 'memory_pools' not in locals():
                    # Only apply postprocessing for fallback case
                    if processed_np.max() <= 1.0:
                        processed_np = (processed_np * 255).astype(np.uint8)
                # For optimized case, processed_np is already ready
                
            # Update timing stats - for optimized txt2img, postprocessing is included in generation
            if TIME_DEBUG:
                if mode == "txt2img" and 'memory_pools' in locals():
                    timing_stats['postprocessing'].append(0.0)  # Already included in generation time
                else:
                    timing_stats['postprocessing'].append(time.time() - postprocessing_start)

            # Output frame via NDI (add alpha channel for the sender)
            if processed_np.shape[2] == 3:
                alpha = np.full((processed_np.shape[0], processed_np.shape[1], 1), 255, dtype=np.uint8)
                processed_np = np.concatenate([processed_np, alpha], axis=2)
            ndi_handler.transmit_frame(processed_np)

            if TIME_DEBUG:
                osc_messages_start = time.time() #timer stats
            
            # Batch OSC messages to reduce communication overhead
            current_frame = frame_count
            osc_batch = [('/frames', current_frame)]
            
            # Batch the FPS calculation and add to batch
            active_input_name = ndi_input_name
            start_time, transmit_count = calculate_fps_and_send_osc(start_time, transmit_count, osc_transmit_port, output_mem_name, True, use_controlnet, active_input_name)
            
            # Send all OSC messages in one batch operation
            if len(osc_batch) > 1:
                batch_osc_operations(osc_batch, osc_transmit_port)
            else:
                send_osc_message('/frames', current_frame, osc_transmit_port)
                
            if TIME_DEBUG:
                timing_stats['osc_messages'].append(time.time() - osc_messages_start) #timer stats
            frame_count += 1
            
            # ADVANCED PERFORMANCE MAINTENANCE: Periodic optimizations to prevent fps degradation
            if device != "cpu":
                # Performance tracking for adaptive optimization (removed problematic memory clearing)
                if frame_count > 10:  # Allow warmup period
                    current_loop_time = time.time() - loop_start_time
                    if current_loop_time > 0:
                        current_fps = 1.0 / current_loop_time
                        peak_fps_tracker.append(current_fps)
                        
                        # Gentle memory management only when performance severely degrades
                        if len(peak_fps_tracker) >= 10:
                            recent_avg_fps = sum(peak_fps_tracker) / len(peak_fps_tracker)
                            if performance_baseline_fps == 0:
                                performance_baseline_fps = recent_avg_fps
                            elif recent_avg_fps < performance_baseline_fps * 0.8:  # Only at 20% degradation
                                # Very gentle cleanup - only empty cache, don't touch active buffers
                                torch.cuda.empty_cache()
                                performance_baseline_fps = recent_avg_fps  # Reset baseline
            
            ##STOP STREAM
            if shared_data.get("stop_stream", False):
                print("\r\n")
                print("Stopping image generation process.")
                break
            # print(f"frame_count: {frame_count}")
            try:
                if TIME_DEBUG:
                    loop_end_time = time.time()
                    loop_time = loop_end_time - loop_start_time
                    loop_times.append(loop_time)
                    if frame_count % 100 == 0:
                        print("\nDetailed timing (average over last 100 frames):")
                        avg_loop_time = sum(loop_times) / len(loop_times)
                        actual_fps = 1 / avg_loop_time if avg_loop_time > 0 else 0
                        print(f"Actual FPS (based on total loop time): {actual_fps:.2f}")
                        
                        total_time = sum(timing_stats['image_generation'])
                        approx_fps = 100 / total_time if total_time > 0 else 0
                        print(f"Approximate FPS (based on image generation): {approx_fps:.2f}")
                        
                        for key, times in timing_stats.items():
                            avg_time = sum(times) / len(times)
                            print(f"{key}: {avg_time:.4f} seconds")
                        
                        unaccounted_time = avg_loop_time - sum(sum(times) / len(times) for times in timing_stats.values())
                        print(f"Unaccounted time: {unaccounted_time:.4f} seconds")
                        
                        timing_stats = {key: [] for key in timing_stats}  # Reset timing stats
                        loop_times = []  # Reset loop times
            except Exception as e:
                pass
            else:
                max_fps = shared_data.get("max_fps", 80)
                if max_fps <= 0.5:
                    max_fps = 0.5
                frame_time = time.time() - loop_start_time
                sleep_time = max(1.0 / max_fps - frame_time, 0.0001)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\r", end='')
            break

    # profiler.disable()  
    # stats = pstats.Stats(profiler).sort_stats('cumulative')
    # Construct absolute path for the output file
    # current_script_dir_for_profile = os.path.dirname(os.path.abspath(__file__))
    # profile_output_path = os.path.join(current_script_dir_for_profile, 'profile_output.pstats')
    # stats.dump_stats(profile_output_path)  # Save stats to a file
    # print(f"\nProfiling data saved to {profile_output_path}")
    # print(f'You can analyze it using: python -m pstats "{profile_output_path}"')


def optimized_postprocess_image(
    image: torch.Tensor,
    memory_pools: dict = None,
    output_type: str = "np",
    do_denormalize: bool = True,
    paused: bool = False
) -> Union[torch.Tensor, np.ndarray]:
    """
    Uses memory pools for optimal performance in normal mode.
    In pause mode, bypasses memory pools for immediate processing but maintains proper data types.
    """
    if not isinstance(image, torch.Tensor):
        raise ValueError(f"Input for postprocessing is in incorrect format: {type(image)}. We only support pytorch tensor")

    if output_type == "latent":
        return image

    # PAUSE MODE: Direct processing without memory pools
    if paused or memory_pools is None:
        # Check device type and use appropriate autocast
        if image.device.type == 'cuda':
            autocast_context = torch.cuda.amp.autocast(enabled=True)
        else:
            autocast_context = nullcontext()

        with autocast_context:
            if do_denormalize:
                image = (image / 2 + 0.5).clamp(0, 1)

            if output_type == "pt":
                return image

            if output_type == "np":
                image = image.mul(255).byte()  # Convert to uint8
                # Move to CPU if on CUDA
                if image.device.type == 'cuda':
                    image = image.cpu()
                return image.permute(0, 2, 3, 1).numpy()

        raise ValueError(f"Unsupported output type: {output_type}")

    # NORMAL MODE: Use memory pools for maximum performance
    else:
        # Use pre-allocated processing buffer
        processing_buffer = memory_pools['output_processing_buffer']
        processing_buffer.copy_(image.float())

        if do_denormalize:
            # In-place denormalization: (image / 2 + 0.5).clamp(0, 1)
            processing_buffer.div_(2).add_(0.5).clamp_(0, 1)

        if output_type == "pt":
            return processing_buffer

        if output_type == "np":
            # Convert to uint8 and move to correct shape using memory pools
            uint8_buffer = memory_pools['output_tensor_buffer']
            uint8_buffer.copy_(processing_buffer.mul(255).byte())
            
            # Handle batch dimension and convert to HWC format
            if uint8_buffer.shape[0] > 1:
                image_chw = uint8_buffer[0]  # (3, H, W)
            else:
                image_chw = uint8_buffer.squeeze(0)  # (3, H, W)
            
            # Use conversion buffer for dimension reordering
            hwc_buffer = memory_pools['hwc_conversion_buffer']
            hwc_buffer.copy_(image_chw.permute(1, 2, 0))
            
            # Transfer to CPU using pinned memory buffer
            output_buffer = memory_pools['output_numpy_buffer']
            output_buffer.copy_(hwc_buffer, non_blocking=True)
            
            return output_buffer.numpy()

        raise ValueError(f"Unsupported output type: {output_type}")


def create_optimized_memory_pools(height: int, width: int, device: torch.device):
    """
    MEMORY POOL OPTIMIZATION: Pre-allocate all buffers to eliminate runtime allocations.
    This dramatically reduces memory fragmentation and allocation overhead.
    """
    pools = {}
    
    # GPU buffers (no pinning for GPU tensors)
    if device.type == 'cuda':
        pools['output_tensor_buffer'] = torch.empty((1, 3, height, width), dtype=torch.uint8, device=device)
        pools['output_processing_buffer'] = torch.empty((1, 3, height, width), dtype=torch.float32, device=device)
        pools['hwc_conversion_buffer'] = torch.empty((height, width, 3), dtype=torch.uint8, device=device)
        
        # CPU staging buffers with pinned memory for fast transfers
        pools['output_numpy_buffer'] = torch.empty((height, width, 3), dtype=torch.uint8, device='cpu', pin_memory=True)
        pools['input_staging_buffer'] = torch.empty((height, width, 3), dtype=torch.uint8, device='cpu', pin_memory=True)
        
        # Stream processing buffers
        pools['stream_sync_event'] = torch.cuda.Event()
    else:
        # CPU-only fallback (no pinning needed)
        pools['output_tensor_buffer'] = torch.empty((1, 3, height, width), dtype=torch.uint8, device=device)
        pools['output_processing_buffer'] = torch.empty((1, 3, height, width), dtype=torch.float32, device=device)
        pools['output_numpy_buffer'] = torch.empty((height, width, 3), dtype=torch.uint8, device='cpu')
    
    # Shared memory optimization buffers (numpy arrays for compatibility)
    pools['shared_mem_buffer'] = np.empty((height, width, 3), dtype=np.uint8)
    
    return pools


def optimized_tensor_to_numpy(tensor: torch.Tensor, numpy_buffer: np.ndarray):
    """
    tensor to numpy conversion using pre-allocated buffers.
    Eliminates all intermediate allocations and memory copies.
    """
    # Use direct memory copy for maximum efficiency
    if tensor.is_contiguous():
        np.copyto(numpy_buffer, tensor.detach().numpy(), casting='no')
    else:
        np.copyto(numpy_buffer, tensor.detach().contiguous().numpy(), casting='no')


def batch_osc_operations(messages: list, osc_out_port: int):
    """
    BATCH OSC OPTIMIZATION: Reduce OSC overhead by batching multiple messages.
    This reduces the number of individual OSC send operations.
    """
    if len(messages) <= 1:
        # Fallback to individual sends for single messages
        for address, value in messages:
            send_osc_message(address, value, osc_out_port)
    else:
        # Batch multiple messages together (if OSC client supports it)
        try:
            client = OSCClientFactory.get_client(osc_out_port)
            # Send all messages in rapid succession to reduce overhead
            for address, value in messages:
                client.send_message(address, value)
        except:
            # Fallback to individual sends
            for address, value in messages:
                send_osc_message(address, value, osc_out_port)

def update_lora_weights(stream_diffusion, lora_dict, use_lcm_lora, lcm_lora_id, wrapper):
# def update_lora_weights(stream_diffusion, lora_dict, use_lcm_lora, lcm_lora_id):
    if lora_dict is not None:
        try:
            wrapper.update_lora_weights(lora_dict)
            if use_lcm_lora:
                if lcm_lora_id:
                    stream_diffusion.load_lcm_lora(lcm_lora_id)
                else:
                    stream_diffusion.load_lcm_lora()
        except Exception as e:
            print(f'\033[91mCaution: Error in update_lora_weights: {e}\033[0m')

def update_attn_processors(stream_diffusion, new_use_cached_attn_settings, disable_cached_attn, device):
    if disable_cached_attn:
        if stream_diffusion.original_processors is not None:
            for key, processor in stream_diffusion.stream.pipe.unet.attn_processors.items():
                if isinstance(processor, (CachedSTXFormersAttnProcessor, CachedSTAttnProcessor2_0)):
                    processor.clear_cache()
            stream_diffusion.stream.pipe.unet.set_attn_processor(stream_diffusion.original_processors.copy())
            # Only run CUDA-specific commands if we're on CUDA
            if device == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            return
    attn_processors = stream_diffusion.pipe.unet.attn_processors
    new_attn_processors = {}
    try:
        for key, attn_processor in attn_processors.items():
            if isinstance(attn_processor, CachedSTXFormersAttnProcessor):
                attn_processor.use_feature_injection = new_use_cached_attn_settings.get("use_feature_injection", True)
                attn_processor.fi_strength = new_use_cached_attn_settings.get("feature_injection_strength", 0.5)
                attn_processor.threshold = new_use_cached_attn_settings.get("feature_similarity_threshold", 0.9)
                attn_processor.interval = new_use_cached_attn_settings.get("cache_interval", 2)
                new_max_frames = new_use_cached_attn_settings.get("cache_maxframes", 1)
                if new_max_frames != attn_processor.cached_key.maxlen:
                    attn_processor.cached_key = deque(attn_processor.cached_key, maxlen=new_max_frames)
                    attn_processor.cached_value = deque(attn_processor.cached_value, maxlen=new_max_frames)
                    attn_processor.cached_output = deque(attn_processor.cached_output, maxlen=new_max_frames)
                attn_processor.use_tome_cache = new_use_cached_attn_settings.get("use_tome_cache", True)
                attn_processor.tome_metric = new_use_cached_attn_settings.get("tome_metric", "keys")
                attn_processor.use_grid = new_use_cached_attn_settings.get("use_grid", False)
                attn_processor.tome_ratio = 0.5 if attn_processor.use_grid else new_use_cached_attn_settings.get("tome_ratio", 0.5)
                new_attn_processors[key] = attn_processor
            else:
                new_attn_processors[key] = CachedSTXFormersAttnProcessor(
                    name=key,
                    use_feature_injection=new_use_cached_attn_settings.get("use_feature_injection", True),
                    feature_injection_strength=new_use_cached_attn_settings.get("feature_injection_strength", 0.5),
                    feature_similarity_threshold=new_use_cached_attn_settings.get("feature_similarity_threshold", 0.9),
                    interval=new_use_cached_attn_settings.get("cache_interval", 2),
                    max_frames=new_use_cached_attn_settings.get("cache_maxframes", 1),
                    use_tome_cache=new_use_cached_attn_settings.get("use_tome_cache", True),
                    tome_metric=new_use_cached_attn_settings.get("tome_metric", "keys"),
                    tome_ratio=0.5 if new_use_cached_attn_settings.get("use_grid", False) else new_use_cached_attn_settings.get("tome_ratio", 0.5),
                    use_grid=new_use_cached_attn_settings.get("use_grid", False),
                )
        stream_diffusion.pipe.unet.set_attn_processor(new_attn_processors)
    except Exception as e:
        return


def update_controlnet_conditioning_scale(stream_diffusion, shared_data):
    new_controlnet_conditioning_scale = shared_data.get("controlnet_conditioning_scale", stream_diffusion.controlnet_conditioning_scale)
    if new_controlnet_conditioning_scale != stream_diffusion.controlnet_conditioning_scale:
        stream_diffusion.controlnet_conditioning_scale = new_controlnet_conditioning_scale

def update_t_list_attributes(stream_diffusion_instance, new_t_list):
    stream_diffusion_instance.t_list = new_t_list
    stream_diffusion_instance.sub_timesteps = [stream_diffusion_instance.timesteps[t] for t in new_t_list]
    sub_timesteps_tensor = torch.tensor(
        stream_diffusion_instance.sub_timesteps, dtype=torch.long, device=stream_diffusion_instance.device
    )
    stream_diffusion_instance.sub_timesteps_tensor = torch.repeat_interleave(
        sub_timesteps_tensor, 
        repeats=stream_diffusion_instance.frame_bff_size if stream_diffusion_instance.use_denoising_batch else 1, 
        dim=0
    )
    c_skip_list = []
    c_out_list = []
    for timestep in stream_diffusion_instance.sub_timesteps:
        c_skip, c_out = stream_diffusion_instance.scheduler.get_scalings_for_boundary_condition_discrete(timestep)
        c_skip_list.append(c_skip)
        c_out_list.append(c_out)
    stream_diffusion_instance.c_skip = torch.stack(c_skip_list).view(len(new_t_list), 1, 1, 1).to(dtype=stream_diffusion_instance.dtype, device=stream_diffusion_instance.device)
    stream_diffusion_instance.c_out = torch.stack(c_out_list).view(len(new_t_list), 1, 1, 1).to(dtype=stream_diffusion_instance.dtype, device=stream_diffusion_instance.device)
    # Calculate alpha_prod_t_sqrt and beta_prod_t_sqrt
    alpha_prod_t_sqrt_list = []
    beta_prod_t_sqrt_list = []
    for timestep in stream_diffusion_instance.sub_timesteps:
        alpha_prod_t_sqrt = stream_diffusion_instance.scheduler.alphas_cumprod[timestep].sqrt()
        beta_prod_t_sqrt = (1 - stream_diffusion_instance.scheduler.alphas_cumprod[timestep]).sqrt()
        alpha_prod_t_sqrt_list.append(alpha_prod_t_sqrt)
        beta_prod_t_sqrt_list.append(beta_prod_t_sqrt)
    alpha_prod_t_sqrt = torch.stack(alpha_prod_t_sqrt_list).view(len(new_t_list), 1, 1, 1).to(dtype=stream_diffusion_instance.dtype, device=stream_diffusion_instance.device)
    beta_prod_t_sqrt = torch.stack(beta_prod_t_sqrt_list).view(len(new_t_list), 1, 1, 1).to(dtype=stream_diffusion_instance.dtype, device=stream_diffusion_instance.device)
    stream_diffusion_instance.alpha_prod_t_sqrt = torch.repeat_interleave(alpha_prod_t_sqrt, repeats=stream_diffusion_instance.frame_bff_size if stream_diffusion_instance.use_denoising_batch else 1, dim=0)
    stream_diffusion_instance.beta_prod_t_sqrt = torch.repeat_interleave(beta_prod_t_sqrt, repeats=stream_diffusion_instance.frame_bff_size if stream_diffusion_instance.use_denoising_batch else 1, dim=0)

def slerp(t, v0, v1, dot_threshold=0.9995):
    dot = torch.sum(v0 * v1 / (torch.norm(v0) * torch.norm(v1)))
    if abs(dot) > dot_threshold:
        result = v0 + t * (v1 - v0)
        return result / torch.norm(result)
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta_t = theta_0 * t
    sin_theta_t = torch.sin(theta_t)
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    return s0 * v0 + s1 * v1

def multi_slerp(embeddings, weights):

    total_weight = sum(weights)
    scale_factor = max(1.0, total_weight)
    if len(embeddings) == 1:
        return embeddings[0] * scale_factor
    scaled_weights = [w / scale_factor for w in weights]
    sorted_pairs = sorted(zip(embeddings, scaled_weights), key=lambda x: x[1], reverse=True)
    sorted_embeddings, sorted_weights = zip(*sorted_pairs)
    result = sorted_embeddings[0]
    accumulated_weight = sorted_weights[0]
    for i in range(1, len(sorted_embeddings)):
        if sorted_weights[i] == 0:
            continue
        t = sorted_weights[i] / (accumulated_weight + sorted_weights[i])
        result = slerp(t, result, sorted_embeddings[i])
        accumulated_weight += sorted_weights[i]
    return result * scale_factor

def cosine_similarity_weighted_average(embeddings, weights):
    embeddings_tensor = torch.stack(embeddings)
    weights_tensor = torch.tensor(weights, device=embeddings_tensor.device, dtype=embeddings_tensor.dtype)
    result = multi_slerp(embeddings, weights)
    return result

def interpolate_embeddings(embeddings, weights, method='slerp', sdxl=False, **kwargs):
    if sdxl:
        return sum(embed * weight for embed, weight in zip(embeddings, weights))
    if method == 'slerp':
        return multi_slerp(embeddings, weights)
    elif method == 'cosine_weighted_interpolation':
        return cosine_similarity_weighted_average(embeddings, weights)
    else:
        return sum(embed * weight for embed, weight in zip(embeddings, weights))

@torch.no_grad()
def update_combined_prompts_and_parameters(stream_diffusion, prompt_list, new_guidance_scale, new_delta, new_negative_prompt, prompt_cache, interpolation_method, shared_data):
    stream_diffusion.guidance_scale = new_guidance_scale
    stream_diffusion.delta = new_delta
    stream_diffusion.stock_noise *= stream_diffusion.delta
    sdxl = stream_diffusion.sdxl
    current_prompts = set()
    embeddings = []
    weights = []
    
    # Determine if we need CFG based on guidance scale and cfg_type
    do_classifier_free_guidance = (stream_diffusion.guidance_scale > 1.0 and 
                                   stream_diffusion.cfg_type != "none")
    
    for idx, (prompt_text, weight) in enumerate(prompt_list):
        if weight == 0:
            continue
        current_prompts.add(idx)
        
        # Create cache key that includes negative prompt for CFG scenarios
        cache_key = f"{idx}_{hash(prompt_text)}_{hash(new_negative_prompt) if do_classifier_free_guidance else 'no_neg'}"
        
        if idx not in prompt_cache or prompt_cache[idx].get('cache_key') != cache_key:
            encoder_output = stream_diffusion.pipe.encode_prompt(
                prompt=prompt_text,
                device=stream_diffusion.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=do_classifier_free_guidance,
                negative_prompt=new_negative_prompt if do_classifier_free_guidance else "",
            )
            
            # Store both positive and negative embeddings if CFG is used
            prompt_cache[idx] = {
                'pos_embed': encoder_output[0], 
                'neg_embed': encoder_output[1] if do_classifier_free_guidance and len(encoder_output) > 1 else None,
                'text': prompt_text,
                'cache_key': cache_key
            }
        
        embeddings.append(prompt_cache[idx]['pos_embed'])
        weights.append(weight)
    
    Interpval1 = shared_data.get("Interpval1", 0.5)
    
    if embeddings and weights:
        embeddings, weights = zip(*sorted(zip(embeddings, weights), key=lambda x: x[1], reverse=True))
        if embeddings:
            combined_pos_embeds = interpolate_embeddings(embeddings, weights, method=interpolation_method, val1=Interpval1, sdxl=sdxl, shared_data=shared_data)
            
            # Handle CFG embedding structure based on cfg_type
            if do_classifier_free_guidance:
                if stream_diffusion.cfg_type == "full":
                    # For "full" CFG, we need negative embeddings for the entire batch
                    # Get negative embeddings from cache and combine them similarly
                    neg_embeddings = []
                    for idx, (prompt_text, weight) in enumerate(prompt_list):
                        if weight == 0:
                            continue
                        neg_embed = prompt_cache[idx]['neg_embed']
                        if neg_embed is not None:
                            neg_embeddings.append(neg_embed)
                    
                    if neg_embeddings:
                        combined_neg_embeds = interpolate_embeddings(neg_embeddings, [w for _, w in prompt_list if w > 0], method=interpolation_method, val1=Interpval1, sdxl=sdxl, shared_data=shared_data)
                        # Repeat embeddings for batch size
                        pos_embeds_repeated = combined_pos_embeds.repeat(stream_diffusion.batch_size, 1, 1)
                        neg_embeds_repeated = combined_neg_embeds.repeat(stream_diffusion.batch_size, 1, 1)
                        # Concatenate negative and positive embeddings for full CFG
                        stream_diffusion.prompt_embeds = torch.cat([neg_embeds_repeated, pos_embeds_repeated], dim=0)
                    else:
                        # Fallback if no negative embeddings available
                        stream_diffusion.prompt_embeds = combined_pos_embeds.repeat(stream_diffusion.batch_size, 1, 1)
                        
                elif stream_diffusion.cfg_type == "initialize":
                    # For "initialize" CFG, we need negative embeddings for frame buffer size
                    neg_embeddings = []
                    for idx, (prompt_text, weight) in enumerate(prompt_list):
                        if weight == 0:
                            continue
                        neg_embed = prompt_cache[idx]['neg_embed']
                        if neg_embed is not None:
                            neg_embeddings.append(neg_embed)
                    
                    if neg_embeddings:
                        combined_neg_embeds = interpolate_embeddings(neg_embeddings, [w for _, w in prompt_list if w > 0], method=interpolation_method, val1=Interpval1, sdxl=sdxl, shared_data=shared_data)
                        # Repeat embeddings appropriately
                        pos_embeds_repeated = combined_pos_embeds.repeat(stream_diffusion.batch_size, 1, 1)
                        neg_embeds_repeated = combined_neg_embeds.repeat(stream_diffusion.frame_bff_size, 1, 1)
                        # Concatenate negative and positive embeddings for initialize CFG
                        stream_diffusion.prompt_embeds = torch.cat([neg_embeds_repeated, pos_embeds_repeated], dim=0)
                    else:
                        # Fallback if no negative embeddings available
                        stream_diffusion.prompt_embeds = combined_pos_embeds.repeat(stream_diffusion.batch_size, 1, 1)
                        
                elif stream_diffusion.cfg_type == "self":
                    # For "self" CFG, we only need positive embeddings (negative guidance comes from stock_noise)
                    stream_diffusion.prompt_embeds = combined_pos_embeds.repeat(stream_diffusion.batch_size, 1, 1)
                else:
                    # Default case
                    stream_diffusion.prompt_embeds = combined_pos_embeds.repeat(stream_diffusion.batch_size, 1, 1)
            else:
                # No CFG, just use positive embeddings
                stream_diffusion.prompt_embeds = combined_pos_embeds.repeat(stream_diffusion.batch_size, 1, 1)
    
    # Clean up unused prompts from cache
    unused_prompts = set(prompt_cache.keys()) - current_prompts
    for prompt in unused_prompts:
        del prompt_cache[prompt]

def blend_noise_tensors(seed_list, noise_bank, stream_diffusion):
    blended_noise = None
    total_weight = 0
    for seed_val, weight in seed_list:
        if weight == 0:
            continue
        noise_tensor = noise_bank.get(seed_val)
        if noise_tensor is None:
            generator = torch.Generator().manual_seed(seed_val)
            noise_tensor = torch.randn(
                (stream_diffusion.batch_size, 4, stream_diffusion.latent_height, stream_diffusion.latent_width),
                generator=generator
            ).to(device=stream_diffusion.device, dtype=stream_diffusion.dtype)
            noise_bank[seed_val] = noise_tensor
        if blended_noise is None:
            blended_noise = noise_tensor * weight
        else:
            blended_noise += noise_tensor * weight
        total_weight += weight
    return blended_noise

def optimized_txt2img_using_prepared_noise(stream_diffusion, expected_batch_size, control_image=None, memory_pools=None, paused=False):
    """
    Uses memory pools and optimized postprocessing for peak performance.
    Maintains full ControlNet support while eliminating memory allocations.
    """
    # Optimize noise tensor selection/adjustment
    if stream_diffusion.init_noise.size(0) > expected_batch_size:
        adjusted_noise = stream_diffusion.init_noise[:expected_batch_size]
    elif stream_diffusion.init_noise.size(0) < expected_batch_size:
        repeats = [expected_batch_size // stream_diffusion.init_noise.size(0)] + [-1] * (stream_diffusion.init_noise.dim() - 1)
        adjusted_noise = stream_diffusion.init_noise.repeat(*repeats)[:expected_batch_size]
    else:
        adjusted_noise = stream_diffusion.init_noise
    
    # Core inference with ControlNet support
    x_0_pred_out = stream_diffusion.predict_x0_batch(adjusted_noise, control_image=control_image)
    x_output = stream_diffusion.decode_image(x_0_pred_out).detach().clone()
    
    # Use unified postprocessing function with proper uint8 conversion
    processed_np = optimized_postprocess_image(
        x_output, 
        memory_pools=memory_pools, 
        output_type="np",
        paused=paused
    )
    
    return processed_np

def custom_txt2img_using_prepared_noise(stream_diffusion, expected_batch_size, output_type='np', control_image=None):
    if stream_diffusion.init_noise.size(0) > expected_batch_size:
        adjusted_noise = stream_diffusion.init_noise[:expected_batch_size]
    elif stream_diffusion.init_noise.size(0) < expected_batch_size:
        repeats = [expected_batch_size // stream_diffusion.init_noise.size(0)] + [-1] * (stream_diffusion.init_noise.dim() - 1)
        adjusted_noise = stream_diffusion.init_noise.repeat(*repeats)[:expected_batch_size]
    else:
        adjusted_noise = stream_diffusion.init_noise
    x_0_pred_out = stream_diffusion.predict_x0_batch(adjusted_noise, control_image=control_image)
    x_output = stream_diffusion.decode_image(x_0_pred_out).detach().clone()
    if output_type == 'np':
        # Use unified postprocessing function with proper uint8 conversion
        x_output = optimized_postprocess_image(x_output, memory_pools=None, output_type=output_type, paused=True)
    return x_output

def safe_basename(path):
    return os.path.basename(path) if path else "Not specified"

def print_info(label, value=None, weight=None, start_phrase=None, color="green"):
    color_codes = {
        "black": "\033[30m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "default": "\033[39m"
    }
    color_code = color_codes.get(color, "\033[32m")  # Default to green if color not found
    if start_phrase:
        print(start_phrase)
    if value and value != "None":
        base_name = safe_basename(value)
        weight_info = f", Weight: {weight}" if weight is not None else ""
        print(f"{label} {base_name}{weight_info}")
        print(f"{color_code}{value}\033[0m\n")

def read_config(config_file_path):
    """Read the minimal launch config and fill in server/engine defaults.

    The file only needs the deployment-specific values (model, prompt,
    resolution, NDI names, ...). Everything else falls back to the defaults
    below, which match the proven headless setup. ``osc_port``/``osc_reply_port``
    default to the TouchDesigner ports 7187/6503.
    """
    with open(config_file_path, 'r') as config_file:
        config = json.load(config_file)

    # CUDA is the only supported acceleration path (NVIDIA box); fall back to CPU
    # only if no GPU is present.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    acceleration = "none"

    model = config.get("model")
    if model and os.path.isfile(model):
        model = model.replace('/', '\\')

    config_dict = {
        # Deployment values (from the file)
        "model_id_or_path": model,
        "prompt": config.get("prompt", ""),
        "negative_prompt": config.get("negative", ""),
        "width": config.get("width", 512),
        "height": config.get("height", 512),
        "seed": config.get("seed", -1),
        "t_index_list": config.get("steps", [17]),
        "guidance_scale": config.get("guidance", 1.0),
        "delta": config.get("delta", 1.0),
        "max_fps": config.get("maxfps", 60),
        "ndi_sender_name": config.get("output"),
        "ndi_input_name": config.get("input"),
        # OSC (defaults keep the TouchDesigner contract)
        "osc_receive_port": config.get("osc_port", 7187),
        "osc_transmit_port": config.get("osc_reply_port", 6503),
        "osc_bind_ip": config.get("osc_bind_ip", "0.0.0.0"),
        "osc_send_ip": config.get("osc_send_ip", "auto"),
        "ndi_control_name": config.get("ndi_control_name"),
        "ndi_extra_ips": config.get("ndi_extra_ips"),
        # Engine defaults (frozen, no longer in the file)
        "lora_dict": None,
        "frame_buffer_size": 1,
        "acceleration": acceleration,
        "use_denoising_batch": True,
        "cfg_type": "self",
        "do_add_noise": True,
        "enable_similar_image_filter": False,
        "similar_image_filter_threshold": 0.98,
        "similar_image_filter_max_skip_frame": 10,
        "mode": "img2img",
        "lcm_lora_id": "skip",
        "vae_id": None,
        "scheduler_name": "Euler",
        "use_karras_sigmas": False,
        "device": device,
        "gpu_id": config.get("gpu_id", 0),
        "use_controlnet": False,
        "controlnet_model": None,
        "controlnet_weight": 0.5,
        "use_cached_attn": False,
        "sd_model_type": None,
        "hf_cache": None,
        "textual_inversion_dict": None,
    }
    return config_dict

def print_config_info(config, config_file_path):
    time.sleep(0.04)
    print(f"\033[36mLoading {config_file_path}\033[0m")
    print_info("Main SD Model:", config['model_id_or_path'], start_phrase="\033[36m=======================================\033[0m")
    time.sleep(0.04)
    print("\033[36mPrompt:\033[0m", config['prompt'] or "(not set)")
    print("\033[36mNegative:\033[0m", config['negative_prompt'] or "(none)")
    print("\033[36mResolution:\033[0m", f"{config['width']} x {config['height']}")
    print("\033[36mSeed:\033[0m", config['seed'], "| Timesteps:", config['t_index_list'])
    print("\033[36mGuidance / Delta:\033[0m", f"{config['guidance_scale']} / {config['delta']}",
          "| Max FPS:", config['max_fps'])
    print("\033[36mNDI out:\033[0m", config['ndi_sender_name'], "| NDI in:", config['ndi_input_name'])
    print("\033[36mOSC listen:\033[0m", f"{config['osc_bind_ip']}:{config['osc_receive_port']}",
          "| status ->", f"{config['osc_send_ip']}:{config['osc_transmit_port']}")
    print("\033[36m=======================================\033[0m\n")

def main():
    child_processes = []

    def signal_handler(sig, frame):
        print("\n\033[31mSignal received, terminating subprocesses...\033[0m")
        terminate_processes(child_processes)
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    parser = argparse.ArgumentParser(description="StreamDiffusion server")
    parser.add_argument('-c', '--config', type=str, default='../configs/config.json', help='Path to the configuration file')
    args = parser.parse_args()
    
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file_path = os.path.join(current_script_dir, args.config)
    
    with Manager() as manager:
        print_banner()
        shared_data = manager.dict()
        
        config = read_config(config_file_path)
        shared_data.update(config)
        device = config['device']
        print("\033[33mOperating System:\033[0m \033[32mLinux\033[0m")
        print(f"\033[33mUsing device:\033[0m {device}")
        if device == "cuda":
            print_gpu_info(gpu_id=config["gpu_id"])
        print()
        print_config_info(config, config_file_path)
        
        osc_process = Process(target=osc_server, args=(shared_data, config['osc_bind_ip'], config['osc_receive_port'], config['osc_transmit_port'], config['osc_send_ip']))
        child_processes.append(osc_process)
        osc_process.start()
        time.sleep(.3)
        
        generation_process = None

        def start_generation_process():
            nonlocal generation_process, config
            if generation_process is None or not generation_process.is_alive():
                new_config = read_config(config_file_path)
                if config != new_config:
                    config = new_config
                    print_config_info(config, config_file_path)
                    shared_data.update(config)
                generation_process = Process(
                    target=image_generation_process,
                    args=(
                        config['model_id_or_path'],
                        config['lora_dict'],
                        config['prompt'],
                        config['negative_prompt'],
                        config['frame_buffer_size'],
                        config['width'],
                        config['height'],
                        config['acceleration'],
                        config['use_denoising_batch'],
                        config['seed'],
                        config['cfg_type'],
                        config['guidance_scale'],
                        config['delta'],
                        config['do_add_noise'],
                        config['enable_similar_image_filter'],
                        config['similar_image_filter_threshold'],
                        config['similar_image_filter_max_skip_frame'],
                        shared_data,
                        config['t_index_list'],
                        config['mode'],
                        config['lcm_lora_id'],
                        config['vae_id'],
                        config['osc_transmit_port'],
                        config['scheduler_name'],
                        config['use_karras_sigmas'],
                        config['device'],
                        config['gpu_id'],
                        config['use_controlnet'],
                        config['controlnet_model'],
                        config['controlnet_weight'],
                        config['use_cached_attn'],
                        config['sd_model_type'],
                        config['max_fps'],
                        config['textual_inversion_dict']
                    ),
                )
                child_processes.append(generation_process)
                generation_process.start()
        
        try:
            start_generation_process()
            while True:
                if shared_data.get("stop_stream", False):
                    print("\n\033[31mStop command received. Initiating shutdown...\033[0m")
                    break
                elif shared_data.get("pause_stream", False):
                    print("\n\033[33;1mPause command received. Pausing generation...\033[0m")
                    shared_data["paused"] = True
                    shared_data["pause_stream"] = False
                elif shared_data.get("play_stream", False):
                    if generation_process is None or not generation_process.is_alive():
                        print("\n\033[32mPlay command received. Starting generation process...\033[0m\n")
                        start_generation_process()
                    else:
                        print("\n\033[92mPlay command received. Resuming generation...\033[0m\n")
                    shared_data["paused"] = False
                    shared_data["play_stream"] = False
                elif shared_data.get("unload_stream", False):
                    if generation_process:
                        print("\n\033[35mUnload command received. Terminating generation process...\033[0m\n")
                        generation_process.terminate()
                        generation_process.join()
                        generation_process = None
                        print_gpu_info(gpu_id=config['gpu_id'])
                    else:
                        print("\n\033[35mUnload command received. No generation process to terminate.\033[0m\n")
                        print_gpu_info(gpu_id=config['gpu_id'])
                    shared_data["unload_stream"] = False
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("\n\033[31mKeyboardInterrupt received, signalling to stop subprocesses...\033[0m")
        finally:
            terminate_processes(child_processes)
            print("\n\033[36mAll subprocesses terminated. Exiting main process...\033[0m")
            sys.exit(0)

if __name__ == "__main__":
    fire.Fire(main)