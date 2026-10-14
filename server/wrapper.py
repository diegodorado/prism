# wrapper.py
import gc
import os
from pathlib import Path
import traceback
from typing import List, Literal, Optional, Union, Dict
import numpy as np
import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionControlNetImg2ImgPipeline, StableDiffusionXLControlNetImg2ImgPipeline, ControlNetModel, StableDiffusionXLControlNetPipeline
# from diffusers.loaders.ip_adapter import load_ip_adapter
from PIL import Image
import time
import re
import huggingface_hub
from diffusers.models.attention_processor import XFormersAttnProcessor, AttnProcessor2_0
from diffusers.utils import DIFFUSERS_CACHE

from io import BytesIO
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import HF_HOME
import requests

from streamdiffusion.image_utils import postprocess_image
from pipeline import StreamDiffusion
from attention import CachedSTXFormersAttnProcessor, CachedSTAttnProcessor2_0  #V2V

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class StreamDiffusionWrapper:
    def __init__(
        self,
        model_id_or_path: str,
        t_index_list: List[int],
        lora_dict: Optional[Dict[str, float]] = None,
        mode: Literal["img2img", "txt2img"] = "img2img",
        output_type: Literal["pil", "pt", "np", "latent"] = "pil",
        lcm_lora_id: Optional[str] = None,
        vae_id: Optional[str] = None,
        gpu_id: int = 0,  # New parameter to specify GPU ID
        device: Literal["cpu", "cuda"] = "cuda",
        dtype: torch.dtype = torch.float16,
        frame_buffer_size: int = 1,
        width: int = 512,
        height: int = 512,
        warmup: int = 10,
        acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt",
        do_add_noise: bool = True,
        device_ids: Optional[List[int]] = None,
        use_lcm_lora: bool = True,
        use_tiny_vae: bool = True,
        enable_similar_image_filter: bool = False,
        similar_image_filter_threshold: float = 0.98,
        similar_image_filter_max_skip_frame: int = 10,
        use_denoising_batch: bool = True,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
        seed: int = 2,
        use_safety_checker: bool = False,
        engine_dir: Optional[Union[str, Path]] = "engines",
        sdxl: bool = None,
        scheduler_name: str = "EulerAncestral",  # Default scheduler name
        use_karras_sigmas: bool = False,  # Default setting for Karras sigmas
        use_controlnet: bool = False,
        controlnet_model: Optional[str] = None,
        use_cached_attn: bool = True,  #V2V - Set default to True
        use_feature_injection: bool = True,  #V2V - Set default to True
        feature_injection_strength: float = 0.8,  #V2V
        feature_similarity_threshold: float = 0.98,  #V2V
        cache_interval: int = 4,  #V2V
        cache_maxframes: int = 1,  #V2V
        use_tome_cache: bool = True,  #V2V - Set default to True
        tome_metric: str = "keys",  #V2V
        tome_ratio: float = 0.5,  #V2V
        use_grid: bool = False,  #V2V
        sd_model_type: Optional[Literal["sd15", "sd21", "sdxl"]] = None,
        textual_inversion_dict: Optional[Dict[str, Optional[str]]] = None
    ):


        """
        Initializes the StreamDiffusionWrapper.

        Parameters
        ----------
        model_id_or_path : str
            The model id or path to load.
        t_index_list : List[int]
            The t_index_list to use for inference.
        lora_dict : Optional[Dict[str, float]], optional
            The lora_dict to load, by default None.
            Keys are the LoRA names and values are the LoRA scales.
            Example: {'LoRA_1' : 0.5 , 'LoRA_2' : 0.7 ,...}
        mode : Literal["img2img", "txt2img"], optional
            txt2img or img2img, by default "img2img".
        output_type : Literal["pil", "pt", "np", "latent"], optional
            The output type of image, by default "pil".
        lcm_lora_id : Optional[str], optional
            The lcm_lora_id to load, by default None.
            If None, the default LCM-LoRA
            ("latent-consistency/lcm-lora-sdv1-5") will be used.
        vae_id : Optional[str], optional
            The vae_id to load, by default None.
            If None, the default TinyVAE
            ("madebyollin/taesd") will be used.
        device : Literal["cpu", "cuda"], optional
            The device to use for inference, by default "cuda".
        dtype : torch.dtype, optional
            The dtype for inference, by default torch.float16.
        frame_buffer_size : int, optional
            The frame buffer size for denoising batch, by default 1.
        width : int, optional
            The width of the image, by default 512.
        height : int, optional
            The height of the image, by default 512.
        warmup : int, optional
            The number of warmup steps to perform, by default 10.
        acceleration : Literal["none", "xformers", "tensorrt"], optional
            The acceleration method, by default "tensorrt".
        do_add_noise : bool, optional
            Whether to add noise for following denoising steps or not,
            by default True.
        device_ids : Optional[List[int]], optional
            The device ids to use for DataParallel, by default None.
        use_lcm_lora : bool, optional
            Whether to use LCM-LoRA or not, by default True.
        use_tiny_vae : bool, optional
            Whether to use TinyVAE or not, by default True.
        enable_similar_image_filter : bool, optional
            Whether to enable similar image filter or not,
            by default False.
        similar_image_filter_threshold : float, optional
            The threshold for similar image filter, by default 0.98.
        similar_image_filter_max_skip_frame : int, optional
            The max skip frame for similar image filter, by default 10.
        use_denoising_batch : bool, optional
            Whether to use denoising batch or not, by default True.
        cfg_type : Literal["none", "full", "self", "initialize"],
        optional
            The cfg_type for img2img mode, by default "self".
            You cannot use anything other than "none" for txt2img mode.
        seed : int, optional
            The seed, by default 2.
        use_safety_checker : bool, optional
            Whether to use safety checker or not, by default False.
        """
        # Initialize sd_turbo and sdxl based on model_id_or_path if sd_model_type is not provided
        self.sd_turbo = "turbo" in model_id_or_path.lower() or "sdxs" in model_id_or_path.lower() or "lightning" in model_id_or_path.lower()
        self.sdxl = "xl" in model_id_or_path.lower()

        # Override sd_turbo and sdxl based on sd_model_type if provided
        if sd_model_type is not None:
            if sd_model_type == "sd15" or sd_model_type == "sd21":
                self.sd_turbo = False
                self.sdxl = False
            elif sd_model_type == "sdxl":
                self.sdxl = True
            elif sd_model_type == "sdxl-turbo":
                self.sd_turbo = True
                self.sdxl = True
        self.vae_model_id = "madebyollin/taesdxl" if self.sdxl else "madebyollin/taesd"
        if mode == "txt2img":
            if cfg_type != "none":
                raise ValueError(
                    f"txt2img mode accepts only cfg_type = 'none', but got {cfg_type}"
                )
            if use_denoising_batch and frame_buffer_size > 1:
                if not self.sd_turbo:
                    raise ValueError(
                        "txt2img mode cannot use denoising batch with frame_buffer_size > 1."
                    )

        if mode == "img2img":
            if not use_denoising_batch:
                raise NotImplementedError(
                    "img2img mode must use denoising batch for now."
                )

        if device == 'cuda':
            self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.dtype = dtype
        self.width = width
        self.height = height
        self.mode = mode
        self.output_type = output_type
        self.frame_buffer_size = frame_buffer_size
        self.batch_size = (
            len(t_index_list) * frame_buffer_size
            if use_denoising_batch
            else frame_buffer_size
        )

        self.use_denoising_batch = use_denoising_batch
        self.use_safety_checker = use_safety_checker

        self.use_cached_attn = use_cached_attn
        self.use_feature_injection = use_feature_injection
        self.feature_injection_strength = feature_injection_strength
        self.feature_similarity_threshold = feature_similarity_threshold
        self.cache_interval = cache_interval
        self.cache_maxframes = cache_maxframes
        self.use_tome_cache = use_tome_cache
        self.tome_metric = tome_metric
        self.tome_ratio = tome_ratio
        self.use_grid = use_grid
        self.use_cached_attn_settings = {
            "use_feature_injection": self.use_feature_injection,
            "feature_injection_strength": self.feature_injection_strength,
            "feature_similarity_threshold": self.feature_similarity_threshold,
            "cache_interval": self.cache_interval,
            "cache_maxframes": self.cache_maxframes,
            "use_tome_cache": self.use_tome_cache,
            "tome_metric": self.tome_metric,
            "tome_ratio": self.tome_ratio,
            "use_grid": self.use_grid,
        }
        self.loaded_loras = {}
        self.loaded_textual_inversions = {}

        self.stream: StreamDiffusion = self._load_model(
            model_id_or_path=model_id_or_path,
            lora_dict=lora_dict,
            lcm_lora_id=lcm_lora_id,
            vae_id=vae_id,
            t_index_list=t_index_list,
            acceleration=acceleration,
            warmup=warmup,
            do_add_noise=do_add_noise,
            use_lcm_lora=use_lcm_lora,
            use_tiny_vae=use_tiny_vae,
            cfg_type=cfg_type,
            seed=seed,
            engine_dir=engine_dir,
            scheduler_name=scheduler_name,
            use_karras_sigmas=use_karras_sigmas,
            use_controlnet=use_controlnet,
            controlnet_model=controlnet_model
        )

        if textual_inversion_dict:
            self.update_textual_inversions(textual_inversion_dict, print_info=True)


        if hasattr(self.stream.unet, 'config'):
            self.stream.unet.config.addition_embed_type = None

        if device_ids is not None:
            self.stream.unet = torch.nn.DataParallel(
                self.stream.unet, device_ids=device_ids
            )

        if enable_similar_image_filter:
            self.stream.enable_similar_image_filter(similar_image_filter_threshold, similar_image_filter_max_skip_frame)

    def prepare(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 1.2,
        delta: float = 1.0,
    ) -> None:
        """
        Prepares the model for inference.

        Parameters
        ----------
        prompt : str
            The prompt to generate images from.
        num_inference_steps : int, optional
            The number of inference steps to perform, by default 50.
        guidance_scale : float, optional
            The guidance scale to use, by default 1.2.
        delta : float, optional
            The delta multiplier of virtual residual noise,
            by default 1.0.
        """
        self.stream.prepare(
            prompt,
            negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            delta=delta,
        )

    def __call__(
        self,
        image: Optional[Union[str, Image.Image, torch.Tensor]] = None,
        prompt: Optional[str] = None,
    ) -> Union[Image.Image, List[Image.Image]]:
        """
        Performs img2img or txt2img based on the mode.

        Parameters
        ----------
        image : Optional[Union[str, Image.Image, torch.Tensor]]
            The image to generate from.
        prompt : Optional[str]
            The prompt to generate images from.

        Returns
        -------
        Union[Image.Image, List[Image.Image]]
            The generated image.
        """
        if self.mode == "img2img":
            return self.img2img(image, prompt)
        else:
            return self.txt2img(prompt)

    def txt2img(
        self, prompt: Optional[str] = None
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        """
        Performs txt2img.

        Parameters
        ----------
        prompt : Optional[str]
            The prompt to generate images from.

        Returns
        -------
        Union[Image.Image, List[Image.Image]]
            The generated image.
        """
        if prompt is not None:
            self.stream.update_prompt(prompt)

        if self.sd_turbo:
            image_tensor = self.stream.txt2img_sd_turbo(self.batch_size)
        else:
            image_tensor = self.stream.txt2img(self.frame_buffer_size)
        image = self.postprocess_image(image_tensor, output_type=self.output_type)

        if self.use_safety_checker:
            safety_checker_input = self.feature_extractor(
                image, return_tensors="pt"
            ).to(self.device)
            _, has_nsfw_concept = self.safety_checker(
                images=image_tensor.to(self.dtype),
                clip_input=safety_checker_input.pixel_values.to(self.dtype),
            )
            image = self.nsfw_fallback_img if has_nsfw_concept[0] else image

        return image

    def img2img(
        self, image: Union[str, Image.Image, torch.Tensor], prompt: Optional[str] = None
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        """
        Performs img2img.

        Parameters
        ----------
        image : Union[str, Image.Image, torch.Tensor]
            The image to generate from.

        Returns
        -------
        Image.Image
            The generated image.
        """
        if prompt is not None:
            self.stream.update_prompt(prompt)

        if isinstance(image, str) or isinstance(image, Image.Image):
            image = self.preprocess_image(image)

        image_tensor = self.stream(image)
        image = self.postprocess_image(image_tensor, output_type=self.output_type)

        if self.use_safety_checker:
            safety_checker_input = self.feature_extractor(
                image, return_tensors="pt"
            ).to(self.device)
            _, has_nsfw_concept = self.safety_checker(
                images=image_tensor.to(self.dtype),
                clip_input=safety_checker_input.pixel_values.to(self.dtype),
            )
            image = self.nsfw_fallback_img if has_nsfw_concept[0] else image

        return image

    def preprocess_image(self, image: Union[str, Image.Image]) -> torch.Tensor:
        """
        Preprocesses the image.

        Parameters
        ----------
        image : Union[str, Image.Image, torch.Tensor]
            The image to preprocess.

        Returns
        -------
        torch.Tensor
            The preprocessed image.
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB").resize((self.width, self.height))
        if isinstance(image, Image.Image):
            image = image.convert("RGB").resize((self.width, self.height))

        return self.stream.image_processor.preprocess(
            image, self.height, self.width
        ).to(device=self.device, dtype=self.dtype)

    def postprocess_image(
        self, image_tensor: torch.Tensor, output_type: str = "pil"
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        """
        Postprocesses the image.

        Parameters
        ----------
        image_tensor : torch.Tensor
            The image tensor to postprocess.

        Returns
        -------
        Union[Image.Image, List[Image.Image]]
            The postprocessed image.
        """
        if self.frame_buffer_size > 1:
            return postprocess_image(image_tensor.cpu(), output_type=output_type)
        else:
            return postprocess_image(image_tensor.cpu(), output_type=output_type)[0]

    def _load_model(
        self,
        model_id_or_path: str,
        t_index_list: List[int],
        lora_dict: Optional[Dict[str, float]] = None,
        lcm_lora_id: Optional[str] = None,
        vae_id: Optional[str] = None,
        acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt",
        warmup: int = 10,
        do_add_noise: bool = True,
        use_lcm_lora: bool = True,
        use_tiny_vae: bool = True,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
        seed: int = 2,
        engine_dir: Optional[Union[str, Path]] = "engines",
        scheduler_name: str = "EulerAncestral",  # Default scheduler name
        use_karras_sigmas: bool = False,  # Default setting for Karras sigmas
        use_controlnet: bool = False,  # Default setting for ControlNet
        controlnet_model: Optional[str] = None,
        textual_inversion_dict: Optional[Dict[str, Optional[str]]] = None,
    ) -> StreamDiffusion:
        """
        Loads the model.

        This method does the following:

        1. Loads the model from the model_id_or_path.
        2. Loads and fuses the LCM-LoRA model from the lcm_lora_id if needed.
        3. Loads the VAE model from the vae_id if needed.
        4. Enables acceleration if needed.
        5. Prepares the model for inference.
        6. Load the safety checker if needed.

        Parameters
        ----------
        model_id_or_path : str
            The model id or path to load.
        t_index_list : List[int]
            The t_index_list to use for inference.
        lora_dict : Optional[Dict[str, float]], optional
            The lora_dict to load, by default None.
            Keys are the LoRA names and values are the LoRA scales.
            Example: {'LoRA_1' : 0.5 , 'LoRA_2' : 0.7 ,...}
        lcm_lora_id : Optional[str], optional
            The lcm_lora_id to load, by default None.
        vae_id : Optional[str], optional
            The vae_id to load, by default None.
        acceleration : Literal["none", "xfomers", "sfast", "tensorrt"], optional
            The acceleration method, by default "tensorrt".
        warmup : int, optional
            The number of warmup steps to perform, by default 10.
        do_add_noise : bool, optional
            Whether to add noise for following denoising steps or not,
            by default True.
        use_lcm_lora : bool, optional
            Whether to use LCM-LoRA or not, by default True.
        use_tiny_vae : bool, optional
            Whether to use TinyVAE or not, by default True.
        cfg_type : Literal["none", "full", "self", "initialize"],
        optional
            The cfg_type for img2img mode, by default "self".
            You cannot use anything other than "none" for txt2img mode.
        seed : int, optional
            The seed, by default 2.

        Returns
        -------
        StreamDiffusion
            The loaded model.
        """
        use_sdxl = self.sdxl    
        sd_turbo = self.sd_turbo
        custom_lora = lcm_lora_id is not None
        use_lora = use_lcm_lora
        if self.sdxl and self.sd_turbo:
            config_type = "sdxl-turbo"
        elif self.sdxl:
            config_type = "sdxl"
        elif self.sd_turbo:
            config_type = "sd21"
        else:
            config_type = "sd15"
        model_type = "sdxl-turbo (sdxl distilled)" if sd_turbo and use_sdxl else "sd-turbo (2.1 distilled)" if sd_turbo else "sdxl" if use_sdxl else "SD1.5"
        lora_type = " + custom LoRA " if use_lora and custom_lora else " + LCM LoRA " if use_lora else " "
        if sd_turbo:
            lora_type = ' '
        print("\033[36m=======================================\033[0m\n")
        print(f"\033[36m...Loading {model_type}{lora_type}model pipeline...\033[0m")
        model_name = os.path.basename(model_id_or_path)
        clip_exists = os.path.exists(os.path.join(HF_HOME, 'hub', 'models--openai--clip-vit-large-patch14'))
        clip_files = ['config.json', 'merges.txt', 'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json']
        clip_exists = all(os.path.exists(os.path.join(HF_HOME, 'hub', 'models--openai--clip-vit-large-patch14', filename)) for filename in clip_files)
        
        if not clip_exists:
            try:
                from huggingface_hub import hf_hub_download
                for filename in clip_files:
                    hf_hub_download("openai/clip-vit-large-patch14", filename, cache_dir=os.path.join(HF_HOME, 'hub'))
                clip_exists = True
            except Exception as e:
                print(f"Failed to download CLIP model files: {e}")
                clip_exists = False
        pipe = None
        def get_sd_model_config(model_type):
            cache_dir = os.getcwd() + "/models/model_configs"
            config_urls = {
                "sd15": "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml",
                "sd21": "https://raw.githubusercontent.com/Stability-AI/generative-models/main/configs/inference/sd_2_1.yaml",
                "sdxl": "https://raw.githubusercontent.com/Stability-AI/generative-models/main/configs/inference/sd_xl_base.yaml",
                "sdxl-turbo": "https://raw.githubusercontent.com/Stability-AI/generative-models/main/configs/inference/sd_xl_base.yaml",
            }

            os.makedirs(cache_dir, exist_ok=True)

            for key, url in config_urls.items():
                config_path = os.path.join(cache_dir, f"{key}_config.yaml")
                if not os.path.exists(config_path):
                    response = requests.get(url)
                    if response.status_code == 200:
                        with open(config_path, "wb") as f:
                            f.write(response.content)
                    else:
                        raise Exception(f"Failed to download config for {key}")

            config_path = os.path.join(cache_dir, f"{model_type}_config.yaml")
            if not os.path.exists(config_path):
                raise ValueError(f"Unsupported model type: {model_type}")

            return config_path

        original_config_path = get_sd_model_config(config_type)
        # print(f"Config file path: {original_config_path}")
        # # Load the config using OmegaConf
        # from omegaconf import OmegaConf
        # original_config = OmegaConf.load(original_config_path)
        # print("Contents of original_config:")
        # print(OmegaConf.to_yaml(original_config))

        def download_model_file(url, cache_dir="/models"):
            dir = os.path.abspath(cache_dir)
            os.makedirs(dir, exist_ok=True)
            filename = url.split("/")[-1]
            return hf_hub_download(repo_id=url.split("/blob/")[0].split("https://huggingface.co/")[1], filename=filename, local_dir=dir, cache_dir=dir)

        def download_controlnet_file(repo_id, filename, cache_dir="/models/ControlNet"):
            dir = os.path.abspath(cache_dir)
            os.makedirs(dir, exist_ok=True)
            return hf_hub_download(repo_id=repo_id, filename=filename, local_dir=dir, cache_dir=dir)
        
        def create_symlink_if_not_exists(target, link_name):
            if not os.path.exists(link_name):
                os.symlink(target, link_name)

        def remove_symlink_if_exists(link_name):
            if os.path.islink(link_name):
                os.unlink(link_name)
                print(f"Removed symlink: {link_name}")

        def print_connection_error(clip_exists):
            print("\n\033[91mERROR: Failed to load model from Hugging Face.\033[0m")
            print("\033[91mThis error is likely due to lack of internet connection.\033[0m")
            print("\033[91mPlease note:\033[0m")
            print("\033[93m1. Hugging Face model IDs must be first downloaded with an internet connection.")
            print("   Once downloaded, they can be used offline in subsequent runs.")
            print("2. You can use local .safetensors files directly.")
            print("3. You can specify paths to local Diffusers model folders.\033[0m")
            if not clip_exists:
                print("\n\033[91mFor a new setup, additional components like CLIP, VAE, and LCM models")
                print("may need to be downloaded to run StreamDiffusion.")
                print("These will also work offline once downloaded.\033[0m")
            print("\n\033[91mPlease check your internet connection and try again, or use a local model option.\033[0m")

        if use_controlnet and controlnet_model:
            cache_dir = os.getcwd() + "/models/ControlNet"
            os.makedirs(cache_dir, exist_ok=True)
            if config_type == "sd21":
                config_path = os.path.join(cache_dir, "cldm_v21.yaml")
            else:
                config_path = os.path.join(cache_dir, "cldm_v15.yaml")
            if not os.path.exists(config_path):
                if config_type == "sd21":
                    config_url = "https://raw.githubusercontent.com/lllyasviel/ControlNet/main/models/cldm_v21.yaml"
                else:
                    config_url = "https://raw.githubusercontent.com/lllyasviel/ControlNet/main/models/cldm_v15.yaml"
                try:
                    config_content = requests.get(config_url).content
                    with open(config_path, "wb") as f:
                        f.write(config_content)
                except requests.exceptions.ConnectionError:
                    print_connection_error(clip_exists)
                    exit()
            with open(config_path, "rb") as f:
                config_file = BytesIO(f.read())
                # print(f"\nUsing ControlNet Config {config_path}")

            controlnet_id = controlnet_model
            cn_model_name = os.path.basename(controlnet_id)
            print(f"\n\033[36m...Loading {config_type} ControlNet model: {cn_model_name}\033[0m") 
            try:
                if controlnet_id.startswith("https://huggingface.co/") and controlnet_id.endswith(".safetensors"):
                    try:
                        controlnet_file = download_model_file(controlnet_id, cache_dir)
                    except requests.exceptions.ConnectionError:
                        print_connection_error(clip_exists)
                        exit()
                    controlnet = ControlNetModel.from_single_file(controlnet_file, config_file=config_file, use_safetensors=True, local_files_only=clip_exists, torch_dtype=torch.float16).to(device=self.device, dtype=self.dtype)
                    print(f"Successfully loaded ControlNet model from downloaded file:")
                    print(f"\033[92m{controlnet_file}\033[0m\n")
                elif os.path.isfile(controlnet_id):
                    use_safetensors = controlnet_id.endswith(".safetensors")
                    controlnet = ControlNetModel.from_single_file(controlnet_id,config_file=config_file, use_safetensors=use_safetensors, torch_dtype=torch.float16).to(device=self.device, dtype=self.dtype)
                    print(f"Successfully loaded ControlNet model from local file:")
                    print(f"\033[92m{controlnet_id}\033[0m\n")
                    # Create symlink if the file is not in the cache_dir
                    link_name = os.path.join(cache_dir, os.path.basename(controlnet_id))
                    create_symlink_if_not_exists(controlnet_id, link_name)
                else:
                    try:
                        kwargs = {"cache_dir": cache_dir, "torch_dtype": torch.float16, "local_files_only": True}
                        try:
                            controlnet = ControlNetModel.from_pretrained(controlnet_id, **kwargs)
                        except:
                            kwargs.pop('local_files_only', None)
                            controlnet = ControlNetModel.from_pretrained(controlnet_id, **kwargs)
                        controlnet = controlnet.to(device=self.device, dtype=self.dtype)
                        print(f"Successfully loaded ControlNet model from Hugging Face:")
                        print(f"\033[92m{controlnet_id}\033[0m\n")
                    except requests.exceptions.ConnectionError:
                        print_connection_error(clip_exists)
                        exit()
            except Exception as e:
                print(f"Failed to load ControlNet model {controlnet_id} due to: {e}")
                traceback.print_exc()
                # Remove symlink if it exists
                if 'link_name' in locals():
                    remove_symlink_if_exists(link_name)
                exit()

        if use_controlnet:
            if use_sdxl:
                pipeline_class = StableDiffusionXLControlNetPipeline
            else:
                pipeline_class = StableDiffusionControlNetImg2ImgPipeline
        else:
            pipeline_class = StableDiffusionXLPipeline if use_sdxl else StableDiffusionPipeline

        print(f"\n\033[36m...Loading {model_type}{lora_type}model{' with ControlNet' if use_controlnet else ''}: {model_name}\033[0m")

        if model_id_or_path.startswith("https://huggingface.co/") and model_id_or_path.endswith(".safetensors"):
            cache_dir = os.getcwd() + "/models/Model"
            os.makedirs(cache_dir, exist_ok=True)
            model_id_or_path = download_model_file(model_id_or_path, cache_dir)

        try:
            kwargs = {"controlnet": controlnet, "local_files_only": clip_exists} if use_controlnet else {"local_files_only": clip_exists}
            try:
                pipe = pipeline_class.from_pretrained(
                    model_id_or_path,
                    **kwargs
                ).to(device=self.device, dtype=self.dtype)
            # except huggingface_hub.utils._errors.LocalEntryNotFoundError:
            except Exception as e:  # Catch any exception, not just LocalEntryNotFoundError
                # print(f"Debug: Not loading from pretrained with local_files_only. Error: {e}")
                # If local files are not found, try again without local_files_only
                kwargs.pop('local_files_only', None)
                pipe = pipeline_class.from_pretrained(
                    model_id_or_path,
                    **kwargs
                ).to(device=self.device, dtype=self.dtype)
            print(f"Successfully loaded {model_name} from Hugging Face:")
            print(f"\033[92m{model_id_or_path}\033[0m\n")
        except ValueError:
            try:
                kwargs = {
                    "controlnet": controlnet,
                    "original_config_file": original_config_path,
                    "local_files_only": clip_exists,
                    "use_safetensors": True
                } if use_controlnet else {
                    "original_config_file": original_config_path,
                    "local_files_only": clip_exists,
                    "use_safetensors": True
                }
                try:
                    pipe = pipeline_class.from_single_file(
                        model_id_or_path,
                        **kwargs
                    ).to(device=self.device, dtype=self.dtype)
                except Exception:
                    # If loading fails, try again without local_files_only
                    kwargs.pop('local_files_only', None)
                    pipe = pipeline_class.from_single_file(
                        model_id_or_path,
                        **kwargs
                    ).to(device=self.device, dtype=self.dtype)
                print(f"Successfully loaded {model_name} from local directory:")
                print(f"\033[92m{model_id_or_path}\033[0m\n")
            except Exception as e:
                traceback.print_exc()
                print(f"Failed to load {model_id_or_path} / {model_type}{lora_type}model from both local and Hugging Face due to: {e}")
                exit()
        except requests.exceptions.ConnectionError:
            print_connection_error(clip_exists)
            exit()
        except Exception as e:
            traceback.print_exc()
            print(f"Failed to load model{'with ControlNet' if use_controlnet else ''} due to: {e}")
            exit()
        finally:
            time.sleep(.5)
            if pipe:
                pipe = pipe.to(device=self.device, dtype=self.dtype)

        stream = StreamDiffusion(
            pipe=pipe,
            t_index_list=t_index_list,
            torch_dtype=self.dtype,
            width=self.width,
            height=self.height,
            do_add_noise=do_add_noise,
            frame_buffer_size=self.frame_buffer_size,
            use_denoising_batch=self.use_denoising_batch,
            cfg_type=cfg_type,
            use_controlnet=use_controlnet,
        )
        if not self.sd_turbo:
            if use_lcm_lora:
                print(f"\n\033[36m...Loading LCM-LoRA Model: {os.path.basename(lcm_lora_id) if lcm_lora_id else 'latent-consistency/lcm-lora-sdv1-5'}\033[0m")
                try:
                    if lcm_lora_id is not None:
                        if lcm_lora_id.startswith("https://huggingface.co/"):
                            lcm_lora_id = download_model_file(lcm_lora_id, cache_dir=os.getcwd() + "/models/LoRA")
                        stream.load_lcm_lora(
                            pretrained_model_name_or_path_or_dict=lcm_lora_id
                        )
                    else:
                        stream.load_lcm_lora()
                        lcm_lora_id = 'latent-consistency/lcm-lora-sdv1-5'
                    stream.fuse_lora()
                    print(f"Successfully loaded LCM-LoRA model:")
                    print(f"\033[92m{lcm_lora_id}\033[0m\n")
                except Exception as e:
                    print(f"\nERROR loading Local LCM-LoRA: {e}\n")

        if hasattr(stream.unet, 'config'):
            stream.unet.config.addition_embed_type = None

        if use_tiny_vae:
            print(f"\n\033[36m...Loading VAE: {vae_id if vae_id else self.vae_model_id}\033[0m")
            if vae_id is not None:
                stream.vae = AutoencoderTiny.from_pretrained(vae_id).to(
                    device=pipe.device, dtype=pipe.dtype,
                )
            else:
                stream.vae = AutoencoderTiny.from_pretrained(self.vae_model_id).to(
                    device=pipe.device, dtype=pipe.dtype,
                )
            print(f"Successfully loaded VAE model:")
            print(f"\033[92m{vae_id if vae_id else self.vae_model_id}\033[0m\n")
        xformers_enabled = False
        try:
            if acceleration == "xformers":
                stream.pipe.enable_xformers_memory_efficient_attention()
                xformers_enabled = True
        except Exception as e:
            print(f"\nERROR enabling XFormers: {e}\n")
            xformers_enabled = False

        self.original_processors = stream.pipe.unet.attn_processors.copy()
        # print(f"Original processors: {self.original_processors}")
        if self.use_cached_attn:  #V2V
            if acceleration == "tensorrt":
                print("\033[91mSkipping V2V Cached Attention since TensorRT was selected...\nTensorRT doesn't allow cached attention.\033[0m")
            else:
                print("Using V2V Cached Attention...")

                attn_processors = stream.pipe.unet.attn_processors  #V2V
                new_attn_processors = {}  #V2V
                for key, attn_processor in attn_processors.items():  #V2V
                    if xformers_enabled:
                        assert isinstance(attn_processor, XFormersAttnProcessor)
                        #   "We only replace 'XFormersAttnProcessor' to 'CachedSTXFormersAttnProcessor'"  #V2V
                    new_attn_processors[key] = CachedSTXFormersAttnProcessor(name=key,  #V2V
                                                                                use_feature_injection=self.use_feature_injection,  #V2V
                                                                                feature_injection_strength=self.feature_injection_strength,  #V2V
                                                                                feature_similarity_threshold=self.feature_similarity_threshold,  #V2V
                                                                                interval=self.cache_interval,  #V2V
                                                                                max_frames=self.cache_maxframes,  #V2V
                                                                                use_tome_cache=self.use_tome_cache,  #V2V
                                                                                tome_metric=self.tome_metric,  #V2V
                                                                                tome_ratio=self.tome_ratio,  #V2V
                                                                                use_grid=self.use_grid)  #V2V
                stream.pipe.unet.set_attn_processor(new_attn_processors)  #V2V

        try:
            if acceleration == "tensorrt": 
                #bl
                print("\n=======================================")
                print("Using TensorRT...")
                from polygraphy import cuda
                from streamdiffusion.acceleration.tensorrt import (
                    TorchVAEEncoder,
                    compile_unet,
                    compile_vae_decoder,
                    compile_vae_encoder,
                )
                from streamdiffusion.acceleration.tensorrt.engine import (
                    AutoencoderKLEngine,
                    UNet2DConditionModelEngine,
                )
                from streamdiffusion.acceleration.tensorrt.models import (
                    VAE,
                    UNet,
                    VAEEncoder,
                )


                def create_prefix(
                    model_id_or_path: str,
                    max_batch_size: int,
                    min_batch_size: int,
                    width: int,
                    height: int,
                ):
                    if width == 512 and height == 512:
                        resolution = ""
                    else:
                        resolution = f"--width-{width}--height-{height}"
                    maybe_path = Path(model_id_or_path)
                    if maybe_path.exists():
                        return f"{maybe_path.stem}--lcm_lora-{use_lcm_lora}--tiny_vae-{use_tiny_vae}--max_batch-{max_batch_size}--min_batch-{min_batch_size}--mode-img2img{resolution}"
                    else:
                        return f"{model_id_or_path}--lcm_lora-{use_lcm_lora}--tiny_vae-{use_tiny_vae}--max_batch-{max_batch_size}--min_batch-{min_batch_size}--mode-img2img{resolution}"


                engine_dir = Path(engine_dir)
                unet_path = os.path.join(
                    engine_dir,
                    create_prefix(
                        model_id_or_path=model_id_or_path,
                        max_batch_size=stream.trt_unet_batch_size,
                        min_batch_size=stream.trt_unet_batch_size,
                        width=self.width,
                        height=self.height,
                    ),
                    "unet.engine",
                )
                vae_encoder_path = os.path.join(
                    engine_dir,
                    create_prefix(
                        model_id_or_path=model_id_or_path,
                        max_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                        min_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                        width=self.width,
                        height=self.height,
                    ),
                    "vae_encoder.engine",
                )
                vae_decoder_path = os.path.join(
                    engine_dir,
                    create_prefix(
                        model_id_or_path=model_id_or_path,
                        max_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                        min_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                        width=self.width,
                        height=self.height,
                    ),
                    "vae_decoder.engine",
                )
                # print("!!! \033[1mSTARTING TENSORRT\033[0m !!!\n--------------------------------")
                # print(f"self.sdxl value: {self.sdxl}")
                engine_build_options = {
                    "opt_image_height": self.height,
                    "opt_image_width": self.width,
                }

                if not os.path.exists(unet_path):
                    os.makedirs(os.path.dirname(unet_path), exist_ok=True)
                    unet_model = UNet(
                        fp16=True,
                        device=stream.device,
                        max_batch_size=stream.trt_unet_batch_size,
                        min_batch_size=stream.trt_unet_batch_size,
                        embedding_dim=stream.text_encoder.config.hidden_size,
                        unet_dim=stream.unet.config.in_channels,
                        # is_xl=self.sdxl,

                    )
                    print("\nCompiling TensorRT UNet...\nThis may take a moment...\n")
                    time.sleep(1)
                    compile_unet(
                        stream.unet,
                        unet_model,
                        unet_path + ".onnx",
                        unet_path + ".opt.onnx",
                        unet_path,
                        opt_batch_size=stream.trt_unet_batch_size,
                        engine_build_options=engine_build_options,
                        # is_xl=self.sdxl,
                    )

                if not os.path.exists(vae_decoder_path):
                    os.makedirs(os.path.dirname(vae_decoder_path), exist_ok=True)
                    stream.vae.forward = stream.vae.decode
                    vae_decoder_model = VAE(
                        device=stream.device,
                        max_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                        min_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                    )
                    print("\nCompiling TensorRT VAE Decoder...\n")
                    compile_vae_decoder(
                        stream.vae,
                        vae_decoder_model,
                        vae_decoder_path + ".onnx",
                        vae_decoder_path + ".opt.onnx",
                        vae_decoder_path,
                        opt_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                        engine_build_options=engine_build_options,

                    )
                    delattr(stream.vae, "forward")

                if not os.path.exists(vae_encoder_path):
                    os.makedirs(os.path.dirname(vae_encoder_path), exist_ok=True)
                    vae_encoder = TorchVAEEncoder(stream.vae).to(torch.device("cuda"))
                    vae_encoder_model = VAEEncoder(
                        device=stream.device,
                        max_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                        min_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                    )
                    print("\nCompiling TensorRT VAE Encoder...\n")
                    compile_vae_encoder(
                        vae_encoder,
                        vae_encoder_model,
                        vae_encoder_path + ".onnx",
                        vae_encoder_path + ".opt.onnx",
                        vae_encoder_path,
                        opt_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                        engine_build_options=engine_build_options,
                    )

                cuda_steram = cuda.Stream()

                vae_config = stream.vae.config
                vae_dtype = stream.vae.dtype

                stream.unet = UNet2DConditionModelEngine(
                    unet_path, cuda_steram, use_cuda_graph=False
                )
                stream.vae = AutoencoderKLEngine(
                    vae_encoder_path,
                    vae_decoder_path,
                    cuda_steram,
                    stream.pipe.vae_scale_factor,
                    use_cuda_graph=False,
                )
                setattr(stream.vae, "config", vae_config)
                setattr(stream.vae, "dtype", vae_dtype)

                gc.collect()
                torch.cuda.empty_cache()

                print("TensorRT acceleration enabled.")
            if acceleration == "sfast":
                from streamdiffusion.acceleration.sfast import (
                    accelerate_with_stable_fast,
                )

                stream = accelerate_with_stable_fast(stream)
                print("StableFast acceleration enabled.")
        except Exception as e:
            # traceback.print_exc()
            self.print_tensorrt_install_message(e)   
            print("\nAcceleration has failed. Falling back to normal mode.")

        if seed < 0: # Random seed
            seed = np.random.randint(0, 1000000)

        stream.prepare(
            "",
            "",
            num_inference_steps=50,
            guidance_scale=1.1
            if stream.cfg_type in ["full", "self", "initialize"]
            else 1.0,
            generator=torch.manual_seed(seed),
            seed=seed,
        )

        if self.use_safety_checker:
            from transformers import CLIPFeatureExtractor
            from diffusers.pipelines.stable_diffusion.safety_checker import (
                StableDiffusionSafetyChecker,
            )

            self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker"
            ).to(pipe.device)
            self.feature_extractor = CLIPFeatureExtractor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            self.nsfw_fallback_img = Image.new("RGB", (512, 512), (0, 0, 0))

        return stream
    
    def print_tensorrt_install_message(self, error=None):
        if error:
            print(f"\n\033[91mERROR: {str(error)[:100]}...\033[0m")  # Print shortened version of the error

        if error and "mat1 and mat2 shapes cannot be multiplied" in str(error):
            print("\n\033[91mERROR: SDXL models are not supported with TensorRT.\033[0m")
            return

        if error and any(module in str(error) for module in ["polygraphy", "onnxgraphsurgeon"]):
            print("\n\033[91mERROR: Failed to load TensorRT. This is likely due to incomplete installation.\033[0m")
            print("\033[93mThis setup is Linux/NVIDIA-only; make sure CUDA is set up correctly.\033[0m")
        else:
            print("\n\033[91mERROR: Failed to load TensorRT due to an unexpected error.\nCheck the installation and restart the stream.\033[0m")

    def update_lora_weights(self, lora_dict: Dict[str, float], print_info: bool = False, print_weight: bool = False):
        try:
            # Track which LoRAs need to be unloaded
            current_loras = set(os.path.splitext(os.path.basename(name))[0] for name in lora_dict.keys())
            to_unload = set(self.loaded_loras.keys()) - current_loras

            # Unload removed LoRAs
            for adapter_name in to_unload:
                if print_info:
                    print(f"\n\033[36m...Unloading LoRA: {adapter_name}\033[0m")
                try:
                    self.stream.pipe.unload_lora_weights(adapter_name=self.loaded_loras[adapter_name]["adapter_name"])
                except Exception as e:
                    print(f"\033[93mWarning while unloading {adapter_name}: {e}\033[0m")
                del self.loaded_loras[adapter_name]

            adapter_names = []
            adapter_weights = []
            updated_lora_dict = {}
            
            # Model type detection
            if self.sdxl:
                model_type = "SDXL"
            elif self.sd_turbo:
                model_type = "SD2.1"
            else:
                model_type = "SD1.5"

            for lora_name, lora_scale in lora_dict.items():
                if lora_name.startswith("https://huggingface.co/"):
                    local_path = os.path.join(os.getcwd(), "models", "LoRA", os.path.basename(lora_name))
                    if not os.path.exists(local_path):
                        if print_info:
                            print(f"\n\033[36m...Downloading LoRA Model: {os.path.basename(lora_name)}\033[0m")
                        lora_name = self.download_model_file(lora_name, cache_dir=os.path.join(os.getcwd(), "models", "LoRA"))
                    else:
                        lora_name = local_path

                base_adapter_name = os.path.splitext(os.path.basename(lora_name))[0]
                updated_lora_dict[lora_name] = lora_scale

                # Check if LoRA is already loaded
                if base_adapter_name in self.loaded_loras:
                    # Just update the weight if it's different
                    self.loaded_loras[base_adapter_name]["scale"] = lora_scale
                    adapter_names.append(self.loaded_loras[base_adapter_name]["adapter_name"])
                    adapter_weights.append(lora_scale)
                    if print_weight:
                        print(f"\n\033[36m...Updating LoRA weight: {base_adapter_name} -> {lora_scale}\033[0m")
                    continue

                # Load new LoRA
                try:
                    if print_info:
                        print(f"\n\033[36m...Loading LoRA Model: {os.path.basename(lora_name)}, Weight: {lora_scale}\033[0m")
                    
                    # Generate unique adapter name with timestamp
                    unique_adapter_name = f"{base_adapter_name}_{int(time.time() * 1000)}"
                    
                    # Clean up any existing adapters before loading
                    try:
                        existing_adapters = self.stream.pipe.unet.get_active_adapters()
                        for adapter in existing_adapters:
                            if base_adapter_name in adapter:
                                self.stream.pipe.unload_lora_weights(adapter_name=adapter)
                    except:
                        pass

                    self.stream.pipe.load_lora_weights(lora_name, adapter_name=unique_adapter_name)
                    self.loaded_loras[base_adapter_name] = {
                        "path": lora_name,
                        "scale": lora_scale,
                        "adapter_name": unique_adapter_name
                    }
                    
                    if print_weight:
                        print(f"Successfully loaded LoRA model:")
                        print(f"\033[92m{lora_name}\033[0m\n")
                    
                    adapter_names.append(unique_adapter_name)
                    adapter_weights.append(lora_scale)

                except Exception as e:
                    error_str = str(e)
                    if "size mismatch" in error_str:
                        print("\n\033[91m=============== LoRA Loading Error ===============\033[0m")
                        print(f"\033[91mArchitecture mismatch detected!\033[0m")
                        print("\n\033[93mDetails:\033[0m")
                        print(f"• Base Model: {model_type}")
                        print(f"• Failed LoRA: {os.path.basename(lora_name)}")
                        print("\n\033[93mProblem:\033[0m")
                        print("The LoRA model architecture might not match your base model.")
                        print("\033[91m===============================================\033[0m\n")
                    else:
                        print(f"\033[91mError loading LoRA '{base_adapter_name}': {e}\033[0m")
                    continue

            # Only set adapters if we have any to set
            if adapter_names:
                self.stream.pipe.set_adapters(adapter_names, adapter_weights)

        except Exception as e:
            print(f'\033[91mCaution: Error updating LoRA weights: {e}\033[0m')
        
        return updated_lora_dict

    def download_model_file(self, url, cache_dir="/models"):
        dir = os.path.abspath(cache_dir)
        os.makedirs(dir, exist_ok=True)
        filename = url.split("/")[-1]
        return hf_hub_download(repo_id=url.split("/blob/")[0].split("https://huggingface.co/")[1], filename=filename, local_dir=dir, cache_dir=dir)

    def load_scheduler(self, scheduler_name: str, use_karras_sigmas: bool = False):
        """
        Dynamically loads a scheduler based on the given name, with the ability to handle
        various initialization parameters such as Karras sigmas.

        Parameters:
        - scheduler_name (str): The name of the scheduler to load.
        - use_karras_sigmas (bool): Whether to use Karras sigmas for applicable schedulers.

        Returns:
        - A scheduler instance from the diffusers library.
        """
        try:
            # Mapping scheduler names to their corresponding module paths and initialization parameters
            scheduler_map = {
                "LMS": ("diffusers.LMSDiscreteScheduler", {"use_karras_sigmas": use_karras_sigmas}),
                "DPMSolverMultistep": ("diffusers.DPMSolverMultistepScheduler", {"use_karras_sigmas": use_karras_sigmas}),
                "KDPM2": ("diffusers.KDPM2DiscreteScheduler", {"use_karras_sigmas": use_karras_sigmas}),
                "KDPM2Ancestral": ("diffusers.KDPM2AncestralDiscreteScheduler", {"use_karras_sigmas": use_karras_sigmas}),
                "Euler": ("diffusers.EulerDiscreteScheduler", {}),
                "EulerAncestral": ("diffusers.EulerAncestralDiscreteScheduler", {}),
                "Heun": ("diffusers.HeunDiscreteScheduler", {}),
                "DEISMultistep": ("diffusers.DEISMultistepScheduler", {}),
                "UniPCMultistep": ("diffusers.UniPCMultistepScheduler", {})
            }

            if scheduler_name not in scheduler_map:
                # Fallback to default scheduler if specified one is not found
                from diffusers import EulerDiscreteScheduler
                print(f"Scheduler '{scheduler_name}' not found. Fallback to default 'EulerDiscreteScheduler'.")
                return EulerDiscreteScheduler()

            # Dynamic import based on the scheduler_map
            module_name, class_name = scheduler_map[scheduler_name][0].rsplit('.', 1)
            scheduler_module = __import__(module_name, fromlist=[class_name])
            scheduler_class = getattr(scheduler_module, class_name)
            
            # Initialize the scheduler with the specified parameters
            scheduler = scheduler_class(**scheduler_map[scheduler_name][1])
            print(f"Loaded scheduler: {scheduler_name} with params: {scheduler_map[scheduler_name][1]}")
            return scheduler

        except Exception as e:
            # Handle unexpected errors in scheduler loading
            from diffusers import EulerDiscreteScheduler
            print(f"Error loading scheduler '{scheduler_name}': {str(e)}. Using default 'EulerDiscreteScheduler'.")
            return EulerDiscreteScheduler()

    def update_textual_inversions(self, ti_dict: Dict[str, Optional[str]], print_info: bool = True):
        """
        Load or update textual inversion embeddings.
        
        Parameters:
        -----------
        ti_dict : Dict[str, Optional[str]]
            Dictionary mapping file paths/URLs to custom tokens. If token is None, 
            will use default token from the embedding.
        print_info : bool
            Whether to print loading information.
        """
        print(f"!!!!!!!!!!Updating Textual Inversions with: {ti_dict}")
        try:
            # Track which embeddings to keep
            current_embeddings = set()
            
            for ti_path, token in ti_dict.items():
                if ti_path.startswith("https://huggingface.co/"):
                    local_path = os.path.join(os.getcwd(), "models", "TextualInversion", os.path.basename(ti_path))
                    if not os.path.exists(local_path):
                        if print_info:
                            print(f"\n\033[36m...Downloading Textual Inversion: {os.path.basename(ti_path)}\033[0m")
                        ti_path = self.download_model_file(ti_path, cache_dir=os.path.join(os.getcwd(), "models", "TextualInversion"))
                    else:
                        ti_path = local_path

                embedding_name = os.path.splitext(os.path.basename(ti_path))[0]
                current_embeddings.add(embedding_name)

                # Skip if already loaded with same token
                if embedding_name in self.loaded_textual_inversions:
                    if self.loaded_textual_inversions[embedding_name]["token"] == token:
                        continue

                if print_info:
                    print(f"\n\033[36m...Loading Textual Inversion: {os.path.basename(ti_path)}\033[0m")
                
                try:
                    self.stream.pipe.load_textual_inversion(
                        ti_path,
                        token=token,
                        text_encoder=self.stream.pipe.text_encoder if not self.sdxl else None,
                        tokenizer=self.stream.pipe.tokenizer if not self.sdxl else None,
                    )
                    
                    # For SDXL, also load into the second text encoder if present
                    if self.sdxl and hasattr(self.stream.pipe, 'text_encoder_2'):
                        self.stream.pipe.load_textual_inversion(
                            ti_path,
                            token=token,
                            text_encoder=self.stream.pipe.text_encoder_2,
                            tokenizer=self.stream.pipe.tokenizer_2,
                        )
                    
                    self.loaded_textual_inversions[embedding_name] = {
                        "path": ti_path,
                        "token": token
                    }
                    
                    if print_info:
                        print(f"Successfully loaded Textual Inversion:")
                        print(f"\033[92m{ti_path}\033[0m\n")
                        
                except Exception as e:
                    print(f"\033[91mError loading Textual Inversion '{embedding_name}': {e}\033[0m")
                    continue

            # Unload embeddings that are no longer needed
            embeddings_to_remove = set(self.loaded_textual_inversions.keys()) - current_embeddings
            for embedding_name in embeddings_to_remove:
                if print_info:
                    print(f"\n\033[36m...Unloading Textual Inversion: {embedding_name}\033[0m")
                
                token = self.loaded_textual_inversions[embedding_name]["token"]
                self.stream.pipe.unload_textual_inversion(
                    tokens=token,
                    text_encoder=self.stream.pipe.text_encoder if not self.sdxl else None,
                    tokenizer=self.stream.pipe.tokenizer if not self.sdxl else None,
                )
                
                # For SDXL, also unload from second text encoder
                if self.sdxl and hasattr(self.stream.pipe, 'text_encoder_2'):
                    self.stream.pipe.unload_textual_inversion(
                        tokens=token,
                        text_encoder=self.stream.pipe.text_encoder_2,
                        tokenizer=self.stream.pipe.tokenizer_2,
                    )
                    
                del self.loaded_textual_inversions[embedding_name]

        except Exception as e:
            print(f'\033[91mCaution: Error updating Textual Inversions: {e}\033[0m')