# pipeline.py
import time
from typing import List, Optional, Union, Any, Dict, Tuple, Literal

import numpy as np
import PIL.Image
import torch
from diffusers import LCMScheduler, StableDiffusionPipeline, StableDiffusionXLPipeline, DiffusionPipeline, StableDiffusionXLControlNetPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    retrieve_latents,
)

from streamdiffusion.image_filter import SimilarImageFilter


# Add these imports at the top of the file
from collections import deque  #V2V
from v2v import get_nn_latent  #V2V




class UNet2DConditionControlNetModel(torch.nn.Module):
    def __init__(self, unet, controlnet) -> None:
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet
        
    def forward(self, sample, timestep, encoder_hidden_states, image):
        # hard-coded since it is not clear how to integrate this argument into tensorrt
        conditioning_scale = 1.0

        down_samples, mid_sample = self.controlnet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=image,
            guess_mode=False,
            return_dict=False,
        )

        down_block_res_samples = [
            down_sample * conditioning_scale
            for down_sample in down_samples
        ]
        mid_block_res_sample = conditioning_scale * mid_sample
        
        noise_pred = self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            return_dict=False,
        )
        return noise_pred



class StreamDiffusion:
    def __init__(
        self,
        pipe: DiffusionPipeline,
        t_index_list: List[int],
        torch_dtype: torch.dtype = torch.float16,
        width: int = 512,
        height: int = 512,
        do_add_noise: bool = True,
        use_denoising_batch: bool = True,
        frame_buffer_size: int = 1,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
        use_controlnet: bool = False,
        use_cached_attn: bool = False, #V2V

    ) -> None:
        self.device = pipe.device
        self.dtype = torch_dtype
        self.generator = None

        self.height = height
        self.width = width

        self.latent_height = int(height // pipe.vae_scale_factor)
        self.latent_width = int(width // pipe.vae_scale_factor)

        self.frame_bff_size = frame_buffer_size
        self.denoising_steps_num = len(t_index_list)

        self.cfg_type = cfg_type

        if use_denoising_batch:
            self.batch_size = self.denoising_steps_num * frame_buffer_size
            if self.cfg_type == "initialize":
                self.trt_unet_batch_size = (
                    self.denoising_steps_num + 1
                ) * self.frame_bff_size
            elif self.cfg_type == "full":
                self.trt_unet_batch_size = (
                    2 * self.denoising_steps_num * self.frame_bff_size
                )
            else:
                self.trt_unet_batch_size = self.denoising_steps_num * frame_buffer_size
        else:
            self.trt_unet_batch_size = self.frame_bff_size
            self.batch_size = frame_buffer_size

        self.t_list = t_index_list

        self.do_add_noise = do_add_noise
        self.use_denoising_batch = use_denoising_batch

        self.similar_image_filter = False
        self.similar_filter = SimilarImageFilter()
        self.prev_image_result = None

        self.pipe = pipe
        self.image_processor = VaeImageProcessor(pipe.vae_scale_factor)

        self.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        # print("scheduler")
        # print(self.scheduler)
        self.text_encoder = pipe.text_encoder
        self.vae = pipe.vae
        
        self.inference_time_ema = 0

        self.sdxl = type(self.pipe) is StableDiffusionXLPipeline or type(self.pipe) is StableDiffusionXLControlNetPipeline

        self.use_controlnet = use_controlnet
        if self.use_controlnet:
            self.controlnet = pipe.controlnet
        else:
            self.controlnet = None
        self.controlnet_conditioning_scale = 1.0
        self.controlnet_start_step: int = 0
        self.controlnet_end_step: int = 49

        # Add these attributes to the StreamDiffusion class
        self.cached_x_t_latent = deque(maxlen=4)  #V2V
        self.use_cached_attn = use_cached_attn #V2V

        # if self.use_controlnet:
        #     self.unet = UNet2DConditionControlNetModel(pipe.unet, self.controlnet)
        # else:
        #     self.unet = pipe.unet
        self.unet = pipe.unet
    # Add these methods to the StreamDiffusion class
    def norm_noise(self, noise):  #V2V
        mean = noise.mean()
        std = noise.std()
        normalized_noise = (noise - mean) / std
        return normalized_noise


    def load_lcm_lora(
        self,
        pretrained_model_name_or_path_or_dict: Union[
            str, Dict[str, torch.Tensor]
        ] = "latent-consistency/lcm-lora-sdv1-5",
        adapter_name: Optional[Any] = None,
        **kwargs,
    ) -> None:
        self.pipe.load_lora_weights(
            pretrained_model_name_or_path_or_dict, adapter_name, **kwargs
        )

    def load_lora(
        self,
        pretrained_lora_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        adapter_name: Optional[Any] = None,
        **kwargs,
    ) -> None:
        self.pipe.load_lora_weights(
            pretrained_lora_model_name_or_path_or_dict, adapter_name, **kwargs
        )

    def fuse_lora(
        self,
        fuse_unet: bool = True,
        fuse_text_encoder: bool = True,
        lora_scale: float = 1.0,
        safe_fusing: bool = False,
    ) -> None:
        self.pipe.fuse_lora(
            fuse_unet=fuse_unet,
            fuse_text_encoder=fuse_text_encoder,
            lora_scale=lora_scale,
            safe_fusing=safe_fusing,
        )

    def enable_similar_image_filter(self, threshold: float = 0.98, max_skip_frame: float = 10) -> None:
        self.similar_image_filter = True
        self.similar_filter.set_threshold(threshold)
        self.similar_filter.set_max_skip_frame(max_skip_frame)

    def disable_similar_image_filter(self) -> None:
        self.similar_image_filter = False

    @torch.no_grad()
    def prepare(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 1.2,
        delta: float = 1.0,
        generator: Optional[torch.Generator] = torch.Generator(),
        seed: int = 2,
    ) -> None:
        self.generator = generator
        self.generator.manual_seed(seed)
        # initialize x_t_latent (it can be any random tensor)
        if self.denoising_steps_num > 1:
            self.x_t_latent_buffer = torch.zeros(
                (
                    (self.denoising_steps_num - 1) * self.frame_bff_size,
                    4,
                    self.latent_height,
                    self.latent_width,
                ),
                dtype=self.dtype,
                device=self.device,
            )
            self.control_image_buffer = torch.zeros(
                (
                    (self.denoising_steps_num - 1) * self.frame_bff_size,
                    3,
                    self.height,
                    self.width,
                ),
                dtype=self.dtype,
                device=self.device,
            )
        else:
            self.x_t_latent_buffer = None
            self.control_image_buffer = None

        if self.cfg_type == "none":
            self.guidance_scale = 1.0
        else:
            self.guidance_scale = guidance_scale
        self.delta = delta

        do_classifier_free_guidance = False
        if self.guidance_scale > 1.0:
            do_classifier_free_guidance = True

        encoder_output = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
        )
        self.prompt_embeds = encoder_output[0].repeat(self.batch_size, 1, 1)

        if self.sdxl:
            self.add_text_embeds = encoder_output[2]
            # Move add_text_embeds to device once during prepare
            self.add_text_embeds = self.add_text_embeds.to(self.device)
            original_size = (self.height, self.width)
            crops_coords_top_left = (0, 0)
            target_size = (self.height, self.width)
            text_encoder_projection_dim = int(self.add_text_embeds.shape[-1])
            base_add_time_ids = self._get_add_time_ids(
                original_size,
                crops_coords_top_left,
                target_size,
                dtype=encoder_output[0].dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
            # Move add_time_ids to device once during prepare
            base_add_time_ids = base_add_time_ids.to(self.device)
            
            # Prepare time_ids for CFG scenarios
            if self.guidance_scale > 1.0 and (self.cfg_type == "initialize" or self.cfg_type == "full"):
                if self.cfg_type == "full":
                    # For full CFG, repeat for the entire doubled batch
                    self.add_time_ids = base_add_time_ids.repeat(2 * self.batch_size, 1)
                elif self.cfg_type == "initialize":
                    # For initialize CFG, repeat for frame buffer + batch size
                    self.add_time_ids = base_add_time_ids.repeat(self.frame_bff_size + self.batch_size, 1)
            else:
                # For no CFG or self CFG, just use the batch size
                self.add_time_ids = base_add_time_ids.repeat(self.batch_size, 1)

        if self.use_denoising_batch and self.cfg_type == "full":
            if encoder_output[1] is not None:
                uncond_prompt_embeds = encoder_output[1].repeat(self.batch_size, 1, 1)
        elif self.cfg_type == "initialize":
            if encoder_output[1] is not None:
                uncond_prompt_embeds = encoder_output[1].repeat(self.frame_bff_size, 1, 1)

        # if self.use_denoising_batch and self.cfg_type == "full":
        #     uncond_prompt_embeds = encoder_output[1].repeat(self.batch_size, 1, 1)
        # elif self.cfg_type == "initialize":
        #     uncond_prompt_embeds = encoder_output[1].repeat(self.frame_bff_size, 1, 1)

        if self.guidance_scale > 1.0 and (
            self.cfg_type == "initialize" or self.cfg_type == "full"
        ):
            self.prompt_embeds = torch.cat(
                [uncond_prompt_embeds, self.prompt_embeds], dim=0
            )

        self.scheduler.set_timesteps(num_inference_steps, self.device)
        self.timesteps = self.scheduler.timesteps.to(self.device)

        # make sub timesteps list based on the indices in the t_list list and the values in the timesteps list
        self.sub_timesteps = []
        for t in self.t_list:
            self.sub_timesteps.append(self.timesteps[t])

        sub_timesteps_tensor = torch.tensor(
            self.sub_timesteps, dtype=torch.long, device=self.device
        )
        self.sub_timesteps_tensor = torch.repeat_interleave(
            sub_timesteps_tensor,
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
        )

        self.init_noise = torch.randn(
            (self.batch_size, 4, self.latent_height, self.latent_width),
            generator=generator,
        ).to(device=self.device, dtype=self.dtype)

        self.stock_noise = torch.zeros_like(self.init_noise)

        c_skip_list = []
        c_out_list = []
        for timestep in self.sub_timesteps:
            c_skip, c_out = self.scheduler.get_scalings_for_boundary_condition_discrete(
                timestep
            )
            c_skip_list.append(c_skip)
            c_out_list.append(c_out)

        self.c_skip = (
            torch.stack(c_skip_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )
        self.c_out = (
            torch.stack(c_out_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )

        alpha_prod_t_sqrt_list = []
        beta_prod_t_sqrt_list = []
        for timestep in self.sub_timesteps:
            alpha_prod_t_sqrt = self.scheduler.alphas_cumprod[timestep].sqrt()
            beta_prod_t_sqrt = (1 - self.scheduler.alphas_cumprod[timestep]).sqrt()
            alpha_prod_t_sqrt_list.append(alpha_prod_t_sqrt)
            beta_prod_t_sqrt_list.append(beta_prod_t_sqrt)
        alpha_prod_t_sqrt = (
            torch.stack(alpha_prod_t_sqrt_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )
        beta_prod_t_sqrt = (
            torch.stack(beta_prod_t_sqrt_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )
        self.alpha_prod_t_sqrt = torch.repeat_interleave(
            alpha_prod_t_sqrt,
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
        )
        self.beta_prod_t_sqrt = torch.repeat_interleave(
            beta_prod_t_sqrt,
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
        )

    @torch.no_grad()
    def update_prompt(self, prompt: str, negative_prompt: str = "") -> None:
        # Determine if we need CFG based on guidance scale and cfg_type
        do_classifier_free_guidance = (self.guidance_scale > 1.0 and 
                                       self.cfg_type != "none")
        
        encoder_output = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt if do_classifier_free_guidance else "",
        )
        
        pos_embeds = encoder_output[0]
        
        # Handle CFG embedding structure based on cfg_type
        if do_classifier_free_guidance:
            if self.cfg_type == "full":
                # For "full" CFG, concatenate negative and positive embeddings
                if len(encoder_output) > 1 and encoder_output[1] is not None:
                    neg_embeds = encoder_output[1]
                    pos_embeds_repeated = pos_embeds.repeat(self.batch_size, 1, 1)
                    neg_embeds_repeated = neg_embeds.repeat(self.batch_size, 1, 1)
                    self.prompt_embeds = torch.cat([neg_embeds_repeated, pos_embeds_repeated], dim=0)
                else:
                    self.prompt_embeds = pos_embeds.repeat(self.batch_size, 1, 1)
                    
            elif self.cfg_type == "initialize":
                # For "initialize" CFG, concatenate with appropriate sizes
                if len(encoder_output) > 1 and encoder_output[1] is not None:
                    neg_embeds = encoder_output[1]
                    pos_embeds_repeated = pos_embeds.repeat(self.batch_size, 1, 1)
                    neg_embeds_repeated = neg_embeds.repeat(self.frame_bff_size, 1, 1)
                    self.prompt_embeds = torch.cat([neg_embeds_repeated, pos_embeds_repeated], dim=0)
                else:
                    self.prompt_embeds = pos_embeds.repeat(self.batch_size, 1, 1)
                    
            elif self.cfg_type == "self":
                # For "self" CFG, only positive embeddings (negative guidance from stock_noise)
                self.prompt_embeds = pos_embeds.repeat(self.batch_size, 1, 1)
            else:
                # Default case
                self.prompt_embeds = pos_embeds.repeat(self.batch_size, 1, 1)
        else:
            # No CFG, just use positive embeddings
            self.prompt_embeds = pos_embeds.repeat(self.batch_size, 1, 1)

    def unload_all_loras(self):
        """Unload all LoRA adapters from the model"""
        if hasattr(self.unet, 'loaded_adapter_names'):
            for adapter_name in self.unet.loaded_adapter_names:
                self.unet.delete_adapters(adapter_name)
        if hasattr(self.text_encoder, 'loaded_adapter_names'):
            for adapter_name in self.text_encoder.loaded_adapter_names:
                self.text_encoder.delete_adapters(adapter_name)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        t_index: int,
    ) -> torch.Tensor:
        noisy_samples = (
            self.alpha_prod_t_sqrt[t_index] * original_samples
            + self.beta_prod_t_sqrt[t_index] * noise
        )
        return noisy_samples

    def scheduler_step_batch(
        self,
        model_pred_batch: torch.Tensor,
        x_t_latent_batch: torch.Tensor,
        idx: Optional[int] = None,
    ) -> torch.Tensor:
        # TODO: use t_list to select beta_prod_t_sqrt
        if idx is None:
            F_theta = (
                x_t_latent_batch - self.beta_prod_t_sqrt * model_pred_batch
            ) / self.alpha_prod_t_sqrt
            denoised_batch = self.c_out * F_theta + self.c_skip * x_t_latent_batch
        else:
            F_theta = (
                x_t_latent_batch - self.beta_prod_t_sqrt[idx] * model_pred_batch
            ) / self.alpha_prod_t_sqrt[idx]
            denoised_batch = (
                self.c_out[idx] * F_theta + self.c_skip[idx] * x_t_latent_batch
            )

        return denoised_batch

    def unet_step(
        self,
        x_t_latent: torch.Tensor,
        t_list: Union[torch.Tensor, list[int]],
        added_cond_kwargs, 
        idx: Optional[int] = None, 
        control_image: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.guidance_scale > 1.0 and (self.cfg_type == "initialize"):
            x_t_latent_plus_uc = torch.concat([x_t_latent[0:1], x_t_latent], dim=0)
            t_list = torch.concat([t_list[0:1], t_list], dim=0)
        elif self.guidance_scale > 1.0 and (self.cfg_type == "full"):
            x_t_latent_plus_uc = torch.concat([x_t_latent, x_t_latent], dim=0)
            t_list = torch.concat([t_list, t_list], dim=0)
        else:
            x_t_latent_plus_uc = x_t_latent

        # Ensure ControlNet input matches the latent input dimensions for CFG
        if self.use_controlnet and self.controlnet is not None and control_image is not None:
            if self.guidance_scale > 1.0 and (self.cfg_type == "initialize"):
                # For initialize CFG, duplicate the first control image
                control_image_plus_uc = torch.concat([control_image[0:1], control_image], dim=0)
            elif self.guidance_scale > 1.0 and (self.cfg_type == "full"):
                # For full CFG, duplicate the entire control batch
                control_image_plus_uc = torch.concat([control_image, control_image], dim=0)
            else:
                control_image_plus_uc = control_image
        else:
            control_image_plus_uc = control_image

        if self.use_controlnet and self.controlnet is not None:
            # print(f"Using controlnet. {'SDXL' if self.sdxl else ''} added_cond_kwargs - \n{added_cond_kwargs}")   
            down_samples, mid_sample = self.controlnet(
                x_t_latent_plus_uc,
                t_list,
                encoder_hidden_states=self.prompt_embeds,
                controlnet_cond=control_image_plus_uc,
                guess_mode=False,
                return_dict=False,
                added_cond_kwargs=added_cond_kwargs,  # Pass added_cond_kwargs to controlnet
            )
            down_block_res_samples = [
                down_sample * self.controlnet_conditioning_scale
                for down_sample in down_samples
            ]
            mid_block_res_sample = self.controlnet_conditioning_scale * mid_sample
            model_pred = self.unet(
                x_t_latent_plus_uc,
                t_list,
                encoder_hidden_states=self.prompt_embeds,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
        else:
            model_pred = self.unet(
                x_t_latent_plus_uc,
                t_list,
                encoder_hidden_states=self.prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

        if self.guidance_scale > 1.0 and (self.cfg_type == "initialize"):
            noise_pred_text = model_pred[1:]
            self.stock_noise = torch.concat(
                [model_pred[0:1], self.stock_noise[1:]], dim=0
            )  # ここコメントアウトでself out cfg
        elif self.guidance_scale > 1.0 and (self.cfg_type == "full"):
            noise_pred_uncond, noise_pred_text = model_pred.chunk(2)
        else:
            noise_pred_text = model_pred
        if self.guidance_scale > 1.0 and (
            self.cfg_type == "self" or self.cfg_type == "initialize"
        ):
            noise_pred_uncond = self.stock_noise * self.delta
        if self.guidance_scale > 1.0 and self.cfg_type != "none":
            # Check for NaN values in noise_pred_uncond
            if torch.isnan(noise_pred_uncond).any():
                # print("NaN detected in noise_pred_uncond, replacing with zeros.")
                noise_pred_uncond = torch.zeros_like(noise_pred_uncond)

            model_pred = noise_pred_uncond + self.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
        else:
            model_pred = noise_pred_text

        # compute the previous noisy sample x_t -> x_t-1
        if self.use_denoising_batch:
            denoised_batch = self.scheduler_step_batch(model_pred, x_t_latent, idx)
            if self.cfg_type == "self" or self.cfg_type == "initialize":
                scaled_noise = self.beta_prod_t_sqrt * self.stock_noise
                delta_x = self.scheduler_step_batch(model_pred, scaled_noise, idx)
                alpha_next = torch.concat(
                    [
                        self.alpha_prod_t_sqrt[1:],
                        torch.ones_like(self.alpha_prod_t_sqrt[0:1]),
                    ],
                    dim=0,
                )
                delta_x = alpha_next * delta_x
                beta_next = torch.concat(
                    [
                        self.beta_prod_t_sqrt[1:],
                        torch.ones_like(self.beta_prod_t_sqrt[0:1]),
                    ],
                    dim=0,
                )
                delta_x = delta_x / beta_next
                init_noise = torch.concat(
                    [self.init_noise[1:], self.init_noise[0:1]], dim=0
                )
                self.stock_noise = init_noise + delta_x

        else:
            # denoised_batch = self.scheduler.step(model_pred, t_list[0], x_t_latent).denoised
            denoised_batch = self.scheduler_step_batch(model_pred, x_t_latent, idx)

        return denoised_batch, model_pred

    def _get_add_time_ids(
        self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
    ):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    def encode_image(self, image_tensors: torch.Tensor) -> torch.Tensor:
        image_tensors = image_tensors.to(
            device=self.device,
            dtype=self.vae.dtype,
        )
        img_latent = retrieve_latents(self.vae.encode(image_tensors), self.generator)
        img_latent = img_latent * self.vae.config.scaling_factor
        x_t_latent = self.add_noise(img_latent, self.init_noise[0], 0)
        return x_t_latent

    def decode_image(self, x_0_pred_out: torch.Tensor) -> torch.Tensor:               
        output_latent = self.vae.decode(
            x_0_pred_out / self.vae.config.scaling_factor, return_dict=False
        )[0]
        return output_latent


    def _prepare_inpaint_control_image(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Prepares the control image for the inpainting ControlNet.
        This is a torch-native version of the `make_inpaint_condition` function from the
        diffusers documentation.
        """
        # Ensure image and mask are in the [0, 1] range.
        # The input `image` tensor from preprocess is already normalized.
        # The `mask` (control_image) is also assumed to be normalized.
        
        # Create a copy of the image to avoid modifying the original
        control_image = image.clone()

        # Where the mask is > 0.5, set the control_image pixel values to -1.0
        # The mask is single-channel, so we need to expand it to match the image's 3 channels.
        mask_expanded = mask.expand_as(control_image)
        control_image[mask_expanded > 0.5] = -1.0
        
        return control_image

    def predict_x0_batch(self, x_t_latent: torch.Tensor, control_image: Optional[torch.Tensor] = None) -> torch.Tensor:
        # print("Entering predict_x0_batch")
        added_cond_kwargs = {}
        prev_latent_batch = self.x_t_latent_buffer

        # ------------------------------------------------------------------
        # FIX: ensure latent batch matches `frame_buffer_size` when only one
        # denoising step is used (len(t_index_list)==1).  In this scenario
        # `self.sub_timesteps_tensor` has length `frame_buffer_size`, but the
        # incoming `x_t_latent` has batch size 1.  We replicate the latent
        # (and control image, if any) so both tensors have identical batch
        # dimensions, avoiding the dimension-mismatch error in UNet.
        # ------------------------------------------------------------------
        if (
            self.use_denoising_batch
            and self.denoising_steps_num == 1
            and self.frame_bff_size > 1
            and x_t_latent.shape[0] == 1
        ):
            x_t_latent = x_t_latent.repeat(self.frame_bff_size, 1, 1, 1)
            if self.use_controlnet and control_image is not None and control_image.shape[0] == 1:
                control_image = control_image.repeat(self.frame_bff_size, 1, 1, 1)
        # ------------------------------------------------------------------

        if self.use_denoising_batch:
            if self.denoising_steps_num > 1:
                if self.use_cached_attn: #V2V
                    x_t_latent = torch.cat((x_t_latent, prev_latent_batch), dim=0)  #V2V
                    self.stock_noise = torch.cat(
                    (self.init_noise[0:1], self.stock_noise[:-1]), dim=0
                    )  #V2V
                else:
                    x_t_latent = torch.cat((x_t_latent, prev_latent_batch), dim=0)
                    # print("Updated x_t_latent with previous batch")



            if self.sdxl:
                # add_text_embeds and add_time_ids are already on self.device from prepare()
                added_cond_kwargs = {"text_embeds": self.add_text_embeds, "time_ids": self.add_time_ids}
                # print("Added condition kwargs for SDXL")

            if self.use_controlnet:
                if control_image is not None:
                    # Ensure control_image is on the correct device and dtype.
                    # The .to() method is a no-op if the tensor is already
                    # in the correct state, so this is safe and efficient.
                    control_image = control_image.to(device=self.device, dtype=self.dtype)

                    if control_image.dim() == 3:
                        control_image = control_image.unsqueeze(0)
                    if self.control_image_buffer is not None:  # Check if buffer is not None
                        control_image = torch.cat((control_image, self.control_image_buffer), dim=0)
                else:
                    control_image = self.control_image_buffer

            t_list = self.sub_timesteps_tensor # Should already be on device
            # Conditional copy for x_t_latent
            if x_t_latent.device != self.device:
                x_t_latent = x_t_latent.to(self.device)
            # t_list = t_list.to(self.device) # Already on device from prepare
            # print("Device transfer complete for tensors")

            x_0_pred_batch, model_pred = self.unet_step(x_t_latent, t_list, added_cond_kwargs=added_cond_kwargs, control_image=control_image)

            #UPDATE BUFFER
            if self.denoising_steps_num > 1:
                x_0_pred_out = x_0_pred_batch[-1].unsqueeze(0)
                if self.do_add_noise:
                    self.x_t_latent_buffer = (
                        self.alpha_prod_t_sqrt[1:] * x_0_pred_batch[:-1]
                        + self.beta_prod_t_sqrt[1:] * self.init_noise[1:]
                    )
                else:
                    self.x_t_latent_buffer = (
                        self.alpha_prod_t_sqrt[1:] * x_0_pred_batch[:-1]
                    )


                #new new 
                if self.use_controlnet:
                    self.control_image_buffer = control_image[:-self.frame_bff_size]

            else:
                # When only one denoising step is used, we replicated the latent
                # purely to satisfy the UNet batch requirement.  Only the first
                # element is the "current" frame; the remaining replicas are
                # identical and can be discarded to keep memory-copy sizes small
                # and downstream code unchanged (expects batch=1).
                x_0_pred_out = x_0_pred_batch[0:1]
                self.x_t_latent_buffer = None
                self.control_image_buffer = None
        else:
            self.init_noise = x_t_latent
            for idx, t in enumerate(self.sub_timesteps_tensor):
                t = t.view(
                    1,
                ).repeat(
                    self.frame_bff_size,
                )
                if self.sdxl:
                    # Use the properly prepared add_time_ids that already account for CFG
                    added_cond_kwargs = {"text_embeds": self.add_text_embeds, "time_ids": self.add_time_ids}
                # x_0_pred, model_pred = self.unet_step(x_t_latent, t, idx=idx, added_cond_kwargs=added_cond_kwargs)
                x_0_pred, model_pred = self.unet_step(x_t_latent, t, idx=idx, added_cond_kwargs=added_cond_kwargs, control_image=control_image)
                if idx < len(self.sub_timesteps_tensor) - 1:
                    if self.do_add_noise:
                        x_t_latent = self.alpha_prod_t_sqrt[
                            idx + 1
                        ] * x_0_pred + self.beta_prod_t_sqrt[
                            idx + 1
                        ] * torch.randn_like(
                            x_0_pred, device=self.device, dtype=self.dtype
                        )
                    else:
                        x_t_latent = self.alpha_prod_t_sqrt[idx + 1] * x_0_pred
            x_0_pred_out = x_0_pred
        return x_0_pred_out

    @torch.no_grad()
    def __call__(
        self, 
        x: Union[torch.Tensor, PIL.Image.Image, np.ndarray] = None,
        control_image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.device == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        if x is not None:
            # Check if x is already a processed tensor on the correct device from main.py
            # main.py sends a (1, 3, H, W) float32 GPU tensor.
            is_likely_preprocessed_input = (
                isinstance(x, torch.Tensor) and
                x.device == self.device and
                x.ndim == 4 and # Has batch dimension
                x.shape[0] == 1 and
                x.shape[1] == 3 and # 3 Channels
                x.shape[2] == self.height and
                x.shape[3] == self.width
            )

            if not is_likely_preprocessed_input:
                # If not preprocessed (e.g., PIL image, numpy, or wrong tensor format),
                # then do the full preprocessing.
                # self.image_processor.preprocess returns a CPU tensor.
                x = self.image_processor.preprocess(x, self.height, self.width).to(
                    device=self.device, dtype=self.dtype, non_blocking=True
                )
            elif x.dtype != self.dtype: # Already on GPU but potentially wrong dtype
                x = x.to(dtype=self.dtype, non_blocking=True) # D2D or no-op if dtype matches
            
            # Now x is assumed to be on self.device and of self.dtype

            if self.similar_image_filter:
                x_filtered = self.similar_filter(x) 
                if x_filtered is None:
                    if self.inference_time_ema > 0: 
                        time.sleep(self.inference_time_ema)
                    return self.prev_image_result
                x = x_filtered 

            x_t_latent = self.encode_image(x)
        else:
            # txt2img path, x_t_latent is created directly on device
            x_t_latent = torch.randn((1, 4, self.latent_height, self.latent_width)).to(
                device=self.device, dtype=self.dtype
            )
        x_0_pred_out = self.predict_x0_batch(x_t_latent, control_image)
        x_output = self.decode_image(x_0_pred_out).detach().clone()

        self.prev_image_result = x_output
        if self.device == "cuda":
            end.record()
            torch.cuda.synchronize() # Ensure all GPU operations are done for timing
            inference_time = start.elapsed_time(end) / 1000
            self.inference_time_ema = 0.9 * self.inference_time_ema + 0.1 * inference_time
        return x_output

    @torch.no_grad()
    def txt2img(self, batch_size: int = 1, control_image: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_0_pred_out = self.predict_x0_batch(
            torch.randn((batch_size, 4, self.latent_height, self.latent_width)).to(
                device=self.device, dtype=self.dtype
            ),
            control_image,
        )
        x_output = self.decode_image(x_0_pred_out).detach().clone()
        return x_output
    

    def txt2img_sd_turbo(self, batch_size: int = 1) -> torch.Tensor:
        x_t_latent = torch.randn(
            (batch_size, 4, self.latent_height, self.latent_width),
            device=self.device,
            dtype=self.dtype,
        )
        model_pred = self.unet(
            x_t_latent,
            self.sub_timesteps_tensor,
            encoder_hidden_states=self.prompt_embeds,
            return_dict=False,
        )[0]
        x_0_pred_out = (
            x_t_latent - self.beta_prod_t_sqrt * model_pred
        ) / self.alpha_prod_t_sqrt
        return self.decode_image(x_0_pred_out)