import logging

import torch

import comfy.sd
import comfy.utils
import folder_paths
import random
from typing import Tuple, Callable

from comfy_extras.nodes_wan import Wan22ImageToVideoLatent, WanImageToVideo
from comfy_extras.nodes_model_advanced import ModelSamplingSD3

from nodes import CLIPTextEncode, KSampler, KSamplerAdvanced

from . import register_recipe
from .base import ModelRecipe, SamplerCFG
from .dataset import HFPromptDataloader, KontextBenchDataLoader

class WAN22SinglePipe:
    def __init__(
            self,
            model,
            clip,
            vae,
            width,
            height,
            length,
            seed=0,
            sampler_cfg: SamplerCFG = None,
            device="cuda",
    ) -> None:
        self.clip = clip
        self.vae = vae
        self.diffusion_model = model

        self.width = width
        self.height = height
        self.length = length

        self.sampler_cfg = sampler_cfg
        self.device = device
        self.seed = seed
        assert self.sampler_cfg is not None, "Sampler configuration is required"

    @torch.inference_mode
    def __call__(self, num_inference_steps, positive_prompt, negative_prompt, image=None, *args, **kwargs):
        positive = CLIPTextEncode().encode(self.clip, positive_prompt)[0]
        negative = CLIPTextEncode().encode(self.clip, negative_prompt)[0]

        latent = Wan22ImageToVideoLatent().execute(width=self.width, height=self.height, length=self.length,
                                                   vae=self.vae, batch_size=1, start_image=image).args[0]

        model = ModelSamplingSD3().patch(self.diffusion_model, self.sampler_cfg.flux_cfg)[0]

        out, denoised_out = KSampler().sample(model, self.seed, num_inference_steps,
                                              self.sampler_cfg.cfg, self.sampler_cfg.sampler_name,
                                              self.sampler_cfg.scheduler, positive=positive,
                                              negative=negative, latent_image=latent,
                                              denoise=self.sampler_cfg.denoise)


@register_recipe
class WAN22SingleRecipe(ModelRecipe):
    @classmethod
    def name(cls) -> str:
        return "wan_22_5b_t2v"

    @classmethod
    def add_model_args(cls, parser):
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument(
            "--unet_path",
            help="Path to diffusion model only (requires test_encoder and VAE)"
        )

        parser.add_argument(
            "--clip_path",
            help="Path to text encoder (required with --unet_path)"
        )

        parser.add_argument(
            "--vae_path",
            help="Path to VAE model (required with --unet_path)",
            required=False,
        )

    def __init__(self, args):
        self.args = args

    def load_model(self) -> Tuple:
        # Load from separate files
        logging.info(f"Loading diffusion model from {self.args.unet_path}")
        model_options = {}
        clip_type = comfy.sd.CLIPType.WAN

        clip_path = folder_paths.get_full_path_or_raise("text_encoders", self.args.clip_path)

        model_patcher = comfy.sd.load_diffusion_model(
            self.args.unet_path,
            model_options=model_options
        )
        clip = comfy.sd.load_clip(
            ckpt_paths=[clip_path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type,
            model_options=model_options
        )

        vae = None  # Not needed for calibration
        if self.args.vae_path:
            vae_path = folder_paths.get_full_path_or_raise("vae", self.args.vae_path)
            sd = comfy.utils.load_torch_file(vae_path)
            vae = comfy.sd.VAE(sd=sd)
            vae.throw_exception_if_invalid()
        return model_patcher, clip, vae

    def create_calibration_pipeline(self, model_components):
        model_patcher, clip, vae = model_components

        return WAN22SinglePipe(
            model=model_patcher,
            clip=clip,
            vae=vae,
            width=self.get_width(),
            height=self.get_height(),
            length=self.get_length(),
            seed=42,
            sampler_cfg=self.get_sampler_cfg(),
            device="cuda"
        )

    def get_forward_loop(self, calib_pipeline, num_calib_steps) -> Callable:
        num_steps = self.get_inference_steps()
        dataloader = self.get_dataset()

        def forward_loop():
            for i in range(num_calib_steps):
                rnd_idx = random.randint(0, len(dataloader) - 1)
                sample = dataloader[rnd_idx]
                prompt_text = sample["prompt"]
                negative_prompt = "low quality"

                logging.debug(f"Calibration step {i + 1}: '{prompt_text[:50]}...'")
                try:
                    calib_pipeline(num_steps, prompt_text, negative_prompt)
                except Exception as e:
                    logging.warning(f"Calibration step {i + 1} failed: {e}")

        return forward_loop

    def get_width(self) -> int:
        return 1280

    def get_height(self) -> int:
        return 704

    def get_length(self) -> int:
        return 121

    def get_default_calib_steps(self) -> int:
        return 32

    def get_sampler_cfg(self) -> SamplerCFG:
        return SamplerCFG(
            cfg=5.0,
            sampler_name="uni_pc",
            scheduler="simple",
            denoise=1.0,
            flux_cfg=8.0
        )

    def get_inference_steps(self) -> int:
        return 20

    def get_dataset(self):
        return HFPromptDataloader()

class WAN22DoublePipe:
    def __init__(
            self,
            high_noise_model,
            low_noise_model,
            clip,
            vae,
            width,
            height,
            length,
            seed=0,
            sampler_cfg: SamplerCFG = None,
            device="cuda",
    ) -> None:
        self.clip = clip
        self.vae = vae
        self.high_noise_model = high_noise_model
        self.low_noise_model = low_noise_model

        self.width = width
        self.height = height
        self.length = length

        self.sampler_cfg = sampler_cfg
        self.device = device
        self.seed = seed
        assert self.sampler_cfg is not None, "Sampler configuration is required"

    @torch.inference_mode
    def __call__(self, num_inference_steps, positive_prompt, negative_prompt, image=None, *args, **kwargs):
        positive = CLIPTextEncode().encode(self.clip, positive_prompt)[0]
        negative = CLIPTextEncode().encode(self.clip, negative_prompt)[0]

        positive, negative, latent_image = WanImageToVideo().execute(width=self.width, height=self.height,
                                                                     length=self.length, batch_size=1,
                                                                     positive=positive, negative=negative,
                                                                     vae=self.vae, start_image=image).args

        high_noise_model = ModelSamplingSD3().patch(self.high_noise_model, self.sampler_cfg.flux_cfg)[0]
        low_noise_model = ModelSamplingSD3().patch(self.low_noise_model, self.sampler_cfg.flux_cfg)[0]

        mid_step = num_inference_steps // 2

        out, denoised_out = KSamplerAdvanced().sample(model=high_noise_model, noise_seed=self.seed,
                                                      steps=num_inference_steps, cfg=self.sampler_cfg.cfg,
                                                      sampler_name=self.sampler_cfg.sampler_name,
                                                      scheduler=self.sampler_cfg.scheduler, positive=positive,
                                                      negative=negative, latent_image=latent_image,
                                                      denoise=self.sampler_cfg.denoise, add_noise=True,
                                                      start_at_step=0, end_at_step=mid_step,
                                                      return_with_leftover_noise=True)

        out, denoised_out = KSamplerAdvanced().sample(model=low_noise_model, noise_seed=self.seed,
                                                      steps=num_inference_steps, cfg=self.sampler_cfg.cfg,
                                                      sampler_name=self.sampler_cfg.sampler_name,
                                                      scheduler=self.sampler_cfg.scheduler, positive=positive,
                                                      negative=negative, latent_image=latent_image,
                                                      denoise=self.sampler_cfg.denoise, add_noise=False,
                                                      start_at_step=mid_step, end_at_step=num_inference_steps,
                                                      return_with_leftover_noise=False)

@register_recipe
class WAN22DoubleRecipe(ModelRecipe):
    @classmethod
    def name(cls) -> str:
        return "wan_22_14b_i2v"

    @classmethod
    def add_model_args(cls, parser):
        parser.add_argument(
            "--unet_path_low_noise",
            help="Path to diffusion model only (requires test_encoder and VAE)"
        )

        parser.add_argument(
            "--unet_path_high_noise",
            help="Path to diffusion model only (requires test_encoder and VAE)"
        )

        parser.add_argument(
            "--clip_path",
            help="Path to text encoder (required with --unet_path)"
        )

        parser.add_argument(
            "--vae_path",
            help="Path to VAE model (required with --unet_path)",
            required=False,
        )

    def __init__(self, args):
        self.args = args

    def load_model(self) -> Tuple:
        # Load from separate files
        logging.info(f"Loading diffusion model from {self.args.unet_path_low_noise}")
        logging.info(f"Loading diffusion model from {self.args.unet_path_high_noise}")
        model_options = {}
        clip_type = comfy.sd.CLIPType.WAN

        clip_path = folder_paths.get_full_path_or_raise("text_encoders", self.args.clip_path)

        model_patcher_low = comfy.sd.load_diffusion_model(
            self.args.unet_path_low_noise,
            model_options=model_options
        )
        model_patcher_high = comfy.sd.load_diffusion_model(
            self.args.unet_path_high_noise,
            model_options=model_options
        )

        clip = comfy.sd.load_clip(
            ckpt_paths=[clip_path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type,
            model_options=model_options
        )

        vae = None  # Not needed for calibration
        if self.args.vae_path:
            vae_path = folder_paths.get_full_path_or_raise("vae", self.args.vae_path)
            sd = comfy.utils.load_torch_file(vae_path)
            vae = comfy.sd.VAE(sd=sd)
            vae.throw_exception_if_invalid()
        return model_patcher_high, model_patcher_low, clip, vae

    def create_calibration_pipeline(self, model_components):
        high_noise_model, low_noise_model, clip, vae = model_components

        return WAN22DoublePipe(
            high_noise_model=high_noise_model,
            low_noise_model=low_noise_model,
            clip=clip,
            vae=vae,
            width=self.get_width(),
            height=self.get_height(),
            length=self.get_length(),
            seed=42,
            sampler_cfg=self.get_sampler_cfg(),
            device="cuda"
        )

    def get_forward_loop(self, calib_pipeline, num_calib_steps) -> Callable:
        num_steps = self.get_inference_steps()
        dataloader = self.get_dataset()

        def forward_loop():
            for i in range(num_calib_steps):
                rnd_idx = random.randint(0, len(dataloader) - 1)
                sample = dataloader[rnd_idx]
                image = sample["image"]
                prompt_text = sample["prompt"]
                negative_prompt = "low quality"

                logging.debug(f"Calibration step {i + 1}: '{prompt_text[:50]}...'")
                try:
                    calib_pipeline(num_steps, prompt_text, negative_prompt, image)
                except Exception as e:
                    logging.warning(f"Calibration step {i + 1} failed: {e}")

        return forward_loop

    def get_width(self) -> int:
        return 720

    def get_height(self) -> int:
        return 480

    def get_length(self) -> int:
        return 81

    def get_default_calib_steps(self) -> int:
        return 32

    def get_inference_steps(self) -> int:
        return 20

    def get_sampler_cfg(self) -> SamplerCFG:
        return SamplerCFG(
            cfg=3.5,
            sampler_name="euler",
            scheduler="simple",
            denoise=1.0,
            flux_cfg=8.0
        )

    def get_dataset(self):
        return KontextBenchDataLoader()
