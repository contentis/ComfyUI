import logging

import torch

import comfy.sd
import comfy.utils
import folder_paths
import random
from typing import Tuple, Callable

from comfy_extras.nodes_flux import FluxGuidance, FluxKontextImageScale, CLIPTextEncodeFlux
from comfy_extras.nodes_sd3 import EmptySD3LatentImage
from comfy_extras.nodes_edit_model import ReferenceLatent

from nodes import CLIPTextEncode, KSampler, VAEEncode, ConditioningZeroOut

from . import register_recipe
from .base import ModelRecipe, SamplerCFG
from .dataset import HFPromptDataloader, KontextBenchDataLoader

class FluxT2IPipe:
    def __init__(
            self,
            model,
            clip,
            vae,
            width,
            height,
            seed=0,
            sampler_cfg: SamplerCFG = None,
            device="cuda",
    ) -> None:
        self.clip = clip
        self.vae = vae
        self.diffusion_model = model

        self.width = width
        self.height = height
        self.sampler_cfg = sampler_cfg
        self.device = device
        self.seed = seed
        assert self.sampler_cfg is not None, "Sampler configuration is required"

    @torch.inference_mode
    def __call__(self, num_inference_steps, positive_prompt, *args, **kwargs):
        positive = CLIPTextEncodeFlux().encode(self.clip, positive_prompt, positive_prompt, self.sampler_cfg.flux_cfg)[0]
        negative = ConditioningZeroOut().zero_out(positive)[0]
        latent = EmptySD3LatentImage().execute(self.width, self.height).args[0]

        KSampler().sample(
                        self.diffusion_model, self.seed, num_inference_steps,
                        self.sampler_cfg.cfg, self.sampler_cfg.sampler_name,
                        self.sampler_cfg.scheduler, positive=positive,
                        negative=negative, latent_image=latent,
                        denoise=self.sampler_cfg.denoise)[0]



class FluxRecipeBase(ModelRecipe):
    @classmethod
    def add_model_args(cls, parser):
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument(
            "--ckpt_path",
            help="Path to full checkpoint (includes diffusion model + text encoder + VAE)"
        )
        group.add_argument(
            "--unet_path",
            help="Path to diffusion model only (requires test_encoder and VAE)"
        )

        parser.add_argument(
            "--clip_path",
            help="Path to text encoder (required with --unet_path)"
        )
        parser.add_argument(
            "--t5_path",
            help="Path to text encoder (required with --unet_path)"
        )

        parser.add_argument(
            "--vae_path",
            help="Path to VAE model (required with --unet_path)",
            required=False,
        )

    def __init__(self, args):
        self.args = args

        # Validate args
        if hasattr(args, 'unet_path') and args.unet_path:
            if not args.clip_path or not args.t5_path:
                raise ValueError("--unet_path requires both --clip_path and --t5_path")

    def load_model(self) -> Tuple:
        if hasattr(self.args, 'ckpt_path') and self.args.ckpt_path:
            # Load from full checkpoint
            logging.info(f"Loading full checkpoint from {self.args.ckpt_path}")
            model_patcher, clip, vae, _ = comfy.sd.load_checkpoint_guess_config(
                self.args.ckpt_path,
                output_vae=True,
                output_clip=True,
                embedding_directory=None
            )
        else:
            # Load from separate files
            logging.info(f"Loading diffusion model from {self.args.unet_path}")
            model_options = {}
            clip_type = comfy.sd.CLIPType.FLUX

            clip_path1 = folder_paths.get_full_path_or_raise("text_encoders", self.args.clip_path)
            clip_path2 = folder_paths.get_full_path_or_raise("text_encoders", self.args.t5_path)

            model_patcher = comfy.sd.load_diffusion_model(
                self.args.unet_path,
                model_options=model_options
            )
            clip = comfy.sd.load_clip(
                ckpt_paths=[clip_path1, clip_path2],
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

        return FluxT2IPipe(
            model=model_patcher,
            clip=clip,
            vae=vae,
            width=self.get_width(),
            height=self.get_height(),
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

                logging.debug(f"Calibration step {i + 1}: '{prompt_text[:50]}...'")
                try:
                    calib_pipeline(num_steps, prompt_text)
                except Exception as e:
                    logging.warning(f"Calibration step {i + 1} failed: {e}")

        return forward_loop

    def get_width(self) -> int:
        return 1024

    def get_height(self) -> int:
        return 1024

    def get_default_calib_steps(self) -> int:
        return 64

    def get_sampler_cfg(self) -> SamplerCFG:
        return SamplerCFG(
            cfg=1.0,
            sampler_name="euler",
            scheduler="simple",
            denoise=1.0,
            flux_cfg=3.5
        )

    def get_inference_steps(self) -> int:
        """Number of sampling steps per calibration iteration."""
        raise NotImplementedError

    def get_dataset(self):
        return HFPromptDataloader()

@register_recipe
class FluxDevRecipe(FluxRecipeBase):
    @classmethod
    def name(cls) -> str:
        return "flux_dev"

    def get_inference_steps(self) -> int:
        return 30

@register_recipe
class FluxSchnellRecipe(FluxRecipeBase):
    @classmethod
    def name(cls) -> str:
        return "flux_schnell"

    def get_inference_steps(self) -> int:
        return 4

class FluxKontextPipe:
    def __init__(
            self,
            model,
            clip,
            vae,
            seed=0,
            sampler_cfg: SamplerCFG = None,
            device="cuda",
    ) -> None:
        self.clip = clip
        self.vae = vae
        self.diffusion_model = model

        self.sampler_cfg = sampler_cfg
        self.device = device
        self.seed = seed
        assert self.sampler_cfg is not None, "Sampler configuration is required"
        assert self.vae is not None, "VAE is required for FluxKontextRecipe"

    @torch.inference_mode
    def __call__(self, num_inference_steps, positive_prompt, image, *args, **kwargs):
        image_preprocessed = FluxKontextImageScale().execute(image).args[0]
        image_encoded = VAEEncode().encode(self.vae, image_preprocessed)[0]

        positive = CLIPTextEncode().encode(self.clip, positive_prompt)[0]
        conditioning_img = ReferenceLatent().execute(positive, image_encoded).args[0]
        conditioning_img = FluxGuidance().execute(conditioning_img, self.sampler_cfg.img_cfg).args[0]

        conditioning_prompt = ConditioningZeroOut().zero_out(positive)[0]
        KSampler().sample(self.diffusion_model, self.seed, num_inference_steps,
                                              self.sampler_cfg.cfg, self.sampler_cfg.sampler_name,
                                              self.sampler_cfg.scheduler, positive=conditioning_img,
                                              negative=conditioning_prompt, latent_image=image_encoded,
                                              denoise=self.sampler_cfg.denoise)[0]

    def _preview_img(self, out):
        pass

@register_recipe
class FluxKontextRecipe(FluxRecipeBase):
    @classmethod
    def name(cls) -> str:
        return "flux_kontext"

    def create_calibration_pipeline(self, model_components):
        model_patcher, clip, vae = model_components
        assert vae is not None, "VAE is required for FluxKontextRecipe"

        return FluxKontextPipe(
            model=model_patcher,
            clip=clip,
            vae=vae,
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
                img = sample["image"]

                logging.debug(f"Calibration step {i+1}: '{prompt_text[:50]}...'")
                try:
                    calib_pipeline(num_steps, prompt_text, img)
                except Exception as e:
                    logging.warning(f"Calibration step {i+1} failed: {e}")

        return forward_loop

    def get_inference_steps(self) -> int:
        return 30

    def get_dataset(self):
        logging.info("Loading KontextBench dataset...")
        return KontextBenchDataLoader()

    def get_sampler_cfg(self) -> SamplerCFG:
        return SamplerCFG(
            cfg=1.0,
            sampler_name="euler",
            scheduler="simple",
            denoise=1.0,
            flux_cfg=2.5
        )
