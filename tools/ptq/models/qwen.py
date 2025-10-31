import logging

import torch

import comfy.sd
import comfy.utils
import folder_paths
import random
from typing import Tuple, Callable

from comfy_extras.nodes_sd3 import EmptySD3LatentImage

from comfy_extras.nodes_model_advanced import ModelSamplingAuraFlow
from comfy_extras.nodes_post_processing import ImageScaleToTotalPixels
from comfy_extras.nodes_qwen import TextEncodeQwenImageEditPlus
from comfy_extras.nodes_cfg import CFGNorm
from nodes import CLIPTextEncode, KSampler, VAEEncode

from . import register_recipe
from .base import ModelRecipe, SamplerCFG
from .dataset import HFPromptDataloader, KontextBenchDataLoader

class QwenPipe:
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
    def __call__(self, num_inference_steps, positive_prompt, negative_prompt, *args, **kwargs):
        positive = CLIPTextEncode().encode(self.clip, positive_prompt)[0]
        negative = CLIPTextEncode().encode(self.clip, negative_prompt)[0]

        model = ModelSamplingAuraFlow().patch_aura(self.diffusion_model, self.sampler_cfg.flux_cfg)[0]

        latent = EmptySD3LatentImage().execute(self.width, self.height).args[0]

        out, denoised_out = KSampler().sample(model, self.seed, num_inference_steps,
                                              self.sampler_cfg.cfg, self.sampler_cfg.sampler_name,
                                              self.sampler_cfg.scheduler, positive=positive,
                                              negative=negative, latent_image=latent,
                                              denoise=self.sampler_cfg.denoise)

    @torch.inference_mode
    def image_edit(self, num_inference_steps, positive_prompt, negative_prompt, image, *args, **kwargs):
        image = ImageScaleToTotalPixels().execute(image, "lanczos", 1.0).args[0]
        latent_image = VAEEncode().encode(self.vae, image)[0]

        positive = TextEncodeQwenImageEditPlus().execute(self.clip, positive_prompt, image).args[0]
        negative = TextEncodeQwenImageEditPlus().execute(self.clip, negative_prompt, image).args[0]



        model = ModelSamplingAuraFlow().patch_aura(self.diffusion_model, self.sampler_cfg.flux_cfg)[0]
        model = CFGNorm().execute(model, 1.0).args[0]


        out, denoised_out = KSampler().sample(model, self.seed, num_inference_steps,
                                              self.sampler_cfg.cfg, self.sampler_cfg.sampler_name,
                                              self.sampler_cfg.scheduler, positive=positive,
                                              negative=negative, latent_image=latent_image,
                                              denoise=self.sampler_cfg.denoise)

class QwenRecipeBase(ModelRecipe):
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
            "--qwen_vl_path",
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
            if not args.qwen_vl_path:
                raise ValueError("--unet_path requires both --qwen_vl_path")

    def load_model(self) -> Tuple:
        """Load FLUX model, CLIP, and VAE."""
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
            clip_type = comfy.sd.CLIPType.QWEN_IMAGE

            clip_path = folder_paths.get_full_path_or_raise("text_encoders", self.args.qwen_vl_path)

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

        return QwenPipe(
            model=model_patcher,
            clip=clip,
            vae=vae,
            width=self.get_width(),
            height=self.get_height(),
            seed=42,
            sampler_cfg=self.get_sampler_cfg(),
            device="cuda"
        )


    def get_width(self) -> int:
        return 1328

    def get_height(self) -> int:
        return 1328

    def get_default_calib_steps(self) -> int:
        return 64

    def get_sampler_cfg(self) -> SamplerCFG:
        return SamplerCFG(
            cfg=2.5,
            sampler_name="euler",
            scheduler="simple",
            denoise=1.0,
            flux_cfg=3.1
        )

    def get_inference_steps(self) -> int:
        return 30



@register_recipe
class QwenImage(QwenRecipeBase):
    @classmethod
    def name(cls) -> str:
        return "qwen_image"

    def get_dataset(self):
        return HFPromptDataloader()

    def get_forward_loop(self, calib_pipeline, num_calib_steps) -> Callable:
        num_steps = self.get_inference_steps()
        dataloader = self.get_dataset()

        def forward_loop():
            for i in range(num_calib_steps):
                rnd_idx = random.randint(0, len(dataloader) - 1)
                sample = dataloader[rnd_idx]
                prompt_text = sample["prompt"]
                negative_text = "low quality"

                logging.debug(f"Calibration step {i + 1}: '{prompt_text[:50]}...'")
                try:
                    calib_pipeline(num_steps, prompt_text, negative_text)
                except Exception as e:
                    logging.warning(f"Calibration step {i + 1} failed: {e}")

        return forward_loop


@register_recipe
class QwenImageEdit(QwenRecipeBase):
    @classmethod
    def name(cls) -> str:
        return "qwen_edit"

    def get_forward_loop(self, calib_pipeline, num_calib_steps) -> Callable:
        num_steps = self.get_inference_steps()
        dataloader = self.get_dataset()
        def forward_loop():
            for i in range(num_calib_steps):
                rnd_idx = random.randint(0, len(dataloader) - 1)
                sample = dataloader[rnd_idx]
                prompt_text = sample["prompt"]
                negative_text = "low quality"
                img = sample["image"]

                logging.debug(f"Calibration step {i+1}: '{prompt_text[:50]}...'")
                try:
                    calib_pipeline.image_edit(num_steps, prompt_text, negative_text, img)
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
            cfg=2.5,
            sampler_name="euler",
            scheduler="simple",
            denoise=1.0,
            flux_cfg=3.0
        )
