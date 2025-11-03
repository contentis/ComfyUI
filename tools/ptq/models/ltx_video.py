import logging
import torch

import comfy.sd
import comfy.utils
import folder_paths
import random
from typing import Tuple, Callable

from comfy_extras.nodes_lt import LTXVConditioning, EmptyLTXVLatentVideo, LTXVScheduler, LTXVImgToVideo

from comfy_extras.nodes_model_advanced import ModelSamplingAuraFlow
from comfy_extras.nodes_custom_sampler import SamplerCustom, KSamplerSelect
from nodes import CLIPTextEncode

from . import register_recipe
from .base import ModelRecipe, SamplerCFG
from .dataset import HFPromptDataloader, KontextBenchDataLoader

class LTXVideoPipe:
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
            custom_kwargs: dict = None,
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

        self.custom_kwargs = custom_kwargs

    @torch.inference_mode
    def __call__(self, num_inference_steps, positive_prompt, negative_prompt, image=None, *args, **kwargs):
        positive = CLIPTextEncode().encode(self.clip, positive_prompt)[0]
        negative = CLIPTextEncode().encode(self.clip, negative_prompt)[0]

        if image is None:
            latent_image = EmptyLTXVLatentVideo().execute(self.width, self.height, self.length).args[0]
        else:
            positive, negative, latent_image = LTXVImgToVideo().execute(positive=positive, negative=negative,
                                                                        vae=self.vae, image=image,
                                                                        width=self.width, height=self.height,
                                                                        length=self.length, strength=0.1, batch_size=1).args

        positive, negative = LTXVConditioning().execute(positive, negative,
                                                        frame_rate=self.custom_kwargs.get("frame_rate", 25.0)).args

        model = ModelSamplingAuraFlow().patch_aura(self.diffusion_model, self.sampler_cfg.flux_cfg)[0]
        sigmas = LTXVScheduler().execute(num_inference_steps, max_shift=self.custom_kwargs.get("max_shift", 2.05),
                                         base_shift=self.custom_kwargs.get("base_shift", 0.95),
                                         stretch=self.custom_kwargs.get("stretch", True),
                                         terminal=self.custom_kwargs.get("terminal", 0.1)).args[0]

        sampler = KSamplerSelect().get_sampler(self.sampler_cfg.sampler_name)[0]
        SamplerCustom().sample(model, positive=positive, negative=negative,
                               sampler=sampler, sigmas=sigmas, latent_image=latent_image,
                               cfg=self.sampler_cfg.cfg, add_noise=True, noise_seed=self.seed)




@register_recipe
class LTXVRecipeBase(ModelRecipe):
    @classmethod
    def name(cls) -> str:
        return "ltxv_t2i"

    @classmethod
    def add_model_args(cls, parser):
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument(
            "--ckpt_path",
            help="Path to full checkpoint (includes diffusion model + text encoder + VAE)"
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

        if not args.t5_path:
            raise ValueError("--unet_path requires both --t5_path")

    def load_model(self) -> Tuple:
        logging.info(f"Loading full checkpoint from {self.args.ckpt_path}")
        model_patcher, clip, vae, _ = comfy.sd.load_checkpoint_guess_config(
            self.args.ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=None
        )

        model_options = {}
        clip_type = comfy.sd.CLIPType.LTXV

        clip_path = folder_paths.get_full_path_or_raise("text_encoders", self.args.t5_path)
        clip = comfy.sd.load_clip(
            ckpt_paths=[clip_path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type,
            model_options=model_options
        )

        if self.args.vae_path:
            vae_path = folder_paths.get_full_path_or_raise("vae", self.args.vae_path)
            sd = comfy.utils.load_torch_file(vae_path)
            vae = comfy.sd.VAE(sd=sd)
            vae.throw_exception_if_invalid()
        return model_patcher, clip, vae

    def create_calibration_pipeline(self, model_components):
        model_patcher, clip, vae = model_components

        return LTXVideoPipe(
            model=model_patcher,
            clip=clip,
            vae=vae,
            width=self.get_width(),
            height=self.get_height(),
            length=self.get_length(),
            seed=42,
            sampler_cfg=self.get_sampler_cfg(),
            device="cuda",
            custom_kwargs=self.get_custom_kwargs()
        )

    def get_width(self) -> int:
        return 768

    def get_height(self) -> int:
        return 512

    def get_length(self) -> int:
        return 97

    def get_default_calib_steps(self) -> int:
        return 64

    def get_inference_steps(self) -> int:
        return 30

    def get_sampler_cfg(self) -> SamplerCFG:
        return SamplerCFG(
            cfg=3.0,
            sampler_name="euler",
        )

    def get_custom_kwargs(self) -> dict:
        return {
            "frame_rate": 25.0,
            "max_shift": 2.05,
            "base_shift": 0.95,
            "stretch": True,
            "terminal": 0.1
        }

@register_recipe
class LTXVText2Video(LTXVRecipeBase):
    @classmethod
    def name(cls) -> str:
        return "ltxv_t2v"

    def get_dataset(self):
        logging.info("Loading KontextBench dataset...")
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

                logging.debug(f"Calibration step {i+1}: '{prompt_text[:50]}...'")
                try:
                    calib_pipeline(num_steps, prompt_text, negative_text)
                except Exception as e:
                    logging.warning(f"Calibration step {i+1} failed: {e}")

        return forward_loop

@register_recipe
class LTXVImg2Video(LTXVRecipeBase):
    @classmethod
    def name(cls) -> str:
        return "ltxv_i2v"

    def get_dataset(self):
        logging.info("Loading KontextBench dataset...")
        return KontextBenchDataLoader()

    def get_forward_loop(self, calib_pipeline, num_calib_steps) -> Callable:
        num_steps = self.get_inference_steps()
        dataloader = self.get_dataset()
        def forward_loop():
            for i in range(num_calib_steps):
                rnd_idx = random.randint(0, len(dataloader) - 1)
                sample = dataloader[rnd_idx]
                prompt_text = sample["prompt"]
                img = sample["image"]
                negative_text = "low quality"

                logging.debug(f"Calibration step {i+1}: '{prompt_text[:50]}...'")
                try:
                    calib_pipeline(num_steps, prompt_text, negative_text, img)
                except Exception as e:
                    logging.warning(f"Calibration step {i+1} failed: {e}")

        return forward_loop
