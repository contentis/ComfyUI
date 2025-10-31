import torch
from typing import Dict

import argparse
import logging
import sys
import torch.utils.data
import modelopt.torch.quantization as mtq
from tools.ptq.utils import log_quant_summary, save_amax_dict, extract_amax_values

import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tools.ptq.models import get_recipe_class, list_recipes
from tools.ptq.utils import register_comfy_ops, FP8_CFG

class PTQPipeline:
    def __init__(self, model_patcher, quant_config: dict):
        self.model_patcher = model_patcher
        self.diffusion_model = model_patcher.model.diffusion_model
        self.quant_config = quant_config

        logging.debug(f"PTQPipeline initialized with config: {quant_config}")

    @torch.no_grad()
    def calibrate_with_pipeline(
        self,
        calib_pipeline,
        num_steps: int,
        recipe
    ):

        logging.info(f"Running calibration with {num_steps} steps...")
        forward_loop = recipe.get_forward_loop(calib_pipeline, num_steps)
        try:
            mtq.quantize(self.diffusion_model, self.quant_config, forward_loop=forward_loop)
        except Exception as e:
            logging.error(f"Calibration failed: {e}")
            raise

        logging.info("Calibration complete")
        log_quant_summary(self.diffusion_model)

    def get_amax_dict(self) -> Dict:
        return extract_amax_values(self.diffusion_model)

    def save_amax_values(self, output_path: str, metadata: dict = None):
        amax_dict = self.get_amax_dict()
        save_amax_dict(amax_dict, output_path, metadata=metadata)
        logging.info(f"Saved amax values to {output_path}")



def main():
    parser = argparse.ArgumentParser(
        description="Quantize ComfyUI models using NVIDIA ModelOptimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model_type",
        required=True,
        choices=list_recipes(),
        help="Model recipe to use"
    )

    args, remaining = parser.parse_known_args()

    recipe_cls = get_recipe_class(args.model_type)
    recipe_cls.add_model_args(parser)
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for amax artefact"
    )
    parser.add_argument(
        "--calib_steps",
        type=int,
        help="Override default calibration steps"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (sets logging to DEBUG and calib_steps to 1)"
    )

    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format='[%(levelname)s] %(name)s: %(message)s'
        )
        logging.info("Debug mode enabled")
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='[%(levelname)s] %(message)s'
        )
    try:
        recipe = recipe_cls(args)
    except Exception as e:
        logging.error(f"Failed to initialize recipe: {e}")
        sys.exit(1)
    if args.debug:
        calib_steps = 1
        logging.debug("Debug mode: forcing calib_steps=1")
    elif args.calib_steps:
        calib_steps = args.calib_steps
    else:
        calib_steps = recipe.get_default_calib_steps()

    logging.info("Registering ComfyUI ops with ModelOptimizer...")
    register_comfy_ops()

    logging.info("[1/5] Loading model...")
    try:
        model_components = recipe.load_model()
        model_patcher = model_components[0]
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        sys.exit(1)

    logging.info("[2/5] Preparing quantization...")
    try:
        pipeline = PTQPipeline(
            model_patcher,
            quant_config=FP8_CFG,
        )
    except Exception as e:
        logging.error(f"Failed to prepare quantization: {e}")
        sys.exit(1)

    logging.info("[3/5] Creating calibration pipeline...")
    try:
        calib_pipeline = recipe.create_calibration_pipeline(model_components)
    except Exception as e:
        logging.error(f"Failed to create calibration pipeline: {e}")
        sys.exit(1)

    logging.info(f"[4/5] Running calibration ({calib_steps} steps)...")
    try:
        pipeline.calibrate_with_pipeline(
            calib_pipeline,
            num_steps=calib_steps,
            recipe=recipe
        )
    except Exception as e:
        logging.error(f"Calibration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    logging.info("[5/5] Extracting and saving amax values...")
    try:
        metadata = {
            "model_type": recipe.name(),
            "calibration_steps": calib_steps,
            "quantization_format": "amax",
            "debug_mode": args.debug
        }

        if hasattr(args, 'ckpt_path') and args.ckpt_path:
            metadata["checkpoint_path"] = args.ckpt_path

        pipeline.save_amax_values(args.output, metadata=metadata)
    except Exception as e:
        logging.error(f"Failed to save amax values: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

