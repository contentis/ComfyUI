from abc import ABC, abstractmethod
import argparse
from typing import Tuple, Any, Callable
from dataclasses import dataclass


class ModelRecipe(ABC):
    @classmethod
    @abstractmethod
    def name(cls) -> str:
        pass

    @classmethod
    @abstractmethod
    def add_model_args(cls, parser: argparse.ArgumentParser):
        pass

    @abstractmethod
    def __init__(self, args):
        pass

    @abstractmethod
    def load_model(self) -> Tuple[Any, ...]:
        pass

    @abstractmethod
    def create_calibration_pipeline(self, model_components) -> Any:
        pass

    @abstractmethod
    def get_forward_loop(self, calib_pipeline, num_calib_steps) -> Callable:
        pass

    @abstractmethod
    def get_default_calib_steps(self) -> int:
        pass

@dataclass
class SamplerCFG:
    cfg: float
    sampler_name: str
    scheduler: str
    denoise: float
    flux_cfg: float
    img_cfg: float = 2.0
