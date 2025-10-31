from typing import Dict, Type
from .base import ModelRecipe


_RECIPE_REGISTRY: Dict[str, Type[ModelRecipe]] = {}


def register_recipe(recipe_cls: Type[ModelRecipe]):
    recipe_name = recipe_cls.name()
    if recipe_name in _RECIPE_REGISTRY:
        raise ValueError(f"Recipe '{recipe_name}' is already registered")

    _RECIPE_REGISTRY[recipe_name] = recipe_cls
    return recipe_cls


def get_recipe_class(name: str) -> Type[ModelRecipe]:
    if name not in _RECIPE_REGISTRY:
        available = ", ".join(sorted(_RECIPE_REGISTRY.keys()))
        raise ValueError(
            f"Unknown model type '{name}'. "
            f"Available recipes: {available}"
        )
    return _RECIPE_REGISTRY[name]


def list_recipes():
    return sorted(_RECIPE_REGISTRY.keys())


# Import recipe modules to trigger registration
from . import flux  # noqa: F401, E402
from . import qwen # noqa: F401, E402
from . import ltx_video # noqa: F401, E402



