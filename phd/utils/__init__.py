from importlib import import_module

from . import geometry

__all__ = ["geometry", "kps", "Renderer"]


def __getattr__(name):
    if name == "kps":
        return import_module(f"{__name__}.kps")
    if name == "Renderer":
        from .renderer import Renderer

        return Renderer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
