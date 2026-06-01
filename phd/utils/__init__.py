from importlib import import_module

__all__ = ["assets", "geometry", "image", "keypoints", "kps", "modeling", "surface", "Renderer"]


def __getattr__(name):
    if name in {"assets", "geometry", "image", "keypoints", "kps", "modeling", "surface"}:
        return import_module(f"{__name__}.{name}")
    if name == "Renderer":
        from .renderer import Renderer

        return Renderer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
