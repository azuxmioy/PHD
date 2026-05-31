"""Self-contained PyTorch OpenPose-135 detector (BODY_25 + 2 hands + 70-pt face).

Vendored / adapted from:
- BODY_25 model + decoder : https://github.com/TracelessLe/OpenPose.PyTorch  (OpenPose license, non-commercial)
- hand / face inference   : https://github.com/lllyasviel/ControlNet-v1-1-nightly  (Apache 2.0 wrapper around CMU weights)
- BODY_25 prototxt        : https://github.com/CMU-Perceptual-Computing-Lab/openpose  (non-commercial)

The CMU OpenPose model weights are licensed for **non-commercial use only**.
"""
__all__ = ["OpenPose135Detector", "write_openpose_json"]


def __getattr__(name):
    # Lazy-load detector.py (which pulls cv2/torch) so utility scripts like
    # `python -m tools.openpose135._fetch_weights` don't require the full
    # inference stack to be importable.
    if name in __all__:
        from . import detector
        return getattr(detector, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
