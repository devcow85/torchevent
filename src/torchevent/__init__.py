import os
import pathlib

# Get TORCHEVENT_CACHE from environment variables or use fallback directory
torchevent_cache = os.environ.get("TORCHEVENT_CACHE") or os.path.join(pathlib.Path.home(), ".cache", "torchevent")
os.environ["TORCHEVENT_CACHE"] = torchevent_cache


if not os.path.exists(torchevent_cache):
    os.makedirs(torchevent_cache, exist_ok=True)

from . import models, artifacts, layers, loss, metrics, transforms, utils

__all__ = [
    "models", 
    "artifacts",
    "layers",
    "loss",
    "metrics", 
    "transforms",
    "utils"
]