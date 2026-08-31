"""GPU / device management for the tokenizer pipeline."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_TORCH_AVAILABLE = False
_torch: Any = None

try:
    import torch as _torch_module

    _torch = _torch_module
    _TORCH_AVAILABLE = True
except ImportError:
    pass


def torch_available() -> bool:
    return _TORCH_AVAILABLE


def cuda_available() -> bool:
    return _TORCH_AVAILABLE and _torch.cuda.is_available()


def mps_available() -> bool:
    return _TORCH_AVAILABLE and hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available()


def resolve_device(preference: str = "auto") -> str:
    """Resolve a device string from preference: auto, cpu, cuda, or mps."""
    pref = preference.lower().strip()
    if pref == "cpu" or not _TORCH_AVAILABLE:
        return "cpu"
    if pref == "cuda":
        if cuda_available():
            return "cuda"
        logger.warning("CUDA requested but unavailable; falling back to CPU")
        return "cpu"
    if pref == "mps":
        if mps_available():
            return "mps"
        logger.warning("MPS requested but unavailable; falling back to CPU")
        return "cpu"
    # auto
    if cuda_available():
        return "cuda"
    if mps_available():
        return "mps"
    return "cpu"


def get_torch_device(preference: str = "auto"):
    """Return a torch.device for the resolved preference."""
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed. Install with: pip install torch")
    return _torch.device(resolve_device(preference))


def device_info(preference: str = "auto") -> dict[str, Any]:
    """Return a dict describing the resolved device and hardware."""
    device = resolve_device(preference)
    info: dict[str, Any] = {
        "torch_available": _TORCH_AVAILABLE,
        "resolved_device": device,
        "preference": preference,
        "cuda_available": cuda_available(),
        "mps_available": mps_available(),
    }
    if device == "cuda" and _TORCH_AVAILABLE:
        idx = _torch.cuda.current_device()
        info["gpu_name"] = _torch.cuda.get_device_name(idx)
        props = _torch.cuda.get_device_properties(idx)
        info["gpu_memory_gb"] = round(props.total_memory / (1024**3), 2)
    return info


def log_device_info(step_logger: logging.Logger, preference: str = "auto") -> str:
    """Log device details and return the resolved device string."""
    info = device_info(preference)
    step_logger.info(
        "Device: %s (preference=%s, torch=%s, cuda=%s, mps=%s)",
        info["resolved_device"],
        info["preference"],
        info["torch_available"],
        info["cuda_available"],
        info["mps_available"],
    )
    if "gpu_name" in info:
        step_logger.info("GPU: %s (%.1f GB)", info["gpu_name"], info["gpu_memory_gb"])
    return info["resolved_device"]
