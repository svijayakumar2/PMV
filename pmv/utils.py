"""
Device management and memory utilities.
"""

import gc
import torch
from typing import List, Optional

_CLEANUP_CALL_COUNT = 0


def get_available_gpus() -> List[int]:
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


def get_device_for_index(idx: int, num_gpus: int) -> str:
    if num_gpus == 0:
        return "cpu"
    return f"cuda:{idx % num_gpus}"


def cleanup_memory(
    device: Optional[str] = None,
    aggressive: bool = False,
    sync: bool = False,
    empty_cache_every: int = 25,
):
    """
    Lightweight cleanup by default to avoid hot-path stalls.
    - Non-aggressive mode: only empties CUDA cache every N calls.
    - Aggressive mode: always empties cache (and optionally synchronizes).
    """
    global _CLEANUP_CALL_COUNT
    gc.collect()
    if not torch.cuda.is_available():
        return
    _CLEANUP_CALL_COUNT += 1
    if not aggressive:
        interval = max(1, int(empty_cache_every))
        if (_CLEANUP_CALL_COUNT % interval) != 0:
            return

    gpu_indices = []
    if device and device.startswith("cuda"):
        gpu_idx = int(device.split(":")[1]) if ":" in device else 0
        gpu_indices = [gpu_idx]
    else:
        gpu_indices = list(range(torch.cuda.device_count()))

    for i in gpu_indices:
        with torch.cuda.device(i):
            if sync:
                torch.cuda.synchronize()
            torch.cuda.empty_cache()


def delete_model(model, device: Optional[str] = None):
    if model is None:
        return
    try:
        for param in model.parameters():
            param.data = torch.empty(0, device=param.device)
            if param.grad is not None:
                param.grad = None
    except Exception:
        pass
    del model
    cleanup_memory(device=device, aggressive=True, sync=False)
