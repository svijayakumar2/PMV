"""
Device management and memory utilities.
"""

import gc
import torch
from typing import List, Optional


def get_available_gpus() -> List[int]:
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


def get_device_for_index(idx: int, num_gpus: int) -> str:
    if num_gpus == 0:
        return "cpu"
    return f"cuda:{idx % num_gpus}"


def cleanup_memory(device: Optional[str] = None):
    gc.collect()
    if not torch.cuda.is_available():
        return
    if device and device.startswith("cuda"):
        gpu_idx = int(device.split(":")[1]) if ":" in device else 0
        with torch.cuda.device(gpu_idx):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    else:
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
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
    cleanup_memory(device)
