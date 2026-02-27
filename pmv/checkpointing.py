"""
Checkpoint helpers for PMV training/evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch


def _load_trusted_checkpoint(path: str):
    """
    Load internal PMV checkpoints that may contain Python objects (e.g., SolutionRecord).
    PyTorch >=2.6 defaults to weights_only=True, so explicitly disable it here.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Backward compatibility for older torch versions without weights_only.
        return torch.load(path, map_location="cpu")


def save_checkpoint(
    path: str,
    round_idx: int,
    ensemble,
    prover=None,
    config_path: Optional[str] = None,
    extra: Optional[dict] = None,
) -> str:
    payload = {
        "round_idx": int(round_idx),
        "config_path": config_path,
        "ensemble": ensemble.state_dict_checkpoint(),
    }
    if prover is not None:
        payload["prover"] = prover.state_dict_checkpoint()
    if extra:
        payload["extra"] = extra

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    return str(out)


def load_checkpoint(
    path: str,
    ensemble=None,
    prover=None,
    strict: bool = True,
):
    ckpt = _load_trusted_checkpoint(path)
    if ensemble is not None and "ensemble" in ckpt:
        ensemble.load_state_dict_checkpoint(ckpt["ensemble"], strict=strict)
    if prover is not None and "prover" in ckpt:
        prover.load_state_dict_checkpoint(ckpt["prover"], strict=strict)
    return ckpt


def save_prover_checkpoint(
    path: str,
    round_idx: int,
    prover,
    config_path: Optional[str] = None,
    extra: Optional[dict] = None,
) -> str:
    payload = {
        "round_idx": int(round_idx),
        "config_path": config_path,
        "prover": prover.state_dict_checkpoint(),
    }
    if extra:
        payload["extra"] = extra

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    return str(out)


def load_prover_checkpoint(
    path: str,
    prover=None,
    strict: bool = True,
):
    ckpt = _load_trusted_checkpoint(path)
    if prover is not None and "prover" in ckpt:
        prover.load_state_dict_checkpoint(ckpt["prover"], strict=strict)
    return ckpt
