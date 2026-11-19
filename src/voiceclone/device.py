"""Torch device and dtype selection."""


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def default_dtype(device: str):
    import torch

    return torch.bfloat16 if device != "cpu" else torch.float32
