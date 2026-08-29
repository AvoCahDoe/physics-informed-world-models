"""Device selection shared by training and evaluation."""

import torch


def get_device(override: str | None = None) -> torch.device:
    if override:
        return torch.device(override)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device)})"
    # A silent CPU fallback would quietly make training 50x slower, so say why.
    reason = "no CUDA build" if not torch.cuda.is_available() else "requested"
    return f"cpu ({reason})"


def log_device(device: torch.device) -> None:
    print(f"[device] using {describe_device(device)}", flush=True)
