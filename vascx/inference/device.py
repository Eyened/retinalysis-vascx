import torch


def auto_device() -> torch.device:
    """Pick the best available PyTorch device for inference."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve an explicit device string or pick one automatically."""
    if device is None:
        return auto_device()
    if isinstance(device, torch.device):
        return device
    return torch.device(device)
