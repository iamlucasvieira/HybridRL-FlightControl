"""Misc helpers"""

import torch

def get_device():
    """Get device to use for PyTorch."""

    # Check if GPU is available
    has_gpu = torch.cuda.is_available()

    # Check if MPS is available
    has_mps = torch.backends.mps.is_available()

    # Use MPS if available, otherwise GPU if available, otherwise CPU
    device = "mps" if has_mps else "gpu" if has_gpu else "cpu"

    return "cpu"
