"""Misc helpers"""

import torch
from datetime import datetime


def get_device():
    """Get device to use for PyTorch."""

    # Check if GPU is available
    has_gpu = torch.cuda.is_available()

    # Check if MPS is available
    has_mps = torch.backends.mps.is_available()

    # Use MPS if available, otherwise GPU if available, otherwise CPU
    device = "mps" if has_mps else "gpu" if has_gpu else "cpu"

    return device


def get_name(file_name):
    """Get a file name with timestamp."""
    if isinstance(file_name, list):
        file_name = "_".join(file_name)

    iso_format = datetime.now().isoformat(timespec='seconds').replace(":", "")
    return f"{file_name}_{iso_format}"
