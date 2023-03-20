import platform
import sys

import pandas as pd
import sklearn as sk
import torch


REQUIRED_PYTHON = "python3"


def main():
    system_major = sys.version_info.major
    if REQUIRED_PYTHON == "python":
        required_major = 2
    elif REQUIRED_PYTHON == "python3":
        required_major = 3
    else:
        raise ValueError(f"Unrecognized python interpreter: {REQUIRED_PYTHON}")

    if system_major != required_major:
        raise TypeError(
            "This project requires Python {}. Found: Python {}".format(
                required_major, sys.version
            )
        )
    else:
        print(">>> Development environment passes all tests!")

    has_gpu = torch.cuda.is_available()
    has_mps = getattr(torch, "has_mps", False)
    device = (
        "mps"
        if getattr(torch, "has_mps", False)
        else "gpu"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Python Platform: {platform.platform()}")
    print(f"PyTorch Version: {torch.__version__}")
    print()
    print(f"Python {sys.version}")
    print(f"Pandas {pd.__version__}")
    print(f"Scikit-Learn {sk.__version__}")
    print("GPU is", "available" if has_gpu else "NOT AVAILABLE")
    print("MPS (Apple Metal) is", "AVAILABLE" if has_mps else "NOT AVAILABLE")
    print(f"Target device is {device}")


if __name__ == "__main__":
    main()
