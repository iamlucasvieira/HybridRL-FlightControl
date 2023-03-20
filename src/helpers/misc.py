"""Misc helpers"""

from datetime import datetime


def get_name(file_name):
    """Get a file name with timestamp."""
    if isinstance(file_name, list):
        file_name = "_".join(file_name)

    iso_format = datetime.now().isoformat(timespec="seconds").replace(":", "")
    return f"{file_name}_{iso_format}"


def verbose_print(message, verbosity, verbose_threshold=0):
    """Prints only if verbose is greater than threshold."""
    if verbosity > verbose_threshold:
        print(message)
