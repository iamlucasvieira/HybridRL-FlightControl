"""Module that defines configuration items that can be automatically set by core."""

ALLOWS_AUTO = ["env", "verbose", "tensorboard_log", "seed"]


def get_auto(item: str) -> str:
    """Returns a string, which the experiment builder replaces with the right variable"""
    if item in ALLOWS_AUTO:
        return f"auto.{item}"
    else:
        raise ValueError(f"Item {item} is not allowed to be set automatically.")


def validate_auto(value: str) -> bool:
    """Validates that an auto item is valid"""
    if value.startswith("auto."):
        item = value.split(".")[-1]
        if item in ALLOWS_AUTO:
            return True
    return False
