"""Module that defines the DSAC policy."""

from agents import BasePolicy

import torch as th

class DSACPolicy(BasePolicy):
    """DSAC policy."""

    def __init__(self):
        """Initialize DSAC policy."""
        super(DSACPolicy, self).__init__()