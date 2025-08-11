"""Top-level package for node importance analysis and diffusion models."""

from .network import Network
from .diffusion import IndependentCascade, WeightedCascade

__all__ = ["Network", "IndependentCascade", "WeightedCascade"]

__version__ = "0.1"
