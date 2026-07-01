"""MLX-native training and inference for Cascadia AI v2."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cascadia-mlx")
except PackageNotFoundError:
    # Immutable cluster source freezes execute directly without installation.
    __version__ = "0.1.0"
