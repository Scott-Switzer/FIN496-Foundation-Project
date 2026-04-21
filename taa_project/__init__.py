"""Whitmore FIN 496 TAA package."""

import os as _os

_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")

__all__ = ["__version__"]

__version__ = "0.1.0"
