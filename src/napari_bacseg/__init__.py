try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

__all__ = "BacSeg"

from napari_bacseg._widget import BacSeg