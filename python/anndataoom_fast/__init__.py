"""anndataoom-fast: multithreaded C++ kernels for anndataoom.

    import anndataoom_fast
    anndataoom_fast.install()          # accelerate anndataoom.chunked_* in place
    anndataoom_fast.set_num_threads(16)

Results do not depend on the number of threads: every reduction runs over
fixed blocks merged in a fixed order.
"""
from importlib.metadata import PackageNotFoundError, version as _version

from . import _core
from ._config import disable, enable, enabled
from ._h5 import Unsupported, open_csr
from ._ops import STATS, PATCHED

try:
    __version__ = _version("anndataoom-fast")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"


def set_num_threads(n: int) -> None:
    """Number of threads used by every C++ pass."""
    _core.set_num_threads(int(n))


def get_num_threads() -> int:
    return _core.get_num_threads()


def install() -> None:
    """Route anndataoom's chunked preprocessing functions through the C++ engine.

    Idempotent.  Unsupported inputs still run anndataoom's original code.
    """
    import anndataoom
    from anndataoom import _chunked_ops
    from . import _ops
    for name in PATCHED:
        if name not in _ops._ORIG:
            _ops._ORIG[name] = getattr(_chunked_ops, name)
        fast = getattr(_ops, name)
        setattr(_chunked_ops, name, fast)
        if hasattr(anndataoom, name):
            setattr(anndataoom, name, fast)


def uninstall() -> None:
    """Restore anndataoom's original functions."""
    import anndataoom
    from anndataoom import _chunked_ops
    from . import _ops
    for name, orig in _ops._ORIG.items():
        setattr(_chunked_ops, name, orig)
        if hasattr(anndataoom, name):
            setattr(anndataoom, name, orig)


__all__ = ["install", "uninstall", "set_num_threads", "get_num_threads", "enable", "disable",
           "enabled", "open_csr", "Unsupported", "STATS", "__version__"]
