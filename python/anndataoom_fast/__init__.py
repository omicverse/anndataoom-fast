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
from ._neighbors import knn, fuzzy_simplicial_set, IVFKNNTransformer
from ._umap import umap
from ._markers import cosg, group_stats

try:
    __version__ = _version("anndataoom-fast")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"


def set_num_threads(n: int) -> None:
    """Number of threads used by every C++ pass."""
    _core.set_num_threads(int(n))


def get_num_threads() -> int:
    return _core.get_num_threads()


def install(omicverse: bool = False) -> None:
    """Route anndataoom's chunked preprocessing functions through the C++ engine.

    With ``omicverse=True`` also route omicverse's kNN, UMAP connectivities and
    layout, Leiden and COSG markers (see :mod:`anndataoom_fast._omicverse`).
    Idempotent.  Unsupported inputs still run the original code.
    """
    if omicverse:
        from . import _omicverse
        _omicverse.install()
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
    """Restore anndataoom's (and omicverse's) original functions."""
    from . import _omicverse
    _omicverse.uninstall()
    import anndataoom
    from anndataoom import _chunked_ops
    from . import _ops
    for name, orig in _ops._ORIG.items():
        setattr(_chunked_ops, name, orig)
        if hasattr(anndataoom, name):
            setattr(anndataoom, name, orig)


try:
    from ._leiden import leiden, leiden_multi
except ImportError:  # the _leiden extension is optional while it is being built
    pass

__all__ = ["install", "uninstall", "set_num_threads", "get_num_threads", "enable", "disable",
           "enabled", "open_csr", "Unsupported", "STATS", "knn", "fuzzy_simplicial_set", "umap",
           "cosg", "group_stats", "IVFKNNTransformer", "leiden", "leiden_multi", "__version__"]
