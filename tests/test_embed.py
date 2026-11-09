"""Exactness tests: anndataoom_fast.pacmap / localmap vs brisc.

brisc is not a test dependency, so its embeddings for a few deterministic
inputs are stored in tests/data/brisc_embed_fixtures.npz (regenerate with
tests/data/make_embed_fixtures.py in an environment that has brisc).
Embeddings must be identical element for element, for any thread count.

PaCMAP matches brisc on every machine.  brisc's LocalMAP starts a Newton step
from the CPU's RSQRTSS estimate, whose bits depend on the CPU; ours starts
from 1/sqrt by default, so LocalMAP is compared with our own machine-
independent outputs (tests/data/localmap_portable.npz).  With
AOF_EMBED_BRISC_RSQRT=1 on the x86 CPU the brisc fixtures were made on,
set AOF_TEST_BRISC_CPU=1 to check LocalMAP against brisc as well.
"""
import importlib.util
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import anndataoom_fast as aof

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("make_embed_fixtures",
                                               HERE / "data" / "make_embed_fixtures.py")
mk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mk)

FIXTURES = np.load(HERE / "data" / "brisc_embed_fixtures.npz")
PORTABLE = np.load(HERE / "data" / "localmap_portable.npz")
X86 = platform.machine().lower() in ("x86_64", "amd64", "i386", "i686", "x86")

# brisc keyword -> ours
_KW = {"num_neighbors": "n_neighbors", "num_extra_neighbors": "n_extra_neighbors",
       "num_mid_near_pairs": "n_mid_near_pairs", "num_further_pairs": "n_further_pairs",
       "num_iterations": "num_iterations", "learning_rate": "learning_rate",
       "max_distance": "max_distance", "seed": "random_state"}
_INPUTS = {}


def inputs(name):
    if name not in _INPUTS:
        n, d, k, _, _ = mk.CASES[name]
        pcs = mk.pcs_from_int8(FIXTURES[f"{name}_pcs_int8"])
        assert pcs.shape == (n, d)
        _INPUTS[name] = (pcs,) + mk.exact_knn(pcs, k)
    return _INPUTS[name]


def kwargs(name, meth):
    kw = dict(mk.CASES[name][4])
    if meth == "localmap":
        kw.update(mk.LOCALMAP_ONLY.get(name, {}))
    return {_KW[k]: v for k, v in kw.items()}


def run(name, meth, n_threads):
    pcs, nbr, dist = inputs(name)
    return getattr(aof, meth)(pcs, knn_indices=nbr, knn_dists=dist, n_threads=n_threads,
                              **kwargs(name, meth))


@pytest.mark.parametrize("name", sorted(mk.CASES))
@pytest.mark.parametrize("meth", ["pacmap", "localmap"])
def test_matches_brisc(name, meth):
    ref = expected(name, meth)
    out = run(name, meth, 4)
    assert out.dtype == np.float32 and out.shape == ref.shape
    assert np.array_equal(out, ref)


def expected(name, meth):
    if meth == "localmap":
        return PORTABLE[f"{name}_localmap"]
    return FIXTURES[f"{name}_{meth}"]


@pytest.mark.skipif(not (X86 and os.environ.get("AOF_TEST_BRISC_CPU") == "1"),
                    reason="needs the x86 CPU the brisc fixtures were made on")
def test_localmap_matches_brisc_with_rsqrtss():
    code = (
        "import sys, numpy as np\n"
        f"sys.path.insert(0, {str(HERE)!r})\n"
        "import test_embed as t\n"
        "for name in sorted(t.mk.CASES):\n"
        "    assert np.array_equal(t.run(name, 'localmap', 3), t.FIXTURES[f'{name}_localmap']), name\n"
        "print('ok')\n")
    env = dict(os.environ, AOF_EMBED_BRISC_RSQRT="1")
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-2000:]


@pytest.mark.parametrize("meth", ["pacmap", "localmap"])
def test_thread_count_independent(meth):
    ref = run("a", meth, 1)
    for nt in (2, 4, 7, 20):
        assert np.array_equal(run("a", meth, nt), ref)


def test_generic_kernels_identical():
    """The portable kernels (no hardware FMA) give the same bits."""
    code = (
        "import sys, numpy as np, anndataoom_fast as aof\n"
        f"sys.path.insert(0, {str(HERE)!r})\n"
        "import test_embed as t\n"
        "for name in sorted(t.mk.CASES):\n"
        "    for meth in ('pacmap', 'localmap'):\n"
        "        out = t.run(name, meth, 3)\n"
        "        assert np.array_equal(out, t.expected(name, meth)), (name, meth)\n"
        "print('ok')\n")
    if not X86:
        pytest.skip("generic kernels are the only kernels off x86")
    env = dict(os.environ, AOF_EMBED_GENERIC="1")
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-2000:]


def test_pairs_and_defaults():
    pcs, nbr, dist = inputs("a")
    out, pairs = aof.localmap(pcs, knn_indices=nbr, knn_dists=dist, return_pairs=True)
    assert np.array_equal(out, PORTABLE["a_localmap"])
    n = len(pcs)
    assert pairs["neighbor_pairs"].shape == (n, 10)
    assert pairs["mid_near_pairs"].shape == (n, 5)
    assert pairs["further_pairs"].shape == (n, 20)
    rows = np.arange(n)[:, None]
    for key in pairs:
        assert (pairs[key] != rows).all()
        assert (pairs[key] < n).all()
    # neighbour pairs are a subset of the kNN
    assert all(set(pairs["neighbor_pairs"][i]) <= set(nbr[i]) for i in range(0, n, 97))
    # a leading self column (scanpy convention) is dropped
    nbr_self = np.hstack([np.arange(n, dtype=np.uint32)[:, None], nbr])
    dist_self = np.hstack([np.zeros((n, 1), np.float32), dist])
    assert np.array_equal(aof.pacmap(pcs, knn_indices=nbr_self, knn_dists=dist_self),
                          FIXTURES["a_pacmap"])


def test_without_knn_runs():
    pcs = inputs("c")[0]
    a = aof.pacmap(pcs, n_threads=2, num_iterations=20)
    b = aof.pacmap(pcs, n_threads=5, num_iterations=20)
    assert a.shape == (len(pcs), 2) and np.isfinite(a).all()
    assert np.array_equal(a, b)


def test_errors():
    pcs, nbr, dist = inputs("c")
    bad = nbr.copy()
    bad[5, 3] = len(pcs) + 3
    with pytest.raises(ValueError):
        aof.pacmap(pcs, knn_indices=bad, knn_dists=dist)
    bad = nbr.copy()
    bad[5, 3] = 5
    with pytest.raises(ValueError):
        aof.pacmap(pcs, knn_indices=bad, knn_dists=dist)
    with pytest.raises(ValueError):
        aof.pacmap(pcs, knn_indices=nbr[:, :12], knn_dists=dist[:, :12])
    with pytest.raises(ValueError):
        aof.localmap(pcs[:, :1])
