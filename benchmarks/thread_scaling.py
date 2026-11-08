"""Time every C++ pass at several thread counts and check the outputs never change.

    python thread_scaling.py COUNTS.h5ad [threads ...]

The file is read once beforehand so all runs see a warm page cache.
"""
import json
import sys
import time

import numpy as np

import anndataoom_fast as aof
from anndataoom_fast import _core
from anndataoom_fast._svd import randomized_svd


def timed(fn):
    t = time.perf_counter()
    out = fn()
    return time.perf_counter() - t, out


def run(path, threads):
    h = aof.open_csr(path)
    n, g = h.shape
    import h5py
    with h5py.File(path) as f:
        names = f["var"][f["var"].attrs["_index"]][:].astype(str)
    gg = np.zeros(g, np.uint8)
    gg[np.char.startswith(names, "MT-")] |= 1
    gg[np.char.startswith(names, "RPS") | np.char.startswith(names, "RPL")] |= 2
    rules = [(2, 0, 0.15), (0, 2, 500.0), (1, 2, 250.0)]  # ov.pp.qc seurat defaults
    results, ref = {}, None
    for nt in threads:
        aof.set_num_threads(nt)
        t = {}
        t["qc"], qc = timed(lambda: _core.qc_pass(h.src, gg, rules))
        keep = qc["keep"]
        genes = (qc["g_stored"] >= 3) & (qc["g_nonzero"] / keep.sum() * 100 >= 0.05)
        ck = genes.astype(np.uint8)
        t["hi"], hi = timed(lambda: _core.hi_pass(h.src, keep, ck, 0.2))
        cmap = np.full(g, -1, np.int32)
        cmap[genes] = np.arange(genes.sum(), dtype=np.int32)
        excl = ((hi["hi_count"] > 0) & genes).astype(np.uint8)
        sg = np.ascontiguousarray(hi["g_sum"][genes])
        clip = float(np.sqrt(keep.sum()))
        t["stats"], st = timed(lambda: _core.stats_pass(h.src, keep, cmap, excl, sg, 5e5, True,
                                                        True, True, 100.0, clip, float(sg.sum())))
        cps = st["cps"][keep.astype(bool)].astype(np.float64)
        su, m = np.unique(cps, return_counts=True)
        t["pearson_zero"], (z1, z2) = timed(lambda: _core.pearson_zero_terms(
            su, m.astype(np.float64), sg, float(sg.sum()), 100.0, clip))
        N = keep.sum()
        rv = (z2 + st["d2"]) / N - ((z1 + st["d1"]) / N) ** 2
        hvg = np.argsort(-rv, kind="stable")[:2000]
        mean = st["s1"] / N
        std = np.sqrt(np.clip(st["s2"] - N * mean ** 2, 0, None) / (N - 1))
        std[std == 0] = 1
        sel = np.sort(hvg)
        gidx = np.where(genes)[0][sel]
        hmap = np.full(g, -1, np.int32)
        hmap[gidx] = np.arange(len(gidx), dtype=np.int32)
        t["extract"], op = timed(lambda: _core.extract_scaled(
            h.src, keep, hmap, np.ascontiguousarray(mean[sel]), np.ascontiguousarray(std[sel]),
            st["recip"], True, 10.0))
        t["svd"], (U, S, Vt) = timed(lambda: randomized_svd(op, 60, n_iter=4, random_state=0))
        t["total"] = sum(t.values())
        sig = {"qc": qc, "hi": hi, "stats": st, "z": (z1, z2), "S": S, "U": U}
        if ref is None:
            ref = sig
            same = True
        else:
            same = all(np.array_equal(sig[k][j], ref[k][j]) for k in ("qc", "hi", "stats")
                       for j in sig[k]) and np.array_equal(sig["S"], ref["S"]) \
                and np.array_equal(sig["U"], ref["U"]) and all(
                    np.array_equal(a, b) for a, b in zip(sig["z"], ref["z"]))
        t["identical_to_first"] = bool(same)
        results[nt] = t
        print(nt, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in t.items()},
              flush=True)
    return {"shape": [int(n), int(g)], "nnz": int(h.indptr[-1]), "kept_cells": int(N),
            "kept_genes": int(genes.sum()), "hvg_nnz": int(op.nnz), "results": results}


if __name__ == "__main__":
    path = sys.argv[1]
    threads = [int(x) for x in sys.argv[2:]] or [1, 2, 4, 8, 16, 20]
    out = run(path, threads)
    print(json.dumps(out, indent=1))
