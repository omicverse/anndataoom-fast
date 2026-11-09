"""Regenerate tests/data/brisc_embed_fixtures.npz with brisc itself.

Run with a Python environment where brisc is installed, e.g.
    python tests/data/make_embed_fixtures.py
For each case the input PCs (Gaussian blobs, stored as int8 and scaled by
1/4 so every squared distance is exact in float32) and an exact kNN graph are
given to brisc.SingleCell.pacmap() / .localmap(); brisc's embeddings with
num_threads=4 are stored, after checking that num_threads=20 and
num_threads=1, match_parallel=True give the same bits.
"""
from pathlib import Path

import numpy as np

# name: (n_cells, n_pcs, n_knn_columns, data seed, brisc keyword arguments)
CASES = {
    "a": (3000, 50, 20, 0, {}),
    "b": (2000, 17, 20, 1, dict(num_neighbors=8, num_extra_neighbors=12, num_mid_near_pairs=4,
                               num_further_pairs=15, num_iterations=(40, 50, 110),
                               learning_rate=0.7, seed=12345)),
    "c": (1500, 10, 25, 2, dict(num_iterations=150, seed=3)),
}
LOCALMAP_ONLY = {"b": dict(max_distance=6.0)}


def make_pcs_int8(n, d, seed, n_blobs=8):
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 20, (n_blobs, d))
    labels = rng.integers(0, n_blobs, n)
    x = centres[labels] + rng.normal(0, 4, (n, d))
    return np.clip(np.rint(x), -127, 127).astype(np.int8)


def pcs_from_int8(q):
    return (q.astype(np.float32) / np.float32(4))


def exact_knn(pcs, k):
    """k nearest neighbours (excluding self), ties broken by index; squared distances.

    The PCs are multiples of 1/4 with |x| <= 32, so every squared distance is a
    multiple of 1/16 below 2^20 and exact in float32."""
    x = pcs.astype(np.float64)
    sq = (x * x).sum(1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (x @ x.T)
    d2 = np.rint(d2 * 16.0) / 16.0  # exact values (removes cancellation noise)
    np.fill_diagonal(d2, np.inf)
    order = np.lexsort((np.broadcast_to(np.arange(len(x)), d2.shape), d2), axis=1)[:, :k]
    dist = np.take_along_axis(d2, order, 1)
    return order.astype(np.uint32), dist.astype(np.float32)


def main():
    import polars as pl
    import scipy.sparse as sp
    from brisc import SingleCell

    out = {}
    for name, (n, d, k, seed, kw) in CASES.items():
        q = make_pcs_int8(n, d, seed)
        pcs = pcs_from_int8(q)
        nbr, dist = exact_knn(pcs, k)
        sc = SingleCell(X=sp.csr_array((n, 1), dtype=np.float32),
                        obs=pl.DataFrame({"cell": [f"c{i}" for i in range(n)]}),
                        var=pl.DataFrame({"gene": ["g"]}),
                        obsm={"pca": pcs, "neighbors": nbr, "distances": dist}, num_threads=4)
        out[f"{name}_pcs_int8"] = q
        for meth in ("pacmap", "localmap"):
            args = dict(kw, **(LOCALMAP_ONLY.get(name, {}) if meth == "localmap" else {}))
            emb = [getattr(sc, meth)(num_threads=t, **args).obsm[meth] for t in (4, 20)]
            emb.append(getattr(sc, meth)(num_threads=1, match_parallel=True, **args).obsm[meth])
            assert all(np.array_equal(emb[0], e) for e in emb[1:])
            out[f"{name}_{meth}"] = emb[0]
        print(name, "done")
    np.savez_compressed(Path(__file__).resolve().parent / "brisc_embed_fixtures.npz", **out)


if __name__ == "__main__":
    main()
