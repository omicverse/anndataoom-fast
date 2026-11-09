"""Regenerate tests/data/brisc_leiden_fixtures.npz with brisc itself.

Run with a Python environment where brisc is installed, e.g.
    python tests/data/make_brisc_fixtures.py
Labels come from brisc.leiden.leiden (the routine behind
brisc.SingleCell.cluster), seed 0.  Each run is also checked against
brisc.leiden.leiden_multiresolution.
"""
import importlib
import importlib.util
from pathlib import Path

import numpy as np

here = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_brisc_graphs", here.parent / "_brisc_graphs.py")
bg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bg)


def brisc_leiden_module():
    for name in ("brisc.x86_64_v4.leiden", "brisc.x86_64_v3.leiden", "brisc.x86_64_v2.leiden",
                 "brisc.leiden"):
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    raise ImportError("brisc is not installed")


def main():
    lm = brisc_leiden_module()
    out = {}
    for gi, params in enumerate(bg.GRAPHS):
        n = params[0]
        G, nbr = bg.make_graph(*params)
        out[f"digest_{gi}"] = np.array(bg.graph_digest(G, nbr))
        for m in bg.MIN_CLUSTER_SIZES:
            nb = nbr if m > 1 else np.array([[]], dtype=np.uint32)
            multi = np.empty((len(bg.RESOLUTIONS), n), dtype=np.uint32)
            nfin = np.empty(len(bg.RESOLUTIONS), dtype=np.uint32)
            lm.leiden_multiresolution(G.data, G.indices, G.indptr, nb, multi, nfin,
                                      np.array(bg.RESOLUTIONS, dtype=np.float32), m, 0,
                                      len(bg.RESOLUTIONS))
            for ri, r in enumerate(bg.RESOLUTIONS):
                lab = np.empty(n, dtype=np.uint32)
                nf, too_large, selfn = lm.leiden(G.data, G.indices, G.indptr, nb, lab, r, m, 0,
                                                 False)
                assert not too_large and not selfn
                assert (lab == multi[ri]).all() and nf == nfin[ri]
                assert lab.max() < 65536
                out[f"labels_{gi}_{ri}_{m}"] = lab.astype(np.uint16)
                print(f"graph {gi} (n={n}, nnz={G.nnz}) r={r} min_cluster_size={m}: "
                      f"{nf} clusters")
    np.savez_compressed(here / "brisc_leiden_fixtures.npz", **out)


if __name__ == "__main__":
    main()
