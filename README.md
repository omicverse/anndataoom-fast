# anndataoom-fast

Multithreaded C++ kernels for single-cell preprocessing, used by
[omicverse](https://github.com/Starlitnightly/omicverse) as its default
`backend='cpp'`.

The kernels reproduce omicverse/scanpy's own algorithms — same QC metrics,
size factors, Pearson-residual HVGs, scaling, PCA, UMAP graph and layout,
Leiden and COSG markers — but run on all cores, and give the same result for
any number of threads (fixed row blocks merged in a fixed order, no
fast-math).

* **Out-of-core** ([anndataoom](https://pypi.org/project/anndataoom/)) h5ad
  files are read by the C++ side directly: h5py describes the chunk layout
  once, then every thread `pread`s and inflates (libdeflate) its own chunks,
  bypassing libhdf5's global lock.
* **In memory** scipy CSR matrices are wrapped without copying.

## Install

```bash
pip install anndataoom-fast
```

Wheels are built for Linux (x86_64, aarch64), macOS (x86_64, arm64) and
Windows (x86_64), CPython 3.9–3.13.

## Use with omicverse

Nothing to do: omicverse uses it by default.

```python
import omicverse as ov
ov.settings.backend = "cpp"      # default; "python" restores the original code
ov.pp.qc(adata, backend="python")  # or switch per call
```

## Standalone

```python
import anndataoom_fast as aof
aof.set_num_threads(16)

idx, dist = aof.knn(X_pca, 15)                      # IVF kNN, self first
G = aof.fuzzy_simplicial_set(idx, dist, 15)         # = umap-learn
emb = aof.umap(G)                                   # UMAP layout, all threads, deterministic
labels = aof.leiden(G, resolution=1.0, random_state=0)   # = leidenalg
labels = aof.leiden_multi(G, [0.25, 0.5, 1.0], random_state=0)
labels = aof.leiden_brisc(G, resolution=1.0, random_state=0)   # = brisc SingleCell.cluster
emb = aof.localmap(X_pca)                           # = brisc SingleCell.localmap (also aof.pacmap)

from anndataoom_fast import memory                  # in-memory CSR kernels
qc = memory.qc_metrics(adata.X, mt=adata.var_names.str.startswith("MT-"))
```

`aof.install()` routes anndataoom's `chunked_*` preprocessing functions
through the C++ engine in place.

## Exactness

| step | vs |
|---|---|
| QC, normalize_total, HVG (pearson), scale | omicverse, identical / ≤1e-13 relative |
| PCA (in memory, covariance_eigh) | sklearn, same components up to sign |
| fuzzy simplicial set | umap-learn, bit-identical |
| UMAP layout (`sgd='parallel'`, default) | same objective, edge schedule and negative sampling as umap-learn; same embedding for any thread count; `sgd='sequential'` = umap-learn's algorithm |
| Leiden (`leiden`, RBConfiguration) | leidenalg, identical partitions |
| Leiden, brisc flavour (`leiden_brisc`) | brisc, identical labels |
| PaCMAP / LocalMAP (`pacmap`, `localmap`) | brisc's algorithms; PaCMAP bit-identical to brisc given the same kNN; LocalMAP identical on every machine (brisc's own LocalMAP depends on the CPU's RSQRTSS) |
| COSG markers | omicverse, identical top-n lists |
| kNN | approximate (IVF), recall > 0.99; `exact=True` for brute force |

## License

GPL-3.0-or-later. Leiden is ported from leidenalg/libleidenalg (GPL-3.0) and
igraph (GPL-2.0-or-later); the UMAP graph/layout follow umap-learn (BSD-3);
`third_party/libdeflate` is MIT. `leiden_brisc` is ported from
[brisc](https://github.com/briscverse/brisc) (MIT, Copyright (c) 2026 Michael
Wainberg and Keon Arbabi), as are `pacmap` / `localmap`.

## Parallel UMAP

umap-learn optimises the layout sequentially (or with Hogwild!, which is not
reproducible).  `sgd='parallel'` keeps umap's update rule — each sampled edge
pulls its head and its tail together, followed by negative samples — and
parallelises it deterministically: vertices are visited in fixed blocks in
umap's own (head-sorted) edge order, each head is updated by one thread from
live coordinates, the tail moves are recorded and applied after the block in
edge order, and negative samples come from a hash of (seed, edge, epoch).  On
SEA-AD (1.38M cells) it matches the sequential layout's kNN preservation
(0.0233 vs 0.0230) and gives identical embeddings for any number of threads.
