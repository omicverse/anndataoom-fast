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
emb = aof.umap(G)                                   # UMAP SGD layout
labels = aof.leiden(G, resolution=1.0, random_state=0)   # = leidenalg
labels = aof.leiden_multi(G, [0.25, 0.5, 1.0], random_state=0)

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
| Leiden (RBConfiguration) | leidenalg, identical partitions |
| COSG markers | omicverse, identical top-n lists |
| kNN | approximate (IVF), recall > 0.99; `exact=True` for brute force |

## License

GPL-3.0-or-later. Leiden is ported from leidenalg/libleidenalg (GPL-3.0) and
igraph (GPL-2.0-or-later); the UMAP graph/layout follow umap-learn (BSD-3);
`third_party/libdeflate` is MIT.
