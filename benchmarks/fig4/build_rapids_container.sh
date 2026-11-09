#!/bin/bash
# RAPIDS 25.08 needs glibc >= 2.28 (Sherlock nodes have 2.17), so run it from
# the official container.  Image, caches and the pip overlay with
# rapids-singlecell + omicverse's dependencies all live in $L_SCRATCH.
set -euo pipefail
export APPTAINER_CACHEDIR=$L_SCRATCH/apptainer_cache APPTAINER_TMPDIR=$L_SCRATCH/apptainer_tmp
mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR
SIF=$L_SCRATCH/rapids2508.sif
LIB=$L_SCRATCH/rapids_pylib
t0=$(date +%s)
[ -f $SIF ] || apptainer -q pull $SIF docker://rapidsai/base:25.08-cuda12.0-py3.12
echo "pull: $(( $(date +%s) - t0 ))s"
OV=/scratch/users/steorra/analysis/omicverse_dev/ovfast/omicverse
t0=$(date +%s)
REQS=$(apptainer exec $SIF python - <<PY
import tomllib; d = tomllib.load(open("$OV/pyproject.toml", "rb"))
print(" ".join(r.replace(" ", "") for r in d["project"]["dependencies"]))
PY
)
apptainer exec --env PIP_CACHE_DIR=$L_SCRATCH/pip_cache $SIF python -m pip install -q --target $LIB \
    rapids-singlecell polars tabulate seaborn anndataoom anndataoom-fast pytest $REQS 2>&1 | grep -iE "^error" | tail -5 || true
echo "pip: $(( $(date +%s) - t0 ))s"
apptainer exec --nv --env PYTHONPATH=$LIB $SIF python -c "import cupy, cuml, rapids_singlecell as rsc; print('cupy', cupy.__version__, 'cuml', cuml.__version__, 'rsc', rsc.__version__)"
