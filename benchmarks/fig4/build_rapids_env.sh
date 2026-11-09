#!/bin/bash
# Build a throw-away RAPIDS 25.08 + rapids-singlecell + omicverse environment in
# node-local scratch ($L_SCRATCH, wiped when the job ends).  Package caches go
# there too, so nothing is written to $HOME.
set -euo pipefail
P=${L_SCRATCH:?}/rapids2508
export CONDA_PKGS_DIRS=$L_SCRATCH/conda_pkgs PIP_CACHE_DIR=$L_SCRATCH/pip_cache
mkdir -p $CONDA_PKGS_DIRS $PIP_CACHE_DIR
t0=$(date +%s)
/home/users/steorra/miniforge3/bin/mamba create -y -q -p $P -c rapidsai -c conda-forge -c nvidia \
    cuml=25.08 cugraph=25.08 cupy python=3.12 cuda-version=12.4 > $L_SCRATCH/mamba.log 2>&1
echo "rapids env: $(( $(date +%s) - t0 ))s"
t0=$(date +%s)
OV=/scratch/users/steorra/analysis/omicverse_dev/ovfast/omicverse
$P/bin/python -m pip install -q rapids-singlecell polars tabulate seaborn anndataoom anndataoom-fast pytest
# omicverse's own dependencies (the code itself is used from $OV via PYTHONPATH)
$P/bin/python -m pip install -q --only-binary=:all: $( $P/bin/python - <<PY
import tomllib; d = tomllib.load(open("$OV/pyproject.toml", "rb"))
print(" ".join("'" + r + "'" for r in d["project"]["dependencies"]))
PY
) 2>&1 | grep -v notice | tail -3 || true
echo "pip: $(( $(date +%s) - t0 ))s"
$P/bin/python -c "import cupy, cuml, rapids_singlecell as rsc, inspect; print('cupy', cupy.__version__, 'cuml', cuml.__version__, 'rsc', rsc.__version__, 'exclude_highly_expressed' in str(inspect.signature(rsc.pp.normalize_total)))"
