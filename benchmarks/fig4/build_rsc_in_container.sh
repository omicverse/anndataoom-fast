#!/bin/bash
# rapids-singlecell >= 0.16 ships CUDA C++ extensions as sdists; the RAPIDS
# images have no compiler, so put gcc + nvcc + cmake (matching the image's
# CUDA) in a prefix on $L_SCRATCH and build there.
#   build_rsc_in_container.sh VERSION TARGET_DIR
# Env: SIF (image or sandbox dir), CUDA (e.g. 12.9), GXX (gcc major), BT (tools prefix)
set -euo pipefail
V=$1; TARGET=$2
SIF=${SIF:-$L_SCRATCH/rapids2508.sif}
CUDA=${CUDA:-12.0}
GXX=${GXX:-12}            # nvcc 12.0 accepts gcc <= 12, 12.9 gcc <= 14
BT=${BT:-$L_SCRATCH/rsc_buildtools_cuda${CUDA}_gcc${GXX}}
export APPTAINER_CACHEDIR=$L_SCRATCH/apptainer_cache
if [ ! -x $BT/bin/nvcc ]; then
  apptainer exec --bind /lscratch --env CONDA_PKGS_DIRS=$L_SCRATCH/conda_pkgs $SIF mamba create -y -q -p $BT \
      -c conda-forge -c nvidia gxx_linux-64=$GXX cuda-nvcc=$CUDA cuda-cudart-dev=$CUDA cuda-cccl=$CUDA \
      libcublas-dev libcusolver-dev cuda-version=$CUDA cmake ninja > $BT.log 2>&1
fi
# rsc pins CMAKE_CUDA_ARCHITECTURES="native", so build with the GPU visible
# (--nv; MPS variables passed through when set).
apptainer exec --nv --bind /scratch,/lscratch,/tmp \
  --env CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-},CUDA_MPS_LOG_DIRECTORY=${CUDA_MPS_LOG_DIRECTORY:-} \
  --env PATH=$BT/bin:/opt/conda/bin:/usr/local/bin:/usr/bin:/bin \
  --env CC=$BT/bin/x86_64-conda-linux-gnu-gcc,CXX=$BT/bin/x86_64-conda-linux-gnu-g++,CUDACXX=$BT/bin/nvcc,CUDAHOSTCXX=$BT/bin/x86_64-conda-linux-gnu-g++,CUDAToolkit_ROOT=$BT \
  --env PIP_CACHE_DIR=$L_SCRATCH/pip_cache,CMAKE_BUILD_PARALLEL_LEVEL=16 \
  $SIF python -m pip install -q --no-deps --target $TARGET "rapids-singlecell==$V"
echo "built rapids-singlecell $V into $TARGET"
