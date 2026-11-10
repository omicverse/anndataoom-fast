#!/bin/bash
#SBATCH --job-name=fig4-gpu
#SBATCH --partition=gpu,xiaojie,owners
#SBATCH --requeue  # owners jobs can be preempted; rerun from the start
#SBATCH --gpus=1
#SBATCH --constraint=GPU_MEM:24GB|GPU_MEM:32GB|GPU_MEM:40GB|GPU_MEM:48GB|GPU_MEM:80GB|GPU_MEM:141GB  # any GPU >= 24 GB
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/users/steorra/analysis/omicverse_dev/ovfast/results/fig4/slurm-gpu-%j.out
# Fig. 4 on SEA-AD (1,378,211 cells), omicverse GPU modes, one GPU (>= 40 GB):
#   mixed: CPU preprocessing (backend='cpp', out of core) + torch GPU for
#          neighbors / UMAP / Leiden  (rapid_singlecell env, glibc 2.17 OK)
#   gpu:   RAPIDS (rapids-singlecell 0.18.1 in the RAPIDS 26.10 container,
#          built in $L_SCRATCH at job start; set-up time is logged, not timed)
set -u
T=${SLURM_CPUS_PER_TASK:-16}
D=/scratch/users/steorra/analysis/omicverse_dev/ovfast/data/seaad/SEAAD_raw.h5ad
O=/scratch/users/steorra/analysis/omicverse_dev/ovfast/results/fig4/gpu_${SLURM_JOB_ID}
B=/scratch/users/steorra/analysis/omicverse_dev/ovfast/anndataoom-fast/benchmarks/fig4
OV=/scratch/users/steorra/analysis/omicverse_dev/ovfast/omicverse
mkdir -p $O
{ lscpu; nvidia-smi; } > $O/hardware.txt 2>&1
export OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T NUMBA_NUM_THREADS=$T NUMEXPR_MAX_THREADS=$T
cd /tmp

# 1) mixed mode (SKIP_MIXED=1 to run only the gpu mode)
if [ -z "${SKIP_MIXED:-}" ]; then
PYTHONPATH=$OV /scratch/users/steorra/env/rapid_singlecell/bin/python $B/run_omicverse.py \
    SEAAD $D $T $O/SEAAD_ovmixed_t$T.csv 1 --backend cpp --mode mixed > $O/SEAAD_ovmixed_t$T.log 2>&1
echo "mixed exit $?" >> $O/status.txt
fi

# 2) gpu (RAPIDS) mode: container + rapids-singlecell 0.18.1 in $L_SCRATCH
export APPTAINER_CACHEDIR=$L_SCRATCH/apptainer_cache APPTAINER_TMPDIR=$L_SCRATCH/apptainer_tmp
mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR
S=$L_SCRATCH/rapids2610_sb; V=$L_SCRATCH/rvenv2610
t0=$(date +%s)
{
  apptainer -q build --sandbox $S docker://rapidsai/base:26.10-cuda12-py3.12
  apptainer exec --bind /scratch,$L_SCRATCH $S python -m venv --system-site-packages $V
  REQS=$(apptainer exec --bind /scratch $S python -c "import tomllib; d=tomllib.load(open('$OV/pyproject.toml','rb')); print(' '.join(r.replace(' ','') for r in d['project']['dependencies']))")
  apptainer exec --bind /scratch,$L_SCRATCH --env PIP_CACHE_DIR=$L_SCRATCH/pip_cache $S $V/bin/python -m pip install -q \
      polars tabulate seaborn anndataoom "anndataoom-fast>=0.1.2" docrep natsort scikit-misc $REQS
  SIF=$S CUDA=12.9 GXX=13 $B/build_rsc_in_container.sh 0.18.1 $L_SCRATCH/rsc_0.18.1
} > $O/gpu_setup.log 2>&1
echo "gpu setup $(( $(date +%s) - t0 ))s" >> $O/status.txt
apptainer exec --nv --bind /scratch,$L_SCRATCH,/tmp --env PYTHONPATH=$L_SCRATCH/rsc_0.18.1:$OV \
    $S $V/bin/python $B/run_omicverse.py SEAAD $D $T $O/SEAAD_ovgpu_t$T.csv 1 --backend cpp --mode gpu \
    > $O/SEAAD_ovgpu_t$T.log 2>&1
echo "gpu exit $?" >> $O/status.txt
