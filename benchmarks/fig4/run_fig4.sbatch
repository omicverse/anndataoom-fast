#!/bin/bash
#SBATCH --job-name=fig4-seaad
#SBATCH --partition=bigmem
#SBATCH --constraint=CLASS:SH4_CSCALE
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/users/steorra/analysis/omicverse_dev/ovfast/results/fig4/slurm-%j.out
# Fig. 4 (SEA-AD MTG, 1,378,211 cells): brisc, omicverse backend='cpp' and
# backend='python', one run each with the job's 32 cores, then
# brisc and omicverse-cpp once more single-threaded.
set -u
T=$(nproc)
D=/scratch/users/steorra/analysis/omicverse_dev/ovfast/data/seaad/SEAAD_raw.h5ad
O=/scratch/users/steorra/analysis/omicverse_dev/ovfast/results/fig4/node_${SLURM_JOB_ID}
B=/scratch/users/steorra/analysis/omicverse_dev/ovfast/anndataoom-fast/benchmarks/fig4
mkdir -p $O
lscpu > $O/lscpu.txt
cd /tmp
for t in $T 1; do
  export OMP_NUM_THREADS=$t OPENBLAS_NUM_THREADS=$t MKL_NUM_THREADS=$t NUMBA_NUM_THREADS=$t NUMEXPR_MAX_THREADS=$t
  /scratch/users/steorra/env/briscbench/bin/python $B/run_brisc.py SEAAD $D $t $O/SEAAD_brisc_t$t.csv 1 donor > $O/SEAAD_brisc_t$t.log 2>&1
  /scratch/users/steorra/env/omicdev/bin/python $B/run_omicverse.py SEAAD $D $t $O/SEAAD_ovcpp_t$t.csv 1 --backend cpp > $O/SEAAD_ovcpp_t$t.log 2>&1
  [ "$t" = "$T" ] && /scratch/users/steorra/env/omicdev/bin/python $B/run_omicverse.py SEAAD $D $t $O/SEAAD_ovpython_t$t.csv 1 --backend python > $O/SEAAD_ovpython_t$t.log 2>&1
done
