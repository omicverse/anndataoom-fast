#!/bin/bash
# Fig. 4 on SEA-AD MTG: brisc, omicverse backend='cpp', omicverse backend='python'; one rep each.
set -u
T=${1:-20}
D=/scratch/users/steorra/analysis/omicverse_dev/ovfast/data/seaad/SEAAD_raw.h5ad
O=/scratch/users/steorra/analysis/omicverse_dev/ovfast/results/fig4
B=/scratch/users/steorra/analysis/omicverse_dev/ovfast/anndataoom-fast/benchmarks/fig4
export OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T NUMBA_NUM_THREADS=$T NUMEXPR_MAX_THREADS=$T
cd /tmp
# /scratch/users/steorra/env/briscbench/bin/python $B/run_brisc.py SEAAD $D $T $O/SEAAD_brisc_t$T.csv 1 donor > $O/SEAAD_brisc_t$T.log 2>&1
/scratch/users/steorra/env/omicdev/bin/python $B/run_omicverse.py SEAAD $D $T $O/SEAAD_ovcpp_t$T.csv 1 --backend cpp > $O/SEAAD_ovcpp_t$T.log 2>&1
/scratch/users/steorra/env/omicdev/bin/python $B/run_omicverse.py SEAAD $D $T $O/SEAAD_ovpython_t$T.csv 1 --backend python > $O/SEAAD_ovpython_t$T.log 2>&1
