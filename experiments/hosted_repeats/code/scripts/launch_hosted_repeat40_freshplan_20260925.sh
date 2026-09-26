#!/usr/bin/env bash
set -eo pipefail
cd /path/to/ImplicitNav
export CUDA_VISIBLE_DEVICES=0 DINO_DEVICE=cuda:0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export ISAACSIM_ROOT=/path/to/isaacsim
source "$ISAACSIM_ROOT/setup_conda_env.sh"
exec /path/to/workspace/envs/goodnav/bin/python scripts/run_hosted_repeat40_freshplan_20260925.py "$@"
