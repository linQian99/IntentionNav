#!/usr/bin/env bash
set -euo pipefail
repo_dir=.
python_bin=/path/to/workspace/envs/goodnav/bin/python
agent_source_dir="${INAV_MTU_AGENT_SOURCE:-$repo_dir/eval/agents}"
cd "$repo_dir"
if [[ " ${*} " == *" --check "* ]]; then
  exec "$python_bin" "$agent_source_dir/agent_mtu3d.py" "$@"
fi
export CUDA_VISIBLE_DEVICES=0 ISAACSIM_ACTIVE_GPU=0 ISAACSIM_PHYSICS_GPU=0 ISAACSIM_MULTI_GPU=0
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
export INAV_RENDER_ANTIALIASING_MODE=2 ISS_RENDER_TICKS=8 INAV_INSTANCE_VISIBILITY_DIAGNOSTIC=1
export ISAACSIM_ROOT=/path/to/isaacsim
export OMNI_USER_PATH="$repo_dir/results/mtu3d_compatibility_20260909/omni"
unset PP_API_KEY OPENAI_API_KEY GEMINI_API_KEY ANTHROPIC_API_KEY
"$python_bin" - <<'PY'
import json
from pathlib import Path
import subprocess
witness=Path('results/mtu3d_compatibility_20260909/gpu_witness.json')
if not witness.is_file():raise SystemExit('Actual MTU3D GPU/model witness is required before navigation')
result=json.loads(witness.read_text())
if not result.get('gpu_witness') or not result.get('model_decision'):
    raise SystemExit('Incomplete model witness')
memory=int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
if memory>500:raise SystemExit('GPU0 is occupied; no process was interrupted')
PY
set +u
source "$ISAACSIM_ROOT/setup_conda_env.sh"
set -u
exec "$python_bin" "$agent_source_dir/agent_mtu3d.py" "$@"
