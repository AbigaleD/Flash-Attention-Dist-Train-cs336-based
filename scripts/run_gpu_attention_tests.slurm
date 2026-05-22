#!/usr/bin/env bash
#SBATCH --job-name=cs336-gpu-attn
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

source /etc/profile 2>/dev/null || true

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

if command -v module >/dev/null 2>&1; then
  module load cuda12.3/toolkit/12.3
fi

SCRATCH_ROOT="${SCRATCH:-/var/scratch/${USER}}"
export UV_PROJECT_ENVIRONMENT="${CS336_GPU_ENV:-${SCRATCH_ROOT}/cs336-systems-gpu-venv}"
export UV_CACHE_DIR="${CS336_UV_CACHE:-${SCRATCH_ROOT}/cs336-systems-uv-cache}"
export TMPDIR="${CS336_TMPDIR:-${SCRATCH_ROOT}/cs336-systems-tmp}"

mkdir -p "${UV_PROJECT_ENVIRONMENT}" "${UV_CACHE_DIR}" "${TMPDIR}"

export PATH="${HOME}/.local/bin:${PATH}"
if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install --user uv
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

uv run --python 3.11 python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside the SLURM allocation")
print("device:", torch.cuda.get_device_name(0))
PY

uv run --python 3.11 pytest -q tests/test_attention.py
