#!/usr/bin/env bash
# If BlueHive requires a SLURM module, load it before submitting this file:
# module load slurm/24.05.0
#SBATCH --job-name=lus-scaling-law
#SBATCH --output=slurm-%x-%j.log
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=kuelap
#SBATCH --gres=gpu:L40S:1
#SBATCH --account=bcastane_lab

set -euo pipefail

module purge || true

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$SUBMIT_DIR"

CONDA_ACTIVATE="${CONDA_ACTIVATE:-/software/anaconda3/5.3.0b/bin/activate}"
CONDA_ENV="${CONDA_ENV:-$SUBMIT_DIR/.conda_bluehive}"

if [[ ! -f "$CONDA_ACTIVATE" ]]; then
  echo "ERROR: conda activate script not found: $CONDA_ACTIVATE" >&2
  echo "Set CONDA_ACTIVATE=/path/to/anaconda/bin/activate when submitting." >&2
  exit 1
fi

if [[ ! -d "$CONDA_ENV" ]]; then
  echo "ERROR: conda environment not found: $CONDA_ENV" >&2
  echo "Set CONDA_ENV=/path/to/env when submitting." >&2
  exit 1
fi

set +u
source "$CONDA_ACTIVATE" "$CONDA_ENV"
set -u

PY="${PYTHON_BIN:-$(command -v python)}"

echo "Job: ${SLURM_JOB_ID:-interactive}"
echo "Node: $(hostname)"
echo "Submit dir: $SUBMIT_DIR"
echo "Conda activate: $CONDA_ACTIVATE"
echo "Conda env: $CONDA_ENV"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-none}"
echo "Python used: $PY"

nvidia-smi

"$PY" --version
"$PY" - <<'PY'
import sys
import torch
import torchvision

print("python:", sys.executable)
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise SystemExit("ERROR: GPU assigned by SLURM, but PyTorch cannot see CUDA.")

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

EXTRA_ARGS=()
if [[ -n "${MAX_RUNS:-}" ]]; then
  EXTRA_ARGS+=(--max-runs "$MAX_RUNS")
fi

ARCHITECTURE="${ARCHITECTURE:-light_unet}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/scaling_law}"

"$PY" scripts/scaling_law_experiment.py \
  --architecture "$ARCHITECTURE" \
  --device cuda \
  --num-workers 2 \
  --batch-size 4 \
  --epochs 200 \
  --sizes 32,64,128,192,256,320,370 \
  --seeds 47,48,49,50,51 \
  --width-multipliers 0.5,1.0,1.5 \
  --output-dir "$OUTPUT_DIR" \
  "${EXTRA_ARGS[@]}"
