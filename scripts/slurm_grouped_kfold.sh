#!/usr/bin/env bash
# If BlueHive requires a SLURM module, load it before submitting this file:
# module load slurm/24.05.0
#SBATCH --job-name=lus-kfold
#SBATCH --output=slurm-%x-%j.log
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
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
echo "Conda env: $CONDA_ENV"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-none}"
echo "Python used: $PY"

nvidia-smi

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

print("gpu:", torch.cuda.get_device_name(0))
PY

ARCHITECTURE="${ARCHITECTURE:-light_unet}"
WIDTH_MULTIPLIER="${WIDTH_MULTIPLIER:-1.0}"
N_FOLDS="${N_FOLDS:-5}"
DATA_SCOPE="${DATA_SCOPE:-train_test}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/grouped_kfold/$ARCHITECTURE}"
EXTRA_ARGS=()

if [[ -n "${MAX_FOLDS:-}" ]]; then
  EXTRA_ARGS+=(--max-folds "$MAX_FOLDS")
fi
if [[ -n "${FOLD_START:-}" ]]; then
  EXTRA_ARGS+=(--fold-start "$FOLD_START")
fi
if [[ -n "${FOLD_STOP:-}" ]]; then
  EXTRA_ARGS+=(--fold-stop "$FOLD_STOP")
fi

"$PY" scripts/grouped_kfold_torch.py \
  --architecture "$ARCHITECTURE" \
  --width-multiplier "$WIDTH_MULTIPLIER" \
  --n-folds "$N_FOLDS" \
  --data-scope "$DATA_SCOPE" \
  --device cuda \
  --num-workers 2 \
  --batch-size 4 \
  --epochs 200 \
  --output-dir "$OUTPUT_DIR" \
  "${EXTRA_ARGS[@]}"
