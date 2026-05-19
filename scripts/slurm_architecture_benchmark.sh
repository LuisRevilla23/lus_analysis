#!/usr/bin/env bash
# If BlueHive requires a SLURM module, load it before submitting this file:
# module load slurm/24.05.0
#SBATCH --job-name=lus-arch-bench
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

ARCHITECTURES="${ARCHITECTURES:-light_unet,residual_unet,attention_unet,unetpp,inception_unet,se_unet,dense_unet}"
SEEDS="${SEEDS:-47,48,49,50,51}"
WIDTH_MULTIPLIER="${WIDTH_MULTIPLIER:-1.0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/architecture_benchmark}"

"$PY" scripts/model_complexity.py \
  --architectures "$ARCHITECTURES" \
  --width-multipliers "$WIDTH_MULTIPLIER" \
  --device cuda \
  --latency-repeats 50 \
  --output "$OUTPUT_ROOT/model_complexity.csv" \
  --json-output "$OUTPUT_ROOT/model_complexity.json"

IFS=',' read -r -a ARCH_ARRAY <<< "$ARCHITECTURES"
IFS=',' read -r -a SEED_ARRAY <<< "$SEEDS"
for ARCH in "${ARCH_ARRAY[@]}"; do
  ARCH="$(echo "$ARCH" | xargs)"
  for SEED in "${SEED_ARRAY[@]}"; do
    SEED="$(echo "$SEED" | xargs)"
    OUT="$OUTPUT_ROOT/$ARCH/seed_$SEED"
    echo "=== Training architecture: $ARCH seed: $SEED -> $OUT ==="
    "$PY" scripts/train_torch.py \
      --architecture "$ARCH" \
      --width-multiplier "$WIDTH_MULTIPLIER" \
      --seed "$SEED" \
      --epochs 200 \
      --batch-size 4 \
      --device cuda \
      --num-workers 2 \
      --paper-augmentations \
      --output-dir "$OUT"

    "$PY" scripts/evaluate_torch.py \
      --checkpoint "$OUT/best.pt" \
      --batch-size 4 \
      --device cuda \
      --num-workers 2 \
      --output "$OUT/test_metrics.json"
  done
done

"$PY" scripts/summarize_architecture_benchmark.py \
  --root "$OUTPUT_ROOT" \
  --output "$OUTPUT_ROOT/architecture_results.csv" \
  --summary-output "$OUTPUT_ROOT/architecture_summary.csv"
