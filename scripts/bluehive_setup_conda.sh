#!/usr/bin/env bash
set -euo pipefail

echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Date: $(date)"

module purge || true

# BlueHive has several Anaconda modules. Override with:
# CONDA_MODULE=anaconda3/2022.05-user bash scripts/bluehive_setup_conda.sh
CONDA_MODULE="${CONDA_MODULE:-anaconda3/2023.07-2}"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-.conda_bluehive}"

module load "$CONDA_MODULE"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda command was not found after loading $CONDA_MODULE"
  exit 1
fi

echo "Conda: $(command -v conda)"
conda --version

source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ ! -d "$CONDA_ENV_DIR" ]]; then
  echo "Creating conda environment at $CONDA_ENV_DIR"
  conda create -y -p "$CONDA_ENV_DIR" python=3.10 pip
fi

conda activate "$CONDA_ENV_DIR"

python --version
python - <<'PY'
import ssl
print("SSL:", ssl.OPENSSL_VERSION)
PY

python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url "${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
python -m pip install -r requirements-common.txt

python - <<'PY'
import torch
import torchvision
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu_name:", torch.cuda.get_device_name(0))
PY

python scripts/check_dataset.py
