#!/usr/bin/env bash
set -euo pipefail

echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Date: $(date)"

module purge || true

# BlueHive's python3/3.10.5 may not include SSL, which breaks pip HTTPS.
# python3/3.11.0 is preferred here. Override with PYTHON_MODULE if needed.
PYTHON_MODULE="${PYTHON_MODULE:-python3/3.11.0}"
module load "$PYTHON_MODULE"

echo "Python candidates:"
command -v python || true
command -v python3 || true
python --version 2>/dev/null || python3 --version

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
VENV_DIR="${VENV_DIR:-.venv_bluehive}"

echo "Using Python: $PYTHON_BIN"
"$PYTHON_BIN" --version
"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(
        "ERROR: Python >= 3.10 is required for current PyTorch wheels. "
        f"Detected Python {sys.version.split()[0]}. "
        "Load a newer Python/Anaconda module before rerunning setup."
    )
PY
"$PYTHON_BIN" - <<'PY'
try:
    import ssl
except Exception as exc:
    raise SystemExit(
        "ERROR: This Python build does not include the ssl module, so pip cannot "
        f"download packages over HTTPS. Details: {exc}. Try another module, e.g. "
        "PYTHON_MODULE=python3/3.11.0 bash scripts/slurm_setup_env.sh"
    )
print("SSL:", ssl.OPENSSL_VERSION)
PY

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating virtual environment: $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "ERROR: $VENV_DIR/bin/activate was not created."
  echo "The Python module loaded on BlueHive may not support venv."
  echo "Try: module avail python"
  echo "Then load an Anaconda/Miniconda/Mamba/Python module and rerun this script."
  exit 1
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip

# GPU build by default for HPC. If this fails on your cluster, switch cu121 to
# the CUDA option recommended by `nvidia-smi` and https://docs.pytorch.org/get-started/locally/.
python -m pip install torch torchvision torchaudio --index-url "${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
python -m pip install -r requirements-pytorch.txt

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu_name:", torch.cuda.get_device_name(0))
PY

python scripts/check_dataset.py
