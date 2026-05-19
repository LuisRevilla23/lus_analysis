# Reproduction Plan

This repository now has two tracks:

1. `src/`: the original TensorFlow/Keras implementation from the paper.
2. `src_torch/` and `scripts/`: a clean PyTorch 3.10 route for reproducibility and first-year-exam analysis.
3. `scripts/evaluate_tf_h5.py`: a no-training inference route using the original `model_lus.h5`.

## Dataset

The local dataset is already arranged as:

- `data/frames/train/images`
- `data/frames/train/masks`
- `data/frames/test/images`
- `data/frames/test/masks`

Masks use integer labels:

- `0`: background
- `1`: ribs
- `2`: pleural line
- `3`: A-line
- `4`: B-line
- `5`: B-line confluence

Run a dataset sanity check:

```powershell
python scripts/check_dataset.py
```

Expected counts in this workspace:

- train: 464 paired images/masks
- test: 100 paired images/masks

## Separate Environments

Use separate environments so local inference and BlueHive training do not step on each other.

- `.venv_tf_infer`: TensorFlow/Keras inference environment for the original `model_lus.h5`.
- `.venv_torch`: PyTorch training/evaluation environment for the reimplementation.

Do not delete an existing `.venv`; create these alongside it.

## TensorFlow Inference Environment

Use this on the local computer to run inference with the pretrained `model_lus.h5`, without training anything.

```powershell
py -3.10 -m venv .venv_tf_infer
.\.venv_tf_infer\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-inference.txt
```

On Windows, TensorFlow can fail to install when the project path is very long. If that happens, create the inference environment in a short path and use it from the project folder:

```powershell
uv venv C:\Users\Lenovo\lus_tf_infer --python 3.10
C:\Users\Lenovo\lus_tf_infer\Scripts\python.exe -m ensurepip --upgrade
C:\Users\Lenovo\lus_tf_infer\Scripts\python.exe -m pip install -r requirements-inference.txt
```

Then run a 10-image inference smoke test:

```powershell
python scripts/evaluate_tf_h5.py --limit-samples 10 --save-predictions
```

Or, with the short-path environment:

```powershell
C:\Users\Lenovo\lus_tf_infer\Scripts\python.exe scripts\evaluate_tf_h5.py --limit-samples 10 --save-predictions
```

Full test-set inference:

```powershell
python scripts/evaluate_tf_h5.py --split test --save-predictions
```

Outputs:

- `outputs/tf_h5_eval/metrics.json`
- `outputs/tf_h5_eval/blas.csv`
- `outputs/tf_h5_eval/pred_masks/*.png`

If Python 3.10 is not installed, install it first from python.org or with Windows `winget`. Do not use Python 3.12 for the first run; Python 3.10 is the safer reproducibility target.

## PyTorch Training Environment

Use this on BlueHive or a training computer. This does not use `model_lus.h5`; it trains/evaluates the PyTorch U-Net implementation.

```powershell
py -3.10 -m venv .venv_torch
.\.venv_torch\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-pytorch.txt
```

On BlueHive, prefer the SLURM setup in `BLUEHIVE_RUN.md`.

## PyTorch Safe Smoke Test With 10 Images

This is the recommended first run. It uses CPU, 10 images, a batch size of 1, and 1 epoch:

```powershell
.\.venv_torch\Scripts\Activate.ps1
python scripts/check_dataset.py
python scripts/train_torch.py --epochs 1 --batch-size 1 --limit-samples 10 --device cpu --num-workers 0
```

This should create:

- `outputs/torch_unet/best.pt`
- `outputs/torch_unet/history.csv`
- `outputs/torch_unet/config.json`

Then test evaluation on 10 test images:

```powershell
python scripts/evaluate_torch.py --checkpoint outputs/torch_unet/best.pt --limit-samples 10 --batch-size 1 --device cpu
```

## Train PyTorch U-Net

Only after the 10-image CPU smoke test works, try a slightly larger CPU run:

```powershell
python scripts/train_torch.py --epochs 2 --batch-size 2 --limit-samples 50 --device cpu --num-workers 0
```

GPU is optional. If you later want to test your RTX 4050 gently, use a tiny batch first:

```powershell
python scripts/train_torch.py --epochs 1 --batch-size 1 --limit-samples 10 --device cuda --num-workers 0
```

Main GPU run, only after smoke tests:

```powershell
python scripts/train_torch.py --epochs 200 --batch-size 4 --device cuda --num-workers 0
```

Outputs:

- `outputs/torch_unet/best.pt`
- `outputs/torch_unet/history.csv`
- `outputs/torch_unet/config.json`

## Evaluate Test Set

```powershell
python scripts/evaluate_torch.py --checkpoint outputs/torch_unet/best.pt
```

Output:

- `outputs/torch_unet/test_metrics.json`

Compare `mean_dice_excluding_background` and per-class Dice to Table 1 in the paper.

## First Year Exam Angle

The strongest engineering analysis is not only reproducing Dice. Use the segmentation masks to analyze BLAS robustness:

1. Calculate BLAS on manual masks.
2. Calculate BLAS on predicted masks.
3. Perturb B-line and B-line confluence regions with controlled false negatives/false positives.
4. Quantify when BLAS crosses approximate categories:
   - `< 0.5`: one/few B-lines
   - `0.5-0.9`: multiple B-lines or some confluence
   - `> 0.9`: white-lung-like artefact
