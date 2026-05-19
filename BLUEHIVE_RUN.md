# Running on BlueHive with SLURM

This assumes you upload/download the whole project folder on BlueHive and run from the repository root.

## 0. Information to Check on BlueHive

Run these after logging in:

```bash
hostname
pwd
module avail python
module avail cuda
sinfo -o "%P %D %G %m %l"
sacctmgr show assoc user=$USER format=Account,Partition,QOS%30 2>/dev/null || true
```

Send me the output if any script needs adjustment. The key unknowns are:

- GPU resource name. Current visible shared option: `gpu:A100:1`.
- GPU partition name.
- Whether BlueHive requires `#SBATCH --account=...`.
- Which Python module name exists.
- Whether PyTorch should use CUDA 12.1, CUDA 12.4, CUDA 12.6, or CPU-only.

## 1. Put the Project on BlueHive

Option A: upload the folder to Box, then download/unzip it on BlueHive.

Option B: if Box is mounted on BlueHive, copy from Box to scratch/project storage.

Use scratch/project storage if possible, not a tiny home directory:

```bash
mkdir -p ~/lus_segmentation
cd ~/lus_segmentation
# put repo contents here
ls
```

You should see:

```bash
data  src  src_torch  scripts  requirements-pytorch.txt  REPRODUCTION.md
```

## 2. Create the Environment

From the repository root:

```bash
bash scripts/bluehive_setup_conda.sh
```

This creates `.conda_bluehive`, installs PyTorch, installs the rest of the dependencies, verifies CUDA, and checks the dataset. Locally, if you also want TensorFlow inference, use a separate `.venv_tf_infer` as described in `REPRODUCTION.md`.

For BlueHive, PyTorch is installed separately from the CUDA wheel index, then the minimal smoke-test dependencies come from `requirements-common.txt`. Heavier analysis packages are in `requirements-analysis.txt` and can be installed later. This avoids accidentally compiling packages such as pandas from source on the cluster.

This setup now loads `anaconda3/2023.07-2` by default because the BlueHive `python3/...` modules may be built without SSL, which breaks `pip` downloads. If that Anaconda module changes, run:

```bash
module avail anaconda
```

Then rerun with a different module:

```bash
CONDA_MODULE=anaconda3/2022.05-user bash scripts/bluehive_setup_conda.sh
```

## 3. Fix SLURM Partition/Account

Open these files:

```bash
nano scripts/slurm_smoke_10.sh
nano scripts/slurm_train_full.sh
```

The scripts now request one A100 explicitly because the provided `sinfo` output shows A100 nodes but no H100 nodes:

```bash
#SBATCH -A bcastane_lab
#SBATCH -p gpu
#SBATCH --gres=gpu:A100:1
```

If you want to use the lab-specific `kuelap` partition from your colleague's example, change the SBATCH header to:

```bash
#SBATCH -A bcastane_lab
#SBATCH -p kuelap
#SBATCH --gres=gpu:1
```

Use `sinfo -o "%P %a %D %G %m %l"` to confirm what is currently available.

## 4. Submit the Safe 10-Image Smoke Test

```bash
sbatch scripts/slurm_smoke_10.sh
```

Watch status:

```bash
squeue -u $USER
```

After it finishes:

```bash
ls outputs/torch_smoke10
cat slurm-lus-smoke10-*.out
cat slurm-lus-smoke10-*.err
```

Expected outputs:

```bash
outputs/torch_smoke10/best.pt
outputs/torch_smoke10/history.csv
outputs/torch_smoke10/test_metrics.json
```

## 5. Submit Full Training Only After Smoke Test Works

```bash
sbatch scripts/slurm_train_full.sh
```

Monitor:

```bash
squeue -u $USER
tail -f slurm-lus-unet-full-*.out
```

Final outputs:

```bash
outputs/torch_unet/best.pt
outputs/torch_unet/history.csv
outputs/torch_unet/config.json
outputs/torch_unet/test_metrics.json
```

The default architecture is the paper lightweight U-Net. To train a different
variant with the same preprocessing, augmentations, loss, epochs, and evaluation
pipeline:

```bash
ARCHITECTURE=attention_unet OUTPUT_DIR=outputs/torch_attention_unet sbatch scripts/slurm_train_full.sh
```

Supported PyTorch architectures:

```bash
light_unet
residual_unet
attention_unet
unetpp
inception_unet
se_unet
dense_unet
```

## 6. Architecture Benchmark

This runs the same training/evaluation recipe for all U-Net variants and also
computes Conv2d MACs/FLOPs plus measured GPU latency. By default it uses the
five seeds `47,48,49,50,51`, matching the paper's five-repeat reporting style.

```bash
sbatch scripts/slurm_architecture_benchmark.sh
```

Useful variants:

```bash
ARCHITECTURES=light_unet,attention_unet,se_unet sbatch scripts/slurm_architecture_benchmark.sh
WIDTH_MULTIPLIER=0.5 ARCHITECTURES=light_unet,residual_unet sbatch scripts/slurm_architecture_benchmark.sh
SEEDS=47 ARCHITECTURES=light_unet,attention_unet sbatch scripts/slurm_architecture_benchmark.sh
```

Final outputs:

```bash
outputs/architecture_benchmark/model_complexity.csv
outputs/architecture_benchmark/model_complexity.json
outputs/architecture_benchmark/architecture_results.csv
outputs/architecture_benchmark/architecture_summary.csv
outputs/architecture_benchmark/<architecture>/seed_<seed>/best.pt
outputs/architecture_benchmark/<architecture>/seed_<seed>/history.csv
outputs/architecture_benchmark/<architecture>/seed_<seed>/test_metrics.json
```

Compare at least:

- Mean Dice excluding background.
- Pixel accuracy.
- BLAS MAE/correlation.
- Parameter count.
- Conv GMAC per 256 x 256 frame.
- Measured GPU latency/FPS.

## 7. Scaling Law by Architecture

The existing scaling-law script still defaults to the paper model. To run a
variant, set `ARCHITECTURE` and keep a separate output directory:

```bash
ARCHITECTURE=attention_unet OUTPUT_DIR=outputs/scaling_law_attention_unet sbatch scripts/slurm_scaling_law.sh
ARCHITECTURE=se_unet OUTPUT_DIR=outputs/scaling_law_se_unet sbatch scripts/slurm_scaling_law.sh
```

You can also pass a comma-separated architecture list directly if running
interactively:

```bash
python scripts/scaling_law_experiment.py \
  --architecture light_unet,attention_unet,se_unet \
  --device cuda \
  --num-workers 2 \
  --batch-size 4 \
  --epochs 200 \
  --output-dir outputs/scaling_law_architectures
```

## 8. Leave-One-Group-Out CV

The LOOCV script leaves out one filename group at a time, where a group is the
part before `-F` in names such as `Image62-F15.png`. This is safer than
leaving out individual frames because ultrasound frames from the same sequence
can be very similar.

Full LOOCV is expensive, so run it in chunks and rerun to resume:

```bash
MAX_FOLDS=10 sbatch scripts/slurm_loocv.sh
MAX_FOLDS=10 sbatch scripts/slurm_loocv.sh
```

Or submit explicit chunks:

```bash
FOLD_START=0 FOLD_STOP=25 sbatch scripts/slurm_loocv.sh
FOLD_START=25 FOLD_STOP=50 sbatch scripts/slurm_loocv.sh
```

By default, LOOCV uses all labelled train+test frames as cross-validation data:

```bash
DATA_SCOPE=train_test
```

Use only the original training split if you want to keep the original test set
untouched:

```bash
DATA_SCOPE=train OUTPUT_DIR=outputs/loocv_train_only/light_unet sbatch scripts/slurm_loocv.sh
```

Outputs:

```bash
outputs/loocv/<architecture>/loocv_results.csv
outputs/loocv/<architecture>/loocv_summary.json
outputs/loocv/<architecture>/folds/<fold>/best.pt
outputs/loocv/<architecture>/folds/<fold>/test_metrics.json
```

## 9. Grouped K-Fold CV

This is cheaper than LOOCV and usually easier to report. It keeps all frames
from the same filename group in the same fold, avoiding frame-level leakage.
Default is 5 folds over all labelled train+test frames:

```bash
sbatch scripts/slurm_grouped_kfold.sh
```

To keep the original paper test split untouched and cross-validate only on the
training split:

```bash
DATA_SCOPE=train OUTPUT_DIR=outputs/grouped_kfold_train_only/light_unet sbatch scripts/slurm_grouped_kfold.sh
```

To run only one fold first:

```bash
MAX_FOLDS=1 sbatch scripts/slurm_grouped_kfold.sh
```

Outputs:

```bash
outputs/grouped_kfold/<architecture>/kfold_results.csv
outputs/grouped_kfold/<architecture>/kfold_summary.json
outputs/grouped_kfold/<architecture>/folds/<fold>/best.pt
outputs/grouped_kfold/<architecture>/folds/<fold>/test_metrics.json
```

## 10. Useful Debug Commands

Interactive GPU session, if BlueHive allows it:

```bash
srun -A bcastane_lab -p gpu --gres=gpu:A100:1 --cpus-per-task=2 --mem=8G --time=00:30:00 --pty bash
```

Inside the interactive session:

```bash
module load anaconda3/2023.07-2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate .conda_bluehive
python scripts/train_torch.py --epochs 1 --batch-size 1 --limit-samples 10 --device cuda --num-workers 0
```

If CUDA is not available:

```bash
python scripts/train_torch.py --epochs 1 --batch-size 1 --limit-samples 10 --device cpu --num-workers 0
```
