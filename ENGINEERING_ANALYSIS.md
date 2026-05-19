# Engineering Analysis Outputs

This analysis evaluates whether the reproduced LUS segmentation model can support a downstream BLAS severity metric and whether that metric is robust enough for portable real-time use.

## Run

The full analysis was run in three passes because the local TensorFlow environment and plotting environment are separate:

```powershell
.\.venv\Scripts\python.exe scripts\engineering_analysis.py --tf-output-dir outputs\tf_h5_eval_test --output-dir outputs\engineering_analysis --sensitivity-repeats 10 --skip-conformal --skip-energy
C:\Users\Lenovo\lus_tf_infer\Scripts\python.exe scripts\engineering_analysis.py --tf-output-dir outputs\tf_h5_eval_test --output-dir outputs\engineering_analysis --skip-agreement --skip-failures --skip-sensitivity
.\.venv\Scripts\python.exe scripts\engineering_analysis.py --tf-output-dir outputs\tf_h5_eval_test --output-dir outputs\engineering_analysis --skip-agreement --skip-failures --skip-sensitivity --skip-energy
```

## Main Outputs

- `blas_agreement_summary.json`: BLAS agreement metrics between manual labels and model predictions.
- `blas_agreement_cases.csv`: per-frame BLAS target, prediction, error, and severity-category mismatch.
- `blas_scatter.png`: manual BLAS vs predicted BLAS.
- `blas_bland_altman.png`: bias and limits of agreement.
- `blas_abs_error_hist.png`: distribution of absolute BLAS error.
- `failure_cases/`: top failure-case panels with image, manual mask, predicted mask, and error map.
- `sensitivity_results.csv`: per-frame BLAS changes after controlled B-line/confluence perturbations.
- `sensitivity_summary.csv`: aggregate sensitivity curves.
- `sensitivity_curves.png`: BLAS sensitivity to false positives, false negatives, erosion, and dilation.
- `conformal_uncertainty_summary.json`: pixel-level conformal/uncertainty summary using a held-out half of the test set.
- `conformal_uncertainty_cases.csv`: per-frame uncertainty, coverage, and BLAS error.
- `uncertainty_vs_blas_error.png`: uncertainty score vs downstream BLAS error.
- `conformal_blas_refined_summary.csv`: conformal/uncertainty analysis repeated on foreground pixels, BLAS ROI pixels, and manual B-line/confluence pixels.
- `conformal_blas_refined_correlations.png`: comparison of localized uncertainty correlations with BLAS error.
- `train_test_similarity_summary.json`: image-level train-test nearest-neighbor similarity analysis.
- `train_test_nearest_neighbors.csv`: top train-set visual neighbors for each test image.
- `train_test_similarity_examples/`: side-by-side panels of the most similar train/test image pairs.
- `portable_energy_summary.json`: model size, approximate MACs, and local TensorFlow inference speed.
- `portable_energy_estimates.csv`: energy-per-frame and battery-runtime estimates for portable deployment scenarios.
- `scaling_law_curve.png`: combined foreground Dice and log-log error scaling-law figure.
- `scaling_law_summary_by_size.csv`: scaling-law means and standard deviations by training-set size and model width.
- `scaling_law_results.csv`: raw completed scaling-law runs.
- `scaling_law_fit.csv`: exploratory power-law fits for test error vs training-set size.

## Key Results From This Run

- Test frames: 100.
- BLAS MAE: 0.158.
- BLAS RMSE: 0.284.
- BLAS bias, prediction minus manual: +0.060.
- Pearson correlation: 0.626.
- Spearman correlation: 0.595.
- BLAS severity-category disagreement: 26/100 frames.
- Largest errors are mostly low-manual-BLAS frames predicted as high BLAS, suggesting false-positive B-line/confluence predictions can strongly affect the downstream severity metric.
- The model has 7.86 million parameters and an estimated 13.9 GMAC per 256 x 256 frame.
- Local CPU TensorFlow inference was about 0.147 s/frame, or 6.82 FPS.
- Refined conformal analysis:
  - all pixels: Spearman entropy vs BLAS error = 0.195.
  - foreground pixels: Spearman entropy vs BLAS error = 0.017.
  - manual BLAS ROI: Spearman entropy vs BLAS error = 0.218.
  - manual B-line/confluence pixels: Spearman entropy vs BLAS error = -0.027.
- Train-test similarity:
  - no train/test sequence IDs are shared in the local split.
  - 17/100 test images have a nearest train image with cosine similarity >= 0.99 after cropped low-resolution normalization.
  - 41/100 test images have nearest-neighbor similarity >= 0.95.
- Scaling-law experiment:
  - 105/105 planned runs completed: 7 training sizes x 3 model widths x 5 seeds.
  - The mean foreground Dice increases with training-set size and plateaus near 0.71 by roughly 320-370 training images.
  - Wider models help most in the low-data regime: at target size 32, mean Dice was 0.164 for 0.5x width, 0.411 for 1.0x, and 0.512 for 1.5x.
  - By the largest training sizes, 1.0x and 1.5x widths converge to similar Dice, suggesting the bottleneck is not only model capacity.

## Interpretation Angle

The reproduced segmentation model is technically useful, but BLAS is more sensitive than pixel accuracy alone suggests. Small visual segmentation errors in B-line and confluence regions can cause clinically meaningful BLAS-category changes. This makes downstream reliability analysis essential before treating the method as a portable decision-support tool.

The conformal analysis is intentionally lightweight: it uses half of the test set as calibration and half as held-out evaluation because a separate validation split is not available locally. This should be presented as an engineering probe, not as final clinical calibration.

The train-test similarity analysis does not prove leakage, because sequence IDs are not shared. It does show that phantom images can be visually redundant across different sequences, so reported test performance may not fully represent scanner-, patient-, or acquisition-site-level generalization.

The scaling-law experiment strengthens this interpretation: more data initially helps, and larger models help when data are scarce, but performance plateaus around the full phantom training-set size. The fitted scaling exponents should be interpreted as exploratory because grouped sampling makes the actual number of training images vary slightly across seeds and because all results remain within the same phantom domain.
