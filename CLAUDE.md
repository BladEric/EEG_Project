# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MISCADA dissertation project decoding EEG from three familiarity/deception recognition experiments. The Python pipeline (`Code/`) is a faithful port of the original MATLAB analysis (`Data/osfstorage-archive/ClassifierModel/EEGanalysis_script_Exp1.m`, `EEGanalysis_script_Exp2n3.m`), then extended with model comparisons, all-channel features, cross-subject generalization, and an EEGNet CNN.

The author (Eric) is a Python/ML beginner who learns step-by-step. Code comments are in Chinese and intentionally verbose/explanatory — **match that style** (Chinese comments, explain the "why") when editing or adding scripts. The running narrative of what has been done, results, and the scientific interpretation lives in `Results/SUMMARY.md` — read it first to understand current state, and keep it updated when you add analyses.

## Python Environment

All code requires the `eeg` conda environment:

```bash
conda run -n eeg python Code/<script>.py     # run from the project root
```

Key packages: numpy 2.4, pandas 3.0, scikit-learn 1.8, scipy 1.17, matplotlib 3.10, mne 1.12, **torch 2.12** (for EEGNet). There is no test suite, linter, or build step — scripts are run directly and validated by their printed output + saved figures.

## Architecture

The pipeline is a **numbered script series** (`Code/01_*.py` … `10_*.py`) sharing one module, `Code/eeg_pipeline.py`. Lower numbers are exploratory/learning steps; the substantive analyses are 05+. Each script is self-contained, run independently, and writes a figure to `Figures/NN_*.png` and a text report to `Results/NN_*.txt` (paths derived from `PROJECT_ROOT`, so always run from anywhere — they resolve absolutely).

`eeg_pipeline.py` is the shared core and the source of truth for parameters and preprocessing:
- **Constants** (`FS`, `N_SAMPLES`, `PRES_TIME`, `DOWNSAMPLE_WIN`, `N_DOWN`, `N_RESTRICT`, `SUBJECTS`, `FACE_CHANNELS`/`FACE_IDX`, `DATA_DIR`) — import these rather than redefining.
- `condition_features(subject, cond)` → per-trial **28-dim** feature vector (4 face channels × first 7 downsampled timepoints).
- `build_dataset()` → reads all 88 Exp1 files once, returns the two binary tasks `{"HighVar": [(X,y)×22], "SingleImg": [...]}`; labels `0=familiar, 1=unfamiliar`.
- `cv_evaluate(task_list, make_clf, augment=...)` → per-subject 10-fold CV; `make_clf` is a **factory** (returns a fresh classifier each fold). Returns `(accuracy, d_prime)` arrays. `augment=True` applies the MATLAB noise augmentation (tile train set ×10 + Gaussian noise SD=`use_noise`).
- `_dprime(prob, y)` → signal-detection d′ with z-scores clipped to ±2.

### Two feature representations / two evaluation regimes
- **Within-subject** (05, 06, 07): train and test on the *same* subject via per-subject 10-fold CV (`cv_evaluate` / `StratifiedKFold`).
- **Cross-subject** (08, 09, 10): pool all subjects and split with `GroupKFold` grouped by subject, so train/test are *different people* — this tests generalization to unseen individuals. This is the headline scientific question.
- Feature width varies by script: **28-dim** (4 face channels, pipeline default), **441-dim** (all 63 channels × 7 timepoints, in 07/08), or **raw `(63 ch × 128 timepoints)` epochs** fed directly to EEGNet (09).

### Key scripts
| Script | Role | Outputs |
|---|---|---|
| `eeg_pipeline.py` | shared loader, features, CV, d′ | — (imported) |
| `01`–`04_*.py` | learning/exploration: read data, plot ERP, condition contrasts, first classifier | `Figures/02,03_*.png` |
| `05_reproduce_exp1.py` | faithful MATLAB reproduction (logistic regression + noise augmentation) | `05_*.png`, `exp1_reproduction.txt` |
| `06_compare_models.py` | 4-model comparison on the 28-dim features | `06_*.png`, `model_comparison.txt` |
| `07_all_channels.py` | 4-channel vs all-63-channel features, 5 models | `07_*.png`, `all_channels_comparison.txt` |
| `08_cross_subject.py` | within- vs cross-subject (GroupKFold), 5 sklearn models, all channels | `08_*.png`, `cross_subject.txt` |
| `09_eegnet.py` | EEGNet CNN on raw epochs, cross-subject | `eegnet_cross_subject.txt` |
| `10_final_cross_subject.py` | summary bar chart of all 6 models cross-subject (hardcodes results from 08/09) | `10_*.png`, `final_cross_subject.txt` |

## Data & the `.mul` File Format

All data is read from `Data/osfstorage-archive/ClassifierModel/Exp{1,2,3}/` (88 Exp1 files = 4 conditions × 22 subjects). There is **no** `Code/Exp1/` directory — scripts point `DATA_DIR` at the archive.

`.mul` = Brain Products export. A file holds **all trials of one subject×condition stacked vertically**: `n_trials * N_SAMPLES` rows × `N_CHANNELS` columns.
- Line 1: metadata (`TimePoints=N Channels=64 SamplingInterval[ms]=0.976563 ...`)
- Line 2: channel names (`Fp1 Fpz Fp2 ...`) — `eeg_pipeline` parses this to map `FACE_CHANNELS` → column indices.
- Lines 3+: whitespace-delimited µV values, one row per time point.

Load with `pd.read_csv(path, sep=r"\s+", skiprows=2, header=None)`, then `.reshape(n_trials, N_SAMPLES, N_CHANNELS)` in **default C order** — because channels are already the last (column) axis, no `order='F'` is needed. (Note: this differs from the old `exp1.py` which used `order='F'`; the current pipeline does not.)

## Preprocessing pipeline (mirrors the MATLAB script)

1. Reshape flat data → `(n_trials, N_SAMPLES, N_CHANNELS)`.
2. **Baseline correct**: subtract the mean of the first `PRES_TIME` (205) pre-stimulus samples.
3. Keep the **1024 post-stimulus samples**; **downsample** by averaging non-overlapping 100-sample windows → 10 timepoints (uses the first 1000 samples).
4. Restrict to the first `N_RESTRICT` (7) timepoints (≈ first 600 ms) and, for the default features, the 4 face channels → 28-dim vector.
5. Classify (per the script): logistic regression with `C=1e3` (≈ no regularization, approximating MATLAB `fitglm`) + noise augmentation; evaluate accuracy, d′, one-sample t-test vs 0.5.

## Key Parameters (must stay in sync with the MATLAB script)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `FS` | 1024 Hz | Sampling rate |
| `N_SAMPLES` | 1229 | Samples per trial |
| `PRES_TIME` | 205 | Pre-stimulus samples (`ceil(1024 * 0.200)`) |
| `DOWNSAMPLE_WIN` | 100 | Downsampling window |
| `N_DOWN` | 10 | Downsampled timepoints (first 1000 post-stim samples) |
| `N_RESTRICT` | 7 | Timepoints kept (≈ first 600 ms) |
| `N_CHANNELS` | 63 | Channels used (64th excluded) |
| `FACE_CHANNELS` | P9/P10/TP9/TP10 | MATLAB indices 9/12/33/36 |
| `use_noise` | 5 | Gaussian SD for training augmentation |
| seed | 13 | Matches `rng(13)`; numbers won't match MATLAB bit-for-bit (different solver/RNG) but magnitude and pattern do |

## Experimental Design

**Exp1** — Familiarity recognition (22 subjects: IDs 1–5, 7–16, 18–24; 6 and 17 skipped). Conditions `_01` HVFam, `_02` LVFam, `_03` HVUnf, `_04` LVUnf. Tasks: **HighVar** = 01 vs 03, **SingleImg** = 02 vs 04.

**Exp2/3** — Deception detection. Conditions: `Lie`, `True`, `Unf`, `ST_CritTrue`, `ST_True`, `ST_Unf`. Not yet ported to the Python pipeline (intended extension).
