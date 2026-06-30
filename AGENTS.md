# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

This is a MISCADA dissertation project analyzing EEG data from three familiarity/deception recognition experiments. The Python code is a port of the original MATLAB analysis pipeline (`Data/osfstorage-archive/ClassifierModel/`).

## Python Environment

All analysis code requires the `eeg` conda environment:

```bash
conda activate eeg
# or run directly:
conda run -n eeg python Code/exp1.py
```

Key packages: numpy 2.4, pandas 3.0, scikit-learn 1.8, scipy 1.17, matplotlib 3.10, mne 1.12.

## Running Scripts

```bash
# Main Experiment 1 analysis (produces a bar chart figure)
conda run -n eeg python Code/exp1.py

# MNE-based loader (for filtering/ICA workflows)
conda run -n eeg python Code/eeg_mne_loader.py

# .mul format tutorial
conda run -n eeg python Code/read_mul_tutorial.py
```

## Repository Layout

```
Code/
  exp1.py               # Main analysis pipeline for Exp1
  eeg_mne_loader.py     # MNE-Python alternative loader
  read_mul_tutorial.py  # .mul file format tutorial/utility
  Exp1/                 # 88 .mul data files (4 conditions × 22 subjects)
  Exp2/                 # .mul files (Lie/True/Unf/ST_* conditions)
  Exp3/                 # .mul files (same conditions as Exp2)
Data/osfstorage-archive/
  ClassifierModel/      # Original MATLAB scripts + reference .mul data
    EEGanalysis_script_Exp1.m
    EEGanalysis_script_Exp2n3.m
    mul_import.m
Literature/             # Literature review PDF
往年优秀案例/            # 16 MISCADA dissertation examples (PDFs)
模版/                   # Dissertation templates (LaTeX + Word)
项目说明/               # MISCADA dissertation regulations PDF
```

## EEG Analysis Pipeline (exp1.py)

The pipeline mirrors the MATLAB script exactly:

1. **Load** `.mul` files via `pd.read_csv(..., sep=r'\s+', skiprows=2)` — skips the 2-line header (metadata + channel names).
2. **Reshape** into `(nChannels, nSamples, nTrials)` using **column-major order** (`order='F'`), matching MATLAB's default reshape behaviour.
3. **Baseline correct** by subtracting the pre-stimulus mean (first `pres_time` = 205 samples at 1024 Hz).
4. **Downsample** by averaging over non-overlapping 100-sample windows.
5. **PCA** on all four concatenated conditions; project onto top N components.
6. **10-fold cross-validated Logistic Regression** with noise augmentation (`use_noise=5`).
7. Compute **d-prime** via signal detection theory (hit rate / false alarm rate → z-scores, clipped to ±2).
8. **One-sample t-test** against chance (0.5) and bar chart output.

## .mul File Format

Brain Products export format used throughout. Structure:
- Line 1: metadata (`TimePoints=N Channels=64 SamplingInterval[ms]=0.976563 ...`)
- Line 2: channel names (`Fp1 Fpz Fp2 ...`)
- Lines 3+: whitespace-delimited numeric data, one row per time point, units µV

`eeg_mne_loader.py` converts `.mul` directly to `mne.io.RawArray` (scales µV → V). `read_mul_tutorial.py` provides a minimal `np.loadtxt`-based reader.

## Experimental Design

**Exp1** — Familiarity recognition (22 subjects: IDs 1–5, 7–16, 18–24):
- `_01`: Familiar, High Variability (HVFam)
- `_02`: Familiar, No Variability (LVFam)
- `_03`: Unfamiliar, High Variability (HVUnf)
- `_04`: Unfamiliar, No Variability (LVUnf)

Classification tasks: HVFam vs HVUnf, and LVFam vs LVUnf.

**Exp2/3** — Deception detection:
- Conditions: `Lie`, `True`, `Unf`, `ST_CritTrue`, `ST_True`, `ST_Unf`

## Key Parameters (must stay in sync with MATLAB)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `fs` | 1024 Hz | Sampling rate |
| `n_samples` | 1229 | Samples per trial |
| `pres_time` | 205 | Pre-stimulus samples (`ceil(1024 * 0.200)`) |
| `n_sample_down` | 100 | Downsampling window |
| `n_channels` | 63 | Channels used (64th excluded) |
| `n_pcomp_used` | 4 | PCA components to test |
| `use_noise` | 5 | Noise SD for training augmentation |
| `np.random.seed` | 13 | Matches `rng(13)` in MATLAB |

The reshape operations **must use `order='F'`** throughout — this is the critical difference from Python's default row-major order and the most common source of bugs when porting from MATLAB.
