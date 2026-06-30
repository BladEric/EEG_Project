# Project: Replicating & Extending Wiese et al. (2021) EEG Face-Familiarity Classifier

## Goal
Reproduce the logistic-regression classifier from Wiese, Anderson, Beierholm,
Tüttenberg, Young & Burton (2021), *Psychophysiology*, e13950, in Python — then
improve **individual-level CIT sensitivity**, especially the concealed-familiarity
condition (the paper's own stated weak point: ~half of participants successfully
hid their familiarity). Reference paper PDF lives at `./Wiese2021_Psychophysiol.pdf`.
OSF data/code: https://osf.io/7xtdy/

## Workflow rules (read these every session)
- Read the paper PDF before changing any analysis logic. Do NOT invent parameters —
  match the paper exactly.
- Work on ONE experiment / ONE stage at a time. Verify against the target numbers
  below before moving on.
- Keep the **faithful-replication** pipeline and the **improved** pipeline as
  SEPARATE scripts, so results stay directly comparable. Never silently "upgrade"
  the baseline.
- Use a fixed random seed everywhere and log it.

## Preprocessing facts — DON'T mix the experiments up
- Channels of interest: **P9, P10, TP9, TP10** (these 4 carry the strongest effects).
- Epoch: −200 to 1000 ms; first 200 ms = baseline (subtract per-channel, per-trial).
- Exp 1 & 2: sampling **1024 Hz**, reference **CPz**.
- Exp 3:     sampling **512 Hz**,  reference **Cz**.
- Re-reference to common average; 40 Hz low-pass; keep correct-response trials only.
- ~36–46 trials per condition after artifact rejection.

## Classifier spec — must match EXACTLY for replication
- Subtract pre-stimulus (first 200 ms) mean per channel.
- Downsample the 4 channels to 1/100 (1/50 for Exp 3) -> **11 timepoints**, ~97.7 ms apart.
- Feature vector = 4 channels x 11 timepoints = **44 dims**.
- Logistic regression; **10-fold CV** across trials; repeat **10x** with shuffled trials.
- Add stochastic noise to **TRAINING data only** (Tikhonov-style regularization).
- Also reproducible if needed: per-timepoint classification (time course) and a
  Searchlight over all channels (spatial map).

## CRITICAL: no data leakage
- ANY learned step (PCA, scaling, channel/feature selection) must be fit on the
  TRAIN fold only, then applied to test. Wrap everything in an sklearn `Pipeline`
  run inside CV.
- The 4-channel choice is theory-driven (prior studies), so using it as a fixed
  prior is fine. If you ever switch to data-driven channel selection, it MUST go
  INSIDE the CV loop.

## Validation targets — replication is "correct" when these match
- Exp 1: N250 main effect F(1,21)=29.67; classifier high-variability **0.68**,
  single-image **0.63**; SFE bootstrapping **21/22** vs **10/22**.
- Exp 2: classifier fam-vs-unfam **0.71**, fam-vs-critical **0.70**,
  unfam-vs-critical **0.56**; false-alarm rate ~**0.16**.
- Exp 3: classifier acknowledged-vs-unfam **0.69**, concealed-vs-unfam **0.61**,
  acknowledged-vs-concealed **0.63**.

## Improvement directions (each a SEPARATE script, compared to baseline above)
- xDAWN spatial filtering before LR (pyRiemann) to lift single-trial SNR.
- Riemannian / tangent-space pipeline: `XdawnCovariances + TangentSpace + LogisticRegression`.
- Use more channels (incl. the central sites the searchlight flagged) + finer time
  sampling, with regularization to control overfitting.
- LDA + shrinkage as a strong, cheap baseline alongside LR.
- Report AUC + sensitivity/specificity (more meaningful than accuracy in a CIT frame).
- Leave-one-subject-out generalization (the paper never does this — strong extension).

## Repo conventions  (Eric: fill these in)
- Data path / where the .mul files live:
- Python env (venv/conda) and key deps (mne, scikit-learn, pyriemann, ...):
- How to run the replication script:
- How to run the improved script:
