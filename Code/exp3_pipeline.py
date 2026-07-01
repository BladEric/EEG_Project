"""
Exp3（测谎 / 隐瞒熟悉度）共享流水线 —— 给 11–16 号脚本复用。

忠实对应老师 MATLAB 脚本 EEGanalysis_script_Exp2n3.m 里 exp2or3=0（Exp3）的分支。
与 Exp1 的关键差异：fs=512、nSamples=614、nChannels=64、降采样窗=50、rng=10。

提供三种取数方式：
  - condition_features(): 降采样后取「指定通道 × 前 N 个时间点」→ 特征向量（默认 4 通道 ×7点 =28 维）
  - load_epochs():        原始时空 epoch (试次, 通道, 时间点)，给 xDAWN+Riemann / CNN 用
  - build_tasks():        一次性组装三个二分类任务（每任务 = 19 个被试各一份 (X,y)）
"""
from pathlib import Path
import numpy as np
import pandas as pd

# ---------- 与 MATLAB 脚本一致的 Exp3 参数 ----------
FS = 512
N_SAMPLES = 614
N_CHANNELS = 64
PRES_TIME = int(np.ceil(FS * 0.200))                  # 103
DOWNSAMPLE_WIN = 50                                   # nSampleDown/2
N_DOWN = 10                                           # 降采样后的时间点数（signDownSamples-1）
N_RESTRICT = int(np.ceil(0.6 * FS / DOWNSAMPLE_WIN))  # 7：默认只取前 ~600ms
FACE_IDX = [8, 11, 32, 35]                            # 0-based：P9/P10/TP9/TP10（MATLAB 9/12/33/36）

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data" / "osfstorage-archive" / "ClassifierModel" / "Exp3"
SUBJECTS = list(range(2, 21))                          # 19 人：Part02–Part20
SEED = 10                                              # MATLAB 这个脚本用 rng(10)

# 三个二分类任务。标签遵循 MATLAB（如 True vs Lie 中 Lie=0, True=1）。
TASKS = {
    "True vs Unf": ("True", "Unf"),
    "Lie vs Unf":  ("Lie",  "Unf"),
    "True vs Lie": ("Lie",  "True"),
}

with open(DATA_DIR / "Part02_True-export.mul") as _f:
    _f.readline()
    CHANNEL_NAMES = _f.readline().split()


def _find_file(subject, cond):
    """Exp3 文件命名大小写不统一（Part02/03 用 Lie/True/Unf，Part04+ 用小写），逐一尝试。"""
    for name in (cond, cond.lower(), cond.capitalize()):
        p = DATA_DIR / f"Part{subject:02d}_{name}-export.mul"
        if p.exists():
            return p
    raise FileNotFoundError(f"找不到 被试{subject:02d} 的 {cond} 文件")


def _downsampled(subject, cond, rng):
    """某被试某条件 → 基线校正 + 降采样后的 (试次, N_DOWN, 64)。这是所有特征的公共底座。"""
    data = pd.read_csv(_find_file(subject, cond), sep=r"\s+", skiprows=2, header=None).to_numpy()
    data = data[:, :N_CHANNELS]
    n = data.shape[0] // N_SAMPLES
    trials = data.reshape(n, N_SAMPLES, N_CHANNELS)
    trials = trials - trials[:, :PRES_TIME, :].mean(axis=1, keepdims=True)          # 基线校正
    post = trials[:, PRES_TIME:PRES_TIME + N_DOWN * DOWNSAMPLE_WIN, :]              # 刺激后前 500 点
    down = post.reshape(n, N_DOWN, DOWNSAMPLE_WIN, N_CHANNELS).mean(axis=2)         # → (试次,10,64)
    return down + rng.normal(0, 1e-4, down.shape)                                   # MATLAB 的极小噪声


def condition_features(subject, cond, rng, n_restrict=N_RESTRICT, channels=FACE_IDX):
    """降采样 → 取前 n_restrict 个时间点 × 指定通道 → 展平成特征向量。"""
    down = _downsampled(subject, cond, rng)
    feat = down[:, :n_restrict, :][:, :, channels]                                  # (试次, n_restrict, n_ch)
    return feat.reshape(down.shape[0], -1)


def build_tasks(rng=None, n_restrict=N_RESTRICT, channels=FACE_IDX, verbose=False):
    """组装三个任务。返回 {task: [(X,y) ×19]}。channels 给 list 选通道；给 'all' 用全 64 通道。"""
    if rng is None:
        rng = np.random.default_rng(SEED)
    if channels == "all":
        channels = list(range(N_CHANNELS))
    feats = {}
    for s in SUBJECTS:
        for c in ["True", "Lie", "Unf"]:
            feats[(s, c)] = condition_features(s, c, rng, n_restrict, channels)
        if verbose:
            print(f"  已处理被试 {s:02d}")
    out = {}
    for task, (c0, c1) in TASKS.items():
        per_subj = []
        for s in SUBJECTS:
            f0, f1 = feats[(s, c0)], feats[(s, c1)]
            X = np.vstack([f0, f1])
            y = np.r_[np.zeros(len(f0)), np.ones(len(f1))]
            per_subj.append((X, y))
        out[task] = per_subj
    return out


def load_epochs(subject, cond, rng, T=128, channels=None):
    """某被试某条件 → 原始时空 epoch (试次, 通道, T)。给 xDAWN+Riemann / CNN 用。
    把刺激后的点降到 T 个时间点；channels=None 用全部 64 通道。"""
    data = pd.read_csv(_find_file(subject, cond), sep=r"\s+", skiprows=2, header=None).to_numpy()
    data = data[:, :N_CHANNELS]
    n = data.shape[0] // N_SAMPLES
    trials = data.reshape(n, N_SAMPLES, N_CHANNELS)
    trials = trials - trials[:, :PRES_TIME, :].mean(axis=1, keepdims=True)
    post_len = N_SAMPLES - PRES_TIME                                                # 511
    factor = post_len // T
    post = trials[:, PRES_TIME:PRES_TIME + T * factor, :]
    post = post.reshape(n, T, factor, N_CHANNELS).mean(axis=2)                      # 降到 T 点
    ep = post.transpose(0, 2, 1).astype(np.float64)                                # (n, 64, T)
    if channels is not None:
        ep = ep[:, channels, :]
    return ep
