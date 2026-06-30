"""
共享数据处理流水线 —— 忠实复现老师 MATLAB 脚本 EEGanalysis_script_Exp1.m 的预处理。

被 05_reproduce_exp1.py（复现）和 06_compare_models.py（新模型）共同调用。

把原始 .mul → 每个试次一个“特征向量”，步骤完全对应老师的脚本：
  1) 切试次   2) 基线校正(减刺激前205点均值)   3) 去掉刺激前，留刺激后1024点
  4) 降采样：前1000点每100点求平均 → 10个时间点
  5) 只取 4 个人脸通道(P9/P10/TP9/TP10) × 前7个时间点(≈前600ms) → 28 维特征
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import StratifiedKFold

# ---------- 与老师脚本一致的参数 ----------
FS = 1024
N_SAMPLES = 1229
N_CHANNELS = 63
PRES_TIME = 205                              # ceil(1024 * 0.200)
SIGN_SAMPLES = N_SAMPLES - PRES_TIME         # 1024：刺激后的点数
DOWNSAMPLE_WIN = 100                         # 每 100 点求一次平均
N_DOWN = SIGN_SAMPLES // DOWNSAMPLE_WIN      # 10：降采样后的时间点数（用前1000点）
N_RESTRICT = int(np.ceil(0.6 * FS / DOWNSAMPLE_WIN))   # 7：只取前 ~600ms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data" / "osfstorage-archive" / "ClassifierModel" / "Exp1"
SUBJECTS = list(range(1, 6)) + list(range(7, 17)) + list(range(18, 25))   # 22 人，跳过6、17

FACE_CHANNELS = ["P9", "P10", "TP9", "TP10"]   # 老师选的人脸通道（MATLAB 9/12/33/36）

with open(DATA_DIR / "Part01_01-export.mul") as _f:
    _f.readline()
    CHANNEL_NAMES = _f.readline().split()
FACE_IDX = [CHANNEL_NAMES.index(n) for n in FACE_CHANNELS]


def condition_features(subject, cond):
    """某被试某条件 → 每个试次的 28 维特征，形状 (试次数, 28)。"""
    path = DATA_DIR / f"Part{subject:02d}_{cond}-export.mul"
    data = pd.read_csv(path, sep=r"\s+", skiprows=2, header=None).to_numpy()
    n_trials = data.shape[0] // N_SAMPLES
    trials = data.reshape(n_trials, N_SAMPLES, N_CHANNELS)              # (试次,时间,电极)

    trials = trials - trials[:, :PRES_TIME, :].mean(axis=1, keepdims=True)  # 基线校正
    post = trials[:, PRES_TIME:PRES_TIME + N_DOWN * DOWNSAMPLE_WIN, :]      # 刺激后前1000点
    down = post.reshape(n_trials, N_DOWN, DOWNSAMPLE_WIN, N_CHANNELS).mean(axis=2)  # 降采样→(试次,10,电极)

    feat = down[:, :N_RESTRICT, :][:, :, FACE_IDX]                     # (试次,7,4)
    return feat.reshape(n_trials, -1)                                  # (试次,28)


def build_dataset(verbose=True):
    """
    读入全部 88 个文件，组装两个二分类任务（每个文件只读一次）。
    返回 {"HighVar": [(X,y)×22], "SingleImg": [(X,y)×22]}；标签 0=熟悉, 1=陌生。
    """
    data = {"HighVar": [], "SingleImg": []}
    for s in SUBJECTS:
        f01, f02 = condition_features(s, "01"), condition_features(s, "02")  # 熟悉：高多样性/单图
        f03, f04 = condition_features(s, "03"), condition_features(s, "04")  # 陌生：高多样性/单图
        data["HighVar"].append((np.vstack([f01, f03]),
                                np.r_[np.zeros(len(f01)), np.ones(len(f03))]))   # 01 vs 03
        data["SingleImg"].append((np.vstack([f02, f04]),
                                  np.r_[np.zeros(len(f02)), np.ones(len(f04))]))  # 02 vs 04
        if verbose:
            print(f"  已处理被试 {s:02d}")
    return data


def _dprime(prob, y):
    """信号检测论 d′：z(命中率) - z(虚报率)，z 值截断到 ±2（与老师一致）。"""
    hit = np.mean(prob[y == 1] > 0.5)
    fa = np.mean(prob[y == 0] > 0.5)
    with np.errstate(all="ignore"):
        return np.clip(norm.ppf(hit), -2, 2) - np.clip(norm.ppf(fa), -2, 2)


def cv_evaluate(task_list, make_clf, augment=False, n_folds=10, seed=13, use_noise=5):
    """
    对一个任务（22 个被试各一份 (X,y)）逐被试做 10 折交叉验证。
    make_clf  : 一个返回“新分类器”的函数（这样每折都用全新的模型）
    augment   : 是否用老师的噪声增强（训练集复制10份 + 加 SD=5 高斯噪声）
    返回 (每被试准确率数组, 每被试 d′ 数组)。
    """
    rng = np.random.default_rng(seed)
    accs, dps = [], []
    for X, y in task_list:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        errs, probs_all, true_all = [], [], []
        for tr, te in skf.split(X, y):
            Xtr, ytr = X[tr], y[tr]
            if augment:
                Xtr = np.tile(Xtr, (10, 1)) + rng.normal(0, use_noise, (len(tr) * 10, X.shape[1]))
                ytr = np.tile(ytr, 10)
            clf = make_clf()
            clf.fit(Xtr, ytr)
            pred = clf.predict(X[te])
            if hasattr(clf, "predict_proba"):
                prob = clf.predict_proba(X[te])[:, 1]
            elif hasattr(clf, "decision_function"):
                prob = 1 / (1 + np.exp(-clf.decision_function(X[te])))
            else:
                prob = pred.astype(float)
            errs.append(np.mean(pred != y[te]))
            probs_all.append(prob)
            true_all.append(y[te])
        accs.append(1 - np.mean(errs))
        dps.append(_dprime(np.concatenate(probs_all), np.concatenate(true_all)))
    return np.array(accs), np.array(dps)
