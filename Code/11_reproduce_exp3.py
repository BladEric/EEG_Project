"""
11 · 忠实复现老师的 Exp3（测谎 / 隐瞒熟悉度）分类分析。

对应老师 MATLAB 脚本 EEGanalysis_script_Exp2n3.m 里 exp2or3=0（Exp3）的分支。

Exp3 是三个实验里最有“司法测谎”价值的：被试对一张【自己真正认识】的脸，
被要求撒谎答“不认识”。核心问题：即使他嘴上否认，EEG 还能不能出卖他？

三个二分类任务（对应论文 Wiese 2021, Figure 2 / 4.2 节）：
  - True vs Unf  (如实承认的熟脸 vs 陌生脸)   —— 论文 0.69，标准熟悉度对照
  - Lie  vs Unf  (撒谎否认的熟脸 vs 陌生脸)   —— 论文 0.61 ★核心：行为上两者都答“不认识”，仅凭脑电能否区分？
  - True vs Lie  (如实 vs 撒谎，都是真熟脸)   —— 论文 0.63，撒谎是否改变了脑电？

与 Exp1 的关键差异（全部来自 MATLAB 脚本）：
  fs=512(不是1024) | nSamples=614 | nChannels=64 | 降采样窗=50 | rng=10
  4 个人脸通道（1-based 索引 9/12/33/36 = P9/P10/TP9/TP10）在 64 通道里同样成立。

运行（项目根目录下）：
    conda run -n eeg python Code/11_reproduce_exp3.py
"""
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eeg_pipeline import cv_evaluate, PROJECT_ROOT   # 这两个是“实验无关”的，直接复用

# ---------- 与 MATLAB 脚本一致的 Exp3 参数 ----------
FS = 512
N_SAMPLES = 614
N_CHANNELS = 64
PRES_TIME = int(np.ceil(FS * 0.200))                  # 103：刺激前点数
DOWNSAMPLE_WIN = 50                                   # nSampleDown/2（因为采样率减半）
N_DOWN = 10                                           # signDownSamples-1（同 Exp1 的有效降采样点数）
N_RESTRICT = int(np.ceil(0.6 * FS / DOWNSAMPLE_WIN))  # 7：只取前 ~600ms
FACE_IDX = [8, 11, 32, 35]                            # 0-based：P9/P10/TP9/TP10（MATLAB 9/12/33/36）

DATA_DIR = PROJECT_ROOT / "Data" / "osfstorage-archive" / "ClassifierModel" / "Exp3"
SUBJECTS = list(range(2, 21))                          # 19 人：Part02–Part20
SEED = 10                                              # MATLAB 这个脚本用 rng(10)


def _find_file(subject, cond):
    """Exp3 文件命名大小写不统一（Part02 用 Lie/True/Unf，Part04+ 用小写），两种都试。"""
    for name in (cond, cond.lower(), cond.capitalize()):
        p = DATA_DIR / f"Part{subject:02d}_{name}-export.mul"
        if p.exists():
            return p
    raise FileNotFoundError(f"找不到 被试{subject:02d} 的 {cond} 文件")


def condition_features(subject, cond, rng):
    """某被试某条件 → 每个试次的 28 维特征 (试次数, 28)。步骤同 Exp1，只是换了 Exp3 的常量。"""
    data = pd.read_csv(_find_file(subject, cond), sep=r"\s+", skiprows=2, header=None).to_numpy()
    data = data[:, :N_CHANNELS]                                   # 取前 64 通道
    n_trials = data.shape[0] // N_SAMPLES
    trials = data.reshape(n_trials, N_SAMPLES, N_CHANNELS)        # (试次,时间,电极)

    trials = trials - trials[:, :PRES_TIME, :].mean(axis=1, keepdims=True)   # 基线校正
    post = trials[:, PRES_TIME:PRES_TIME + N_DOWN * DOWNSAMPLE_WIN, :]       # 刺激后前 500 点
    down = post.reshape(n_trials, N_DOWN, DOWNSAMPLE_WIN, N_CHANNELS).mean(axis=2)  # 降采样→(试次,10,电极)
    down = down + rng.normal(0, 1e-4, down.shape)                # MATLAB 加的极小噪声（防某些通道恒为0）

    feat = down[:, :N_RESTRICT, :][:, :, FACE_IDX]              # (试次,7,4)
    return feat.reshape(n_trials, -1)                           # (试次,28)


# ---------- 读入全部数据，组装三个二分类任务 ----------
print("读取并预处理 Exp3 全部文件 ...")
rng = np.random.default_rng(SEED)
# 三个任务：(标签0条件, 标签1条件)。标签遵循 MATLAB（True vs Lie 里 Lie=0,True=1）。
TASKS = {
    "True vs Unf": ("True", "Unf"),
    "Lie vs Unf":  ("Lie",  "Unf"),
    "True vs Lie": ("Lie",  "True"),
}
feats = {}
for s in SUBJECTS:
    for c in ["True", "Lie", "Unf"]:
        feats[(s, c)] = condition_features(s, c, rng)
    print(f"  已处理被试 {s:02d}")

task_data = {}
for task, (c0, c1) in TASKS.items():
    per_subj = []
    for s in SUBJECTS:
        f0, f1 = feats[(s, c0)], feats[(s, c1)]
        X = np.vstack([f0, f1])
        y = np.r_[np.zeros(len(f0)), np.ones(len(f1))]
        per_subj.append((X, y))
    task_data[task] = per_subj

# ---------- 逐被试 10 折交叉验证 + 逻辑回归 + 噪声增强 ----------
print("\n跑分类（逐被试 10 折 CV + 噪声增强）...")
make_logreg = lambda: LogisticRegression(C=1e3, max_iter=1000)   # C 很大≈不正则，靠噪声增强防过拟合
results = {}
for task in TASKS:
    acc, dp = cv_evaluate(task_data[task], make_logreg, augment=True, seed=SEED)
    results[task] = (acc, dp)


def sem(a):
    return a.std(ddof=1) / np.sqrt(len(a))


# 论文 Wiese 2021 的对标值
PAPER = {"True vs Unf": 0.69, "Lie vs Unf": 0.61, "True vs Lie": 0.63}

lines = ["====== Exp3 复现结果（逻辑回归, N=19 被试）======",
         f"{'任务':<14}{'本复现':>10}{'论文值':>9}{'d-prime':>10}{'   t检验 vs 50%'}"]
for task in TASKS:
    acc, dp = results[task]
    t, p = ttest_1samp(acc, 0.5)
    star = "  *显著*" if p < 0.05 else ""
    lines.append(f"{task:<14}{acc.mean():>9.1%}{PAPER[task]:>9.0%}{dp.mean():>10.2f}"
                 f"   t={t:.2f}, p={p:.3g}{star}")
report = "\n".join(lines)
print("\n" + report)

results_dir = PROJECT_ROOT / "Results"
results_dir.mkdir(exist_ok=True)
(results_dir / "exp3_reproduction.txt").write_text(report + "\n", encoding="utf-8")

# ---------- 柱状图：本复现 vs 论文 ----------
tasks = list(TASKS)
mine = [results[t][0].mean() for t in tasks]
sems = [sem(results[t][0]) for t in tasks]
paper = [PAPER[t] for t in tasks]
x = np.arange(len(tasks))
w = 0.38
plt.figure(figsize=(8, 5.5))
plt.bar(x - w / 2, mine, w, yerr=sems, capsize=5, label="This reproduction (Python)", color="steelblue")
plt.bar(x + w / 2, paper, w, label="Paper (Wiese 2021)", color="indianred", alpha=0.85)
plt.axhline(0.5, ls="--", color="gray", label="chance (50%)")
plt.xticks(x, tasks)
plt.ylim(0.4, 0.8)
plt.ylabel("Classification accuracy")
plt.title("Exp3 reproduction: detecting (concealed) familiarity\n(logistic regression, N=19)")
plt.legend()
plt.tight_layout()
fig_dir = PROJECT_ROOT / "Figures"
fig_dir.mkdir(exist_ok=True)
plt.savefig(fig_dir / "11_exp3_reproduction.png", dpi=150)
print("\n图已保存: Figures/11_exp3_reproduction.png")
print("结果已保存: Results/exp3_reproduction.txt")
