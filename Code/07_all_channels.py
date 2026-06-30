"""
拓展实验：用【全部 63 个通道】当特征，复杂模型会不会反超线性基线？

对比两套特征（同样只取前 7 个时间点 ≈ 前 600ms）：
  - 4 通道  : P9/P10/TP9/TP10 × 7  = 28 维（老师的版本）
  - 全通道  : 63 个通道       × 7  = 441 维
5 个模型：逻辑回归 / LDA / SVM(RBF) / 随机森林 / 神经网络(MLP)。
评估方式与前面完全相同：逐被试 10 折交叉验证（公平对比）。

运行（项目根目录下）：
    conda run -n eeg python Code/07_all_channels.py
"""
import warnings
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, ttest_rel
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import ConvergenceWarning
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eeg_pipeline import (N_SAMPLES, N_CHANNELS, PRES_TIME, DOWNSAMPLE_WIN,
                          N_DOWN, N_RESTRICT, DATA_DIR, SUBJECTS, FACE_IDX,
                          cv_evaluate, PROJECT_ROOT)

warnings.filterwarnings("ignore", category=ConvergenceWarning)   # MLP 在小样本上可能不收敛，忽略提示


def load_tensor(subject, cond):
    """返回某被试某条件的 (试次, 7时间点, 63通道) 张量（基线校正+降采样+取前7点）。"""
    data = pd.read_csv(DATA_DIR / f"Part{subject:02d}_{cond}-export.mul",
                       sep=r"\s+", skiprows=2, header=None).to_numpy()
    n = data.shape[0] // N_SAMPLES
    trials = data.reshape(n, N_SAMPLES, N_CHANNELS)
    trials = trials - trials[:, :PRES_TIME, :].mean(axis=1, keepdims=True)
    post = trials[:, PRES_TIME:PRES_TIME + N_DOWN * DOWNSAMPLE_WIN, :]
    down = post.reshape(n, N_DOWN, DOWNSAMPLE_WIN, N_CHANNELS).mean(axis=2)
    return down[:, :N_RESTRICT, :]                                # (试次, 7, 63)


def build():
    """读 88 个文件（仅一次），同时组装 4 通道 与 全通道 两套特征。"""
    sets = {"4ch": {"HighVar": [], "SingleImg": []},
            "all": {"HighVar": [], "SingleImg": []}}
    for s in SUBJECTS:
        t = {c: load_tensor(s, c) for c in ["01", "02", "03", "04"]}
        for task, (cf, cu) in {"HighVar": ("01", "03"), "SingleImg": ("02", "04")}.items():
            fam, unf = t[cf], t[cu]
            y = np.r_[np.zeros(len(fam)), np.ones(len(unf))]
            # 4 通道：只取人脸通道再展平
            Xf = np.vstack([fam[:, :, FACE_IDX].reshape(len(fam), -1),
                            unf[:, :, FACE_IDX].reshape(len(unf), -1)])
            # 全通道：63 个通道全展平
            Xa = np.vstack([fam.reshape(len(fam), -1), unf.reshape(len(unf), -1)])
            sets["4ch"][task].append((Xf, y))
            sets["all"][task].append((Xa, y))
    return sets


# LDA 用 shrinkage 以应对“特征多于样本”；其余模型套标准化
MODELS = {
    "Logistic Reg": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "LDA":          lambda: make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    "SVM (RBF)":    lambda: make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=13)),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=300, random_state=13),
    "Neural Net":   lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-2, max_iter=800, random_state=13)),
}
TASKS = ["HighVar", "SingleImg"]

print("读取并预处理全部 88 个文件 ...")
sets = build()

print("跑分类（5 模型 × 2 特征集 × 2 任务）...")
results = {}   # (model, feat, task) -> (acc数组, dp数组)
for model_name, make in MODELS.items():
    for feat in ["4ch", "all"]:
        for task in TASKS:
            acc, dp = cv_evaluate(sets[feat][task], make, augment=False)
            results[(model_name, feat, task)] = (acc, dp)
    print(f"  完成: {model_name}")

# ---------- 结果表 ----------
lines = ["====== 4 通道(28维) vs 全 63 通道(441维) 准确率, N=22 ======",
         f"{'模型':<15}{'HighVar 4ch':>13}{'HighVar all':>13}{'SingleImg 4ch':>15}{'SingleImg all':>15}"]
for m in MODELS:
    hv4 = results[(m, '4ch', 'HighVar')][0].mean()
    hva = results[(m, 'all', 'HighVar')][0].mean()
    si4 = results[(m, '4ch', 'SingleImg')][0].mean()
    sia = results[(m, 'all', 'SingleImg')][0].mean()
    lines.append(f"{m:<15}{hv4:>12.1%}{hva:>13.1%}{si4:>14.1%}{sia:>15.1%}")
report = "\n".join(lines)
print("\n" + report)

# ---------- 全通道 vs 4通道 的提升：逐被试配对 t 检验 ----------
paired_lines = ["", "====== 全通道 − 4通道 的提升（配对 t 检验, N=22）======"]
for task in TASKS:
    paired_lines.append(f"-- {'High Var' if task == 'HighVar' else 'Single Img'} --")
    for m in MODELS:
        a4 = results[(m, "4ch", task)][0]
        aa = results[(m, "all", task)][0]
        t, p = ttest_rel(aa, a4)
        flag = "  *显著*" if p < 0.05 else ""
        paired_lines.append(f"  {m:<14}{(aa - a4).mean():+.1%}  (p={p:.3g}){flag}")
paired_report = "\n".join(paired_lines)
print(paired_report)

results_dir = PROJECT_ROOT / "Results"
results_dir.mkdir(exist_ok=True)
(results_dir / "all_channels_comparison.txt").write_text(report + "\n" + paired_report + "\n", encoding="utf-8")

# ---------- 图：每个任务一张子图，每个模型并排画 4通道 vs 全通道 ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
for ax, task in zip(axes, TASKS):
    x = np.arange(len(MODELS))
    w = 0.38
    m4 = [results[(m, "4ch", task)][0].mean() for m in MODELS]
    ma = [results[(m, "all", task)][0].mean() for m in MODELS]
    e4 = [results[(m, "4ch", task)][0].std(ddof=1) / np.sqrt(22) for m in MODELS]
    ea = [results[(m, "all", task)][0].std(ddof=1) / np.sqrt(22) for m in MODELS]
    ax.bar(x - w / 2, m4, w, yerr=e4, capsize=3, label="4 channels (28 feat)", color="lightsteelblue")
    ax.bar(x + w / 2, ma, w, yerr=ea, capsize=3, label="all 63 channels (441 feat)", color="steelblue")
    ax.axhline(0.5, ls="--", color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(list(MODELS), rotation=20, ha="right")
    ax.set_ylim(0.4, 0.8)
    ax.set_title("High Var (01 vs 03)" if task == "HighVar" else "Single Img (02 vs 04)")
axes[0].set_ylabel("Classification accuracy")
axes[0].legend(loc="upper right", fontsize=8)
fig.suptitle("Does using all 63 channels help? (per-subject 10-fold CV, N=22)")
plt.tight_layout()
fig_dir = PROJECT_ROOT / "Figures"
fig_dir.mkdir(exist_ok=True)
plt.savefig(fig_dir / "07_all_channels.png", dpi=150)
print("\n图已保存: Figures/07_all_channels.png")
print("结果已保存: Results/all_channels_comparison.txt")
