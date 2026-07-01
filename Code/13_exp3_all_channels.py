"""
13 · Exp3 通道数对比：4 个人脸通道 vs 全 64 通道（对标 Exp1 的 07_all_channels.py）。

问题：老师只用 4 个人脸通道（28 维）。如果把全部 64 通道喂进去（7 点 ×64 = 448 维），
各模型能不能涨？尤其是最难的 Lie vs Unf。用逐被试配对 t 检验看提升是否显著。

运行：
    conda run -n eeg python Code/13_exp3_all_channels.py
"""
import warnings
import numpy as np
from scipy.stats import ttest_rel
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

from eeg_pipeline import cv_evaluate, PROJECT_ROOT
from exp3_pipeline import build_tasks, TASKS, SEED

warnings.filterwarnings("ignore", category=ConvergenceWarning)

MODELS = {
    "Logistic Reg":  lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "LDA":           lambda: make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    "SVM (RBF)":     lambda: make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=13)),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=300, random_state=13),
    "Neural Net":    lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-2, max_iter=800, random_state=13)),
}

print("读取 4 通道(28维) 与 全64通道(448维) 两套特征 ...")
data_4ch = build_tasks(channels=None or [8, 11, 32, 35])   # 4 个人脸通道
data_all = build_tasks(channels="all")                      # 全 64 通道

print("\n跑分类（5 模型 × 3 任务 × 2 套特征, 逐被试 10 折 CV）...")
acc4, accA = {}, {}
for task in TASKS:
    for m, make in MODELS.items():
        acc4[(m, task)] = cv_evaluate(data_4ch[task], make, augment=False, seed=SEED)[0]
        accA[(m, task)] = cv_evaluate(data_all[task], make, augment=False, seed=SEED)[0]
    print(f"  完成: {task}")

# ---------- 表（含配对 t 检验：全通道 vs 4通道）----------
lines = ["====== Exp3：4 通道 vs 全 64 通道（准确率, N=19）======"]
for task in TASKS:
    lines.append(f"\n-- {task} --")
    lines.append(f"  {'模型':<16}{'4通道':>10}{'全64通道':>11}{'  Δ':>8}{'  配对t检验'}")
    for m in MODELS:
        a4, aA = acc4[(m, task)], accA[(m, task)]
        t, p = ttest_rel(aA, a4)
        star = " *" if p < 0.05 else ""
        lines.append(f"  {m:<16}{a4.mean():>9.1%}{aA.mean():>10.1%}{aA.mean()-a4.mean():>+8.1%}"
                     f"   p={p:.3f}{star}")
report = "\n".join(lines)
print("\n" + report)
(PROJECT_ROOT / "Results" / "exp3_all_channels.txt").write_text(report + "\n", encoding="utf-8")

# ---------- 图 ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for ax, task in zip(axes, TASKS):
    x = np.arange(len(MODELS))
    w = 0.38
    m4 = [acc4[(m, task)].mean() for m in MODELS]
    mA = [accA[(m, task)].mean() for m in MODELS]
    ax.bar(x - w/2, m4, w, label="4 channels (28-d)", color="mediumseagreen")
    ax.bar(x + w/2, mA, w, label="all 64 channels (448-d)", color="slateblue")
    ax.axhline(0.5, ls="--", color="gray")
    ax.set_xticks(x); ax.set_xticklabels(list(MODELS), rotation=20, ha="right")
    ax.set_ylim(0.4, 0.8); ax.set_title(task)
axes[0].set_ylabel("Classification accuracy")
axes[0].legend(loc="upper right", fontsize=8)
fig.suptitle("Exp3: 4 face channels vs all 64 channels (N=19)")
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "Figures" / "13_exp3_all_channels.png", dpi=150)
print("\n图已保存: Figures/13_exp3_all_channels.png")
print("结果已保存: Results/exp3_all_channels.txt")
