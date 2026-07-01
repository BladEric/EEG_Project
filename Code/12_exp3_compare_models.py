"""
12 · Exp3 多模型对比（对标 Exp1 的 06_compare_models.py）。

在同一套 28 维特征（4 人脸通道 ×7 时间点）上，对比 5 个模型在 Exp3 三个任务上的表现。
为公平起见：统一标准化、统一不加噪声增强（所以这里的逻辑回归与 11 号 baseline 会略有差异，属正常）。

运行：
    conda run -n eeg python Code/12_exp3_compare_models.py
"""
import warnings
import numpy as np
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

print("读取并预处理 Exp3（28 维特征）...")
task_data = build_tasks(verbose=True)

print("\n跑 5 个模型 × 3 个任务（逐被试 10 折 CV）...")
acc = {}   # acc[(model, task)] = 每被试准确率数组
for task in TASKS:
    for m, make in MODELS.items():
        acc[(m, task)] = cv_evaluate(task_data[task], make, augment=False, seed=SEED)[0]
    print(f"  完成: {task}")

# ---------- 表 ----------
lines = ["====== Exp3 多模型对比（准确率, 28 维特征, N=19）======",
         f"{'模型':<16}" + "".join(f"{t:>14}" for t in TASKS)]
for m in MODELS:
    lines.append(f"{m:<16}" + "".join(f"{acc[(m, t)].mean():>13.1%}" for t in TASKS))
report = "\n".join(lines)
print("\n" + report)
(PROJECT_ROOT / "Results" / "exp3_model_comparison.txt").write_text(report + "\n", encoding="utf-8")

# ---------- 图 ----------
x = np.arange(len(TASKS))
w = 0.16
plt.figure(figsize=(10, 5.5))
for i, m in enumerate(MODELS):
    means = [acc[(m, t)].mean() for t in TASKS]
    errs = [acc[(m, t)].std(ddof=1) / np.sqrt(19) for t in TASKS]
    plt.bar(x + (i - 2) * w, means, w, yerr=errs, capsize=2, label=m)
plt.axhline(0.5, ls="--", color="gray", label="chance (50%)")
plt.xticks(x, list(TASKS))
plt.ylim(0.4, 0.8)
plt.ylabel("Classification accuracy")
plt.title("Exp3: model comparison on 28-dim face-channel features (N=19)")
plt.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "Figures" / "12_exp3_model_comparison.png", dpi=150)
print("\n图已保存: Figures/12_exp3_model_comparison.png")
print("结果已保存: Results/exp3_model_comparison.txt")
