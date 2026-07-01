"""
15 · Exp3 的“真·提升”尝试：xDAWN + 黎曼几何（Riemannian geometry）。

这是 ERP 解码公认的 SOTA。和“先抽4通道特征再逻辑回归”不同，它：
  1) 用 xDAWN 学到能放大 ERP 的空间滤波器；
  2) 把每个试次表示成【协方差矩阵】（捕捉通道间的时空协变，信息远比 4 通道丰富）；
  3) 在协方差矩阵所在的弯曲流形上，投到切空间（TangentSpace）后再做逻辑回归。
它专门擅长低信噪比、小样本的 ERP，且仍是线性可解释的——非常适合写进论文。

用全部 64 通道的原始 epoch（64×128），逐被试 10 折 CV，与 28 维逻辑回归 baseline 对比。

运行：
    conda run -n eeg python Code/15_exp3_riemann.py
"""
import warnings
import numpy as np
from scipy.stats import ttest_1samp, ttest_rel
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eeg_pipeline import cv_evaluate, PROJECT_ROOT
from exp3_pipeline import build_tasks, load_epochs, TASKS, SUBJECTS, SEED

warnings.filterwarnings("ignore")
PAPER = {"True vs Unf": 0.69, "Lie vs Unf": 0.61, "True vs Lie": 0.63}


def make_riemann():
    return make_pipeline(
        XdawnCovariances(nfilter=4, estimator="lwf", xdawn_estimator="lwf"),
        TangentSpace(metric="riemann"),
        LogisticRegression(max_iter=1000),
    )


def riemann_cv(X, y, n_folds=10, seed=SEED):
    """单被试 10 折 CV，返回平均准确率。X:(试次,64,128)。"""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(X, y):
        clf = make_riemann()
        clf.fit(X[tr], y[tr])
        accs.append(np.mean(clf.predict(X[te]) == y[te]))
    return np.mean(accs)


# ---------- baseline：28 维逻辑回归（含噪声增强）----------
print("baseline：28 维逻辑回归 ...")
base_tasks = build_tasks()
base_acc = {t: cv_evaluate(base_tasks[t], lambda: LogisticRegression(C=1e3, max_iter=1000),
                           augment=True, seed=SEED)[0] for t in TASKS}

# ---------- 读原始 epoch（全 64 通道）----------
print("读取原始 epoch（64×128）...")
rng = np.random.default_rng(SEED)
epochs = {(s, c): load_epochs(s, c, rng) for s in SUBJECTS for c in ["True", "Lie", "Unf"]}

print("\n跑 xDAWN + Riemann（逐被试 10 折 CV）...")
rie_acc = {}
for task in TASKS:
    c0, c1 = TASKS[task]
    per_subj = []
    for s in SUBJECTS:
        e0, e1 = epochs[(s, c0)], epochs[(s, c1)]
        X = np.concatenate([e0, e1])
        y = np.r_[np.zeros(len(e0)), np.ones(len(e1))].astype(int)
        per_subj.append(riemann_cv(X, y))
    rie_acc[task] = np.array(per_subj)
    print(f"  完成: {task}")

# ---------- 表 ----------
lines = ["====== Exp3：xDAWN+Riemann vs 逻辑回归 baseline（准确率, N=19）======",
         f"{'任务':<14}{'baseline(28d)':>15}{'Riemann(64ch)':>15}{'  Δ':>8}{'  配对t检验':>14}{'  论文':>8}"]
for task in TASKS:
    b, r = base_acc[task], rie_acc[task]
    t, p = ttest_rel(r, b)
    star = " *" if p < 0.05 else ""
    lines.append(f"{task:<14}{b.mean():>14.1%}{r.mean():>15.1%}{r.mean()-b.mean():>+8.1%}"
                 f"   p={p:.3f}{star}{PAPER[task]:>8.0%}")
lines.append("")
for task in TASKS:
    r = rie_acc[task]
    t, p = ttest_1samp(r, 0.5)
    lines.append(f"{task}: Riemann={r.mean():.1%}, vs 50% t={t:.2f} p={p:.3g}")
report = "\n".join(lines)
print("\n" + report)
(PROJECT_ROOT / "Results" / "exp3_riemann.txt").write_text(report + "\n", encoding="utf-8")

# ---------- 图 ----------
x = np.arange(len(TASKS))
w = 0.38
plt.figure(figsize=(8.5, 5.5))
bm = [base_acc[t].mean() for t in TASKS]
rm = [rie_acc[t].mean() for t in TASKS]
be = [base_acc[t].std(ddof=1) / np.sqrt(19) for t in TASKS]
re = [rie_acc[t].std(ddof=1) / np.sqrt(19) for t in TASKS]
plt.bar(x - w/2, bm, w, yerr=be, capsize=4, label="Baseline: LogReg (28-d)", color="steelblue")
plt.bar(x + w/2, rm, w, yerr=re, capsize=4, label="xDAWN + Riemann (64 ch)", color="darkorange")
plt.axhline(0.5, ls="--", color="gray", label="chance (50%)")
plt.xticks(x, list(TASKS))
plt.ylim(0.4, 0.8)
plt.ylabel("Classification accuracy")
plt.title("Exp3: xDAWN + Riemannian geometry vs baseline (N=19)")
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "Figures" / "15_exp3_riemann.png", dpi=150)
print("\n图已保存: Figures/15_exp3_riemann.png")
print("结果已保存: Results/exp3_riemann.txt")
