"""
08 · 跨被试 vs 被试内（全通道特征，5 个 sklearn 模型）

两种评估方式的对比：
  - 被试内 (within) : 每个被试各自 10 折 CV（之前的做法）——训练/测试是同一个人
  - 跨被试 (cross)  : 把 22 人合起来，用 GroupKFold 按【被试】分组做 5 折 CV
                      —— 训练集和测试集是【不同的人】，真正测“泛化到新人”

特征：全部 63 通道 × 前 7 个时间点 = 441 维。

运行（项目根目录下）：
    conda run -n eeg python Code/08_cross_subject.py
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
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
                          N_DOWN, N_RESTRICT, DATA_DIR, SUBJECTS, cv_evaluate, PROJECT_ROOT)

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def load_features(subject, cond):
    """某被试某条件 → (试次, 441) 全通道特征。"""
    data = pd.read_csv(DATA_DIR / f"Part{subject:02d}_{cond}-export.mul",
                       sep=r"\s+", skiprows=2, header=None).to_numpy()
    n = data.shape[0] // N_SAMPLES
    tr = data.reshape(n, N_SAMPLES, N_CHANNELS)
    tr = tr - tr[:, :PRES_TIME, :].mean(axis=1, keepdims=True)
    post = tr[:, PRES_TIME:PRES_TIME + N_DOWN * DOWNSAMPLE_WIN, :]
    down = post.reshape(n, N_DOWN, DOWNSAMPLE_WIN, N_CHANNELS).mean(axis=2)
    return down[:, :N_RESTRICT, :].reshape(n, -1)


print("读取并预处理全部 88 个文件 ...")
feats = {}                               # feats[(subject, cond)] = (n, 441)
for s in SUBJECTS:
    for c in ["01", "02", "03", "04"]:
        feats[(s, c)] = load_features(s, c)

# 为两个任务分别准备：被试内(每人一份 X,y) 与 跨被试(合并 X,y,groups)
TASKS = {"High Var (01 vs 03)": ("01", "03"), "Single Img (02 vs 04)": ("02", "04")}
within_data, cross_data = {}, {}
for task, (cf, cu) in TASKS.items():
    per_subj, Xall, yall, gall = [], [], [], []
    for s in SUBJECTS:
        fam, unf = feats[(s, cf)], feats[(s, cu)]
        X = np.vstack([fam, unf])
        y = np.r_[np.zeros(len(fam)), np.ones(len(unf))]
        per_subj.append((X, y))
        Xall.append(X); yall.append(y); gall.append(np.full(len(y), s))
    within_data[task] = per_subj
    cross_data[task] = (np.vstack(Xall), np.concatenate(yall), np.concatenate(gall))

MODELS = {
    "Logistic Reg": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "LDA":          lambda: make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    "SVM (RBF)":    lambda: make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=13)),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=300, random_state=13),
    "Neural Net":   lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-2, max_iter=800, random_state=13)),
}


def cross_subject_cv(X, y, g, make, n_splits=5):
    """按被试分组的 5 折 CV（训练/测试是不同的人）。"""
    gkf = GroupKFold(n_splits=n_splits)
    accs = []
    for tr, te in gkf.split(X, y, g):
        clf = make()
        clf.fit(X[tr], y[tr])
        accs.append(np.mean(clf.predict(X[te]) == y[te]))
    return np.array(accs)


print("跑分类（被试内 + 跨被试）...")
within_acc, cross_acc = {}, {}
for task in TASKS:
    for m, make in MODELS.items():
        within_acc[(m, task)] = cv_evaluate(within_data[task], make, augment=False)[0]
        Xall, yall, gall = cross_data[task]
        cross_acc[(m, task)] = cross_subject_cv(Xall, yall, gall, make)
    print(f"  完成: {task}")

# ---------- 表 ----------
lines = ["====== 被试内 vs 跨被试（准确率, 全通道）======"]
for task in TASKS:
    lines.append(f"-- {task} --")
    lines.append(f"  {'模型':<14}{'被试内':>10}{'跨被试':>10}")
    for m in MODELS:
        lines.append(f"  {m:<14}{within_acc[(m, task)].mean():>9.1%}{cross_acc[(m, task)].mean():>10.1%}")
report = "\n".join(lines)
print("\n" + report)
results_dir = PROJECT_ROOT / "Results"
results_dir.mkdir(exist_ok=True)
(results_dir / "cross_subject.txt").write_text(report + "\n", encoding="utf-8")

# ---------- 图 ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
for ax, task in zip(axes, TASKS):
    x = np.arange(len(MODELS))
    w = 0.38
    wm = [within_acc[(m, task)].mean() for m in MODELS]
    cm = [cross_acc[(m, task)].mean() for m in MODELS]
    we = [within_acc[(m, task)].std(ddof=1) / np.sqrt(22) for m in MODELS]
    ce = [cross_acc[(m, task)].std(ddof=1) / np.sqrt(5) for m in MODELS]
    ax.bar(x - w / 2, wm, w, yerr=we, capsize=3, label="within-subject", color="mediumseagreen")
    ax.bar(x + w / 2, cm, w, yerr=ce, capsize=3, label="cross-subject", color="slateblue")
    ax.axhline(0.5, ls="--", color="gray")
    ax.set_xticks(x); ax.set_xticklabels(list(MODELS), rotation=20, ha="right")
    ax.set_ylim(0.4, 0.8); ax.set_title(task)
axes[0].set_ylabel("Classification accuracy")
axes[0].legend(loc="upper right", fontsize=8)
fig.suptitle("Within-subject vs cross-subject decoding (all 63 channels, N=22)")
plt.tight_layout()
fig_dir = PROJECT_ROOT / "Figures"
fig_dir.mkdir(exist_ok=True)
plt.savefig(fig_dir / "08_cross_subject.png", dpi=150)
print("\n图已保存: Figures/08_cross_subject.png")
print("结果已保存: Results/cross_subject.txt")
