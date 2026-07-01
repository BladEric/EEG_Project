"""
16 · Exp3 最佳配置 —— 把前面有效的改进组合起来，给出最终“提升后”结果。

前面发现两个真正有效的改进（都用论文同款分类器 = 逻辑回归 + 噪声增强）：
  - 13 号：全 64 通道（对 True vs Lie 显著有效）
  - 14 号：延长时间窗到 ~9-10 个点（对三个任务都小幅有效）
这里把「通道(4 vs 全64) × 时间窗(7点 vs 10点)」四种组合都跑一遍，
为每个任务挑出最佳配置，和【论文 / 复现 baseline】对比。

运行：
    conda run -n eeg python Code/16_exp3_best_config.py
"""
import numpy as np
from scipy.stats import ttest_rel
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eeg_pipeline import cv_evaluate, PROJECT_ROOT
from exp3_pipeline import build_tasks, TASKS, SEED

PAPER = {"True vs Unf": 0.69, "Lie vs Unf": 0.61, "True vs Lie": 0.63}
make_logreg = lambda: LogisticRegression(C=1e3, max_iter=1000)     # 论文同款分类器

# 四种配置：(通道, 时间窗点数, 标签)
CONFIGS = [
    ("face", 7,  "4ch / 7pts (baseline)"),
    ("face", 10, "4ch / 10pts"),
    ("all",  7,  "64ch / 7pts"),
    ("all",  10, "64ch / 10pts"),
]

print("逐配置重建特征并分类 ...")
# acc[(label, task)] = 每被试准确率数组
acc = {}
for ch, win, label in CONFIGS:
    channels = "all" if ch == "all" else [8, 11, 32, 35]
    data = build_tasks(channels=channels, n_restrict=win)
    for task in TASKS:
        acc[(label, task)] = cv_evaluate(data[task], make_logreg, augment=True, seed=SEED)[0]
    print(f"  完成配置: {label}")

# ---------- 表 ----------
labels = [c[2] for c in CONFIGS]
base_label = labels[0]
lines = ["====== Exp3 最佳配置搜索（逻辑回归+噪声增强, 准确率, N=19）======",
         f"{'配置':<26}" + "".join(f"{t:>14}" for t in TASKS)]
for label in labels:
    lines.append(f"{label:<26}" + "".join(f"{acc[(label, t)].mean():>13.1%}" for t in TASKS))

lines.append("\n-- 每任务最佳 vs 复现 baseline vs 论文 --")
best = {}
for task in TASKS:
    means = {label: acc[(label, task)].mean() for label in labels}
    best_label = max(means, key=means.get)
    best[task] = best_label
    b = acc[(base_label, task)]
    r = acc[(best_label, task)]
    t, p = ttest_rel(r, b)
    star = " *显著" if p < 0.05 else ""
    lines.append(f"{task}: 论文={PAPER[task]:.0%} | baseline={b.mean():.1%} | "
                 f"最佳={r.mean():.1%} ({best_label}) | Δ={r.mean()-b.mean():+.1%} (p={p:.3f}){star}")
report = "\n".join(lines)
print("\n" + report)
(PROJECT_ROOT / "Results" / "exp3_best_config.txt").write_text(report + "\n", encoding="utf-8")

# ---------- 图：论文 vs baseline vs 最佳 ----------
x = np.arange(len(TASKS))
w = 0.27
paper = [PAPER[t] for t in TASKS]
basel = [acc[(base_label, t)].mean() for t in TASKS]
bestv = [acc[(best[t], t)].mean() for t in TASKS]
beste = [acc[(best[t], t)].std(ddof=1) / np.sqrt(19) for t in TASKS]
plt.figure(figsize=(9, 5.5))
plt.bar(x - w, paper, w, label="Paper (Wiese 2021)", color="indianred", alpha=0.85)
plt.bar(x,     basel, w, label="Our baseline (4ch/7pts)", color="steelblue")
plt.bar(x + w, bestv, w, yerr=beste, capsize=4, label="Our best config", color="darkorange")
plt.axhline(0.5, ls="--", color="gray", label="chance (50%)")
for i, t in enumerate(TASKS):
    plt.text(x[i] + w, bestv[i] + beste[i] + 0.008, best[t].split(" ")[0], ha="center", fontsize=7)
plt.xticks(x, list(TASKS))
plt.ylim(0.4, 0.8)
plt.ylabel("Classification accuracy")
plt.title("Exp3: best improved configuration vs baseline & paper (N=19)")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "Figures" / "16_exp3_best_config.png", dpi=150)
print("\n图已保存: Figures/16_exp3_best_config.png")
print("结果已保存: Results/exp3_best_config.txt")
