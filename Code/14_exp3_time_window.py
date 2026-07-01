"""
14 · Exp3 时间窗扫描 —— 论文驱动的“提升”实验（参数对比）。

老师代码只取刺激后前 7 个降采样点（≈前 680ms）。但论文明确说分类峰值出现在更晚：
  - 如实承认(acknowledged) vs 陌生：峰值 500–700ms
  - 隐瞒(concealed) vs 陌生：峰值 200–400ms
所以“用多长的时间窗”是个真正值得扫的参数。这里把窗口从 1 个点扫到 10 个点（≈全 epoch），
看三个任务（尤其最难的 Lie vs Unf）准确率怎么变。

每个降采样点 = 50/512 ≈ 97.7ms。窗口 = n 个点 ≈ 0 ~ n×97.7ms。

运行：
    conda run -n eeg python Code/14_exp3_time_window.py
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eeg_pipeline import cv_evaluate, PROJECT_ROOT
from exp3_pipeline import build_tasks, TASKS, SEED, N_DOWN, DOWNSAMPLE_WIN, FS

MS_PER_POINT = DOWNSAMPLE_WIN / FS * 1000           # ≈97.7ms
WINDOWS = list(range(1, N_DOWN + 1))                # 1..10 个时间点
make_logreg = lambda: LogisticRegression(C=1e3, max_iter=1000)   # 同 baseline（含噪声增强）

print("逐窗口重建特征并分类（逻辑回归 + 噪声增强, 逐被试 10 折 CV）...")
# acc_curve[task] = 每个窗口的平均准确率
acc_curve = {t: [] for t in TASKS}
sem_curve = {t: [] for t in TASKS}
for n in WINDOWS:
    data = build_tasks(n_restrict=n)                # 取前 n 个时间点 × 4 通道
    for task in TASKS:
        acc = cv_evaluate(data[task], make_logreg, augment=True, seed=SEED)[0]
        acc_curve[task].append(acc.mean())
        sem_curve[task].append(acc.std(ddof=1) / np.sqrt(len(acc)))
    print(f"  窗口 {n:>2} 点 (≈{n*MS_PER_POINT:.0f}ms) 完成")

# ---------- 表 ----------
BASE = N_DOWN if False else 7                        # 老师默认窗口 = 7 点
lines = ["====== Exp3 时间窗扫描（逻辑回归, 准确率, N=19）======",
         f"{'窗口(点)':<10}{'≈时长':>10}" + "".join(f"{t:>14}" for t in TASKS)]
for i, n in enumerate(WINDOWS):
    lines.append(f"{n:<10}{n*MS_PER_POINT:>8.0f}ms" + "".join(f"{acc_curve[t][i]:>13.1%}" for t in TASKS))
lines.append("")
for task in TASKS:
    base = acc_curve[task][7 - 1]                    # 7 点处
    best_i = int(np.argmax(acc_curve[task]))
    best = acc_curve[task][best_i]
    lines.append(f"{task}: 默认7点={base:.1%} | 最佳={best:.1%} @ {WINDOWS[best_i]}点"
                 f"(≈{WINDOWS[best_i]*MS_PER_POINT:.0f}ms) | Δ={best-base:+.1%}")
report = "\n".join(lines)
print("\n" + report)
(PROJECT_ROOT / "Results" / "exp3_time_window.txt").write_text(report + "\n", encoding="utf-8")

# ---------- 图：准确率随时间窗的曲线 ----------
plt.figure(figsize=(9, 5.5))
ms = [n * MS_PER_POINT for n in WINDOWS]
for task in TASKS:
    plt.errorbar(ms, acc_curve[task], yerr=sem_curve[task], marker="o", capsize=3, label=task)
plt.axhline(0.5, ls="--", color="gray", label="chance (50%)")
plt.axvline(7 * MS_PER_POINT, ls=":", color="black", alpha=0.6, label="paper/teacher window (7 pts)")
plt.xlabel("Time window end (ms post-stimulus)")
plt.ylabel("Classification accuracy")
plt.ylim(0.45, 0.8)
plt.title("Exp3: does a longer time window help? (logistic regression, N=19)")
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "Figures" / "14_exp3_time_window.png", dpi=150)
print("\n图已保存: Figures/14_exp3_time_window.png")
print("结果已保存: Results/exp3_time_window.txt")
