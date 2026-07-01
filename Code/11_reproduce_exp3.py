"""
11 · 忠实复现老师的 Exp3（测谎 / 隐瞒熟悉度）分类分析 —— baseline。

对应老师 MATLAB 脚本 EEGanalysis_script_Exp2n3.m 里 exp2or3=0（Exp3）的分支。
Exp3 是三个实验里最有“司法测谎”价值的：被试对一张【自己真正认识】的脸，
被要求撒谎答“不认识”。核心问题：即使他嘴上否认，EEG 还能不能出卖他？

三个二分类任务（对应论文 Wiese 2021, Figure 2 / 4.2 节）：
  - True vs Unf  (如实承认的熟脸 vs 陌生脸)   —— 论文 0.69，标准熟悉度对照
  - Lie  vs Unf  (撒谎否认的熟脸 vs 陌生脸)   —— 论文 0.61 ★核心：行为上两者都答“不认识”，仅凭脑电能否区分？
  - True vs Lie  (如实 vs 撒谎，都是真熟脸)   —— 论文 0.63，撒谎是否改变了脑电？

参数/加载全部来自共享模块 exp3_pipeline.py（fs=512、nSamples=614、nChannels=64、降采样窗=50）。

运行（项目根目录下）：
    conda run -n eeg python Code/11_reproduce_exp3.py
"""
import numpy as np
from scipy.stats import ttest_1samp
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eeg_pipeline import cv_evaluate, PROJECT_ROOT          # 实验无关，直接复用
from exp3_pipeline import build_tasks, TASKS, SEED

# 论文 Wiese 2021 的对标值
PAPER = {"True vs Unf": 0.69, "Lie vs Unf": 0.61, "True vs Lie": 0.63}

print("读取并预处理 Exp3 全部文件 ...")
task_data = build_tasks(verbose=True)                        # 默认 4 通道 ×7 时间点 = 28 维

print("\n跑分类（逐被试 10 折 CV + 噪声增强）...")
make_logreg = lambda: LogisticRegression(C=1e3, max_iter=1000)   # C 很大≈不正则，靠噪声增强防过拟合
results = {t: cv_evaluate(task_data[t], make_logreg, augment=True, seed=SEED) for t in TASKS}


def sem(a):
    return a.std(ddof=1) / np.sqrt(len(a))


lines = ["====== Exp3 复现结果（逻辑回归 + 噪声增强, N=19 被试）======",
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
