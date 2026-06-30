"""
第 2 步 · 忠实复现老师的 Exp1 分类分析。

逐被试做 10 折交叉验证 + 逻辑回归 + 噪声增强，两个分类任务：
  - High Var  : 高多样性下 熟(01) vs 生(03)
  - Single Img: 单图下     熟(02) vs 生(04)
输出：每个任务的平均准确率、d′、对 50% 的 t 检验，以及一张柱状图。

运行（项目根目录下）：
    conda run -n eeg python Code/05_reproduce_exp1.py
"""
import numpy as np
from scipy.stats import ttest_1samp
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use("Agg")            # 无窗口模式，直接存图
import matplotlib.pyplot as plt

from eeg_pipeline import build_dataset, cv_evaluate, PROJECT_ROOT

print("读取并预处理全部 88 个文件 ...")
data = build_dataset()

# 逻辑回归：C 设很大≈不加正则，靠老师的噪声增强来防过拟合（贴近 MATLAB 的 fitglm）
make_logreg = lambda: LogisticRegression(C=1e3, max_iter=1000)

print("\n跑分类（逐被试 10 折交叉验证 + 噪声增强）...")
accHV, dpHV = cv_evaluate(data["HighVar"], make_logreg, augment=True)
accLV, dpLV = cv_evaluate(data["SingleImg"], make_logreg, augment=True)


def sem(a):
    return a.std(ddof=1) / np.sqrt(len(a))


lines = ["====== Exp1 复现结果（逻辑回归, N=22 被试）======"]
for name, acc, dp in [("High Var  (01 vs 03)", accHV, dpHV),
                      ("Single Img(02 vs 04)", accLV, dpLV)]:
    t, p = ttest_1samp(acc, 0.5)
    star = "  *显著高于随机*" if p < 0.05 else ""
    lines.append(f"{name}: 准确率 {acc.mean():.1%} ± {sem(acc):.1%}(SEM) | "
                 f"d'={dp.mean():.2f} | t检验 vs 50%: t={t:.2f}, p={p:.4g}{star}")
report = "\n".join(lines)
print("\n" + report)

# ---------- 保存结果文本 ----------
results_dir = PROJECT_ROOT / "Results"
results_dir.mkdir(exist_ok=True)
(results_dir / "exp1_reproduction.txt").write_text(report + "\n", encoding="utf-8")

# ---------- 柱状图（对应老师的主图）----------
means = [accHV.mean(), accLV.mean()]
sems = [sem(accHV), sem(accLV)]
plt.figure(figsize=(6, 5))
plt.bar(["High Var\n(01 vs 03)", "Single Img\n(02 vs 04)"], means, yerr=sems,
        capsize=6, color=["steelblue", "indianred"], width=0.6)
plt.axhline(0.5, ls="--", color="gray", label="chance (50%)")
for x, acc in enumerate([accHV, accLV]):
    _, p = ttest_1samp(acc, 0.5)
    if p < 0.05:
        plt.text(x, means[x] + sems[x] + 0.012, "*", ha="center", fontsize=18)
plt.ylim(0.4, 0.8)
plt.ylabel("Classification accuracy")
plt.title("Exp1 reproduction: familiar vs unfamiliar\n(logistic regression, N=22)")
plt.legend()
plt.tight_layout()
fig_dir = PROJECT_ROOT / "Figures"
fig_dir.mkdir(exist_ok=True)
plt.savefig(fig_dir / "05_exp1_reproduction.png", dpi=150)
print("\n图已保存: Figures/05_exp1_reproduction.png")
print("结果已保存: Results/exp1_reproduction.txt")
