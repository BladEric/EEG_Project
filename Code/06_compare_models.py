"""
第 3 步 · 在同一套特征上，比较 4 个分类模型。

模型：逻辑回归(基线) / LDA / SVM(RBF核) / 随机森林。
用与复现完全相同的“逐被试 10 折交叉验证”，保证公平对比。
（这里不加噪声增强；除随机森林外都套 StandardScaler 标准化，让各模型各展所长。）

输出：准确率对比表 + 分组柱状图。

运行（项目根目录下）：
    conda run -n eeg python Code/06_compare_models.py
"""
import numpy as np
from scipy.stats import ttest_1samp
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eeg_pipeline import build_dataset, cv_evaluate, PROJECT_ROOT

# 4 个模型：每个都是“返回新模型”的函数（换模型只改这里——就是当初说的那一行！）
MODELS = {
    "Logistic Reg": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "LDA":          lambda: make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
    "SVM (RBF)":    lambda: make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=13)),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=200, random_state=13),
}
TASKS = {"High Var (01 vs 03)": "HighVar", "Single Img (02 vs 04)": "SingleImg"}

print("读取并预处理全部 88 个文件 ...")
data = build_dataset(verbose=False)

# results[task][model] = (准确率数组, d′数组)
results = {task: {} for task in TASKS}
for task_name, key in TASKS.items():
    print(f"\n=== 任务: {task_name} ===")
    for model_name, make in MODELS.items():
        acc, dp = cv_evaluate(data[key], make, augment=False)
        results[task_name][model_name] = (acc, dp)
        _, p = ttest_1samp(acc, 0.5)
        print(f"  {model_name:14s}: 准确率 {acc.mean():.1%}  d'={dp.mean():.2f}  (p={p:.2g})")

# ---------- 结果表（文本）----------
lines = ["====== Exp1 模型对比（准确率, N=22 被试）======",
         f"{'模型':<16}{'High Var':>12}{'Single Img':>14}"]
for model_name in MODELS:
    a_hv = results["High Var (01 vs 03)"][model_name][0].mean()
    a_si = results["Single Img (02 vs 04)"][model_name][0].mean()
    lines.append(f"{model_name:<16}{a_hv:>11.1%}{a_si:>13.1%}")
report = "\n".join(lines)
print("\n" + report)

results_dir = PROJECT_ROOT / "Results"
results_dir.mkdir(exist_ok=True)
(results_dir / "model_comparison.txt").write_text(report + "\n", encoding="utf-8")

# ---------- 分组柱状图 ----------
model_names = list(MODELS)
task_names = list(TASKS)
x = np.arange(len(task_names))
width = 0.2
plt.figure(figsize=(9, 5.5))
for i, model_name in enumerate(model_names):
    means = [results[t][model_name][0].mean() for t in task_names]
    sems = [results[t][model_name][0].std(ddof=1) / np.sqrt(22) for t in task_names]
    plt.bar(x + (i - 1.5) * width, means, width, yerr=sems, capsize=4, label=model_name)
plt.axhline(0.5, ls="--", color="gray", label="chance (50%)")
plt.xticks(x, task_names)
plt.ylim(0.4, 0.8)
plt.ylabel("Classification accuracy")
plt.title("Exp1: comparison of 4 classifiers (10-fold CV, N=22)")
plt.legend(ncol=2, fontsize=9)
plt.tight_layout()
fig_dir = PROJECT_ROOT / "Figures"
fig_dir.mkdir(exist_ok=True)
plt.savefig(fig_dir / "06_model_comparison.png", dpi=150)
print("\n图已保存: Figures/06_model_comparison.png")
print("结果已保存: Results/model_comparison.txt")
