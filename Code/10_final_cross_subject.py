"""
10 · 最终汇总图：跨被试设定下 6 个模型（含 EEGNet）的准确率对比。

数字来自本项目已跑过的：
  - 08_cross_subject.py  → 5 个 sklearn 模型（按被试分组 5 折 CV，全通道）
  - 09_eegnet.py         → EEGNet（同样按被试分组 5 折 CV，原始 63×128 epoch）

运行（项目根目录下）：
    conda run -n eeg python Code/10_final_cross_subject.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from eeg_pipeline import PROJECT_ROOT

MODELS = ["Logistic Reg", "LDA", "SVM (RBF)", "Random Forest", "Neural Net (MLP)", "EEGNet (CNN)"]
HIGHVAR = [62.5, 63.8, 64.4, 64.3, 62.2, 64.2]      # High Var (01 vs 03) 跨被试准确率 %
SINGLEIMG = [54.7, 55.6, 55.8, 55.4, 54.9, 54.4]    # Single Img (02 vs 04) 跨被试准确率 %

x = np.arange(len(MODELS))
w = 0.38
plt.figure(figsize=(11, 5.5))
plt.bar(x - w / 2, HIGHVAR, w, label="High Var (01 vs 03)", color="steelblue")
plt.bar(x + w / 2, SINGLEIMG, w, label="Single Img (02 vs 04)", color="indianred")
plt.axhline(50, ls="--", color="gray", label="chance (50%)")
plt.xticks(x, MODELS, rotation=18, ha="right")
plt.ylim(40, 75)
plt.ylabel("Cross-subject accuracy (%)")
plt.title("Cross-subject decoding: no model (incl. EEGNet) beats the simple baselines")
plt.legend()
plt.tight_layout()

fig_dir = PROJECT_ROOT / "Figures"
fig_dir.mkdir(exist_ok=True)
plt.savefig(fig_dir / "10_final_cross_subject.png", dpi=150)

# 同时存一份表
lines = ["====== 跨被试准确率汇总（6 模型, 按被试分组 5 折 CV）======",
         f"{'模型':<18}{'High Var':>10}{'Single Img':>12}"]
for m, hv, si in zip(MODELS, HIGHVAR, SINGLEIMG):
    lines.append(f"{m:<18}{hv:>9.1f}%{si:>11.1f}%")
report = "\n".join(lines)
results_dir = PROJECT_ROOT / "Results"
results_dir.mkdir(exist_ok=True)
(results_dir / "final_cross_subject.txt").write_text(report + "\n", encoding="utf-8")
print(report)
print("\n图已保存: Figures/10_final_cross_subject.png")
