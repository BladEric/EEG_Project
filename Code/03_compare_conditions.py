"""
第 3 步：把“熟悉的脸”和“陌生的脸”的脑电反应画在一起对比。

这是你论文核心问题的第一张图：熟 vs 生，大脑反应到底有没有差别？
为了干净可信，我们把全部 22 个被试一起平均（grand average，大平均）。
这一步用到的“写函数 + 遍历所有被试”正是后面复现实验要反复用的套路。

运行（项目根目录下）：
    conda run -n eeg python Code/03_compare_conditions.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- 0. 参数 ----------
FS = 1024
N_SAMPLES = 1229
N_CHANNELS = 63
PRES_TIME = 205

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data" / "osfstorage-archive" / "ClassifierModel" / "Exp1"

# 老师用的 22 个被试编号（注意跳过了 6 号和 17 号）
SUBJECTS = list(range(1, 6)) + list(range(7, 17)) + list(range(18, 25))

# ---------- 1. 先从任意一个文件读出 63 个电极名字 ----------
with open(DATA_DIR / "Part01_01-export.mul") as f:
    f.readline()
    channel_names = f.readline().split()
face_idx = [channel_names.index(n) for n in ["P9", "P10", "TP9", "TP10"]]


# ---------- 2. 定义一个“函数”：算某条件下、所有被试的大平均 ERP ----------
# def = “定义一个能反复调用的小工具”。括号里的 cond 是“参数”，
# 调用时传进去（如 "01" 或 "03"），函数内部就用它去拼文件名。
def grand_average_erp(cond):
    """读入所有被试在某条件下的数据，返回大平均 ERP，形状 (时间点, 电极)。"""
    per_subject = []                                  # 装每个被试 ERP 的列表
    for s in SUBJECTS:
        # f"...{s:02d}..." 把数字补成两位：1 → "01"，文件名才对得上
        path = DATA_DIR / f"Part{s:02d}_{cond}-export.mul"
        if not path.exists():
            print(f"  跳过(文件缺失): {path.name}")
            continue
        data = pd.read_csv(path, sep=r"\s+", skiprows=2, header=None).to_numpy()
        n_trials = data.shape[0] // N_SAMPLES
        trials = data.reshape(n_trials, N_SAMPLES, N_CHANNELS)
        erp = trials.mean(axis=0)                      # 该被试该条件：跨试次平均
        erp = erp - erp[:PRES_TIME].mean(axis=0)       # 基线校正
        per_subject.append(erp)
    print(f"  条件 {cond}: 共纳入 {len(per_subject)} 个被试")
    # np.stack 把 22 个 (时间,电极) 摞成 (22,时间,电极)，再沿被试方向求平均
    return np.stack(per_subject).mean(axis=0)


print("正在读取【熟悉】条件 (_01) ...")
erp_fam = grand_average_erp("01")
print("正在读取【陌生】条件 (_03) ...")
erp_unf = grand_average_erp("03")

# ---------- 3. 把 4 个人脸通道再平均成一条波形（便于比较）----------
fam_wave = erp_fam[:, face_idx].mean(axis=1)
unf_wave = erp_unf[:, face_idx].mean(axis=1)
time_ms = (np.arange(N_SAMPLES) - PRES_TIME) / FS * 1000

# ---------- 4. 画对比图 ----------
plt.figure(figsize=(10, 6))
plt.plot(time_ms, fam_wave, label="Familiar (_01)", color="crimson", lw=2)
plt.plot(time_ms, unf_wave, label="Unfamiliar (_03)", color="steelblue", lw=2)
plt.fill_between(time_ms, fam_wave, unf_wave, color="gray", alpha=0.15)  # 阴影=两者之差

plt.axvline(0, color="black", lw=1)
plt.axvspan(130, 200, color="orange", alpha=0.12)
plt.xlabel("Time (ms, 0 = face onset)")
plt.ylabel("Amplitude (µV)")
plt.title("Familiar vs Unfamiliar faces — grand-average ERP at face channels (N=22)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

fig_dir = PROJECT_ROOT / "Figures"
fig_dir.mkdir(exist_ok=True)
out_path = fig_dir / "03_familiar_vs_unfamiliar.png"
plt.savefig(out_path, dpi=150)
print("图已保存到:", out_path)
plt.show()
