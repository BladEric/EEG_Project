"""
第 2 步：把脑电数据画成波形图，亲眼看见大脑“看到一张脸”的反应（著名的 N170）。

运行方法（在项目根目录下）：
    conda run -n eeg python Code/02_visualize_erp.py

跑完会在 Figures/ 文件夹生成一张 PNG 图（可能还会弹出一个窗口）。
"""

from pathlib import Path
import numpy as np                 # 新工具①：做数组数学运算（比如求平均）
import pandas as pd
import matplotlib.pyplot as plt    # 新工具②：画图

# ---------- 0. 固定参数（来自老师的实验设定）----------
FS = 1024            # 采样率：每秒 1024 个点
N_SAMPLES = 1229     # 每个试次的时间点数
N_CHANNELS = 63      # 电极数
PRES_TIME = 205      # 刺激出现前的采样点数（前约 200 毫秒是“基线”）

# ---------- 1. 找到并读入数据（和第 1 步一样）----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data" / "osfstorage-archive" / "ClassifierModel" / "Exp1"
mul_file = DATA_DIR / "Part01_01-export.mul"   # 被试01，条件_01（熟悉的脸 + 高多样性）

# 先读第 2 行，拿到 63 个电极的名字（后面要靠名字找到“人脸通道”）
with open(mul_file) as f:
    f.readline()                      # 跳过第 1 行元信息
    channel_names = f.readline().split()

# 读数字部分，并用 .to_numpy() 把 pandas 表格转成 numpy 数组（更适合做数学运算）
data = pd.read_csv(mul_file, sep=r"\s+", skiprows=2, header=None).to_numpy()
print("读入数据形状 (时间点, 电极):", data.shape)

# ---------- 2. 把竖着摞在一起的试次切开 ----------
# 现在 data 是 (54076, 63)，而 54076 = 44 试次 × 1229 点。
# reshape 把它重排成三维：(试次, 时间点, 电极)。
n_trials = data.shape[0] // N_SAMPLES       # // 是“整除”，54076 // 1229 = 44
trials = data.reshape(n_trials, N_SAMPLES, N_CHANNELS)
print(f"切成试次后形状 (试次, 时间点, 电极): {trials.shape}  → 共 {n_trials} 个试次")

# ---------- 3. 把所有试次平均起来，得到 ERP ----------
# 关键概念：单个试次里，真正的脑反应被随机噪声淹没，几乎看不出来。
# 但把几十个试次一平均，随机噪声会相互抵消，而每次都被“脸”稳定引发的
# 那部分信号会保留下来——这就是 ERP（事件相关电位）。
# axis=0 表示“沿试次方向求平均”，结果形状变成 (时间点, 电极)。
erp = trials.mean(axis=0)
print("平均后 ERP 形状 (时间点, 电极):", erp.shape)

# ---------- 4. 基线校正 ----------
# 用刺激出现前(前 205 个点)的平均值当作每个电极的“0 基准”，再整体减掉。
# 这样波形的起点对齐到 0，刺激后的起伏才看得清。
baseline = erp[:PRES_TIME, :].mean(axis=0)   # 每个电极一个基线值，形状 (63,)
erp = erp - baseline

# ---------- 5. 造一个以“脸出现 = 0 毫秒”为原点的时间轴 ----------
# 第 0~204 个点是刺激前，第 205 个点是脸出现的瞬间(=0ms)。
time_ms = (np.arange(N_SAMPLES) - PRES_TIME) / FS * 1000   # 约 -200ms ~ +1000ms

# ---------- 6. 找出老师选的 4 个“人脸通道”各自是第几列 ----------
face_channels = ["P9", "P10", "TP9", "TP10"]
face_idx = [channel_names.index(name) for name in face_channels]
print("人脸通道对应列号:", dict(zip(face_channels, face_idx)))

# ---------- 7. 画图 ----------
plt.figure(figsize=(10, 6))
for name, idx in zip(face_channels, face_idx):
    plt.plot(time_ms, erp[:, idx], label=name)

ylim = plt.ylim()
plt.axvline(0, color="black", linewidth=1)                 # 脸出现的竖线
plt.text(15, ylim[1] * 0.92, "face onset", fontsize=9)
plt.axvspan(130, 200, color="orange", alpha=0.15)          # N170 大致窗口
plt.text(165, ylim[1] * 0.75, "N170", color="darkorange", ha="center")

plt.xlabel("Time (ms, 0 = face onset)")
plt.ylabel("Amplitude (µV)")
plt.title("Subject 01, Familiar faces: average ERP at the 4 face-selective channels")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# ---------- 8. 保存 + 显示 ----------
fig_dir = PROJECT_ROOT / "Figures"
fig_dir.mkdir(exist_ok=True)                  # 没有 Figures 文件夹就自动建一个
out_path = fig_dir / "02_erp_face_channels.png"
plt.savefig(out_path, dpi=150)
print("图已保存到:", out_path)
plt.show()                                    # 可能会弹出一个窗口（弹不出也没关系，图已存好）
