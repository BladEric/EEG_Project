# ================================================================
# 完整教程：如何用 Python 读取、显示、转换 .mul EEG 文件
# 把这个文件放在你的项目目录里，和 .mul 文件放在一起运行
# ================================================================

# ── 需要安装的库 ──────────────────────────────────────────────
# pip install numpy pandas matplotlib scipy scikit-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ================================================================
# 1. 核心函数：读取 .mul 文件
# ================================================================
def read_mul(filepath):
    """
    读取 Brain Products .mul EEG 文件。

    .mul 文件结构：
        第1行 — 元数据 (TimePoints, Channels, SamplingInterval 等)
        第2行 — 通道名 (Fp1, Fpz, Fp2, F7 ...)
        第3行起 — 数值数据，每列一个通道，单位 µV

    返回值：
        data     : numpy array, shape = (时间点数, 通道数)
        channels : 通道名列表，如 ['Fp1', 'Fpz', 'Fp2', ...]
        meta     : 第一行元数据字符串（含采样率等信息）
    """
    with open(filepath, "r") as f:
        meta     = f.readline().strip()   # 第1行
        channels = f.readline().split()   # 第2行，按空白拆分

    data = np.loadtxt(filepath, skiprows=2)  # 第3行起全部读成数组
    return data, channels, meta


# ================================================================
# 2. 使用例子
# ================================================================
if __name__ == "__main__":

    # 改成你自己的文件路径，例如：
    # filepath = "Exp1/Part01_01-export.mul"
    filepath = "/Users/eric/Documents/Durham/Project/code/Exp1/Part01_01-export.mul"  # 教程用小样本

    data, channels, meta = read_mul(filepath)

    # ── 基本信息 ──────────────────────────────────────────────
    print("元数据：", meta)
    print(f"通道数：{len(channels)}")
    print(f"时间点数：{data.shape[0]}")
    print(f"数据形状（时间点 × 通道）：{data.shape}")
    print(f"电压范围：{data.min():.3f} ~ {data.max():.3f} µV\n")

    # ── 取出单个通道 ──────────────────────────────────────────
    ch_name  = "Fz"
    ch_index = channels.index(ch_name)
    fz       = data[:, ch_index]
    print(f"{ch_name} 通道（索引 {ch_index}）：{fz}\n")

    # ── 格式转换 ──────────────────────────────────────────────

    # → pandas DataFrame（每列=一个通道，每行=一个时间点）
    df = pd.DataFrame(data, columns=channels)
    print("DataFrame 前3行：")
    print(df.head(3), "\n")
    '''
    # 保存为 CSV（可用 Excel 打开）
    df.to_csv("eeg_data.csv", index=False)
    print("已保存：eeg_data.csv")

    # 保存为 numpy .npy（在 Python 里读写最快）
    np.save("eeg_data.npy", data)
    print("已保存：eeg_data.npy")

    # 读取 .npy
    data_reloaded = np.load("eeg_data.npy")
    print(f"重新载入 .npy，形状：{data_reloaded.shape}\n")
    '''
    # ── 可视化 ────────────────────────────────────────────────
    # 采样间隔从元数据里手动提取（也可以写成解析函数）
    sampling_interval_ms = 0.976563   # Exp1/Exp2 是 1/1024*1000 ≈ 0.977 ms
    time_ms = np.arange(data.shape[0]) * sampling_interval_ms

    fig, axes = plt.subplots(2, 1, figsize=(11, 7))

    # 图1：3个通道的时间序列
    ax = axes[0]
    for name in ["Fz", "Cz", "Pz"]:
        idx = channels.index(name)
        ax.plot(time_ms, data[:, idx], label=name, linewidth=2)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Voltage (µV)")
    ax.set_title("EEG Time Series")
    ax.legend()
    ax.grid(alpha=0.3)

    # 图2：第一个时间点，所有通道的电压快照
    ax2 = axes[1]
    ax2.bar(range(len(channels)), data[0, :], color="steelblue", alpha=0.7)
    ax2.axhline(0, color="red", linestyle="--")
    ax2.set_xlabel("Channel Index")
    ax2.set_ylabel("Voltage (µV)")
    ax2.set_title("Snapshot at t=0: all channels")
    ax2.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("eeg_visualization.png", dpi=130)
    plt.show()
    print("图表已保存：eeg_visualization.png")
