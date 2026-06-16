"""
第 1 步：把一个 .mul 文件读进来，看看它长什么样。

这是整个项目的第一行代码，目标只有一个：搞清楚一个数据文件里有什么。

运行方法（在项目根目录 /Users/eric/Documents/Project 下，终端里输入）：
    conda run -n eeg python Code/01_explore_data.py
"""

# import = “借用别人写好的工具箱”。
# pathlib 是 Python 自带的、用来处理“文件路径”的工具。
# pandas 是数据分析最常用的库，可以把表格数据读成一个叫 DataFrame 的东西。
from pathlib import Path
import pandas as pd


# ---------- 1. 先找到要读的数据文件 ----------
# __file__ 是“本脚本自己的位置”；.parent 是“上一级文件夹”。
# 连用两次 .parent 就从 Code/ 退回到项目根目录。这样不管你在哪运行都能找到数据。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data" / "osfstorage-archive" / "ClassifierModel" / "Exp1"

# 我们先只看一个文件：被试 01、条件 _01（= 熟悉的脸 + 高多样性）
mul_file = DATA_DIR / "Part01_01-export.mul"
print("正在读取文件：", mul_file.name)


# ---------- 2. 先用最原始的方式看前两行（表头）----------
# open(...) 打开文件，readline() 一次读一行。
with open(mul_file) as f:
    line1 = f.readline().strip()   # 第 1 行：元信息
    line2 = f.readline().strip()   # 第 2 行：电极名字

print("\n【第 1 行 · 元信息】")
print(line1)

# split() 把一行字符串按空格切成一个列表（list）
channel_names = line2.split()
print(f"\n【第 2 行 · 电极名字】一共 {len(channel_names)} 个电极：")
print(channel_names)


# ---------- 3. 用 pandas 读“数字部分”----------
# skiprows=2  ：跳过前面那 2 行表头
# sep=r'\s+'  ：用“一个或多个空格”作为分隔符（数据是用空格对齐的）
# header=None ：告诉 pandas 数据里没有列名，别把第一行数据当成标题
data = pd.read_csv(mul_file, sep=r"\s+", skiprows=2, header=None)

# .shape 返回 (行数, 列数)。行 = 时间点，列 = 电极。
print("\n【数据部分的形状】(行=时间采样点, 列=电极)：", data.shape)


# ---------- 4. 算出这个文件里包含多少个“试次(trial)”----------
fs = 1024            # 采样率：每秒采 1024 个点（来自第 1 行的 SamplingInterval）
n_samples = 1229     # 每个试次固定 1229 个时间点（老师实验设定）
n_rows = data.shape[0]
n_trials = n_rows / n_samples
print(f"\n总时间点 {n_rows} ÷ 每试次 {n_samples} 点 = {n_trials} 个试次")


# ---------- 5. 看一眼最前面几个真实数值 ----------
# .iloc[行范围, 列范围] 用来按“位置”取出表格的一小块。单位是微伏 µV。
print("\n【前 3 个时间点 × 前 5 个电极 的原始脑电值（单位 µV）】")
print(data.iloc[:3, :5])
