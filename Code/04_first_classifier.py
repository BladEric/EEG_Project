"""
第 4 步 · 机器学习第一课：教电脑根据脑电判断“熟悉 or 陌生”。

⚠️ 这是一个【简化版】，目的是先把“分类”这件事彻底搞懂。
   特征用最直观的：每个试次在 200~600ms（你看到熟/生分叉那段）内，
   4 个人脸通道的平均幅度 —— 把每个试次浓缩成 4 个数字。
   下一步再做忠实复现老师的完整版（降采样特征 + 10 折交叉验证 + d′）。

运行（项目根目录下）：
    conda run -n eeg python Code/04_first_classifier.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ---------- 0. 参数 ----------
FS, N_SAMPLES, N_CHANNELS, PRES_TIME = 1024, 1229, 63, 205
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data" / "osfstorage-archive" / "ClassifierModel" / "Exp1"
SUBJECTS = list(range(1, 6)) + list(range(7, 17)) + list(range(18, 25))

with open(DATA_DIR / "Part01_01-export.mul") as f:
    f.readline()
    channel_names = f.readline().split()
face_idx = [channel_names.index(n) for n in ["P9", "P10", "TP9", "TP10"]]

# 特征时间窗 200~600ms，换算成采样点位置
w0 = PRES_TIME + int(0.200 * FS)
w1 = PRES_TIME + int(0.600 * FS)


def load_trials(subject, cond):
    """读一个被试一个条件，返回基线校正后的试次，形状 (试次, 时间, 电极)。"""
    data = pd.read_csv(DATA_DIR / f"Part{subject:02d}_{cond}-export.mul",
                       sep=r"\s+", skiprows=2, header=None).to_numpy()
    n_trials = data.shape[0] // N_SAMPLES
    trials = data.reshape(n_trials, N_SAMPLES, N_CHANNELS)
    trials = trials - trials[:, :PRES_TIME, :].mean(axis=1, keepdims=True)  # 每试次各自基线校正
    return trials


def features(trials):
    """把每个试次浓缩成 4 个数字：200~600ms 内 4 个人脸通道的平均幅度。"""
    return trials[:, w0:w1, :].mean(axis=1)[:, face_idx]   # (试次, 4)


# ---------- 1. 收集所有被试的特征 X 和标签 y ----------
X_list, y_list = [], []
for s in SUBJECTS:
    fam = load_trials(s, "01")   # 熟悉
    unf = load_trials(s, "03")   # 陌生
    X_list += [features(fam), features(unf)]
    y_list += [np.zeros(len(fam)), np.ones(len(unf))]   # 0=熟悉, 1=陌生

X = np.vstack(X_list)            # 所有试次的特征 (总试次数, 4)
y = np.concatenate(y_list)       # 每个试次的正确答案
print(f"一共 {len(y)} 个试次；每个试次 {X.shape[1]} 个特征")
print(f"  其中 熟悉 {int((y == 0).sum())} 个，陌生 {int((y == 1).sum())} 个")

# ---------- 2. 切成“训练集”和“测试集” ----------
# 70% 训练（学规律），30% 藏起来当考题（模型没见过）。stratify 保证两类在两边都均衡。
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=13, stratify=y)
print(f"训练集 {len(y_train)} 个，测试集 {len(y_test)} 个")

# ---------- 3. 训练逻辑回归，然后“考试” ----------
clf = LogisticRegression()
clf.fit(X_train, y_train)             # 学习：看着带答案的训练集找规律
pred = clf.predict(X_test)            # 预测：对没见过的测试集做判断
acc = accuracy_score(y_test, pred)    # 对比预测 vs 真答案

print("\n========== 结果 ==========")
print("瞎猜的水平 (chance) : 50.0%")
print(f"分类器测试集准确率  : {acc:.1%}")
