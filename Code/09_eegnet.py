"""
09 · EEGNet（卷积神经网络）跨被试解码。

与前面"先抽特征再分类"不同，EEGNet 直接吃【原始时空数据】：
每个试次 = (63 通道 × 128 时间点)，由它自己学空间滤波 + 时间滤波。
按被试分组 5 折 CV（训练/测试是不同的人），看它在跨被试设定下能否超过简单模型。

运行（项目根目录下）：
    conda run -n eeg python Code/09_eegnet.py
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import GroupKFold

from eeg_pipeline import N_SAMPLES, N_CHANNELS, PRES_TIME, DATA_DIR, SUBJECTS, PROJECT_ROOT

torch.manual_seed(13)
np.random.seed(13)
DEVICE = "cpu"          # 网络很小、数据不大，CPU 足够且稳
T = 128                 # 把刺激后 1024 点降到 128 点喂给网络
EPOCHS = 50


def load_epochs(subject, cond):
    """某被试某条件 → (试次, 63通道, 128时间点) 的原始 epoch。"""
    data = pd.read_csv(DATA_DIR / f"Part{subject:02d}_{cond}-export.mul",
                       sep=r"\s+", skiprows=2, header=None).to_numpy()
    n = data.shape[0] // N_SAMPLES
    tr = data.reshape(n, N_SAMPLES, N_CHANNELS)
    tr = tr - tr[:, :PRES_TIME, :].mean(axis=1, keepdims=True)        # 基线校正
    post = tr[:, PRES_TIME:PRES_TIME + 1024, :]                        # 刺激后 1024 点
    factor = 1024 // T
    post = post[:, :T * factor, :].reshape(n, T, factor, N_CHANNELS).mean(axis=2)  # 降到 128 点
    return post.transpose(0, 2, 1).astype(np.float32)                 # (n, 63, 128)


class EEGNet(nn.Module):
    """紧凑版 EEGNet (Lawhern et al. 2018)。"""
    def __init__(self, C=63, T=128, F1=8, D=2, F2=16, p=0.5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1))
        self.block2 = nn.Sequential(                       # 空间滤波（跨通道）
            nn.Conv2d(F1, F1 * D, (C, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(p))
        self.block3 = nn.Sequential(                       # 可分离时间滤波
            nn.Conv2d(F1 * D, F1 * D, (1, 16), groups=F1 * D, padding=(0, 8), bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(p))
        with torch.no_grad():                              # 动态求展平维度
            flat = self.block3(self.block2(self.block1(torch.zeros(1, 1, C, T)))).numel()
        self.head = nn.Linear(flat, 1)

    def forward(self, x):
        x = self.block3(self.block2(self.block1(x)))
        return self.head(x.flatten(1)).squeeze(1)


def train_eval(Xtr, ytr, Xte, yte):
    """训练 EEGNet 并返回测试准确率。"""
    mu, sd = Xtr.mean((0, 2), keepdims=True), Xtr.std((0, 2), keepdims=True) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd            # 用训练集统计做标准化
    Xtr_t = torch.tensor(Xtr).unsqueeze(1)                 # 加一个“图像通道”维度
    Xte_t = torch.tensor(Xte).unsqueeze(1)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)

    model = EEGNet(C=Xtr.shape[1], T=Xtr.shape[2]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    dl = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=64, shuffle=True)
    model.train()
    for _ in range(EPOCHS):
        for xb, yb in dl:
            opt.zero_grad()
            loss = lossf(model(xb.to(DEVICE)), yb.to(DEVICE))
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(Xte_t.to(DEVICE))).cpu().numpy()
    return float(np.mean((prob > 0.5) == yte))


print("读取 epoch（88 个文件）...")
epochs_data = {(s, c): load_epochs(s, c) for s in SUBJECTS for c in ["01", "02", "03", "04"]}

TASKS = {"High Var (01 vs 03)": ("01", "03"), "Single Img (02 vs 04)": ("02", "04")}
lines = ["====== EEGNet 跨被试（按被试分组 5 折 CV）======"]
for task, (cf, cu) in TASKS.items():
    X, y, g = [], [], []
    for s in SUBJECTS:
        fam, unf = epochs_data[(s, cf)], epochs_data[(s, cu)]
        X += [fam, unf]
        y += [np.zeros(len(fam)), np.ones(len(unf))]
        g += [np.full(len(fam), s), np.full(len(unf), s)]
    X, y, g = np.concatenate(X), np.concatenate(y).astype(np.float32), np.concatenate(g)

    accs = []
    for i, (tr, te) in enumerate(GroupKFold(5).split(X, y, g), 1):
        acc = train_eval(X[tr], y[tr], X[te], y[te])
        accs.append(acc)
        print(f"  {task}  fold {i}: {acc:.1%}")
    lines.append(f"{task}: EEGNet 跨被试准确率 {np.mean(accs):.1%} (±{np.std(accs):.1%})")
    print(f"  >> {lines[-1]}")

report = "\n".join(lines)
print("\n" + report)
results_dir = PROJECT_ROOT / "Results"
results_dir.mkdir(exist_ok=True)
(results_dir / "eegnet_cross_subject.txt").write_text(report + "\n", encoding="utf-8")
print("结果已保存: Results/eegnet_cross_subject.txt")
