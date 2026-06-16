import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from scipy.stats import norm, ttest_1samp
import matplotlib.pyplot as plt
import os
import warnings

# 忽略类似 MATLAB 的收敛警告
warnings.filterwarnings('ignore')

# 1. 核心参数设置 (完全对应 MATLAB 变量)
np.random.seed(13) # 对应 rng(13)
fs = 1024 # 采样率
subject_range = list(range(1, 6)) + list(range(7, 17)) + list(range(18, 25))
n_samples = 1229 # 每个 trial 的采样点数
pres_time = int(np.ceil(fs * 0.200)) # 刺激前时间 (~205个采样点)
n_sample_down = 100 
down_sample_win = int(np.ceil(fs * (n_sample_down / fs))) # 降采样窗口
n_channels = 63 # 通道数
use_noise = 5 # 训练数据增强的噪声比例
n_pcomp_used = 4 # 选用的主成分数量
# 自动获取当前脚本所在的绝对路径，并拼接上 Exp1
script_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(script_dir, 'Exp1')

# 降采样后的有效点数计算
sign_samples = n_samples - pres_time
sign_down_samples = int(np.ceil(sign_samples / down_sample_win))
n_effect_samples = sign_down_samples - 1 
n_subjects = len(subject_range)

# ==========================================
# 工具函数：加载并预处理单个数据文件
# ==========================================
def load_and_preprocess(filepath):
    # 模拟 dlmread 或 mul_import，跳过前两行头文件
    # delim_whitespace=True 能够处理任意数量的空格分隔符
    df = pd.read_csv(filepath, sep=r'\s+', skiprows=2, header=None)
    raw_data = df.iloc[:, :n_channels].values
    
    n_trials = raw_data.shape[0] // n_samples
    
    # 核心坑点：MATLAB 的 reshape 默认按列填充 (order='F')
    # Reshape 成 (nChannels, nSamples, nTrials)
    data_temp = np.reshape(raw_data.T, (n_channels, n_samples, n_trials), order='F')
    
    # 去除基线 (Baseline Correction): 减去 pres_time 窗口内的均值
    mean_pre_stim = np.mean(data_temp[:, :pres_time, :], axis=1, keepdims=True)
    data_bl = data_temp[:, pres_time:, :] - mean_pre_stim
    
    # 将矩阵展平回二维，模拟 MATLAB: reshape(..., nChannels, ...)'
    data_bl_flat = np.reshape(data_bl, (n_channels, sign_samples * n_trials), order='F').T
    
    # 降采样 (Downsampling)
    if down_sample_win > 1:
        data_ds = np.zeros(((sign_down_samples - 1) * n_trials, n_channels))
        for idx_tri in range(n_trials):
            for j in range(n_channels):
                # 提取对应 trial 和 通道 的数据段
                start_idx = idx_tri * sign_samples
                end_idx = start_idx + (sign_down_samples - 1) * down_sample_win
                trial_chan_data = data_bl_flat[start_idx:end_idx, j]
                
                # 在时间窗口内取均值
                reshaped_data = np.reshape(trial_chan_data, (down_sample_win, sign_down_samples - 1), order='F')
                mean_ds = np.mean(reshaped_data, axis=0)
                
                # 保存结果
                out_start = idx_tri * (sign_down_samples - 1)
                out_end = out_start + (sign_down_samples - 1)
                data_ds[out_start:out_end, j] = mean_ds
                
        # 添加极小噪声防止全0
        data_ds += np.random.randn(*data_ds.shape) * 1e-4
        return data_ds, n_trials
    return data_bl_flat, n_trials

# 存储结果的矩阵
cmp_err_diag_HFvsU = np.zeros((n_subjects, n_pcomp_used))
cmp_err_diag_LFvsU = np.zeros((n_subjects, n_pcomp_used))
d_prime_HFvsU = np.zeros((n_subjects, n_pcomp_used))
d_prime_LFvsU = np.zeros((n_subjects, n_pcomp_used))

# ==========================================
# 主循环：遍历所有被试进行分析
# ==========================================
for idx_sub, subject in enumerate(subject_range):
    print(f"\nProcessing Subject N: {idx_sub + 1} (ID: {subject})")
    subj_str = f"{subject + 100}"[-2:] # 格式化被试编号如 '01', '02'
    
    # 1. 导入四种条件的数据并预处理
    data_HVFam, n_trials_HVFam = load_and_preprocess(os.path.join(path, f"Part{subj_str}_01-export.mul"))
    data_LVFam, n_trials_LVFam = load_and_preprocess(os.path.join(path, f"Part{subj_str}_02-export.mul"))
    data_HVUnf, n_trials_HVUnf = load_and_preprocess(os.path.join(path, f"Part{subj_str}_03-export.mul"))
    data_LVUnf, n_trials_LVUnf = load_and_preprocess(os.path.join(path, f"Part{subj_str}_04-export.mul"))
    
    # 2. 主成分分析 (PCA)
    data_all = np.vstack([data_HVFam, data_HVUnf, data_LVFam, data_LVUnf])
    pca = PCA()
    pca.fit(data_all)
    u = pca.components_.T # 提取主成分特征向量
    
    # 3. 为逻辑回归准备交叉验证
    condition_HFvsU = np.vstack([np.zeros((n_trials_HVFam, 1)), np.ones((n_trials_HVUnf, 1))]).ravel()
    condition_LFvsU = np.vstack([np.zeros((n_trials_LVFam, 1)), np.ones((n_trials_LVUnf, 1))]).ravel()
    
    kf = KFold(n_splits=10, shuffle=False) # 对应 MATLAB 的 cvpartition
    
    for cmp_idx in range(1, n_pcomp_used + 1):
        restrict_time = int(np.ceil(0.6 * fs / down_sample_win)) # 只取前 600ms
        
        # 将原始数据投影到前 cmp_idx 个主成分上，并进行三维重排提取时间限制
        def get_ncomp_data(data, n_trials, cmp_idx):
            # 相当于 MATLAB 中的 temp = reshape(data * u(:, 1:cmpIdx), nEffectSamples, nTrials, cmpIdx)
            projected = np.dot(data, u[:, :cmp_idx])
            temp = np.reshape(projected, (n_effect_samples, n_trials, cmp_idx), order='F')
            # 取截断时间并排列
            temp_restricted = temp[:restrict_time, :, :]
            return np.reshape(np.transpose(temp_restricted, (0, 2, 1)), (restrict_time * cmp_idx, n_trials), order='F').T
            
        data_HVFam_ncomp = get_ncomp_data(data_HVFam, n_trials_HVFam, cmp_idx)
        data_HVUnf_ncomp = get_ncomp_data(data_HVUnf, n_trials_HVUnf, cmp_idx)
        data_LVFam_ncomp = get_ncomp_data(data_LVFam, n_trials_LVFam, cmp_idx)
        data_LVUnf_ncomp = get_ncomp_data(data_LVUnf, n_trials_LVUnf, cmp_idx)
        
        # ------------------------------------
        # 任务 1: HighVar Fam vs Unf (高变异组对比)
        # ------------------------------------
        data_NComp_H = np.vstack([data_HVFam_ncomp, data_HVUnf_ncomp])
        err_HF = []
        preds_HF_all, true_HF_all = [], []
        
        for train_index, test_index in kf.split(data_NComp_H):
            X_train, X_test = data_NComp_H[train_index], data_NComp_H[test_index]
            y_train, y_test = condition_HFvsU[train_index], condition_HFvsU[test_index]
            
            # 加入噪声进行数据增强 (Data Augmentation)
            if use_noise:
                X_train = np.tile(X_train, (10, 1)) + np.random.randn(X_train.shape[0] * 10, X_train.shape[1]) * use_noise
                y_train = np.tile(y_train, 10)
                
            # Logistic Regression 对应 MATLAB 的 fitglm(..., 'binomial', 'logit')
            clf = LogisticRegression(solver='liblinear') 
            clf.fit(X_train, y_train)
            
            # 使用 predict_proba 获取概率结果
            preds = clf.predict_proba(X_test)[:, 1]
            preds_class = (preds > 0.5).astype(int)
            err_HF.append(np.mean(preds_class != y_test))
            
            preds_HF_all.extend(preds)
            true_HF_all.extend(y_test)
            
        cmp_err_diag_HFvsU[idx_sub, cmp_idx-1] = np.mean(err_HF)
        
        # 计算 d-prime (信号检测论指标)
        preds_HF_arr, true_HF_arr = np.array(preds_HF_all), np.array(true_HF_all)
        hit_rate = np.sum((preds_HF_arr > 0.5) & (true_HF_arr == 1)) / np.sum(true_HF_arr == 1)
        fa_rate = np.sum((preds_HF_arr > 0.5) & (true_HF_arr == 0)) / np.sum(true_HF_arr == 0)
        
        d_prime_HFvsU[idx_sub, cmp_idx-1] = np.clip(norm.ppf(hit_rate), -2, 2) - np.clip(norm.ppf(fa_rate), -2, 2)
        
        # ------------------------------------
        # 任务 2: LowVar Fam vs Unf (低变异/单一图像组对比)
        # ------------------------------------
        data_NComp_L = np.vstack([data_LVFam_ncomp, data_LVUnf_ncomp])
        err_LF = []
        preds_LF_all, true_LF_all = [], []
        
        for train_index, test_index in kf.split(data_NComp_L):
            X_train, X_test = data_NComp_L[train_index], data_NComp_L[test_index]
            y_train, y_test = condition_LFvsU[train_index], condition_LFvsU[test_index]
            
            if use_noise:
                X_train = np.tile(X_train, (10, 1)) + np.random.randn(X_train.shape[0] * 10, X_train.shape[1]) * use_noise
                y_train = np.tile(y_train, 10)
                
            clf = LogisticRegression(solver='liblinear')
            clf.fit(X_train, y_train)
            
            preds = clf.predict_proba(X_test)[:, 1]
            preds_class = (preds > 0.5).astype(int)
            err_LF.append(np.mean(preds_class != y_test))
            
            preds_LF_all.extend(preds)
            true_LF_all.extend(y_test)
            
        cmp_err_diag_LFvsU[idx_sub, cmp_idx-1] = np.mean(err_LF)

        preds_LF_arr, true_LF_arr = np.array(preds_LF_all), np.array(true_LF_all)
        
        hit_rate_L = np.sum((preds_LF_arr > 0.5) & (true_LF_arr == 1)) / np.sum(true_LF_arr == 1)
        fa_rate_L = np.sum((preds_LF_arr > 0.5) & (true_LF_arr == 0)) / np.sum(true_LF_arr == 0)
        d_prime_LFvsU[idx_sub, cmp_idx-1] = np.clip(norm.ppf(hit_rate_L), -2, 2) - np.clip(norm.ppf(fa_rate_L), -2, 2)

# ==========================================
# 可视化与统计推断 (Main Figure)
# ==========================================
print("\n--- Final Analysis ---")
n_comp = 3 # 对应 MATLAB 的 nComp=4 (Python 索引从0开始)
data_type_perc = True

if data_type_perc:
    data_compare = 1 - cmp_err_diag_HFvsU[:, n_comp]
    control_compare = 1 - cmp_err_diag_LFvsU[:, n_comp]
else:
    data_compare = d_prime_HFvsU[:, n_comp]
    control_compare = d_prime_LFvsU[:, n_comp]

# 单样本 T 检验对比猜测水平 0.5
t_stat_d, p_val_d = ttest_1samp(data_compare, 0.5)
t_stat_c, p_val_c = ttest_1samp(control_compare, 0.5)

# 画图
plt.figure(figsize=(6, 6))
plt.bar([1], [np.mean(data_compare)], width=0.3, color='b', label='High Var')
plt.bar([2], [np.mean(control_compare)], width=0.3, color='r', label='Single Img')

# 添加标准误误差棒 (Standard Error)
std_err_d = np.std(data_compare, ddof=1) / np.sqrt(len(data_compare))
std_err_c = np.std(control_compare, ddof=1) / np.sqrt(len(control_compare))

plt.errorbar([1], [np.mean(data_compare)], yerr=std_err_d, fmt='k.', linewidth=2, capsize=5)
plt.errorbar([2], [np.mean(control_compare)], yerr=std_err_c, fmt='k.', linewidth=2, capsize=5)

# 添加显著性星号 (模拟 MATLAB 的绘图逻辑)
if p_val_d < 0.05:
    plt.text(1, np.mean(data_compare) + std_err_d + 0.02, '*', fontsize=16, ha='center')
if p_val_c < 0.05:
    plt.text(2, np.mean(control_compare) + std_err_c + 0.02, '*', fontsize=16, ha='center')

if data_type_perc:
    plt.ylim([0.4, 0.8])
    plt.title('Percentage correct')
    plt.ylabel('Percentage correct')
else:
    plt.ylim([0, 2])
    plt.title('d Prime')
    plt.ylabel('d Prime')

plt.xticks([1, 2], ['High Var', 'Single Img'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()