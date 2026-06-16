import pandas as pd
import numpy as np
import mne

def load_mul_to_mne(filepath):
    """
    替代 mul_import.m，将 .mul 文本文件直接读取并转换为 mne.io.Raw 对象，
    方便后续进行滤波、ICA 等高级操作。
    """
    # 1. 读取第一行解析元数据
    with open(filepath, 'r', encoding='utf-8') as f:
        meta_line = f.readline().strip()
        
    metadata = {}
    for item in meta_line.split():
        if '=' in item:
            key, value = item.split('=', 1)
            metadata[key] = value

    # 计算采样率 (Sampling Frequency)
    # 比如 SamplingInterval[ms]=0.976563 -> fs = 1024 Hz
    sampling_interval_ms = float(metadata.get('SamplingInterval[ms]', 1.0))
    sfreq = 1000.0 / sampling_interval_ms

    # 2. 读取数据矩阵
    df = pd.read_csv(filepath, sep=r'\s+', skiprows=1)
    ch_names = list(df.columns)
    
    # 3. 准备 MNE 所需的数据格式
    # MNE 要求数据形状为 (n_channels, n_times)
    # 并且 MNE 的标准单位是伏特 (V)，如果原始数据是微伏 (uV)，需乘以 1e-6
    data_matrix = df.values.T * 1e-6 
    
    # 将所有的通道类型设置为 'eeg'
    ch_types = ['eeg'] * len(ch_names)
    
    # 创建 MNE 的 Info 对象和 Raw 对象
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data_matrix, info)
    
    # (可选) 设置标准的脑电极位置映射，比如标准的 10-20 系统
    try:
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='ignore')
    except Exception as e:
        print(f"设置电极位置时提示: {e}")

    return raw

# 测试代码
if __name__ == "__main__":
    filepath = "ClassifierModel/Exp1/Part01_01-export.mul"
    raw = load_mul_to_mne(filepath)
    
    # 打印 MNE 对象信息
    print(raw.info)
    
    # MNE 自带非常强大的交互式绘图功能，可以直接可视化！
    # raw.plot(duration=2.0, n_channels=20, scalings='auto')