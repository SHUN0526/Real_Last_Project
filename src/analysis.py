import pandas as pd
import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt

# — 시간 영역 지표
def compute_sdnn(ibi):
    return np.std(ibi)
def compute_rmssd(ibi):
    diffs = np.diff(ibi)
    return np.sqrt(np.mean(diffs**2)) if diffs.size>0 else np.nan

# — 밴드 파워 계산
def band_power(f, pxx, f_low, f_high):
    mask = (f>=f_low)&(f<f_high)
    return np.trapezoid(pxx[mask], f[mask])

def compute_freq(ibi, fs=4.0):
    ibi_s = ibi/1000.0
    t = np.cumsum(ibi_s)
    ti = np.arange(t[0], t[-1], 1/fs)
    interp = np.interp(ti, t, ibi_s)
    interp = interp - np.mean(interp)  # DC 제거
    f, pxx = welch(interp, fs=fs, nperseg=min(256, len(interp))) ## fft적용
    lf = band_power(f, pxx, 0.04, 0.15) # LF: 0.04-0.15 Hz
    hf = band_power(f, pxx, 0.15, 0.4) # HF: 0.15-0.4 Hz
    tot= band_power(f, pxx, 0.0033, 0.4) # Total: 0.0033-0.4 Hz
    return lf, hf, tot

# 1) 데이터 로드 & 요약 계산
csv_path = r"C:\Users\soong\Downloads\data_.csv"
df = pd.read_csv(csv_path)

records = []
for (pid, lbl), grp in df.groupby(['participant','label']):
    ibi = grp['ibi_ms'].values
    sd = compute_sdnn(ibi)
    rm = compute_rmssd(ibi)
    lf, hf, tot = compute_freq(ibi)
    records.append(dict(participant=pid, label=lbl,
                        sdnn=sd, rmssd=rm,
                        lf_power=lf, hf_power=hf, total_power=tot))
summary = pd.DataFrame(records)

# 2) wide-format으로 변환
metrics = ['sdnn','rmssd','lf_power','hf_power','total_power']
wide = summary.set_index(['participant','label'])[metrics].unstack('label')
wide.columns = [f"{m}_L{l}" for m,l in wide.columns]
wide.reset_index(inplace=True)

# 3) 서브플롯 그리기
participants = wide['participant'].astype(str)
x = np.arange(len(participants))
width = 0.35

fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
axes = axes.flatten()

# 각 지표별 타이틀
titles = ['SDNN', 'RMSSD', 'LF Power', 'HF Power', 'Total Power']
for ax, metric, title in zip(axes, metrics, titles):
    ax.bar(x - width/2, wide[f'{metric}_L0'], width, label='label 0')
    ax.bar(x + width/2, wide[f'{metric}_L1'], width, label='label 1')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(participants, rotation=90)
    ax.legend()
    if 'power' in metric:
        ax.set_ylabel('Power (unit²/Hz)')
    else:
        ax.set_ylabel(metric.upper())

# 마지막 빈 서브플롯 지우기
fig.delaxes(axes[-1])

plt.tight_layout()
plt.show()
