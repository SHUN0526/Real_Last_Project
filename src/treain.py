import pandas as pd
import numpy as np
import json
import logging
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit, ParameterGrid
from sklearn.neural_network import MLPClassifier
from joblib import dump

# --- Logging 설정 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)   
logger = logging.getLogger(__name__)

# --- HRV Feature 계산 함수 ---
def compute_sdnn(ibi: np.ndarray) -> float:
    """IBI 배열의 표준편차(SDNN) 계산"""
    return float(np.std(ibi))


def compute_rmssd(ibi: np.ndarray) -> float:
    """인접 IBI 차분의 제곱평균제곱근(RMSSD) 계산"""
    diffs = np.diff(ibi)
    return float(np.sqrt(np.mean(diffs ** 2))) if diffs.size else np.nan


def compute_freq_bands(ibi: np.ndarray, fs: float = 4.0) -> tuple:
    """
    IBI(ms) 배열에서 LF/HF/Total 주파수 대역 파워 계산
    - fs: 리샘플링 주파수 (Hz)
    """
    # ms -> s 변환 및 시간축 생성
    ibi_s = ibi / 1000.0
    t_orig = np.cumsum(ibi_s)
    t_interp = np.arange(t_orig[0], t_orig[-1], 1.0 / fs)
    ibi_interp = np.interp(t_interp, t_orig, ibi_s)
    ibi_interp -= np.mean(ibi_interp)  # DC 제거

    # Welch PSD
    freqs, pxx = welch(ibi_interp, fs=fs, nperseg=min(256, len(ibi_interp)))

    def band(low, high):
        mask = (freqs >= low) & (freqs < high)
        return float(np.trapz(pxx[mask], freqs[mask]))

    lf = band(0.04, 0.15)
    hf = band(0.15, 0.40)
    total = band(0.0033, 0.40)
    return lf, hf, total


# --- 주요 파라미터 ---
WINDOW_SIZE = 100  # 윈도우 크기(IBI 수)
STEP = 5        # 슬라이딩 스텝
CSV_PATH = r"C:\Users\soong\Downloads\data_.csv"
OUTPUT_MODEL = "hrv_emotion_model.pkl"
OUTPUT_JSON = "best_hrv_params.json"


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    참가자별, 라벨별 IBI 윈도우에서 HRV 피처 계산
    반환: DataFrame(columns=['participant','label','sdnn','rmssd','lf_power','hf_power','total_power'])
    """
    records = []
    for (pid, lbl), group in df.groupby(['participant', 'label']):
        ibis = group['ibi_ms'].values
        for start in range(0, len(ibis) - WINDOW_SIZE + 1, STEP):
            window = ibis[start:start + WINDOW_SIZE]
            sdnn = compute_sdnn(window)
            rmssd = compute_rmssd(window)
            lf, hf, total = compute_freq_bands(window)
            records.append({
                'participant': pid,
                'label': lbl,
                'sdnn': sdnn,
                'rmssd': rmssd,
                'lf_power': lf,
                'hf_power': hf,
                'total_power': total
            })
    return pd.DataFrame(records)


def main():
    # 1. 데이터 로드 & 피처 추출
    logger.info(f"Loading data from {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    features = extract_features(df)
    X = features[['sdnn','rmssd','lf_power','hf_power','total_power']].values
    y = features['label'].values
    groups = features['participant'].values
    logger.info(f"Extracted feature table: {features.shape}")

    # 2. 그룹별 샘플 분할 & 그리드 서치
    gss = GroupShuffleSplit(n_splits=5, test_size=5, random_state=42)
    param_grid = {
        'hidden_layer_sizes': [(64,32), (128,64), (256,128)],
        'alpha': [0.001, 0.0005, 0.0001, 0.00001],
        'learning_rate_init': [1e-3, 1e-4],
        'batch_size': [64, 128, 256],
    }

    results = []
    total = len(list(ParameterGrid(param_grid)))

    for idx, params in enumerate(ParameterGrid(param_grid), start=1):
        logger.info(f"Testing params {idx}/{total}: {params}")
        scores = []
        for split_num, (train_idx, test_idx) in enumerate(gss.split(X, y, groups), start=1):
            # (추가) 이번 분할에서 테스트로 쓰인 참가자 목록 출력
            test_parts = np.unique(groups[test_idx])
            logger.info(f" Split {split_num}: Test participants = {test_parts}")

            # 학습/테스트 분리
            Xtr, Xte = X[train_idx], X[test_idx]
            ytr, yte = y[train_idx], y[test_idx]

            # 스케일링
            scaler = StandardScaler().fit(Xtr)
            Xtr = scaler.transform(Xtr)
            Xte = scaler.transform(Xte)

            # 모델 학습 및 평가
            model = MLPClassifier(
                **params,
                solver='adam',
                max_iter=2000,
                early_stopping=True,
                n_iter_no_change=10,
                random_state=42
            )
            model.fit(Xtr, ytr)
            score = model.score(Xte, yte)
            logger.info(f"  Split {split_num} accuracy: {score:.3f}")
            scores.append(score)

        results.append({**params, 'mean_accuracy': np.mean(scores)})

    # 3. 결과 정리 & 최적 파라미터 저장
    res_df = pd.DataFrame(results).sort_values('mean_accuracy', ascending=False)
    print(res_df.to_string(index=False))
    best = res_df.iloc[0].drop('mean_accuracy').to_dict()
    with open(OUTPUT_JSON, 'w') as jf:
        json.dump(best, jf, indent=2)
    logger.info(f"Saved best params to {OUTPUT_JSON}: {best}")

    # 4. 전체 데이터로 재훈련 및 저장
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    final_model = MLPClassifier(
        **best,
        solver='adam',
        max_iter=2000,
        random_state=42
    )
    final_model.fit(X_scaled, y)
    dump({'scaler': scaler, 'model': final_model}, OUTPUT_MODEL)
    logger.info(f"Trained model saved to {OUTPUT_MODEL}")


if __name__ == '__main__':
    main()
