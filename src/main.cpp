// ===== main_fft.cpp  (ESP32 Feather + EmotiBit + BLE UART) =====
// 완전 무선: 센서 → BLE(Nordic UART Service) → 안드로이드 앱
//
// 주파수 영역 HRV 계산을 "파이썬 compute_freq_bands()" 흐름에 맞춰 FFT로 구현.
//  - IBI(ms) -> s 변환
//  - t_orig = cumsum(ibi_s)   (파이썬 원본과 동일; 0에서 시작 안 함)
//  - 균일시간축 t = [t_start : t_end : 1/fs]
//  - 선형보간, DC 제거
//  - FFT → LF/HF/Total (0.04~0.40 Hz 밴드) 파워 적분
//
// NOTE:
//  * 파이썬쪽 Welch(구간 평균)까지는 구현 안 함. 짧은 창(<=64)에서는 단일 FFT도 비슷한 경향.
//  * 학습 코드와 최대한 피처 스케일을 맞추기 위해 cumsum-기반 시간축을 그대로 사용.
//  * 필요시 정석(0초 시작) 또는 Welch버전 별도 구현 가능.
//
// 빌드 전 반드시 hrv_model_params.h 에 스케일러/MLP 파라미터 정의되어 있어야 함.

#include <Arduino.h>
#include <NimBLEDevice.h>         // ESP32 NimBLE 스택 (가볍고 안정적)
#include <ArduinoJson.h>          // JSON 직렬화
#include "EmotiBit.h"
#include "SharedVitals.h"
#include "hrv_model_params.h"    // scaler_mean[], scaler_scale[], W1[][], b1[], W2[][], b2[]

// ───────────────────────────────────── CONSTANTS ─────────────────────────────
#define SERIAL_BAUD      115200
#define WINDOW_SIZE      30       // HRV 윈도우 크기 (IBI 샘플 수)
#define STEP_SAMPLES      5       // 5 샘플마다 모델 추론
#define LOG_INTERVAL   1000       // ms 단위 디버그 로그 주기

// 파이썬 compute_freq_bands()와 맞춤 리샘플링 주파수
#define HRV_FS_RESAMP   4.0f      // Hz
#define HRV_RESAMP_MAX  64        // 리샘플 후 최대 포인트 수 (필요하면 ↑)

// Nordic UART Service UUID 정의
static const char* NUS_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E";
static const char* NUS_TX_UUID      = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E";  // Notify only

// ──────────────────────────────── GLOBAL OBJECTS ─────────────────────────────
EmotiBit emotibit;
NimBLECharacteristic* txChar = nullptr;   // BLE Notify characteristic

// IBI 버퍼 & 상태 변수
float  ibi_buf[WINDOW_SIZE];
int    ibi_idx      = 0;
bool   buf_full     = false;
int    newSamples   = 0;
unsigned long lastLogMs       = 0;
unsigned long lastProcessedMs = 0;

// ── HRV FFT 계산용 워크 버퍼 ────────────────────────────────────────────────
static float hrv_t_orig[WINDOW_SIZE];          // 누적시간(초)  (파이썬: cumsum)
static float hrv_ibi_s[WINDOW_SIZE];           // IBI(s)
static float hrv_interp[HRV_RESAMP_MAX];       // 균일 보간 시계열 (s)
static float hrv_fft_re[HRV_RESAMP_MAX];
static float hrv_fft_im[HRV_RESAMP_MAX];

// ──────────────────────────── UTILITY FUNCTIONS ──────────────────────────────
static inline float sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

// 다음 2의 거듭제곱 찾기
static int nextPow2(int n) {
  int p = 1;
  while (p < n && p < HRV_RESAMP_MAX) p <<= 1;
  return p;
}

// 간단 Cooley-Tukey radix-2 FFT (in-place)
static void fftRadix2(float *re, float *im, int N) {
  int j = 0;
  for (int i = 0; i < N; ++i) {
    if (i < j) {
      float tr = re[i]; re[i] = re[j]; re[j] = tr;
      float ti = im[i]; im[i] = im[j]; im[j] = ti;
    }
    int m = N >> 1;
    while (m > 0 && j >= m) { j -= m; m >>= 1; }
    j += m;
  }
  for (int len = 2; len <= N; len <<= 1) {
    float ang = -2.0f * PI / (float)len;
    float wlen_r = cosf(ang);
    float wlen_i = sinf(ang);
    for (int i = 0; i < N; i += len) {
      float wr = 1.0f, wi = 0.0f;
      for (int k = 0; k < len / 2; ++k) {
        int i0 = i + k;
        int i1 = i + k + len / 2;
        float ur = re[i0];
        float ui = im[i0];
        float vr = re[i1] * wr - im[i1] * wi;
        float vi = re[i1] * wi + im[i1] * wr;
        re[i0] = ur + vr;
        im[i0] = ui + vi;
        re[i1] = ur - vr;
        im[i1] = ui - vi;
        float tmp = wr;
        wr = wr * wlen_r - wi * wlen_i;
        wi = tmp * wlen_i + wi * wlen_r;
      }
    }
  }
}

// 작은 배열 선형보간 (np.interp 대체)
static float linInterp(const float *x, const float *y, int N, float q) {
  if (q <= x[0])   return y[0];
  if (q >= x[N-1]) return y[N-1];
  int i = 1;
  while (i < N && x[i] < q) ++i;
  int i0 = i - 1;
  int i1 = i;
  float x0 = x[i0], x1 = x[i1];
  float y0 = y[i0], y1 = y[i1];
  float f = (q - x0) / (x1 - x0);
  return y0 + f * (y1 - y0);
}

// 파이썬 compute_freq_bands() 로직을 따라 HRV LF/HF/Total 파워 계산 (FFT 사용)
// 입력: ibi_ms[] (ms)
// 출력: lf, hf, total
// 실패 시 false
static bool computeFreqBands_likePythonFFT(const float *ibi_ms, int N,
                                           float &lf, float &hf, float &total)
{
  if (N < 2) return false;

  // IBI -> s, cumsum (파이썬: t_orig = np.cumsum(ibi_s))
  float acc = 0.0f;
  for (int i = 0; i < N; ++i) {
    hrv_ibi_s[i] = ibi_ms[i] / 1000.0f;
    acc += hrv_ibi_s[i];
    hrv_t_orig[i] = acc;            // 첫 값은 ibi_s[0]
  }
  float t_start = hrv_t_orig[0];
  float t_end   = hrv_t_orig[N-1];
  float dur     = t_end - t_start;
  if (dur <= 0) return false;

  // 균일 시간축 생성: [t_start, t_end) step=1/fs
  const float dt = 1.0f / HRV_FS_RESAMP;
  int M = 0;
  for (float t = t_start; t < t_end && M < HRV_RESAMP_MAX; t += dt) {
    hrv_interp[M++] = linInterp(hrv_t_orig, hrv_ibi_s, N, t);
  }
  if (M < 8) return false;   // 포인트 부족

  // DC 제거
  float mean = 0.0f;
  for (int i = 0; i < M; ++i) mean += hrv_interp[i];
  mean /= (float)M;
  for (int i = 0; i < M; ++i) {
    hrv_fft_re[i] = hrv_interp[i] - mean;
    hrv_fft_im[i] = 0.0f;
  }

  // zero padding to pow2
  int Nfft = nextPow2(M);
  for (int i = M; i < Nfft; ++i) {
    hrv_fft_re[i] = 0.0f;
    hrv_fft_im[i] = 0.0f;
  }

  // FFT
  fftRadix2(hrv_fft_re, hrv_fft_im, Nfft);

  // 파워 적분 (0..Fs/2)
  lf = hf = total = 0.0f;
  float df = HRV_FS_RESAMP / (float)Nfft;
  int kmax = Nfft / 2;
  for (int k = 1; k < kmax; ++k) {
    float fk = k * df;
    float pr = hrv_fft_re[k];
    float pi = hrv_fft_im[k];
    float p  = (pr*pr + pi*pi) * (2.0f / (float)Nfft); // 대략적 스케일
    if      (fk >= 0.04f   && fk < 0.15f) lf    += p;
    else if (fk >= 0.15f   && fk < 0.40f) hf    += p;
    if      (fk >= 0.0033f && fk < 0.40f) total += p;
  }
  return true;
}

// ───────────────────────── BLE INITIALISATION ───────────────────────────────
void initBLE()
{
  // 1) Device 이름 초기화 (GAP name)
  NimBLEDevice::init("Vitals");

  // 2) 서버/서비스/특성 설정
  NimBLEServer* server = NimBLEDevice::createServer();
  NimBLEService* service = server->createService(NUS_SERVICE_UUID);
  txChar = service->createCharacteristic(
              NUS_TX_UUID,
              NIMBLE_PROPERTY::NOTIFY
           );
  service->start();

  // 3) Advertising 설정
  NimBLEAdvertising* adv = NimBLEDevice::getAdvertising();
  adv->addServiceUUID(NUS_SERVICE_UUID);
  adv->setName("Vitals");
  adv->start();

  Serial.println("[BLE] Advertising started (Vitals)");
}

// ───────────────────────────────── SETUP ─────────────────────────────────────
void setup()
{
  Serial.begin(SERIAL_BAUD);
  while (!Serial);
  emotibit.setup();
  initBLE();
  emotibit.acquireData.eda       = true;
  emotibit.acquireData.heartRate = true;

  Serial.println("=== System Started ===");
}

// ───────────────────────────────── LOOP ──────────────────────────────────────
void loop()
{
  emotibit.update();

  unsigned long now = millis();
  float ibi = SharedVitals::ibi;          // ms
  float hr  = SharedVitals::heartRate;    // bpm

  // ───── 1) 주기적 진행 상황 로그 ─────
  if (now - lastLogMs >= LOG_INTERVAL) {
    lastLogMs = now;

    if (!buf_full) {
      float winPct = 100.0f * ibi_idx / WINDOW_SIZE;
      Serial.printf("[%6lums] Collecting IBI window: %2.0f%%\n", now, winPct);
    } else {
      Serial.printf("[%6lums] HR: %.1f bpm  IBI: %.1f ms  Samp: %d/%d\n",
                    now, hr, ibi, newSamples, STEP_SAMPLES);
    }
  }

  // ───── 2) 새 IBI 샘플 처리 ─────
  if (ibi > 0.0f && (now - lastProcessedMs) >= (unsigned long)ibi) {
    lastProcessedMs = now;

    // 버퍼에 IBI 저장
    ibi_buf[ibi_idx++] = ibi;
    if (ibi_idx >= WINDOW_SIZE) {
      ibi_idx = 0;
      if (!buf_full) Serial.println(">> IBI window is now FULL. Starting feature computations.");
      buf_full = true;
    }

    // STEP_SAMPLES마다 HRV 계산 + 모델 추론
    if (buf_full) {
      if (++newSamples < STEP_SAMPLES) return;
      newSamples = 0;

      // ─ A) 시간 영역 피처 ─ (학습 코드와 일치: 모집단 분산)
      float mean = 0, sq = 0;
      for (int i = 0; i < WINDOW_SIZE; ++i) mean += ibi_buf[i];
      mean /= WINDOW_SIZE;
      for (int i = 0; i < WINDOW_SIZE; ++i) {
        float d = ibi_buf[i] - mean;
        sq += d * d;
      }
      float sdnn = sqrtf(sq / WINDOW_SIZE);

      float sumd2 = 0;
      for (int i = 1; i < WINDOW_SIZE; ++i) {
        float d = ibi_buf[i] - ibi_buf[i - 1];
        sumd2 += d * d;
      }
      float rmssd = sqrtf(sumd2 / (WINDOW_SIZE - 1));

      // ─ B) 주파수 영역 피처 ─ (리샘플 + FFT, 파이썬 compute_freq_bands와 호환)
      float lf = 0, hf = 0, total = 0;
      bool ok = computeFreqBands_likePythonFFT(ibi_buf, WINDOW_SIZE, lf, hf, total);
      if (!ok) { lf = hf = total = 0; }

      Serial.printf("    DBG: sdnn=%.2f rmssd=%.2f lf=%.6f hf=%.6f tot=%.6f\n",
                    sdnn, rmssd, lf, hf, total);

      // ─ C) MLP 추론 ─
      float x[5]  = { sdnn, rmssd, lf, hf, total };
      float xs[5];
      for (int i = 0; i < 5; ++i) xs[i] = (x[i] - scaler_mean[i]) / scaler_scale[i];

      const int H = sizeof(b1) / sizeof(b1[0]);
      float hbuf[H];
      for (int j = 0; j < H; ++j) {
        float s = b1[j];
        for (int i = 0; i < 5; ++i) s += xs[i] * W1[i][j];
        hbuf[j] = sigmoid(s);
      }
      float o = b2[0];
      for (int j = 0; j < H; ++j) o += hbuf[j] * W2[j][0];
      float tense = sigmoid(o);
      float calm  = 1.0f - tense;

      // ─ D) 결과 패킷(JSON) 생성 & 전송 ─
      StaticJsonDocument<128> doc;
      doc["temp"]    = SharedVitals::deviceTemperature;
      doc["battery"] = SharedVitals::battery;
      doc["hr"]      = hr;
      doc["calm"]    = calm;
      doc["tense"]   = tense;

      char payload[128];
      size_t len = serializeJson(doc, payload);
      payload[len++] = '\n';          // 가독성용 줄바꿈

      // ① 시리얼 디버그
      Serial.write(payload, len);

      // ② BLE Notify (NUS)
      if (txChar) {
        txChar->setValue((uint8_t*)payload, len);
        txChar->notify();
      }
    }
  }
}

// ==========================================================================
