// ===== main.cpp  (ESP32 Feather + EmotiBit + BLE UART) =====
// 완전 무선: 센서 → BLE(Nordic UART Service) → 안드로이드 앱

#include <Arduino.h>
#include <NimBLEDevice.h>         // ESP32 NimBLE 스택 (가볍고 안정적)
#include <ArduinoJson.h>          // JSON 직렬화
#include "EmotiBit.h"
#include "SharedVitals.h"
#include "hrv_model_params.h"

// ───────────────────────────────────── CONSTANTS ─────────────────────────────
#define SERIAL_BAUD      115200
#define WINDOW_SIZE      30       // HRV 윈도우 크기 (IBI 샘플 수)
#define STEP_SAMPLES      5       // 5 샘플마다 모델 추론
#define LOG_INTERVAL   1000       // ms 단위 디버그 로그 주기

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

// ──────────────────────────── UTILITY FUNCTIONS ──────────────────────────────
static inline float sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
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

  // (a) 서비스 UUID 포함 → Service Data (AD type 0x21) 에 붙음
  adv->addServiceUUID(NUS_SERVICE_UUID);

  // (b) 스캔 리스폰스에 로컬 이름 & Complete List of 128-bit UUID 포함
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
  float ibi = SharedVitals::ibi;
  float hr  = SharedVitals::heartRate;

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

      // ─ A) 시간 영역 피처 ─
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

      // ─ B) 주파수 영역 피처 ─ (DFT, 0.04–0.40 Hz)
      const float fs = 4.0f;
      float lf = 0, hf = 0, total = 0;
      for (int k = 1; k <= WINDOW_SIZE / 2; ++k) {
        float re = 0, im = 0;
        for (int n = 0; n < WINDOW_SIZE; ++n) {
          float a = 2 * PI * k * n / float(WINDOW_SIZE);
          re += ibi_buf[n] * cosf(a);
          im -= ibi_buf[n] * sinf(a);
        }
        float power = (re * re + im * im) / float(WINDOW_SIZE);
        float fk = k * fs / float(WINDOW_SIZE);
        if      (fk >= 0.04f   && fk < 0.15f) lf    += power;
        else if (fk >= 0.15f   && fk < 0.40f) hf    += power;
        if      (fk >= 0.0033f && fk < 0.40f) total += power;
      }

      Serial.printf("    DBG: sdnn=%.2f rmssd=%.2f lf=%.1f hf=%.1f tot=%.1f\n",
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

      // ② BLE Notify (NUS) - getSubscribedCount()로 구독 확인
      if (txChar) {
        txChar->setValue((uint8_t*)payload, len);
        txChar->notify();
      }
    }
  }
}

// ==========================================================================
