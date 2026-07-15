# SAFER FDS — Model Card v2

**Model Name**: SAFER Ensemble Fraud Scoring Engine  
**Version**: v2  
**Trained**: 2026-07-15 18:55  
**Status**: ✅ Production Ready  

---

## Model Overview

SAFER menggunakan ensemble dari dua model gradient boosting (XGBoost + LightGBM) untuk mendeteksi transaksi fraud pada ekosistem pembayaran digital Indonesia. Skor akhir dihitung dari rata-rata probabilitas kedua model.

### Algorithms
| Model | Type | Library |
|-------|------|---------|
| XGBoost | Extreme Gradient Boosting | xgboost |
| LightGBM | Light Gradient Boosting | lightgbm |

---

## Dataset

| Metric | Train | Test |
|--------|-------|------|
| Total Records | 80,000 | 20,000 |
| Fraud Records | 8,000 | 2,000 |
| Fraud Ratio | 10.0% | 10.0% |
| Features | 29 | 29 |

### Data Source
Dataset sintetis yang dirancang untuk mereplikasi pola transaksi digital Indonesia secara realistis. Mencakup distribusi bank lokal (BCA, BRI, Mandiri, BNI, BSI, dll.), rail pembayaran (QRIS, BI-FAST, E-Wallet, dll.), dan geolokasi 20 kota besar Indonesia.

---

## Performance Metrics

| Metric | XGBoost | LightGBM | **Ensemble** |
|--------|---------|----------|--------------|
| Accuracy | 0.9990 | 0.9989 | **0.9989** |
| Precision | 0.9985 | 0.9985 | **0.9985** |
| Recall | 0.9920 | 0.9905 | **0.9905** |
| F1 Score | 0.9952 | 0.9945 | **0.9945** |
| ROC-AUC | 0.9999 | 0.9999 | **0.9999** |
| PR-AUC | 0.9995 | 0.9995 | **0.9995** |

---

## Fraud Patterns Detected (8 Scenarios)

| # | Pattern | Deskripsi | Indikator Utama |
|---|---------|-----------|-----------------|
| 1 | **Mule Ring** | Jaringan rekening penampung money laundering | Velocity anomaly, unusual beneficiary, multiple rapid transfers |
| 2 | **Device Farm** | Satu perangkat mengoperasikan banyak akun | Device fingerprint sharing, device mismatch, new device |
| 3 | **Account Takeover** | Pengambilalihan akun oleh pihak tak berwenang | New device, SIM swap, geo mismatch, suspicious IP |
| 4 | **Impossible Travel** | Transaksi dari 2 lokasi berjauhan dalam waktu singkat | Extreme geo distance, velocity anomaly |
| 5 | **Smurfing/Structuring** | Pemecahan transaksi besar menjadi banyak kecil | High velocity, unusual beneficiary, consistent amounts |
| 6 | **Judi Online Laundering** | Deposit berulang ke merchant gambling | Risky merchant, off-hours, high frequency |
| 7 | **Deepfake Social Engineering** | Identitas palsu + akun baru + transfer besar | New account, SIM swap, high value, new beneficiary |
| 8 | **Slot Cashout Ring** | Merchant palsu menerima pembayaran dari banyak akun | Multiple senders to same merchant, unusual beneficiary |

---

## Feature List (29 features)

### Binary Risk Indicators (12)
- `is_new_device` — Perangkat baru belum pernah terdaftar
- `is_velocity_anomaly` — Pola transaksi beruntun dalam waktu singkat
- `is_geo_mismatch` — Jarak geografis antara pengirim dan penerima ekstrem
- `is_off_hours` — Transaksi pada jam 24:00-04:00 WIB
- `is_high_value_for_rail` — Nominal melebihi threshold rail pembayaran
- `is_suspicious_ip` — IP dari VPN/hosting provider
- `is_risky_merchant` — Merchant kategori Crypto/Gambling/Lending
- `is_new_account` — Umur akun < 30 hari
- `has_failed_attempts` — Kegagalan otentikasi sebelum transaksi
- `is_device_mismatch` — Device fingerprint tidak cocok baseline
- `is_sim_swap` — SIM swap terdeteksi < 48 jam
- `is_unusual_beneficiary` — Penerima belum pernah menerima dari pengirim

### Categorical (9)
- `sender_bank`, `receiver_bank`, `payment_rail`, `ewallet_provider`
- `merchant`, `merchant_category`, `channel`, `device_type`, `device_brand`

### Numeric (8)
- `amount` — Nominal transaksi (IDR)
- `account_age_days` — Umur akun dalam hari
- `velocity_count` — Jumlah transaksi dalam jendela waktu pendek
- `geo_distance_km` — Jarak geografis transaksi (km)
- `sender_lat`, `sender_lng`, `receiver_lat`, `receiver_lng` — Koordinat

---

## Limitasi & Bias

1. **Synthetic Data**: Model dilatih pada data sintetis, belum divalidasi pada data transaksi riil
2. **Indonesia-Specific**: Hanya memahami pola transaksi dan bank Indonesia
3. **Temporal Bias**: Belum memiliki fitur temporal tingkat lanjut (seasonal patterns)
4. **No Online Learning**: Model tidak belajar otomatis dari data produksi (by design, untuk keamanan)

---

## Retraining Safety Protocol

- Model hanya diretrain secara **offline** oleh admin via `retrain_pipeline.py`
- Data training harus lolos validasi (fraud ratio 1-30%, no extreme outliers)
- Model baru **otomatis ditolak** jika performa lebih rendah dari threshold:
  - Recall ≥ 70%
  - Precision ≥ 60%
  - ROC-AUC ≥ 85%
- Setiap versi model disimpan dengan versioning (rollback capability)
