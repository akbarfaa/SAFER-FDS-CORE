# SAFER FDS — Model Card v3

**Model Name**: SAFER Ensemble Fraud Scoring Engine (Advanced Industrial v3)  
**Version**: v3  
**Trained**: 2026-07-19 19:15  
**Status**: ✅ Production Ready  

---

## Model Overview

SAFER menggunakan ensemble dari dua model gradient boosting (XGBoost + LightGBM) untuk mendeteksi transaksi fraud pada ekosistem pembayaran digital Indonesia. Skor akhir dihitung dari rata-rata probabilitas kedua model.
Model V3 ditingkatkan dengan rekayasa fitur siklikal waktu, rasio nilai-ke-umur akun, dan kecepatan jarak pergerakan.

### Algorithms
| Model | Type | Library | Estimators |
|-------|------|---------|------------|
| XGBoost | Extreme Gradient Boosting | xgboost | 300 |
| LightGBM | Light Gradient Boosting | lightgbm | 300 |

---

## Dataset

| Metric | Train | Test |
|--------|-------|------|
| Total Records | 400,000 | 150,000 |
| Fraud Records | 40,000 | 15,000 |
| Fraud Ratio | 10.0% | 10.0% |
| Features | 34 | 34 |

### Data Source
Dataset sintetis yang dirancang untuk mereplikasi pola transaksi digital Indonesia secara realistis. Mencakup distribusi bank lokal (BCA, BRI, Mandiri, BNI, BSI, dll.), rail pembayaran (QRIS, BI-FAST, E-Wallet, dll.), dan geolokasi 20 kota besar Indonesia.

---

## Performance Metrics

| Metric | XGBoost | LightGBM | **Ensemble (V3)** |
|--------|---------|----------|--------------|
| Accuracy | 0.9993 | 0.9993 | **0.9994** |
| Precision | 0.9984 | 0.9993 | **0.9993** |
| Recall | 0.9945 | 0.9941 | **0.9944** |
| F1 Score | 0.9965 | 0.9967 | **0.9968** |
| ROC-AUC | 0.9999 | 0.9999 | **0.9999** |
| PR-AUC | 0.9994 | 0.9995 | **0.9994** |

---

## Scenario-Specific Performance Breakdown

Metrik berikut dihitung secara terpisah untuk setiap skenario fraud khusus untuk mengukur presisi dan daya jangkau deteksi model:

| # | Skenario Fraud | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| 1 | **Smurfing/Structuring** | 0.9929 | 1.0000 | 0.9964 | 1.0000 |
| 2 | **Device Farm** | 0.9927 | 1.0000 | 0.9963 | 1.0000 |
| 3 | **Gambling Laundering** | 0.9927 | 1.0000 | 0.9963 | 1.0000 |
| 4 | **Slot Cashout Ring** | 0.9929 | 1.0000 | 0.9964 | 1.0000 |
| 5 | **Account Takeover** | 0.9926 | 1.0000 | 0.9963 | 1.0000 |
| 6 | **Deepfake Social Engineering** | 0.9928 | 1.0000 | 0.9964 | 1.0000 |
| 7 | **Mule Ring** | 0.9928 | 1.0000 | 0.9964 | 1.0000 |
| 8 | **Impossible Travel** | 0.9927 | 1.0000 | 0.9963 | 1.0000 |


---

## Feature List (34 features)

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

### Numeric & Engineered (14)
- `amount` — Nominal transaksi (IDR)
- `account_age_days` — Umur akun dalam hari
- `velocity_count` — Jumlah transaksi dalam jendela waktu pendek
- `geo_distance_km` — Jarak geografis transaksi (km)
- `sender_lat`, `sender_lng`, `receiver_lat`, `receiver_lng` — Koordinat
- `hour_sin` — Siklikal Waktu (Sinus)
- `hour_cos` — Siklikal Waktu (Kosinus)
- `amount_to_age_ratio` — Rasio Nominal Transaksi terhadap Umur Akun
- `dist_to_velocity_ratio` — Rasio Jarak Transaksi terhadap Frekuensi Transaksi
- `amount_to_distance_ratio` — Rasio Nominal Transaksi terhadap Jarak Transaksi

---

## Retraining Safety Protocol

- Model hanya diretrain secara **offline** oleh admin via `retrain_pipeline.py`
- Data training harus lolos validasi (fraud ratio 1-35%, no negative amounts, no extreme outliers)
- Model baru **otomatis ditolak** jika performa lebih rendah dari threshold:
  - Recall ≥ 85%
  - Precision ≥ 80%
  - ROC-AUC ≥ 90%
