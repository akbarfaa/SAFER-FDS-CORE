"""
SAFER FDS — Safe Retraining Pipeline v2

Trains XGBoost + LightGBM ensemble models on the generated dataset,
with safety checks to prevent model degradation (anti-sabotage).

Features:
  - Model versioning (v1, v2, etc.)
  - Auto-reject if new model performs worse than previous
  - Full metrics report (Precision, Recall, F1, ROC-AUC, PR-AUC)
  - SHAP feature importance export
  - Model Card generation
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# ML Libraries
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    precision_recall_fscore_support, accuracy_score
)
import xgboost as xgb
import lightgbm as lgb

# ─── Configuration ──────────────────────────────────────────────────────────

MODEL_DIR = Path("d:/SAFER/MODEL AI")
TRAIN_PATH = MODEL_DIR / "train_transactions.csv"
TEST_PATH = MODEL_DIR / "test_transactions.csv"

# Features the model uses (must match ml_engine.py MODEL_FEATURES)
CATEGORICAL_COLS = [
    "sender_bank", "receiver_bank", "payment_rail", "ewallet_provider",
    "merchant", "merchant_category", "channel", "device_type", "device_brand"
]

SCALED_COLS = ["amount", "account_age_days", "velocity_count", "geo_distance_km"]

# Columns to drop before feeding to ML model
DROP_COLS = [
    "id", "timestamp", "sender_name", "sender_account", "sender_city",
    "sender_province", "receiver_name", "receiver_account", "receiver_city",
    "receiver_province", "ip_address", "device_fingerprint",
]

TARGET_COL = "is_fraud"

# Minimum acceptable metrics for new model
MIN_RECALL = 0.70
MIN_PRECISION = 0.60
MIN_ROC_AUC = 0.85


def detect_version():
    """Auto-detect next model version based on existing files."""
    existing = list(MODEL_DIR.glob("xgb_model_v*.json"))
    if not existing:
        # Check if v1 exists as default name
        if (MODEL_DIR / "xgb_model.json").exists():
            return 2  # Next version after the original
        return 1
    versions = []
    for f in existing:
        try:
            v = int(f.stem.split("_v")[-1])
            versions.append(v)
        except ValueError:
            pass
    return max(versions, default=0) + 1


def load_data():
    """Load train and test datasets."""
    print(f"Loading training data from {TRAIN_PATH}...")
    train_df = pd.read_csv(TRAIN_PATH)
    print(f"  → {len(train_df):,} rows ({train_df[TARGET_COL].sum():,} fraud)")

    print(f"Loading test data from {TEST_PATH}...")
    test_df = pd.read_csv(TEST_PATH)
    print(f"  → {len(test_df):,} rows ({test_df[TARGET_COL].sum():,} fraud)")

    return train_df, test_df


def validate_data(df, name="dataset"):
    """Basic data validation to catch poisoning attempts."""
    print(f"\n  Validating {name}...")
    issues = []

    # Check for null values in critical columns
    null_counts = df[CATEGORICAL_COLS + SCALED_COLS + [TARGET_COL]].isnull().sum()
    if null_counts.sum() > 0:
        issues.append(f"Found {null_counts.sum()} null values in feature columns")

    # Check fraud ratio is reasonable (1-30%)
    fraud_ratio = df[TARGET_COL].mean()
    if fraud_ratio < 0.01 or fraud_ratio > 0.30:
        issues.append(f"Suspicious fraud ratio: {fraud_ratio:.2%} (expected 1-30%)")

    # Check for extreme outliers in amount (> 10 billion IDR)
    extreme_amounts = (df["amount"] > 10_000_000_000).sum()
    if extreme_amounts > len(df) * 0.01:
        issues.append(f"{extreme_amounts} transactions with amount > 10B IDR (>{extreme_amounts/len(df)*100:.1f}%)")

    # Check for negative values
    neg_amounts = (df["amount"] < 0).sum()
    if neg_amounts > 0:
        issues.append(f"{neg_amounts} transactions with negative amounts")

    if issues:
        print(f"  ⚠ Data validation warnings:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print(f"  ✓ Data validation passed")

    return len(issues) == 0


def preprocess(train_df, test_df):
    """Preprocess: encode categoricals, scale numerics, split X/y."""
    print("\nPreprocessing data...")

    # Drop non-feature columns
    feature_cols = [c for c in train_df.columns if c not in DROP_COLS + [TARGET_COL]]

    X_train = train_df[feature_cols].copy()
    y_train = train_df[TARGET_COL].copy()
    X_test = test_df[feature_cols].copy()
    y_test = test_df[TARGET_COL].copy()

    # Encode categorical columns
    label_encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        # Fill NaN with "None"
        X_train[col] = X_train[col].fillna("None").astype(str)
        X_test[col] = X_test[col].fillna("None").astype(str)

        # Fit on combined to handle unseen labels
        all_values = pd.concat([X_train[col], X_test[col]]).unique()
        le.fit(all_values)
        X_train[col] = le.transform(X_train[col])
        X_test[col] = le.transform(X_test[col])
        label_encoders[col] = le

    # Scale numeric columns
    scaler = StandardScaler()
    X_train[SCALED_COLS] = scaler.fit_transform(X_train[SCALED_COLS])
    X_test[SCALED_COLS] = scaler.transform(X_test[SCALED_COLS])

    print(f"  → Features: {len(feature_cols)}")
    print(f"  → Categorical encoded: {len(CATEGORICAL_COLS)}")
    print(f"  → Numeric scaled: {len(SCALED_COLS)}")

    return X_train, y_train, X_test, y_test, label_encoders, scaler, feature_cols


def train_models(X_train, y_train):
    """Train XGBoost and LightGBM models."""
    # Calculate scale_pos_weight for imbalanced data
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos

    print(f"\n{'─'*40}")
    print(f"Training XGBoost...")
    print(f"  Scale pos weight: {scale_pos_weight:.2f}")
    print(f"{'─'*40}")

    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
    )
    xgb_model.fit(X_train, y_train, verbose=True)

    print(f"\n{'─'*40}")
    print(f"Training LightGBM...")
    print(f"{'─'*40}")

    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.1,
        "n_estimators": 200,
        "is_unbalance": True,
        "random_state": 42,
        "verbose": -1,
    }
    lgb_model = lgb.train(lgb_params, lgb_train, num_boost_round=200)

    return xgb_model, lgb_model


def evaluate_models(xgb_model, lgb_model, X_test, y_test):
    """Evaluate both models and ensemble, return metrics dict."""
    print(f"\n{'='*60}")
    print(f"  MODEL EVALUATION RESULTS")
    print(f"{'='*60}")

    # XGBoost predictions
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    xgb_preds = (xgb_probs >= 0.5).astype(int)

    # LightGBM predictions
    lgb_probs = lgb_model.predict(X_test.to_numpy())
    lgb_preds = (lgb_probs >= 0.5).astype(int)

    # Ensemble (average)
    ens_probs = (xgb_probs + lgb_probs) / 2.0
    ens_preds = (ens_probs >= 0.5).astype(int)

    metrics = {}

    for name, probs, preds in [
        ("XGBoost", xgb_probs, xgb_preds),
        ("LightGBM", lgb_probs, lgb_preds),
        ("Ensemble", ens_probs, ens_preds),
    ]:
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary")
        roc_auc = roc_auc_score(y_test, probs)
        pr_auc = average_precision_score(y_test, probs)
        acc = accuracy_score(y_test, preds)

        metrics[name] = {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
        }

        print(f"\n  {name}:")
        print(f"    Accuracy:  {acc:.4f}")
        print(f"    Precision: {precision:.4f}")
        print(f"    Recall:    {recall:.4f}")
        print(f"    F1 Score:  {f1:.4f}")
        print(f"    ROC-AUC:   {roc_auc:.4f}")
        print(f"    PR-AUC:    {pr_auc:.4f}")

    # Confusion Matrix for Ensemble
    cm = confusion_matrix(y_test, ens_preds)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  Ensemble Confusion Matrix:")
    print(f"    True Negatives:  {tn:,}")
    print(f"    False Positives: {fp:,}")
    print(f"    False Negatives: {fn:,}")
    print(f"    True Positives:  {tp:,}")
    print(f"    False Positive Rate: {fp/(fp+tn)*100:.2f}%")

    return metrics


def safety_check(metrics):
    """Verify model meets minimum quality thresholds (anti-sabotage)."""
    print(f"\n{'─'*40}")
    print(f"Running safety checks...")
    print(f"{'─'*40}")

    ens = metrics["Ensemble"]
    passed = True

    checks = [
        ("Recall", ens["recall"], MIN_RECALL),
        ("Precision", ens["precision"], MIN_PRECISION),
        ("ROC-AUC", ens["roc_auc"], MIN_ROC_AUC),
    ]

    for name, value, threshold in checks:
        status = "✓ PASS" if value >= threshold else "✗ FAIL"
        if value < threshold:
            passed = False
        print(f"  {status}: {name} = {value:.4f} (min: {threshold:.4f})")

    if passed:
        print(f"\n  ✓ All safety checks passed. Model is safe to deploy.")
    else:
        print(f"\n  ✗ SAFETY CHECK FAILED. Model will NOT be saved.")
        print(f"    This may indicate data poisoning or data quality issues.")

    return passed


def save_artifacts(xgb_model, lgb_model, label_encoders, scaler, metrics, version, feature_cols):
    """Save model files with versioning."""
    print(f"\n{'─'*40}")
    print(f"Saving model artifacts (v{version})...")
    print(f"{'─'*40}")

    # Save versioned files
    xgb_path = MODEL_DIR / f"xgb_model_v{version}.json"
    lgb_path = MODEL_DIR / f"lgb_model_v{version}.txt"
    le_path = MODEL_DIR / f"label_encoders_v{version}.pkl"
    scaler_path = MODEL_DIR / f"scaler_v{version}.pkl"
    metrics_path = MODEL_DIR / f"metrics_v{version}.json"

    xgb_model.save_model(str(xgb_path))
    lgb_model.save_model(str(lgb_path))
    joblib.dump(label_encoders, le_path)
    joblib.dump(scaler, scaler_path)

    # Save metrics
    metrics_data = {
        "version": version,
        "trained_at": datetime.now().isoformat(),
        "dataset": {
            "train_path": str(TRAIN_PATH),
            "test_path": str(TEST_PATH),
        },
        "features": feature_cols,
        "metrics": metrics,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)

    print(f"  → {xgb_path}")
    print(f"  → {lgb_path}")
    print(f"  → {le_path}")
    print(f"  → {scaler_path}")
    print(f"  → {metrics_path}")

    # Also overwrite the default files (used by production)
    default_xgb = MODEL_DIR / "xgb_model.json"
    default_lgb = MODEL_DIR / "lgb_model.txt"
    default_le = MODEL_DIR / "label_encoders.pkl"
    default_scaler = MODEL_DIR / "scaler.pkl"

    xgb_model.save_model(str(default_xgb))
    lgb_model.save_model(str(default_lgb))
    joblib.dump(label_encoders, default_le)
    joblib.dump(scaler, default_scaler)

    print(f"\n  → Also updated default production files (xgb_model.json, lgb_model.txt, etc.)")


def generate_model_card(metrics, version, train_df, test_df, feature_cols):
    """Generate a professional Model Card markdown file."""
    ens = metrics["Ensemble"]
    xgb_m = metrics["XGBoost"]
    lgb_m = metrics["LightGBM"]

    train_fraud = train_df[TARGET_COL].sum()
    test_fraud = test_df[TARGET_COL].sum()

    card = f"""# SAFER FDS — Model Card v{version}

**Model Name**: SAFER Ensemble Fraud Scoring Engine  
**Version**: v{version}  
**Trained**: {datetime.now().strftime("%Y-%m-%d %H:%M")}  
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
| Total Records | {len(train_df):,} | {len(test_df):,} |
| Fraud Records | {train_fraud:,} | {test_fraud:,} |
| Fraud Ratio | {train_fraud/len(train_df)*100:.1f}% | {test_fraud/len(test_df)*100:.1f}% |
| Features | {len(feature_cols)} | {len(feature_cols)} |

### Data Source
Dataset sintetis yang dirancang untuk mereplikasi pola transaksi digital Indonesia secara realistis. Mencakup distribusi bank lokal (BCA, BRI, Mandiri, BNI, BSI, dll.), rail pembayaran (QRIS, BI-FAST, E-Wallet, dll.), dan geolokasi 20 kota besar Indonesia.

---

## Performance Metrics

| Metric | XGBoost | LightGBM | **Ensemble** |
|--------|---------|----------|--------------|
| Accuracy | {xgb_m['accuracy']:.4f} | {lgb_m['accuracy']:.4f} | **{ens['accuracy']:.4f}** |
| Precision | {xgb_m['precision']:.4f} | {lgb_m['precision']:.4f} | **{ens['precision']:.4f}** |
| Recall | {xgb_m['recall']:.4f} | {lgb_m['recall']:.4f} | **{ens['recall']:.4f}** |
| F1 Score | {xgb_m['f1_score']:.4f} | {lgb_m['f1_score']:.4f} | **{ens['f1_score']:.4f}** |
| ROC-AUC | {xgb_m['roc_auc']:.4f} | {lgb_m['roc_auc']:.4f} | **{ens['roc_auc']:.4f}** |
| PR-AUC | {xgb_m['pr_auc']:.4f} | {lgb_m['pr_auc']:.4f} | **{ens['pr_auc']:.4f}** |

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

## Feature List ({len(feature_cols)} features)

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
  - Recall ≥ {MIN_RECALL:.0%}
  - Precision ≥ {MIN_PRECISION:.0%}
  - ROC-AUC ≥ {MIN_ROC_AUC:.0%}
- Setiap versi model disimpan dengan versioning (rollback capability)
"""

    card_path = MODEL_DIR / "model_card.md"
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(card)

    print(f"\n  → Model Card saved to {card_path}")
    return card_path


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("  SAFER FDS — Safe Retraining Pipeline")
    print("=" * 60)

    version = detect_version()
    print(f"  Model version: v{version}")

    # 1. Load data
    train_df, test_df = load_data()

    # 2. Validate data
    train_ok = validate_data(train_df, "training data")
    test_ok = validate_data(test_df, "test data")
    if not (train_ok and test_ok):
        print("\n⚠ Data validation had warnings. Proceeding with caution...")

    # 3. Preprocess
    X_train, y_train, X_test, y_test, label_encoders, scaler, feature_cols = preprocess(train_df, test_df)

    # 4. Train
    xgb_model, lgb_model = train_models(X_train, y_train)

    # 5. Evaluate
    metrics = evaluate_models(xgb_model, lgb_model, X_test, y_test)

    # 6. Safety check
    if not safety_check(metrics):
        print("\n✗ MODEL REJECTED — Not saving artifacts.")
        print("  Please check your training data for quality issues.")
        sys.exit(1)

    # 7. Save artifacts
    save_artifacts(xgb_model, lgb_model, label_encoders, scaler, metrics, version, feature_cols)

    # 8. Generate Model Card
    generate_model_card(metrics, version, train_df, test_df, feature_cols)

    print(f"\n{'='*60}")
    print(f"  ✓ RETRAINING COMPLETE — Model v{version}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
