"""
SAFER Scoring Service — SHAP Explainer

Computes SHAP feature contributions for a transaction
and generates analyst-quality natural language explanations in Bahasa Indonesia.
"""

import os
import sys
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.config import get_severity

# Friendly Indonesian labels for features
FEATURE_LABELS = {
    "sender_bank": "Bank Pengirim",
    "sender_lat": "Latitude Pengirim",
    "sender_lng": "Longitude Pengirim",
    "receiver_bank": "Bank Penerima",
    "receiver_lat": "Latitude Penerima",
    "receiver_lng": "Longitude Penerima",
    "amount": "Nominal Transaksi",
    "payment_rail": "Rail Pembayaran",
    "ewallet_provider": "Penyedia E-Wallet",
    "merchant": "Nama Merchant",
    "merchant_category": "Kategori Merchant",
    "channel": "Channel Transaksi",
    "device_type": "Tipe Perangkat",
    "device_brand": "Merek Perangkat",
    "is_new_device": "Perangkat Baru",
    "account_age_days": "Umur Akun Pengirim",
    "is_velocity_anomaly": "Anomali Kecepatan Transaksi",
    "is_geo_mismatch": "Geolokasi Tidak Cocok",
    "is_off_hours": "Transaksi Jam Tidak Wajar",
    "is_high_value_for_rail": "Transaksi Nilai Tinggi",
    "is_suspicious_ip": "IP Address Mencurigakan",
    "is_risky_merchant": "Merchant Berisiko Tinggi",
    "is_new_account": "Akun Baru Dibuat",
    "has_failed_attempts": "Percobaan Gagal Berulang",
    "is_device_mismatch": "Sidik Jari Perangkat Berbeda",
    "is_sim_swap": "SIM Swap Terdeteksi",
    "is_unusual_beneficiary": "Penerima Tidak Biasa",
    "velocity_count": "Frekuensi Transaksi Menit Terakhir",
    "geo_distance_km": "Jarak Geografis Transaksi"
}

# Detailed descriptions for triggered indicators
FEATURE_DESCRIPTIONS = {
    "is_new_device": "Transaksi dilakukan dari perangkat baru yang belum pernah terdaftar.",
    "is_velocity_anomaly": "Terdeteksi pola transaksi beruntun (velocity anomaly) dalam jangka waktu singkat.",
    "is_geo_mismatch": "Terdapat deviasi jarak geografis yang ekstrem antara kota pengirim dan penerima.",
    "is_off_hours": "Transaksi dieksekusi pada jam tidak wajar (off-hours), umumnya antara pukul 24:00 - 04:00 WIB.",
    "is_high_value_for_rail": "Nominal transaksi melampaui rata-rata historis (high-value threshold) untuk rail pembayaran ini.",
    "is_suspicious_ip": "Alamat IP pengirim terindikasi berasal dari hosting provider atau network yang sering diasosiasikan dengan fraud.",
    "is_risky_merchant": "Dana dikirim ke merchant berkategori risiko tinggi (seperti Crypto Exchange, Gambling, atau Pinjol).",
    "is_new_account": "Umur akun pengirim masih sangat baru (kurang dari 30 hari) sejak registrasi.",
    "has_failed_attempts": "Terdapat beberapa kali kegagalan otentikasi atau input PIN/OTP sebelum transaksi berhasil.",
    "is_device_mismatch": "Identitas sidik jari perangkat (device fingerprint) tidak cocok dengan baseline profil akun.",
    "is_sim_swap": "Terdeteksi indikasi penggantian kartu SIM (SIM swap) dalam waktu kurang dari 48 jam terakhir.",
    "is_unusual_beneficiary": "Akun penerima (beneficiary) tidak pernah menerima dana dari pengirim ini sebelumnya.",
    "amount": "Nominal transaksi relatif besar dibanding aktivitas normal.",
    "velocity_count": "Frekuensi transaksi abnormal yang tinggi pada akun ini dalam jendela waktu sempit.",
    "geo_distance_km": "Jarak transaksi geografis terlampau jauh melebihi mobilitas fisik normal pengirim."
}


class SHAPExplainer:
    """Computes SHAP values using XGBoost and yields natural language reasoning."""

    def __init__(self, xgb_model):
        self.xgb_model = xgb_model
        self.explainer = None
        if xgb_model is not None:
            try:
                # TreeExplainer is fast and native for XGBoost
                self.explainer = shap.TreeExplainer(xgb_model)
                print("[SHAPExplainer] SHAP TreeExplainer initialized successfully.")
            except Exception as e:
                print(f"[SHAPExplainer] Failed to initialize SHAP TreeExplainer: {e}")

    def explain(self, tx: dict, features_df: pd.DataFrame, risk_score: int, severity: str, prob: float) -> dict:
        """
        Explain the transaction using SHAP and return:
        - raw SHAP values dict
        - sorted primary risk factors
        - Indonesian natural language reasoning paragraph
        - suggested action
        """
        shap_vals_dict = {}
        primary_risk_factors = []
        ai_reasoning = ""
        suggested_action = ""

        # Default recommendations
        if severity == "critical":
            suggested_action = "HOLD TRANSACTION IMMEDIATELY & FREEZE ACCOUNT"
        elif severity == "high":
            suggested_action = "HOLD TRANSACTION FOR MANUAL VERIFICATION"
        elif severity == "medium":
            suggested_action = "MONITOR CLOSELY & ENABLE STEP-UP AUTH"
        else:
            suggested_action = "ALLOW TRANSACTION"

        # If SHAP is not initialized or preprocess failed
        if self.explainer is None or features_df is None:
            # Fallback to rule-based explanation
            ai_reasoning = self._generate_fallback_reasoning(tx, risk_score, severity, prob, suggested_action)
            return {
                "shap_values": {},
                "primary_risk_factors": [],
                "ai_reasoning": ai_reasoning,
                "suggested_action": suggested_action
            }

        try:
            # Compute SHAP values for the single row
            sv = self.explainer.shap_values(features_df)
            
            # For newer shap/xgboost, shape is (1, n_features) or (1, n_features, 2)
            # We want the values for class 1 (Fraud)
            if len(sv.shape) == 3:  # (1, n_features, 2)
                row_sv = sv[0, :, 1]
            else:  # (1, n_features)
                row_sv = sv[0]

            feature_names = features_df.columns.tolist()

            # Map to dictionary
            for name, val in zip(feature_names, row_sv):
                shap_vals_dict[name] = float(val)

            # Sort features by positive SHAP values descending (risk drivers)
            sorted_features = sorted(shap_vals_dict.items(), key=lambda x: x[1], reverse=True)

            # Extract features contributing POSITIVELY to risk (SHAP value > 0)
            risk_factors = []
            for feat, val in sorted_features:
                if val > 0.01:  # Only significant positive contributions
                    label = FEATURE_LABELS.get(feat, feat)
                    risk_factors.append({
                        "feature": feat,
                        "shap_value": val,
                        "label": label
                    })
                    if len(risk_factors) >= 5:  # Top 5 factors
                        break

            primary_risk_factors = risk_factors

            # Build narrative explanation
            ai_reasoning = self._build_indonesian_narrative(tx, risk_score, severity, prob, risk_factors, suggested_action)

        except Exception as e:
            print(f"[SHAPExplainer] SHAP computation failed: {e}")
            ai_reasoning = self._generate_fallback_reasoning(tx, risk_score, severity, prob, suggested_action)

        return {
            "shap_values": shap_vals_dict,
            "primary_risk_factors": primary_risk_factors,
            "ai_reasoning": ai_reasoning,
            "suggested_action": suggested_action
        }

    def _build_indonesian_narrative(self, tx: dict, risk_score: int, severity: str, prob: float, risk_factors: list, recommendation: str) -> str:
        """Constructs a narrative, cohesive analysis paragraph in Indonesian."""
        # 1. Opening
        if severity == "critical":
            opening = f"**PERINGATAN KRITIS**: Sistem mendeteksi transaksi dengan **profil risiko ekstrem** (Skor Risiko: **{risk_score}/100**, Probabilitas: **{prob*100:.1f}%**)."
        elif severity == "high":
            opening = f"**ANALISIS RISIKO TINGGI**: Transaksi terdeteksi memiliki **deviasi signifikan** dari pola normal customer (Skor Risiko: **{risk_score}/100**, Probabilitas: **{prob*100:.1f}%**)."
        elif severity == "medium":
            opening = f"**MONITORING RISIKO SEDANG**: Transaksi menunjukkan indikasi **anomali minor** (Skor Risiko: **{risk_score}/100**)."
        else:
            opening = f"**TRANSAKSI NORMAL**: Transaksi ini berjalan dalam **parameter wajar** (Skor Risiko: **{risk_score}/100**)."

        # 2. Context line
        amount_fmt = f"Rp {float(tx.get('amount', 0)):,.0f}".replace(",", ".")
        sender_info = f"**{tx.get('sender_name', 'Customer')}** (rekening **{tx.get('sender_account', 'N/A')}** di **{tx.get('sender_bank', 'N/A')}**, kota **{tx.get('sender_city', 'N/A')}**)"
        receiver_info = f"**{tx.get('receiver_name', 'Beneficiary')}** (rekening **{tx.get('receiver_account', 'N/A')}** di **{tx.get('receiver_bank', 'N/A')}**, kota **{tx.get('receiver_city', 'N/A')}**)"
        
        context = f"Detail transaksi: Pengiriman dana sebesar **{amount_fmt}** via **{tx.get('payment_rail', 'N/A')}** oleh {sender_info} ke {receiver_info}."

        # 3. Factor explanations
        factor_narratives = []
        for rf in risk_factors:
            feat = rf["feature"]
            desc = FEATURE_DESCRIPTIONS.get(feat)
            if desc:
                # Add specific values if helpful
                if feat == "amount":
                    desc = f"Nominal transaksi sebesar **{amount_fmt}** dinilai tidak biasa."
                elif feat == "geo_distance_km" and "geo_distance_km" in tx:
                    desc = f"Jarak geolokasi transaksi sangat jauh (**{float(tx['geo_distance_km']):.1f} km**) melebihi mobilitas fisik normal."
                elif feat == "velocity_count" and "velocity_count" in tx:
                    desc = f"Frekuensi transaksi beruntun sangat tinggi (**{int(tx['velocity_count'])} kali** dalam waktu singkat)."
                factor_narratives.append(desc)

        if factor_narratives:
            analysis = "Indikator risiko utama yang mendorong penilaian ini meliputi: " + " ".join(factor_narratives)
        else:
            analysis = "Tidak ditemukan indikator anomali yang mencurigakan secara signifikan pada transaksi ini."

        # 4. Action/Conclusion
        if severity in ("critical", "high"):
            action = f"**Rekomendasi Tindakan**: **{recommendation}**. Tahan dana segera dan assign tiket investigasi ini ke unit Fraud Operations."
        elif severity == "medium":
            action = f"**Rekomendasi Tindakan**: **{recommendation}**. Monitor pola aktivitas akun dalam 24 jam ke depan."
        else:
            action = f"**Rekomendasi Tindakan**: **{recommendation}**. Proses transaksi seperti biasa."

        return f"{opening}\n\n{context}\n\n{analysis}\n\n{action}"

    def _generate_fallback_reasoning(self, tx: dict, risk_score: int, severity: str, prob: float, recommendation: str) -> str:
        """Simple rule-based fallback reasoning in case SHAP is unavailable."""
        amount_fmt = f"Rp {float(tx.get('amount', 0)):,.0f}".replace(",", ".")
        
        # Simple analysis text
        triggered = []
        for k in ["is_velocity_anomaly", "is_geo_mismatch", "is_off_hours", "is_high_value_for_rail", "is_suspicious_ip", "is_risky_merchant", "is_new_device", "is_device_mismatch", "is_sim_swap", "is_unusual_beneficiary"]:
            val = tx.get(k)
            # Handle boolean or int representation
            if val is True or val == 1 or val == "1":
                triggered.append(FEATURE_LABELS.get(k, k))

        if triggered:
            analysis = f"Sistem mengidentifikasi beberapa faktor anomali behavioral terpicu pada transaksi ini: **{', '.join(triggered)}**."
        else:
            analysis = "Tidak terdeteksi indikator risiko khusus yang terpicu secara manual."

        if severity == "critical":
            opening = f"**PERINGATAN KRITIS**: Transaksi berisiko ekstrem (Skor: **{risk_score}/100**)."
            action = f"**Rekomendasi Tindakan**: **{recommendation}** untuk investigasi mendalam."
        elif severity == "high":
            opening = f"**ANALISIS RISIKO TINGGI**: Deteksi deviasi pola transaksi (Skor: **{risk_score}/100**)."
            action = f"**Rekomendasi Tindakan**: **{recommendation}**."
        elif severity == "medium":
            opening = f"**MONITORING RISIKO SEDANG**: Terdeteksi anomali minor (Skor: **{risk_score}/100**)."
            action = f"**Rekomendasi Tindakan**: **{recommendation}**."
        else:
            opening = f"**TRANSAKSI NORMAL**: Transaksi dinilai aman (Skor: **{risk_score}/100**)."
            action = f"**Rekomendasi Tindakan**: **{recommendation}**."

        context = f"Dana **{amount_fmt}** dikirim oleh **{tx.get('sender_name', 'N/A')}** ke **{tx.get('receiver_name', 'N/A')}** via **{tx.get('payment_rail', 'N/A')}**."

        return f"{opening}\n\n{context}\n\n{analysis}\n\n{action}"
