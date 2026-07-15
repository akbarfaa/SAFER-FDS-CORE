"""
SAFER Graph Service — FastAPI Application

Handles fraud graph scenarios and NetworkX relational graph layouts.
Runs on port 8002.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Any, List

from shared.config import CORS_ORIGINS, GRAPH_PORT
from graph_service.graph_intel import analyze_transaction_network

app = FastAPI(
    title="SAFER Graph Intelligence Service",
    description="NetworkX-based community detection, PageRank scoring, and layouts.",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeGraphRequest(BaseModel):
    transactions: List[dict]
# Predefined static scenarios matching the frontend definitions (as a baseline)
SCENARIOS_DATA = [
    {
        "id": "mule-ring",
        "name": "Sindikat Rekening Bagong (Mule Ring)",
        "description": "Aliran dana melingkar di mana beberapa rekening penampung mentransfer uang ke satu akun induk utama yang diakses dari satu perangkat fisik smartphone yang sama.",
        "severity": "critical",
        "nodes": [
            { "id": "a1", "label": "ACC-7821 (Utama)", "type": "account", "risk": "critical", "x": 380, "y": 220, "details": { "Nama Pemilik": "Hendra Wijaya", "Bank": "BCA", "No. Rekening": "022839120", "Lokasi": "Jakarta Pusat", "Status": "Dalam Pemantauan" } },
            { "id": "a2", "label": "ACC-3219 (Mule)", "type": "account", "risk": "high", "x": 200, "y": 130, "details": { "Nama Pemilik": "Andi Susanto", "Bank": "BNI", "No. Rekening": "018274932", "Lokasi": "Depok", "Status": "Mencurigakan" } },
            { "id": "a3", "label": "ACC-9912 (Mule)", "type": "account", "risk": "high", "x": 200, "y": 310, "details": { "Nama Pemilik": "Siti Nurul", "Bank": "BRI", "No. Rekening": "074291843", "Lokasi": "Bekasi", "Status": "Mencurigakan" } },
            { "id": "a4", "label": "ACC-1188 (Mule)", "type": "account", "risk": "medium", "x": 560, "y": 130, "details": { "Nama Pemilik": "Rizki Fajar", "Bank": "Mandiri", "No. Rekening": "100293848", "Lokasi": "Tangerang", "Status": "Tinjauan Ulang" } },
            { "id": "a5", "label": "ACC-4451 (Mule)", "type": "account", "risk": "medium", "x": 560, "y": 310, "details": { "Nama Pemilik": "Dewi Lestari", "Bank": "CIMB Niaga", "No. Rekening": "053918239", "Lokasi": "Bogor", "Status": "Tinjauan Ulang" } },
            { "id": "a6", "label": "ACC-2207 (Target)", "type": "account", "risk": "low", "x": 740, "y": 220, "details": { "Nama Pemilik": "Budi Hartono", "Bank": "Bank Jago", "No. Rekening": "502938192", "Lokasi": "Jakarta Selatan", "Status": "Aktif / Normal" } },
            { "id": "d1", "label": "DEV-AX91 (Shared)", "type": "device", "risk": "critical", "x": 100, "y": 220, "details": { "Tipe Perangkat": "Xiaomi Redmi 12", "Fingerprint": "ax918f0a2e7c3b1", "Koneksi": "Telkomsel LTE", "Lokasi Terakhir": "Depok" } },
            { "id": "d2", "label": "DEV-KL44", "type": "device", "risk": "high", "x": 380, "y": 70, "details": { "Tipe Perangkat": "Samsung Galaxy S22", "Fingerprint": "kl447b9d1e2a8f0", "Koneksi": "Biznet Home", "Lokasi Terakhir": "Jakarta Pusat" } },
            { "id": "m1", "label": "QRIS-Toko-Berkah", "type": "merchant", "risk": "high", "x": 380, "y": 370, "details": { "Nama Merchant": "Toko Berkah Kelontong", "Kategori": "Retail / QRIS", "ID Merchant": "MID992810", "Kota": "Jakarta Pusat" } },
            { "id": "m2", "label": "QRIS-Sumber-Jaya", "type": "merchant", "risk": "medium", "x": 700, "y": 370, "details": { "Nama Merchant": "Sumber Jaya Elektronik", "Kategori": "E-Commerce / QRIS", "ID Merchant": "MID881920", "Kota": "Bandung" } },
            { "id": "i1", "label": "IP 103.84.x.x", "type": "ip", "risk": "critical", "x": 100, "y": 70, "details": { "Alamat IP": "103.84.221.18", "ISP": "Indosat Ooredoo", "Tipe": "Seluler / Dynamic", "Reputasi IP": "Buruk / Masuk Blacklist" } }
        ],
        "edges": [
            { "from": "a2", "to": "a1", "label": "Rp 24M" },
            { "from": "a3", "to": "a1", "label": "Rp 18M" },
            { "from": "a1", "to": "a4", "label": "Rp 30M" },
            { "from": "a1", "to": "a5", "label": "Rp 22M" },
            { "from": "a5", "to": "a6", "label": "Transfer" },
            { "from": "d1", "to": "a2" },
            { "from": "d1", "to": "a3" },
            { "from": "d2", "to": "a1" },
            { "from": "d2", "to": "a4" },
            { "from": "i1", "to": "d1" },
            { "from": "a1", "to": "m1" },
            { "from": "a4", "to": "m1" },
            { "from": "a5", "to": "m2" }
        ],
        "insights": [
            "Indikasi Device Sharing: Rekening penampung ACC-3219 dan ACC-9912 diakses dari HP yang sama (DEV-AX91) secara bergantian dalam hitungan menit.",
            "Pola Fan-In / Fan-Out: Dana dari beberapa rekening kecil ditarik secara massal ke rekening utama ACC-7821 sebelum dialirkan ke merchant QRIS retail dalam jumlah besar.",
            "IP Address Berisiko Tinggi: Koneksi device menggunakan IP yang masuk dalam daftar pantauan sindikat pencucian uang OJK."
        ]
    },
    {
        "id": "device-sharing",
        "name": "Pertukaran Device Farm (Device Sharing)",
        "description": "Satu perangkat fisik digunakan untuk login ke 5 akun nasabah yang berbeda secara berurutan dalam kurun waktu 2 jam — indikasi khas operasi robot atau sindikat pinjaman online fiktif.",
        "severity": "high",
        "nodes": [
            { "id": "a1", "label": "ACC-9011", "type": "account", "risk": "high", "x": 280, "y": 150, "details": { "Nama Pemilik": "Dimas Saputra", "Bank": "BCA", "No. Rekening": "089123019", "Lokasi": "Surabaya", "Status": "Tinjauan Risiko" } },
            { "id": "a2", "label": "ACC-5044", "type": "account", "risk": "high", "x": 200, "y": 280, "details": { "Nama Pemilik": "Maya Permata", "Bank": "BNI", "No. Rekening": "022910394", "Lokasi": "Surabaya", "Status": "Tinjauan Risiko" } },
            { "id": "a3", "label": "ACC-8812", "type": "account", "risk": "high", "x": 380, "y": 330, "details": { "Nama Pemilik": "Aldo Wijaya", "Bank": "BRI", "No. Rekening": "073910382", "Lokasi": "Sidoarjo", "Status": "Tinjauan Risiko" } },
            { "id": "a4", "label": "ACC-2731", "type": "account", "risk": "medium", "x": 440, "y": 150, "details": { "Nama Pemilik": "Agus Santoso", "Bank": "Mandiri", "No. Rekening": "100293123", "Lokasi": "Gresik", "Status": "Normal" } },
            { "id": "a5", "label": "ACC-1049", "type": "account", "risk": "medium", "x": 540, "y": 280, "details": { "Nama Pemilik": "Lina Puspita", "Bank": "Bank Jago", "No. Rekening": "502910398", "Lokasi": "Surabaya", "Status": "Normal" } },
            { "id": "d1", "label": "DEV-OPPO-A5", "type": "device", "risk": "critical", "x": 340, "y": 230, "details": { "Tipe Perangkat": "Oppo A5 2020", "Fingerprint": "opp5a20208c9d1", "Koneksi": "Tri Cellular", "Lokasi Terakhir": "Surabaya" } },
            { "id": "i1", "label": "IP 114.122.x.x", "type": "ip", "risk": "high", "x": 100, "y": 220, "details": { "Alamat IP": "114.122.91.50", "ISP": "Hutchison 3", "Lokasi": "Surabaya", "Tipe": "Seluler" } },
            { "id": "i2", "label": "IP 180.244.x.x", "type": "ip", "risk": "medium", "x": 620, "y": 220, "details": { "Alamat IP": "180.244.18.23", "ISP": "Telkomsel", "Lokasi": "Surabaya", "Tipe": "Seluler" } }
        ],
        "edges": [
            { "from": "d1", "to": "a1", "label": "Login 11:02" },
            { "from": "d1", "to": "a2", "label": "Login 11:15" },
            { "from": "d1", "to": "a3", "label": "Login 11:32" },
            { "from": "d1", "to": "a4", "label": "Login 11:48" },
            { "from": "d1", "to": "a5", "label": "Login 12:10" },
            { "from": "i1", "to": "d1" },
            { "from": "i2", "to": "d1" }
        ],
        "insights": [
            "Deteksi Device Farm: Satu perangkat fisik (DEV-OPPO-A5) digunakan untuk mengakses 5 akun bank berbeda dalam kurun waktu kurang dari 2 jam.",
            "Konektivitas IP Seluler: Perangkat berpindah-pindah koneksi (Tri dan Telkomsel) demi menghindari deteksi firewall berbasis alamat IP statis.",
            "Pola Pendaftaran Sindikat: Seluruh akun dibuat dalam kurun waktu 14 hari yang sama, mengindikasikan serangan registrasi identitas palsu terkoordinasi."
        ]
    },
    {
        "id": "slot-cashout",
        "name": "Aliran Judi Online (Slot Cashout Ring)",
        "description": "Kasus pengumpulan dana kilat (fan-in) dari berbagai akun individu dengan transaksi berfrekuensi tinggi, yang langsung dilarikan ke merchant terafiliasi perjudian dan bursa aset kripto.",
        "severity": "critical",
        "nodes": [
            { "id": "a1", "label": "ACC-1100", "type": "account", "risk": "low", "x": 150, "y": 100, "details": { "Nama Pemilik": "Dimas Saputra", "Bank": "BCA", "No. Rekening": "022938190", "Lokasi": "Jakarta", "Status": "Normal" } },
            { "id": "a2", "label": "ACC-1101", "type": "account", "risk": "low", "x": 120, "y": 200, "details": { "Nama Pemilik": "Wulan Dari", "Bank": "BNI", "No. Rekening": "019283912", "Lokasi": "Bandung", "Status": "Normal" } },
            { "id": "a3", "label": "ACC-1102", "type": "account", "risk": "medium", "x": 150, "y": 300, "details": { "Nama Pemilik": "Bayu Wijaya", "Bank": "Mandiri", "No. Rekening": "100293129", "Lokasi": "Semarang", "Status": "Dalam Tinjauan" } },
            { "id": "a4", "label": "ACC-1103", "type": "account", "risk": "low", "x": 560, "y": 100, "details": { "Nama Pemilik": "Sari Indah", "Bank": "CIMB Niaga", "No. Rekening": "053928192", "Lokasi": "Surabaya", "Status": "Normal" } },
            { "id": "a5", "label": "ACC-1104", "type": "account", "risk": "medium", "x": 590, "y": 200, "details": { "Nama Pemilik": "Faisal Reza", "Bank": "Bank Jago", "No. Rekening": "502938198", "Lokasi": "Malang", "Status": "Dalam Tinjauan" } },
            { "id": "a6", "label": "ACC-1105", "type": "account", "risk": "low", "x": 560, "y": 300, "details": { "Nama Pemilik": "Mega Utami", "Bank": "Nobu", "No. Rekening": "060293812", "Lokasi": "Medan", "Status": "Normal" } },
            { "id": "a7", "label": "ACC-9922 (Penampung)", "type": "account", "risk": "critical", "x": 340, "y": 200, "details": { "Nama Pemilik": "OnlineBet Collection", "Bank": "BCA", "No. Rekening": "022938111", "Lokasi": "Medan", "Status": "Mencurigakan / High Risk" } },
            { "id": "m1", "label": "OnlineBet88", "type": "merchant", "risk": "critical", "x": 340, "y": 70, "details": { "Nama Merchant": "OnlineBet88 Global", "Kategori": "Perjudian / Risky", "ID Merchant": "MID441029", "Lisensi": "None / Tidak Terdaftar" } },
            { "id": "m2", "label": "CryptoXchange ID", "type": "merchant", "risk": "critical", "x": 340, "y": 330, "details": { "Nama Merchant": "CryptoXchange Indonesia", "Kategori": "Virtual Asset / Crypto", "ID Merchant": "MID119280", "Lokasi": "Jakarta Pusat" } }
        ],
        "edges": [
            { "from": "a1", "to": "a7", "label": "Rp 5.2M" },
            { "from": "a2", "to": "a7", "label": "Rp 8.1M" },
            { "from": "a3", "to": "a7", "label": "Rp 12.0M" },
            { "from": "a4", "to": "a7", "label": "Rp 4.5M" },
            { "from": "a5", "to": "a7", "label": "Rp 15.2M" },
            { "from": "a6", "to": "a7", "label": "Rp 6.0M" },
            { "from": "a7", "to": "m1", "label": "Rp 22.0M" },
            { "from": "a7", "to": "m2", "label": "Rp 29.0M" }
        ],
        "insights": [
            "Pengumpulan Dana Cepat (Fan-In): Puluhan transaksi kecil dari rekening pribadi mengalir deras ke rekening penampung ACC-9922 dalam interval waktu sangat rapat.",
            "Pelarian Dana (Fan-Out): Rekening penampung langsung mengosongkan saldo dengan mentransfer dana keluar ke merchant terafiliasi situs judi online dan bursa kripto luar negeri.",
            "Karakteristik Off-Hours: Transaksi didominasi terjadi pada jam tengah malam antara pukul 01:00 hingga 04:30 WIB."
        ]
    }
]


@app.get("/scenarios")
def get_scenarios():
    """Get predefined scenarios."""
    return SCENARIOS_DATA


@app.post("/analyze")
def analyze_network(req: AnalyzeGraphRequest):
    """
    Constructs a relation graph using NetworkX and runs topological analysis
    to dynamically generate a layout.
    """
    try:
        raw_txs = req.transactions
        analysis_result = analyze_transaction_network(raw_txs)
        return analysis_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"service": "graph", "status": "ok", "port": GRAPH_PORT}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("graph_service.main:app", host="0.0.0.0", port=GRAPH_PORT, reload=True)
