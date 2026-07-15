"""
SAFER FDS — Enhanced Dataset Generator v2

Generates 100,000 synthetic Indonesian financial transactions with 8 structured
fraud patterns for training the XGBoost + LightGBM ensemble models.

Fraud Patterns Covered:
  1. Mule Ring — Jaringan rekening penampung money laundering
  2. Device Farm — Satu perangkat mengoperasikan banyak akun
  3. Account Takeover — Pengambilalihan akun (new device + SIM swap + big transfer)
  4. Impossible Travel — Transaksi dari 2 lokasi berjauhan dalam waktu singkat
  5. Smurfing/Structuring — Pemecahan transaksi besar menjadi banyak kecil
  6. Judi Online Laundering — Deposit/withdraw cepat ke gambling merchant
  7. Deepfake Social Engineering — New account + SIM swap + large transfer to new beneficiary
  8. Slot Cashout Ring — Merchant palsu menerima pembayaran berulang dari banyak akun

Changes from v1:
  - 100K total (80K train, 20K test) vs 60K (50K/10K)
  - Fraud rate ~10% (more realistic vs 15%)
  - 4 new fraud scenarios added
  - Better fraud indicator correlations
"""

import os
import csv
import random
import uuid
from datetime import datetime, timedelta
import math

# --- Lookup Tables (Matching SAFER FDS UI/System Specifications) ---

FIRST_NAMES = [
    "Andi", "Sari", "Budi", "Maya", "Rizki", "Dewi", "Faisal", "Putri",
    "Aldo", "Indah", "Agus", "Rina", "Dimas", "Lina", "Hendra", "Nita",
    "Yusuf", "Wulan", "Arif", "Ratna", "Bayu", "Siti", "Reza", "Ayu",
    "Taufik", "Mega", "Irfan", "Dian", "Kurnia", "Fitri", "Eko", "Nurul",
    "Wahyu", "Rini", "Surya", "Intan", "Fajar", "Lestari", "Gilang", "Amelia"
]

LAST_NAMES = [
    "Prasetyo", "Wulandari", "Hartono", "Kusuma", "Hidayat", "Permata",
    "Ramadhan", "Nugraheni", "Santoso", "Wijaya", "Susanto", "Purnama",
    "Suryadi", "Laksmi", "Setiawan", "Maharani", "Nugroho", "Anggraini",
    "Saputra", "Handayani", "Utomo", "Rahayu", "Firmansyah", "Puspita",
    "Kurniawan", "Safitri", "Wahyudi", "Hapsari", "Pratama", "Damayanti"
]

MERCHANTS_RETAIL = [
    "Alfamart Jl. Sudirman", "Indomaret Kebayoran", "Circle K Senopati",
    "Warung Makan Sederhana", "Bakso Pak Kumis", "Kopi Kenangan Sudirman",
    "Starbucks Plaza Indonesia", "McDonald's Sarinah", "KFC Thamrin",
    "J.CO Donuts Grand Indonesia"
]

MERCHANTS_ECOMMERCE = [
    "Tokopedia", "Shopee", "Bukalapak", "Lazada", "Blibli",
    "JD.ID", "Zalora", "Sociolla", "Bhinneka", "Orami"
]

MERCHANTS_SERVICES = [
    "Grab", "GoFood", "Gojek", "Traveloka", "Tiket.com",
    "PLN Prepaid", "Telkomsel Pulsa", "BPJS Kesehatan", "PGN Gas",
    "Indosat Prepaid"
]

MERCHANTS_RISKY = [
    "CryptoXchange ID", "BitTrade Asia", "OnlineBet88", "LuckySlot ID",
    "FastCash Pinjol", "QuickLoan Digital", "JudiNet99", "SlotMania ID",
    "CashBoss Lending", "BetKing Indo"
]

MERCHANTS_CORPORATE = [
    "PT Sumber Makmur Sentosa", "CV Jaya Abadi", "PT Nusantara Logistik",
    "Apotek Kimia Farma", "RS Pondok Indah", "Universitas Indonesia",
    "PT Pertamina (Persero)", "PT Telkom Indonesia", "PT Bank Central Asia",
    "PT Astra International"
]

# Fake/shell merchants for slot cashout ring
MERCHANTS_SHELL = [
    "Toko Maju Jaya Digital", "CV Berkah Sentosa", "UD Mulia Elektronik",
    "PT Karya Digital Utama", "Toko Online Nusantara", "CV Cemerlang Makmur"
]

BANKS = [
    {"code": "BCA", "name": "Bank Central Asia", "prefix": "0"},
    {"code": "BNI", "name": "Bank Negara Indonesia", "prefix": "0"},
    {"code": "BRI", "name": "Bank Rakyat Indonesia", "prefix": "0"},
    {"code": "Mandiri", "name": "Bank Mandiri", "prefix": "1"},
    {"code": "BSI", "name": "Bank Syariah Indonesia", "prefix": "7"},
    {"code": "CIMB", "name": "CIMB Niaga", "prefix": "0"},
    {"code": "Danamon", "name": "Bank Danamon", "prefix": "0"},
    {"code": "Permata", "name": "PermataBank", "prefix": "0"},
    {"code": "BNP", "name": "Bank BTPN", "prefix": "9"},
    {"code": "Mega", "name": "Bank Mega", "prefix": "0"},
    {"code": "OCBC", "name": "OCBC NISP", "prefix": "0"},
    {"code": "Jago", "name": "Bank Jago", "prefix": "5"},
    {"code": "Nobu", "name": "Bank Nobu", "prefix": "0"},
    {"code": "Seabank", "name": "SeaBank", "prefix": "9"}
]

CITIES = [
    {"name": "Jakarta Pusat", "province": "DKI Jakarta", "weight": 14, "lat": -6.186, "lng": 106.834},
    {"name": "Jakarta Selatan", "province": "DKI Jakarta", "weight": 12, "lat": -6.261, "lng": 106.810},
    {"name": "Jakarta Barat", "province": "DKI Jakarta", "weight": 8, "lat": -6.168, "lng": 106.758},
    {"name": "Jakarta Utara", "province": "DKI Jakarta", "weight": 5, "lat": -6.121, "lng": 106.837},
    {"name": "Jakarta Timur", "province": "DKI Jakarta", "weight": 6, "lat": -6.225, "lng": 106.900},
    {"name": "Surabaya", "province": "Jawa Timur", "weight": 10, "lat": -7.250, "lng": 112.751},
    {"name": "Bandung", "province": "Jawa Barat", "weight": 7, "lat": -6.917, "lng": 107.619},
    {"name": "Medan", "province": "Sumatera Utara", "weight": 6, "lat": 3.595, "lng": 98.672},
    {"name": "Semarang", "province": "Jawa Tengah", "weight": 5, "lat": -6.966, "lng": 110.420},
    {"name": "Makassar", "province": "Sulawesi Selatan", "weight": 4, "lat": -5.147, "lng": 119.432},
    {"name": "Denpasar", "province": "Bali", "weight": 4, "lat": -8.650, "lng": 115.220},
    {"name": "Tangerang", "province": "Banten", "weight": 4, "lat": -6.178, "lng": 106.630},
    {"name": "Bekasi", "province": "Jawa Barat", "weight": 4, "lat": -6.241, "lng": 106.992},
    {"name": "Depok", "province": "Jawa Barat", "weight": 3, "lat": -6.402, "lng": 106.794},
    {"name": "Bogor", "province": "Jawa Barat", "weight": 3, "lat": -6.597, "lng": 106.806},
    {"name": "Yogyakarta", "province": "DI Yogyakarta", "weight": 3, "lat": -7.797, "lng": 110.370},
    {"name": "Malang", "province": "Jawa Timur", "weight": 2, "lat": -7.978, "lng": 112.630},
    {"name": "Palembang", "province": "Sumatera Selatan", "weight": 2, "lat": -2.976, "lng": 104.775},
    {"name": "Balikpapan", "province": "Kalimantan Timur", "weight": 2, "lat": -1.267, "lng": 116.829},
    {"name": "Manado", "province": "Sulawesi Utara", "weight": 1, "lat": 1.474, "lng": 124.842}
]

PAYMENT_RAILS = [
    "QRIS", "BI-FAST", "RTGS", "SKN", "E-Wallet",
    "Virtual Account", "Kartu Debit", "Kartu Kredit", "Transfer"
]

RAIL_WEIGHTS = {
    "QRIS": 28, "BI-FAST": 18, "RTGS": 3, "SKN": 5, "E-Wallet": 24,
    "Virtual Account": 8, "Kartu Debit": 6, "Kartu Kredit": 4, "Transfer": 4
}

EWALLET_PROVIDERS = ["GoPay", "OVO", "DANA", "ShopeePay", "LinkAja"]

DEVICE_TYPES = ["Android", "iOS", "Web Browser", "Mobile Web"]
DEVICE_BRANDS_ANDROID = ["Samsung", "Xiaomi", "OPPO", "Vivo", "Realme", "Infinix", "POCO"]
DEVICE_BRANDS_IOS = ["iPhone 13", "iPhone 14", "iPhone 15", "iPhone 12", "iPhone SE"]
DEVICE_BRANDS_WEB = ["Chrome 125", "Safari 18", "Firefox 127", "Edge 125"]

HIGH_VALUE_THRESHOLDS = {
    "QRIS": 2000000,
    "BI-FAST": 50000000,
    "RTGS": 500000000,
    "SKN": 50000000,
    "E-Wallet": 1000000,
    "Virtual Account": 10000000,
    "Kartu Debit": 5000000,
    "Kartu Kredit": 15000000,
    "Transfer": 25000000
}

# --- Helper functions ---

def get_weighted_choice(choices_dict):
    total = sum(choices_dict.values())
    r = random.uniform(0, total)
    upto = 0
    for key, weight in choices_dict.items():
        if upto + weight >= r:
            return key
        upto += weight
    return list(choices_dict.keys())[-1]

def get_weighted_city():
    weights = [c["weight"] for c in CITIES]
    return random.choices(CITIES, weights=weights, k=1)[0]

def generate_account_number(bank):
    length = 13 if bank["code"] == "Mandiri" else 10
    num = bank["prefix"]
    for _ in range(length - len(num)):
        num += str(random.randint(0, 9))
    return num

def generate_device_fingerprint():
    chars = "0123456789abcdef"
    return "".join(random.choice(chars) for _ in range(16))

def generate_ip():
    prefixes = ["103.84", "114.142", "36.68", "180.244", "110.136", "103.28", "202.134", "112.215"]
    pfx = random.choice(prefixes)
    return f"{pfx}.{random.randint(1, 254)}.{random.randint(1, 254)}"

def generate_suspicious_ip():
    """Generate IPs from known VPN/hosting ranges."""
    prefixes = ["45.76", "104.238", "185.199", "198.51", "203.0", "91.108"]
    pfx = random.choice(prefixes)
    return f"{pfx}.{random.randint(1, 254)}.{random.randint(1, 254)}"

def generate_device_info():
    device_type = random.choice(DEVICE_TYPES)
    if device_type == "iOS":
        brand = random.choice(DEVICE_BRANDS_IOS)
    elif device_type == "Android":
        brand = random.choice(DEVICE_BRANDS_ANDROID)
    else:
        brand = random.choice(DEVICE_BRANDS_WEB)
    return device_type, brand, generate_device_fingerprint()

def generate_amount(rail, is_fraud=False, is_high_val=False):
    if is_high_val or (is_fraud and random.random() < 0.6):
        thresh = HIGH_VALUE_THRESHOLDS.get(rail, 10000000)
        return int(thresh * random.uniform(1.2, 5.0))
    
    if rail == "QRIS":
        return random.randint(5000, 500000) if random.random() < 0.7 else random.randint(500000, 5000000)
    elif rail == "BI-FAST":
        return random.randint(100000, 10000000) if random.random() < 0.6 else random.randint(10000000, 250000000)
    elif rail == "RTGS":
        return random.randint(100000000, 2000000000)
    elif rail == "SKN":
        return random.randint(1000000, 100000000)
    elif rail == "E-Wallet":
        return random.randint(1000, 500000) if random.random() < 0.8 else random.randint(500000, 2000000)
    elif rail == "Virtual Account":
        return random.randint(50000, 50000000)
    elif rail == "Kartu Debit":
        return random.randint(10000, 10000000)
    elif rail == "Kartu Kredit":
        return random.randint(50000, 50000000)
    else:
        return random.randint(50000, 10000000)

def generate_merchant(rail, is_fraud=False):
    if is_fraud and random.random() < 0.7:
        return random.choice(MERCHANTS_RISKY)
    
    if rail == "QRIS":
        return random.choice(MERCHANTS_RETAIL) if random.random() < 0.7 else random.choice(MERCHANTS_SERVICES)
    elif rail == "E-Wallet":
        return random.choice(MERCHANTS_SERVICES) if random.random() < 0.5 else random.choice(MERCHANTS_ECOMMERCE)
    elif rail in ["RTGS", "SKN"]:
        return random.choice(MERCHANTS_CORPORATE)
    elif rail == "Kartu Kredit":
        return random.choice(MERCHANTS_ECOMMERCE) if random.random() < 0.6 else random.choice(MERCHANTS_RETAIL)
    else:
        return random.choice(MERCHANTS_RETAIL + MERCHANTS_ECOMMERCE + MERCHANTS_SERVICES)

def get_merchant_category(merchant):
    if merchant in MERCHANTS_RETAIL:
        return "Retail"
    if merchant in MERCHANTS_ECOMMERCE:
        return "E-Commerce"
    if merchant in MERCHANTS_SERVICES:
        return "Services"
    if merchant in MERCHANTS_CORPORATE:
        return "Corporate"
    if merchant in MERCHANTS_SHELL:
        return "Retail"  # Shell merchants disguise as retail
    if merchant in MERCHANTS_RISKY:
        if "Crypto" in merchant or "Bit" in merchant:
            return "Crypto"
        if "Bet" in merchant or "Slot" in merchant or "Judi" in merchant:
            return "Gambling"
        return "Lending"
    return "General"

def get_geo_distance(lat1, lng1, lat2, lng2):
    dlat = lat1 - lat2
    dlng = lng1 - lng2
    return round(math.sqrt(dlat**2 + dlng**2) * 111, 2)

# --- Generator for a Single Transaction ---

def generate_single_transaction(is_fraud=False, overrides=None):
    if overrides is None:
        overrides = {}
        
    rail = overrides.get("rail", get_weighted_choice(RAIL_WEIGHTS))
    
    sender_city = overrides.get("sender_city", get_weighted_city())
    if isinstance(sender_city, str):
        sender_city = next((c for c in CITIES if c["name"] == sender_city), CITIES[0])
        
    receiver_city = overrides.get("receiver_city", get_weighted_city())
    if isinstance(receiver_city, str):
        receiver_city = next((c for c in CITIES if c["name"] == receiver_city), CITIES[0])
        
    sender_bank = overrides.get("sender_bank", random.choice(BANKS))
    if isinstance(sender_bank, str):
        sender_bank = next((b for b in BANKS if b["code"] == sender_bank), BANKS[0])
        
    receiver_bank = overrides.get("receiver_bank", random.choice(BANKS))
    if isinstance(receiver_bank, str):
        receiver_bank = next((b for b in BANKS if b["code"] == receiver_bank), BANKS[0])
    
    device_type, device_brand, device_fp = generate_device_info()
    device_type = overrides.get("device_type", device_type)
    device_brand = overrides.get("device_brand", device_brand)
    device_fp = overrides.get("device_fingerprint", device_fp)
    
    timestamp = overrides.get("timestamp", datetime.now() - timedelta(days=random.randint(0, 30), hours=random.randint(0, 23), minutes=random.randint(0, 59)))
    
    is_high_val = overrides.get("is_high_value_for_rail", False)
    amount = overrides.get("amount", generate_amount(rail, is_fraud, is_high_val))
    
    merchant = overrides.get("merchant", generate_merchant(rail, is_fraud))
    merchant_category = get_merchant_category(merchant)
    
    account_age_days = overrides.get("account_age_days", random.randint(30, 3650))
    if is_fraud and random.random() < 0.4 and "account_age_days" not in overrides:
        account_age_days = random.randint(1, 25)
        
    is_off_hours = overrides.get("is_off_hours", timestamp.hour < 5 or timestamp.hour > 23)
    is_new_device = overrides.get("is_new_device", True if (is_fraud and random.random() < 0.6) else (random.random() < 0.05))
    is_device_mismatch = overrides.get("is_device_mismatch", True if (is_fraud and random.random() < 0.5) else (random.random() < 0.02))
    is_suspicious_ip = overrides.get("is_suspicious_ip", True if (is_fraud and random.random() < 0.4) else (random.random() < 0.01))
    has_failed_attempts = overrides.get("has_failed_attempts", True if (is_fraud and random.random() < 0.5) else (random.random() < 0.02))
    is_sim_swap = overrides.get("is_sim_swap", True if (is_fraud and random.random() < 0.3) else (random.random() < 0.01))
    is_unusual_beneficiary = overrides.get("is_unusual_beneficiary", True if (is_fraud and random.random() < 0.7) else (random.random() < 0.08))
    
    velocity_count = overrides.get("velocity_count", random.randint(8, 25) if (is_fraud and random.random() < 0.6) else random.randint(1, 3))
    is_velocity_anomaly = overrides.get("is_velocity_anomaly", velocity_count >= 8)
    
    geo_distance = get_geo_distance(sender_city["lat"], sender_city["lng"], receiver_city["lat"], receiver_city["lng"])
    is_geo_mismatch = overrides.get("is_geo_mismatch", geo_distance > 300 if (is_fraud and random.random() < 0.5) else False)
    if is_geo_mismatch and geo_distance < 300:
        geo_distance = random.randint(400, 2500)
        
    thresh = HIGH_VALUE_THRESHOLDS.get(rail, 10000000)
    is_high_value_for_rail = is_high_val or (amount > thresh)
    
    is_new_account = overrides.get("is_new_account", account_age_days < 30)
    is_risky_merchant = merchant_category in ["Crypto", "Gambling", "Lending"]
    
    ip_address = overrides.get("ip_address", generate_suspicious_ip() if (is_fraud and is_suspicious_ip) else generate_ip())
    
    tx = {
        "id": f"TX-{random.randint(100000, 999999)}",
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "sender_name": overrides.get("sender_name", f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"),
        "sender_account": overrides.get("sender_account", generate_account_number(sender_bank)),
        "sender_bank": sender_bank["code"],
        "sender_city": sender_city["name"],
        "sender_province": sender_city["province"],
        "sender_lat": sender_city["lat"],
        "sender_lng": sender_city["lng"],
        "receiver_name": overrides.get("receiver_name", f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"),
        "receiver_account": overrides.get("receiver_account", generate_account_number(receiver_bank)),
        "receiver_bank": receiver_bank["code"],
        "receiver_city": receiver_city["name"],
        "receiver_province": receiver_city["province"],
        "receiver_lat": receiver_city["lat"],
        "receiver_lng": receiver_city["lng"],
        "amount": amount,
        "payment_rail": rail,
        "ewallet_provider": random.choice(EWALLET_PROVIDERS) if rail == "E-Wallet" else "None",
        "merchant": merchant,
        "merchant_category": merchant_category,
        "channel": "Mobile App" if rail == "E-Wallet" else "QR Scan" if rail == "QRIS" else random.choice(["Mobile Banking", "Internet Banking", "ATM"]),
        "device_type": device_type,
        "device_brand": device_brand,
        "device_fingerprint": device_fp,
        "ip_address": ip_address,
        "is_new_device": int(is_new_device),
        "account_age_days": account_age_days,
        "is_velocity_anomaly": int(is_velocity_anomaly),
        "is_geo_mismatch": int(is_geo_mismatch),
        "is_off_hours": int(is_off_hours),
        "is_high_value_for_rail": int(is_high_value_for_rail),
        "is_suspicious_ip": int(is_suspicious_ip),
        "is_risky_merchant": int(is_risky_merchant),
        "is_new_account": int(is_new_account),
        "has_failed_attempts": int(has_failed_attempts),
        "is_device_mismatch": int(is_device_mismatch),
        "is_sim_swap": int(is_sim_swap),
        "is_unusual_beneficiary": int(is_unusual_beneficiary),
        "velocity_count": velocity_count,
        "geo_distance_km": geo_distance,
        "is_fraud": int(is_fraud)
    }
    
    return tx

# ============================================================================
# FRAUD SCENARIO GENERATORS (8 Patterns)
# ============================================================================

def generate_mule_ring_scenarios(count=80):
    """Pattern 1: Mule Ring — Multiple senders -> 1 collector -> cashout to risky merchant."""
    transactions = []
    for _ in range(count):
        collector_bank = random.choice(BANKS)
        collector_acc = generate_account_number(collector_bank)
        collector_name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
        collector_city = get_weighted_city()
        
        num_mules = random.randint(4, 8)
        base_time = datetime.now() - timedelta(days=random.randint(0, 30))
        
        for i in range(num_mules):
            tx_time = base_time + timedelta(minutes=i * random.randint(1, 10))
            tx = generate_single_transaction(is_fraud=True, overrides={
                "receiver_account": collector_acc, "receiver_bank": collector_bank,
                "receiver_name": collector_name, "receiver_city": collector_city,
                "timestamp": tx_time, "amount": random.randint(2000000, 10000000),
                "rail": "BI-FAST", "is_unusual_beneficiary": True,
                "velocity_count": random.randint(5, 12),
            })
            transactions.append(tx)
            
        cashout_time = base_time + timedelta(minutes=num_mules * 10 + random.randint(5, 15))
        cashout_tx = generate_single_transaction(is_fraud=True, overrides={
            "sender_account": collector_acc, "sender_bank": collector_bank,
            "sender_name": collector_name, "sender_city": collector_city,
            "merchant": random.choice(MERCHANTS_RISKY),
            "amount": random.randint(15000000, 60000000), "rail": "Transfer",
            "timestamp": cashout_time, "is_high_value_for_rail": True,
            "velocity_count": random.randint(10, 20),
        })
        transactions.append(cashout_tx)
    return transactions

def generate_device_farm_scenarios(count=60):
    """Pattern 2: Device Farm — 1 device fingerprint, many different accounts."""
    transactions = []
    for _ in range(count):
        dev_type, dev_brand, shared_fp = generate_device_info()
        num_accounts = random.randint(8, 15)
        base_time = datetime.now() - timedelta(days=random.randint(0, 30))
        for i in range(num_accounts):
            tx_time = base_time + timedelta(minutes=i * random.randint(1, 5))
            tx = generate_single_transaction(is_fraud=True, overrides={
                "device_fingerprint": shared_fp, "device_type": dev_type,
                "device_brand": dev_brand, "timestamp": tx_time,
                "is_device_mismatch": True, "is_new_device": True,
                "has_failed_attempts": random.choice([True, False]),
                "velocity_count": random.randint(8, 18),
            })
            transactions.append(tx)
    return transactions

def generate_account_takeover_scenarios(count=100):
    """Pattern 3: Account Takeover — Legitimate account + new device + different geo + SIM swap."""
    transactions = []
    for _ in range(count):
        victim_bank = random.choice(BANKS)
        victim_acc = generate_account_number(victim_bank)
        victim_name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
        victim_city = get_weighted_city()
        
        far_cities = [c for c in CITIES if get_geo_distance(
            victim_city["lat"], victim_city["lng"], c["lat"], c["lng"]) > 200]
        attacker_city = random.choice(far_cities) if far_cities else get_weighted_city()
        
        base_time = datetime.now() - timedelta(days=random.randint(0, 30))
        num_transfers = random.randint(1, 3)
        for i in range(num_transfers):
            tx_time = base_time + timedelta(minutes=i * random.randint(2, 8))
            tx = generate_single_transaction(is_fraud=True, overrides={
                "sender_account": victim_acc, "sender_bank": victim_bank,
                "sender_name": victim_name, "sender_city": attacker_city,
                "timestamp": tx_time, "amount": random.randint(5000000, 50000000),
                "rail": random.choice(["BI-FAST", "Transfer"]),
                "is_new_device": True, "is_device_mismatch": True,
                "is_sim_swap": True, "is_geo_mismatch": True,
                "is_unusual_beneficiary": True,
                "has_failed_attempts": random.choice([True, False]),
                "account_age_days": random.randint(180, 3000),
                "ip_address": generate_suspicious_ip(), "is_suspicious_ip": True,
            })
            transactions.append(tx)
    return transactions

def generate_impossible_travel_scenarios(count=80):
    """Pattern 4: Impossible Travel — 2 transactions from distant cities in minutes."""
    transactions = []
    for _ in range(count):
        sender_bank = random.choice(BANKS)
        sender_acc = generate_account_number(sender_bank)
        sender_name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
        
        city1 = get_weighted_city()
        far_cities = [c for c in CITIES if get_geo_distance(
            city1["lat"], city1["lng"], c["lat"], c["lng"]) > 500]
        city2 = random.choice(far_cities) if far_cities else CITIES[-1]
        
        base_time = datetime.now() - timedelta(days=random.randint(0, 30))
        
        tx1 = generate_single_transaction(is_fraud=True, overrides={
            "sender_account": sender_acc, "sender_bank": sender_bank,
            "sender_name": sender_name, "sender_city": city1,
            "timestamp": base_time, "is_geo_mismatch": True,
            "velocity_count": random.randint(3, 8),
        })
        transactions.append(tx1)
        
        tx2_time = base_time + timedelta(minutes=random.randint(3, 15))
        tx2 = generate_single_transaction(is_fraud=True, overrides={
            "sender_account": sender_acc, "sender_bank": sender_bank,
            "sender_name": sender_name, "sender_city": city2,
            "timestamp": tx2_time, "is_geo_mismatch": True,
            "is_velocity_anomaly": True, "velocity_count": random.randint(5, 12),
        })
        transactions.append(tx2)
    return transactions

def generate_smurfing_scenarios(count=60):
    """Pattern 5: Smurfing — Breaking large amount into many small transactions."""
    transactions = []
    for _ in range(count):
        sender_bank = random.choice(BANKS)
        sender_acc = generate_account_number(sender_bank)
        sender_name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
        sender_city = get_weighted_city()
        
        target_total = random.randint(50000000, 200000000)
        num_splits = random.randint(8, 20)
        per_tx = target_total // num_splits
        
        base_time = datetime.now() - timedelta(days=random.randint(0, 30))
        
        receivers = []
        for _ in range(random.randint(2, 4)):
            rcv_bank = random.choice(BANKS)
            receivers.append({
                "bank": rcv_bank, "account": generate_account_number(rcv_bank),
                "name": f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
                "city": get_weighted_city(),
            })
        
        for i in range(num_splits):
            rcv = random.choice(receivers)
            amount = per_tx + random.randint(-500000, 500000)
            amount = max(100000, min(amount, 49000000))
            tx_time = base_time + timedelta(minutes=i * random.randint(3, 20))
            tx = generate_single_transaction(is_fraud=True, overrides={
                "sender_account": sender_acc, "sender_bank": sender_bank,
                "sender_name": sender_name, "sender_city": sender_city,
                "receiver_account": rcv["account"], "receiver_bank": rcv["bank"],
                "receiver_name": rcv["name"], "receiver_city": rcv["city"],
                "timestamp": tx_time, "amount": amount,
                "rail": random.choice(["BI-FAST", "Transfer", "SKN"]),
                "is_unusual_beneficiary": True, "velocity_count": random.randint(10, 25),
                "is_velocity_anomaly": True,
            })
            transactions.append(tx)
    return transactions

def generate_gambling_laundering_scenarios(count=80):
    """Pattern 6: Judi Online Laundering — Rapid deposits to gambling merchants at off-hours."""
    transactions = []
    gambling_merchants = [m for m in MERCHANTS_RISKY if "Bet" in m or "Slot" in m or "Judi" in m]
    
    for _ in range(count):
        sender_bank = random.choice(BANKS)
        sender_acc = generate_account_number(sender_bank)
        sender_name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
        sender_city = get_weighted_city()
        
        merchant = random.choice(gambling_merchants)
        base_time = datetime.now() - timedelta(days=random.randint(0, 30))
        base_time = base_time.replace(hour=random.choice([0, 1, 2, 3, 23]))
        
        num_deposits = random.randint(3, 8)
        deposit_amount = random.choice([100000, 200000, 500000, 1000000])
        
        for i in range(num_deposits):
            tx_time = base_time + timedelta(minutes=i * random.randint(2, 10))
            tx = generate_single_transaction(is_fraud=True, overrides={
                "sender_account": sender_acc, "sender_bank": sender_bank,
                "sender_name": sender_name, "sender_city": sender_city,
                "merchant": merchant, "timestamp": tx_time,
                "amount": deposit_amount + random.randint(-10000, 10000),
                "rail": random.choice(["E-Wallet", "Virtual Account", "QRIS"]),
                "is_off_hours": True, "velocity_count": random.randint(5, 15),
                "is_velocity_anomaly": True,
            })
            transactions.append(tx)
    return transactions

def generate_deepfake_social_engineering_scenarios(count=80):
    """Pattern 7: Deepfake Social Engineering — New account + SIM swap + big transfer."""
    transactions = []
    for _ in range(count):
        sender_bank = random.choice(BANKS)
        sender_acc = generate_account_number(sender_bank)
        sender_name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
        sender_city = get_weighted_city()
        
        base_time = datetime.now() - timedelta(days=random.randint(0, 14))
        num_transfers = random.randint(1, 2)
        for i in range(num_transfers):
            tx_time = base_time + timedelta(minutes=i * random.randint(5, 15))
            tx = generate_single_transaction(is_fraud=True, overrides={
                "sender_account": sender_acc, "sender_bank": sender_bank,
                "sender_name": sender_name, "sender_city": sender_city,
                "timestamp": tx_time, "amount": random.randint(10000000, 100000000),
                "rail": random.choice(["BI-FAST", "Transfer"]),
                "account_age_days": random.randint(0, 7), "is_new_account": True,
                "is_new_device": True, "is_sim_swap": True,
                "is_unusual_beneficiary": True,
                "is_suspicious_ip": random.choice([True, False]),
                "has_failed_attempts": True, "is_high_value_for_rail": True,
            })
            transactions.append(tx)
    return transactions

def generate_slot_cashout_ring_scenarios(count=50):
    """Pattern 8: Slot Cashout Ring — Fake shell merchants receiving payments from many accounts."""
    transactions = []
    for _ in range(count):
        shell_merchant = random.choice(MERCHANTS_SHELL)
        shell_city = get_weighted_city()
        _, _, merchant_device_fp = generate_device_info()
        
        base_time = datetime.now() - timedelta(days=random.randint(0, 30))
        num_payers = random.randint(6, 12)
        for i in range(num_payers):
            tx_time = base_time + timedelta(minutes=i * random.randint(5, 30))
            tx = generate_single_transaction(is_fraud=True, overrides={
                "merchant": shell_merchant, "receiver_city": shell_city,
                "timestamp": tx_time,
                "amount": random.choice([500000, 1000000, 2000000, 5000000]),
                "rail": random.choice(["QRIS", "Virtual Account", "Transfer"]),
                "is_unusual_beneficiary": True, "velocity_count": random.randint(3, 8),
            })
            transactions.append(tx)
    return transactions

# ============================================================================
# MAIN DATASET GENERATION
# ============================================================================

def main():
    print("=" * 60)
    print("  SAFER FDS — Enhanced Dataset Generator v2")
    print("  100K transactions | 8 fraud patterns")
    print("=" * 60)
    
    os.makedirs("d:/SAFER/MODEL AI", exist_ok=True)
    
    train_size = 80000
    test_size = 20000
    
    # Fraud Target Ratio: ~10% (more realistic)
    train_fraud_target = int(train_size * 0.10)
    test_fraud_target = int(test_size * 0.10)
    
    # ─── TRAIN DATA ─────────────────────────────────────────────
    print(f"\n{'─'*40}")
    print(f"Generating TRAIN dataset ({train_size:,} rows)...")
    print(f"{'─'*40}")
    train_txs = []
    
    scenarios = [
        ("Mule Ring", generate_mule_ring_scenarios, 200),
        ("Device Farm", generate_device_farm_scenarios, 100),
        ("Account Takeover", generate_account_takeover_scenarios, 200),
        ("Impossible Travel", generate_impossible_travel_scenarios, 150),
        ("Smurfing/Structuring", generate_smurfing_scenarios, 100),
        ("Gambling Laundering", generate_gambling_laundering_scenarios, 150),
        ("Deepfake Social Engineering", generate_deepfake_social_engineering_scenarios, 150),
        ("Slot Cashout Ring", generate_slot_cashout_ring_scenarios, 80),
    ]
    
    for name, gen_func, count in scenarios:
        print(f"  Injecting {name} fraud scenarios...")
        txs = gen_func(count=count)
        train_txs.extend(txs)
        print(f"    -> Generated {len(txs)} transactions")
    
    current_fraud = sum(1 for x in train_txs if x["is_fraud"] == 1)
    needed_fraud = max(0, train_fraud_target - current_fraud)
    
    print(f"  Generating {needed_fraud} individual fraud transactions...")
    for _ in range(needed_fraud):
        train_txs.append(generate_single_transaction(is_fraud=True))
    
    needed_normal = train_size - len(train_txs)
    print(f"  Generating {needed_normal:,} normal transactions...")
    for _ in range(needed_normal):
        train_txs.append(generate_single_transaction(is_fraud=False))
    
    random.shuffle(train_txs)
    
    # ─── TEST DATA ──────────────────────────────────────────────
    print(f"\n{'─'*40}")
    print(f"Generating TEST dataset ({test_size:,} rows)...")
    print(f"{'─'*40}")
    test_txs = []
    
    test_scenarios = [
        ("Mule Ring", generate_mule_ring_scenarios, 40),
        ("Device Farm", generate_device_farm_scenarios, 25),
        ("Account Takeover", generate_account_takeover_scenarios, 40),
        ("Impossible Travel", generate_impossible_travel_scenarios, 30),
        ("Smurfing/Structuring", generate_smurfing_scenarios, 25),
        ("Gambling Laundering", generate_gambling_laundering_scenarios, 30),
        ("Deepfake Social Engineering", generate_deepfake_social_engineering_scenarios, 30),
        ("Slot Cashout Ring", generate_slot_cashout_ring_scenarios, 15),
    ]
    
    for name, gen_func, count in test_scenarios:
        print(f"  Injecting {name} fraud scenarios...")
        txs = gen_func(count=count)
        test_txs.extend(txs)
        print(f"    -> Generated {len(txs)} transactions")
    
    current_fraud_test = sum(1 for x in test_txs if x["is_fraud"] == 1)
    needed_fraud_test = max(0, test_fraud_target - current_fraud_test)
    
    print(f"  Generating {needed_fraud_test} individual fraud transactions...")
    for _ in range(needed_fraud_test):
        test_txs.append(generate_single_transaction(is_fraud=True))
    
    needed_normal_test = test_size - len(test_txs)
    print(f"  Generating {needed_normal_test:,} normal transactions...")
    for _ in range(needed_normal_test):
        test_txs.append(generate_single_transaction(is_fraud=False))
    
    random.shuffle(test_txs)
    
    # ─── SAVE ───────────────────────────────────────────────────
    headers = list(train_txs[0].keys())
    
    train_path = "d:/SAFER/MODEL AI/train_transactions.csv"
    test_path = "d:/SAFER/MODEL AI/test_transactions.csv"
    
    print(f"\nSaving to {train_path}...")
    with open(train_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(train_txs)
        
    print(f"Saving to {test_path}...")
    with open(test_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(test_txs)
    
    # ─── SUMMARY ────────────────────────────────────────────────
    train_fraud_count = sum(1 for x in train_txs if x["is_fraud"] == 1)
    test_fraud_count = sum(1 for x in test_txs if x["is_fraud"] == 1)
    
    print(f"\n{'='*60}")
    print(f"  DATASET GENERATION COMPLETE!")
    print(f"{'='*60}")
    print(f"  Train: {len(train_txs):,} records ({train_fraud_count:,} fraud = {train_fraud_count/len(train_txs)*100:.1f}%)")
    print(f"  Test:  {len(test_txs):,} records ({test_fraud_count:,} fraud = {test_fraud_count/len(test_txs)*100:.1f}%)")
    print(f"  Total: {len(train_txs) + len(test_txs):,} records")
    print(f"\n  Fraud patterns injected: 8")
    print(f"  -> Mule Ring, Device Farm, Account Takeover, Impossible Travel")
    print(f"  -> Smurfing, Gambling Laundering, Deepfake Social Eng., Slot Cashout Ring")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
