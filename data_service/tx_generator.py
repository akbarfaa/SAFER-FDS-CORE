"""
SAFER Data Service — On-demand Transaction Generator

Lightweight port of generate_dataset.py for real-time demo transaction
generation.  Produces transactions matching the ML model's training schema.
"""

import random
import math
from datetime import datetime, timedelta, timezone

# ─── Lookup Tables (identical to training data) ────────────────────────────

FIRST_NAMES = [
    "Andi", "Sari", "Budi", "Maya", "Rizki", "Dewi", "Faisal", "Putri",
    "Aldo", "Indah", "Agus", "Rina", "Dimas", "Lina", "Hendra", "Nita",
    "Yusuf", "Wulan", "Arif", "Ratna", "Bayu", "Siti", "Reza", "Ayu",
    "Taufik", "Mega", "Irfan", "Dian", "Kurnia", "Fitri", "Eko", "Nurul",
    "Wahyu", "Rini", "Surya", "Intan", "Fajar", "Lestari", "Gilang", "Amelia",
]

LAST_NAMES = [
    "Prasetyo", "Wulandari", "Hartono", "Kusuma", "Hidayat", "Permata",
    "Ramadhan", "Nugraheni", "Santoso", "Wijaya", "Susanto", "Purnama",
    "Suryadi", "Laksmi", "Setiawan", "Maharani", "Nugroho", "Anggraini",
    "Saputra", "Handayani", "Utomo", "Rahayu", "Firmansyah", "Puspita",
    "Kurniawan", "Safitri", "Wahyudi", "Hapsari", "Pratama", "Damayanti",
]

MERCHANTS_RETAIL = [
    "Alfamart Jl. Sudirman", "Indomaret Kebayoran", "Circle K Senopati",
    "Warung Makan Sederhana", "Bakso Pak Kumis", "Kopi Kenangan Sudirman",
    "Starbucks Plaza Indonesia", "McDonald's Sarinah", "KFC Thamrin",
    "J.CO Donuts Grand Indonesia",
]
MERCHANTS_ECOMMERCE = [
    "Tokopedia", "Shopee", "Bukalapak", "Lazada", "Blibli",
    "JD.ID", "Zalora", "Sociolla", "Bhinneka", "Orami",
]
MERCHANTS_SERVICES = [
    "Grab", "GoFood", "Gojek", "Traveloka", "Tiket.com",
    "PLN Prepaid", "Telkomsel Pulsa", "BPJS Kesehatan", "PGN Gas",
    "Indosat Prepaid",
]
MERCHANTS_RISKY = [
    "CryptoXchange ID", "BitTrade Asia", "OnlineBet88", "LuckySlot ID",
    "FastCash Pinjol", "QuickLoan Digital",
]
MERCHANTS_CORPORATE = [
    "PT Sumber Makmur Sentosa", "CV Jaya Abadi", "PT Nusantara Logistik",
    "Apotek Kimia Farma", "RS Pondok Indah", "Universitas Indonesia",
    "PT Pertamina (Persero)", "PT Telkom Indonesia", "PT Bank Central Asia",
    "PT Astra International",
]

BANKS = [
    {"code": "BCA", "prefix": "0"}, {"code": "BNI", "prefix": "0"},
    {"code": "BRI", "prefix": "0"}, {"code": "Mandiri", "prefix": "1"},
    {"code": "BSI", "prefix": "7"}, {"code": "CIMB", "prefix": "0"},
    {"code": "Danamon", "prefix": "0"}, {"code": "Permata", "prefix": "0"},
    {"code": "BNP", "prefix": "9"}, {"code": "Mega", "prefix": "0"},
    {"code": "OCBC", "prefix": "0"}, {"code": "Jago", "prefix": "5"},
    {"code": "Nobu", "prefix": "0"}, {"code": "Seabank", "prefix": "9"},
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
    {"name": "Manado", "province": "Sulawesi Utara", "weight": 1, "lat": 1.474, "lng": 124.842},
]

PAYMENT_RAILS = ["QRIS", "BI-FAST", "RTGS", "SKN", "E-Wallet", "Virtual Account", "Kartu Debit", "Kartu Kredit", "Transfer"]
RAIL_WEIGHTS = {"QRIS": 28, "BI-FAST": 18, "RTGS": 3, "SKN": 5, "E-Wallet": 24, "Virtual Account": 8, "Kartu Debit": 6, "Kartu Kredit": 4, "Transfer": 4}
EWALLET_PROVIDERS = ["GoPay", "OVO", "DANA", "ShopeePay", "LinkAja"]
DEVICE_TYPES = ["Android", "iOS", "Web Browser", "Mobile Web"]
DEVICE_BRANDS = {"Android": ["Samsung", "Xiaomi", "OPPO", "Vivo", "Realme", "Infinix", "POCO"], "iOS": ["iPhone 13", "iPhone 14", "iPhone 15", "iPhone 12", "iPhone SE"], "Web Browser": ["Chrome 125", "Safari 18", "Firefox 127", "Edge 125"], "Mobile Web": ["Chrome 125", "Safari 18", "Firefox 127", "Edge 125"]}

HIGH_VALUE_THRESHOLDS = {"QRIS": 2_000_000, "BI-FAST": 50_000_000, "RTGS": 500_000_000, "SKN": 50_000_000, "E-Wallet": 1_000_000, "Virtual Account": 10_000_000, "Kartu Debit": 5_000_000, "Kartu Kredit": 15_000_000, "Transfer": 25_000_000}

_MERCHANT_CATEGORIES = {}
for m in MERCHANTS_RETAIL: _MERCHANT_CATEGORIES[m] = "Retail"
for m in MERCHANTS_ECOMMERCE: _MERCHANT_CATEGORIES[m] = "E-Commerce"
for m in MERCHANTS_SERVICES: _MERCHANT_CATEGORIES[m] = "Services"
for m in MERCHANTS_CORPORATE: _MERCHANT_CATEGORIES[m] = "Corporate"
for m in MERCHANTS_RISKY:
    if "Crypto" in m or "Bit" in m: _MERCHANT_CATEGORIES[m] = "Crypto"
    elif "Bet" in m or "Slot" in m: _MERCHANT_CATEGORIES[m] = "Gambling"
    else: _MERCHANT_CATEGORIES[m] = "Lending"

_tx_counter = 100_000
_tx_counter_initialized = False

def get_next_tx_id():
    global _tx_counter, _tx_counter_initialized
    if not _tx_counter_initialized:
        try:
            from data_service.database import SessionLocal
            from data_service.models import Transaction
            db = SessionLocal()
            try:
                # Query the lexicographically largest ID (which matches our format TX-XXXXXX)
                max_id_row = db.query(Transaction.id).order_by(Transaction.id.desc()).first()
                if max_id_row and max_id_row[0] and max_id_row[0].startswith("TX-"):
                    try:
                        num = int(max_id_row[0].split("-")[1])
                        if num >= _tx_counter:
                            _tx_counter = num
                    except ValueError:
                        pass
            finally:
                db.close()
        except Exception as e:
            # Fallback if DB not ready or circular import
            pass
        _tx_counter_initialized = True

    _tx_counter += 1
    return f"TX-{_tx_counter:06d}"


# ─── Helpers ────────────────────────────────────────────────────────────────

def _weighted_choice(choices: dict) -> str:
    items = list(choices.items())
    total = sum(w for _, w in items)
    r = random.uniform(0, total)
    upto = 0
    for key, weight in items:
        upto += weight
        if upto >= r:
            return key
    return items[-1][0]


def _weighted_city():
    return random.choices(CITIES, weights=[c["weight"] for c in CITIES], k=1)[0]


def _acct_num(bank):
    length = 13 if bank["code"] == "Mandiri" else 10
    return bank["prefix"] + "".join(str(random.randint(0, 9)) for _ in range(length - len(bank["prefix"])))


def _fp():
    return "".join(random.choice("0123456789abcdef") for _ in range(16))


def _ip():
    pfx = random.choice(["103.84", "114.142", "36.68", "180.244", "110.136", "103.28", "202.134", "112.215"])
    return f"{pfx}.{random.randint(1,254)}.{random.randint(1,254)}"


def _amount(rail, is_fraud=False):
    if is_fraud and random.random() < 0.6:
        thresh = HIGH_VALUE_THRESHOLDS.get(rail, 10_000_000)
        return int(thresh * random.uniform(1.2, 5.0))
    ranges = {"QRIS": (5000, 500000), "BI-FAST": (100000, 10000000), "RTGS": (100000000, 2000000000), "SKN": (1000000, 100000000), "E-Wallet": (1000, 500000), "Virtual Account": (50000, 50000000), "Kartu Debit": (10000, 10000000), "Kartu Kredit": (50000, 50000000), "Transfer": (50000, 10000000)}
    lo, hi = ranges.get(rail, (10000, 10000000))
    return random.randint(lo, hi)


def _merchant(rail, is_fraud=False):
    if is_fraud and random.random() < 0.7:
        return random.choice(MERCHANTS_RISKY)
    if rail == "QRIS": return random.choice(MERCHANTS_RETAIL if random.random() < 0.7 else MERCHANTS_SERVICES)
    if rail == "E-Wallet": return random.choice(MERCHANTS_SERVICES if random.random() < 0.5 else MERCHANTS_ECOMMERCE)
    if rail in ("RTGS", "SKN"): return random.choice(MERCHANTS_CORPORATE)
    if rail == "Kartu Kredit": return random.choice(MERCHANTS_ECOMMERCE if random.random() < 0.6 else MERCHANTS_RETAIL)
    return random.choice(MERCHANTS_RETAIL + MERCHANTS_ECOMMERCE + MERCHANTS_SERVICES)


def _geo_dist(lat1, lng1, lat2, lng2):
    return round(math.sqrt((lat1 - lat2)**2 + (lng1 - lng2)**2) * 111, 2)


# ─── Main Generator ────────────────────────────────────────────────────────

def generate_transaction(is_fraud: bool = False) -> dict:
    """Generate a single synthetic transaction as a flat dict (DB-ready)."""
    tx_id = get_next_tx_id()

    rail = _weighted_choice(RAIL_WEIGHTS)
    sender_city = _weighted_city()
    receiver_city = _weighted_city()
    sender_bank = random.choice(BANKS)
    receiver_bank = random.choice(BANKS)
    device_type = random.choice(DEVICE_TYPES)
    device_brand = random.choice(DEVICE_BRANDS.get(device_type, ["Unknown"]))

    now = datetime.now(timezone.utc)
    timestamp = now - timedelta(seconds=random.randint(0, 60))

    amount = _amount(rail, is_fraud)
    merchant = _merchant(rail, is_fraud)
    merchant_category = _MERCHANT_CATEGORIES.get(merchant, "General")
    account_age_days = random.randint(1, 25) if (is_fraud and random.random() < 0.4) else random.randint(30, 3650)

    # Fraud indicators
    is_off_hours = (is_fraud and random.random() < 0.5) or timestamp.hour < 5 or timestamp.hour > 23
    is_new_device = (is_fraud and random.random() < 0.6) or random.random() < 0.05
    is_device_mismatch = (is_fraud and random.random() < 0.5) or random.random() < 0.02
    is_suspicious_ip = (is_fraud and random.random() < 0.4) or random.random() < 0.01
    has_failed_attempts = (is_fraud and random.random() < 0.5) or random.random() < 0.02
    is_sim_swap = (is_fraud and random.random() < 0.3) or random.random() < 0.01
    is_unusual_beneficiary = (is_fraud and random.random() < 0.7) or random.random() < 0.08
    velocity_count = random.randint(8, 25) if (is_fraud and random.random() < 0.6) else random.randint(1, 3)
    is_velocity_anomaly = velocity_count >= 8
    geo_distance = _geo_dist(sender_city["lat"], sender_city["lng"], receiver_city["lat"], receiver_city["lng"])
    is_geo_mismatch = (is_fraud and random.random() < 0.5 and geo_distance > 300)
    if is_geo_mismatch and geo_distance < 300:
        geo_distance = random.randint(400, 2500)
    thresh = HIGH_VALUE_THRESHOLDS.get(rail, 10_000_000)
    is_high_value = amount > thresh
    is_new_account = account_age_days < 30
    is_risky_merchant = merchant_category in ("Crypto", "Gambling", "Lending")

    if is_off_hours and is_fraud:
        timestamp = timestamp.replace(hour=random.randint(2, 4), minute=random.randint(0, 59))

    channel = "Mobile App" if rail == "E-Wallet" else "QR Scan" if rail == "QRIS" else random.choice(["Mobile Banking", "Internet Banking", "ATM"])

    return {
        "id": tx_id,
        "timestamp": timestamp,
        "sender_name": f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
        "sender_account": _acct_num(sender_bank),
        "sender_bank": sender_bank["code"],
        "sender_city": sender_city["name"],
        "sender_province": sender_city["province"],
        "sender_lat": sender_city["lat"],
        "sender_lng": sender_city["lng"],
        "receiver_name": f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
        "receiver_account": _acct_num(receiver_bank),
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
        "channel": channel,
        "device_type": device_type,
        "device_brand": device_brand,
        "device_fingerprint": _fp(),
        "ip_address": _ip(),
        "is_new_device": is_new_device,
        "account_age_days": account_age_days,
        "is_velocity_anomaly": is_velocity_anomaly,
        "is_geo_mismatch": is_geo_mismatch,
        "is_off_hours": is_off_hours,
        "is_high_value_for_rail": is_high_value,
        "is_suspicious_ip": is_suspicious_ip,
        "is_risky_merchant": is_risky_merchant,
        "is_new_account": is_new_account,
        "has_failed_attempts": has_failed_attempts,
        "is_device_mismatch": is_device_mismatch,
        "is_sim_swap": is_sim_swap,
        "is_unusual_beneficiary": is_unusual_beneficiary,
        "velocity_count": velocity_count,
        "geo_distance_km": geo_distance,
        "is_fraud": is_fraud,
    }


def generate_batch(count: int = 5, fraud_ratio: float = 0.18) -> list[dict]:
    """Generate a batch of transactions with a given fraud ratio."""
    txs = []
    for _ in range(count):
        is_fraud = random.random() < fraud_ratio
        txs.append(generate_transaction(is_fraud=is_fraud))
    return txs
