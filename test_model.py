import joblib
import json
import xgboost as xgb
import lightgbm as lgb
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from shared.config import (
    XGB_MODEL_PATH,
    LGB_MODEL_PATH,
    LABEL_ENCODERS_PATH,
    SCALER_PATH,
)

print("XGB Path:", XGB_MODEL_PATH)
print("LGB Path:", LGB_MODEL_PATH)
print("Enc Path:", LABEL_ENCODERS_PATH)
print("Scaler Path:", SCALER_PATH)

if os.path.exists(LABEL_ENCODERS_PATH):
    encoders = joblib.load(LABEL_ENCODERS_PATH)
    print("Label Encoders keys:", list(encoders.keys()))
    for k, v in encoders.items():
        print(f"  {k}: {type(v)} with {len(v.classes_)} classes")
else:
    print("Label Encoders not found!")

if os.path.exists(SCALER_PATH):
    scaler = joblib.load(SCALER_PATH)
    print("Scaler type:", type(scaler))
    if hasattr(scaler, "n_features_in_"):
        print("Scaler features in:", scaler.n_features_in_)
        if hasattr(scaler, "feature_names_in_"):
            print("Scaler features:", scaler.feature_names_in_)
else:
    print("Scaler not found!")

if os.path.exists(XGB_MODEL_PATH):
    try:
        model = xgb.XGBClassifier()
        model.load_model(str(XGB_MODEL_PATH))
        print("XGB features:", getattr(model, "feature_names_in_", None))
    except Exception as e:
        print("XGB loading error:", e)
else:
    print("XGB Model not found!")

if os.path.exists(LGB_MODEL_PATH):
    try:
        booster = lgb.Booster(model_file=str(LGB_MODEL_PATH))
        print("LGB features:", booster.feature_name())
    except Exception as e:
        print("LGB loading error:", e)
else:
    print("LGB Model not found!")
