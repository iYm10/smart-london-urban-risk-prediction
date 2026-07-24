# Run this as ONE final cell in the notebook after model_results is created.
import os
import shutil
import joblib
import numpy as np

save_folder = "saved_models"
if os.path.exists(save_folder):
    shutil.rmtree(save_folder)
os.makedirs(save_folder, exist_ok=True)

best_row = model_results.loc[model_results["RMSE"].idxmin()]
best_model_name = best_row["Model"]

if best_model_name == "LSTM":
    # Keras models are stored separately, while the bundle keeps their path.
    model_path = os.path.join(save_folder, "best_model.keras")
    lstm_model.save(model_path)
    model_object = None
elif best_model_name == "Random Forest":
    model_object = rf_model
elif best_model_name == "XGBoost":
    model_object = xgb_model
else:
    raise ValueError(f"Unknown model: {best_model_name}")

# Raw default values used to complete fields not shown in the customer UI.
# Medians are safer than zeros for continuous training features.
raw_source = df_model.drop(columns=["collision_count"], errors="ignore").copy()
feature_defaults = {}
for col in raw_source.columns:
    if pd.api.types.is_numeric_dtype(raw_source[col]):
        feature_defaults[col] = float(raw_source[col].median())
    else:
        mode = raw_source[col].mode(dropna=True)
        feature_defaults[col] = mode.iloc[0] if not mode.empty else "Unknown"

# Data-driven thresholds for Low / Moderate / High predicted collision counts.
q50 = float(y.quantile(0.50))
q90 = float(y.quantile(0.90))

bundle = {
    "model_name": best_model_name,
    "model": model_object,
    "keras_model_path": "best_model.keras" if best_model_name == "LSTM" else None,
    "feature_columns": list(X.columns),
    "feature_defaults": feature_defaults,
    "scaler": scaler if best_model_name == "LSTM" else None,
    "pca": pca if best_model_name == "LSTM" else None,
    "metrics": {
        "mae": float(best_row["MAE"]),
        "mse": float(best_row["MSE"]),
        "rmse": float(best_row["RMSE"]),
        "r2": float(best_row["R2 Score"]),
    },
    "risk_thresholds": {
        "low_max": q50,
        "high_min": q90,
    },
    "target_mean": float(y.mean()),
    "target_p99": float(y.quantile(0.99)),
}

# LSTM needs a small wrapper adjustment in app.py; tree models work directly.
joblib.dump(bundle, os.path.join(save_folder, "model_bundle.pkl"))

shutil.make_archive("smart_london_streamlit_assets", "zip", save_folder)
print("Best model:", best_model_name)
print("Created:", os.path.join(save_folder, "model_bundle.pkl"))
print("Created: smart_london_streamlit_assets.zip")
