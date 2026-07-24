"""
Paste this in place of the final "Save and download the best model" cell
in the training notebook. It saves the model exactly as before, but writes
`best_model_name.txt` as the rich JSON metadata object that app.py expects
(model_name, feature_columns, feature_defaults, metrics, risk_thresholds),
instead of a plain model-name string.

Assumes the following already exist earlier in the notebook, unchanged:
X, X_train, model_results, best_model_name,
lstm_model / rf_model / xgb_model, mae_rf/mse_rf/rmse_rf/r2_rf,
mae_xgb/mse_xgb/rmse_xgb/r2_xgb.
"""

import os
import json
import shutil

import numpy as np

save_folder = "saved_models"

if os.path.exists(save_folder):
    shutil.rmtree(save_folder)
os.makedirs(save_folder, exist_ok=True)

best_model_name = model_results.loc[model_results["RMSE"].idxmin(), "Model"]
print("Best Model:", best_model_name)

# ---- Save the model file itself (unchanged from the original cell) ----
if best_model_name == "LSTM":
    best_model_path = os.path.join(save_folder, "best_model.keras")
    lstm_model.save(best_model_path)
    metrics = {"mae": None, "mse": float(mse_lstm), "rmse": float(np.sqrt(mse_lstm)), "r2": float(r2_lstm)}

elif best_model_name == "Random Forest":
    import joblib
    best_model_path = os.path.join(save_folder, "best_model.pkl")
    joblib.dump(rf_model, best_model_path)
    metrics = {"mae": float(mae_rf), "mse": float(mse_rf), "rmse": float(rmse_rf), "r2": float(r2_rf)}

elif best_model_name == "XGBoost":
    best_model_path = os.path.join(save_folder, "best_model.json")
    xgb_model.save_model(best_model_path)
    metrics = {"mae": float(mae_xgb), "mse": float(mse_xgb), "rmse": float(rmse_xgb), "r2": float(r2_xgb)}

else:
    raise ValueError(f"Unknown model name: {best_model_name}")

# ---- Build the rich metadata the Streamlit app relies on ----
feature_columns = X_train.columns.tolist()

# Sensible default for every raw/dummy column: mean for numeric,
# 0 for one-hot dummy columns (site_*, season_*) so the app can
# safely start every scenario from these values.
feature_defaults = {}
for col in feature_columns:
    if col.startswith("site_") or col.startswith("season_"):
        feature_defaults[col] = 0.0
    else:
        feature_defaults[col] = float(X_train[col].mean())

# Risk bands from the tertiles of the chosen model's predictions on the
# test set — adapts automatically to whichever model/data you trained on.
if best_model_name == "XGBoost":
    preds = y_pred_xgb
elif best_model_name == "Random Forest":
    preds = y_pred_rf
else:
    preds = np.asarray(y_pred_lstm).reshape(-1)

preds = np.clip(np.asarray(preds).reshape(-1), 0, None)
risk_thresholds = {
    "low_max": float(np.percentile(preds, 33)),
    "high_min": float(np.percentile(preds, 66)),
}

metadata = {
    "model_name": best_model_name,
    "feature_columns": feature_columns,
    "feature_defaults": feature_defaults,
    "metrics": metrics,
    "risk_thresholds": risk_thresholds,
}

model_name_path = os.path.join(save_folder, "best_model_name.txt")
with open(model_name_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print("\nBest model saved successfully.")
print("Saved model:", best_model_path)
print("Saved metadata:", model_name_path)

# ---- Zip and download, same as before ----
zip_file = shutil.make_archive("smart_london_best_model", "zip", save_folder)
print("\nZIP file created:", zip_file)

from google.colab import files
files.download(zip_file)
