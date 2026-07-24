"""
Smart London Urban Risk Prediction
-----------------------------------
Streamlit application that predicts the expected number of road
collisions in London for a given hour, based on air quality,
weather, and bike-sharing activity conditions.

Tuwaiq Academy — Data Science Bootcamp
"""

import os
import json
import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Smart London Urban Risk Prediction",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_DIR = "saved_models"
MODEL_NAME_FILE = os.path.join(MODEL_DIR, "best_model_name.txt")

LONDON_SITES = [
    "London Bexley",
    "London Bloomsbury",
    "London Eltham",
    "London Haringey Priory Park South",
    "London Harlington",
    "London Hillingdon",
    "London Marylebone Road",
    "London N. Kensington",
    "London Westminster",
]

SEASON_ORDER = ["Winter", "Spring", "Summer", "Autumn"]

WEATHER_CODE_LABELS = {
    1.0: "Clear",
    2.0: "Scattered clouds",
    3.0: "Broken clouds",
    4.0: "Cloudy",
    7.0: "Rain",
    10.0: "Rain with thunderstorm",
    26.0: "Snowfall",
    94.0: "Freezing fog",
}

# Metrics captured from the training notebook (test set)
MODEL_METRICS = pd.DataFrame(
    {
        "Model": ["XGBoost", "Random Forest", "LSTM"],
        "MAE": [0.8051, 1.1713, np.nan],
        "MSE": [1.6584, 2.9575, 3.0586],
        "RMSE": [1.2878, 1.7197, 1.7489],
        "R2 Score": [0.7265, 0.5123, 0.4956],
    }
).sort_values("RMSE").reset_index(drop=True)


# ------------------------------------------------------------------
# Custom styling
# ------------------------------------------------------------------
def inject_css():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #0b1220 0%, #10182b 100%);
        }
        .main-header {
            padding: 1.6rem 2rem;
            border-radius: 16px;
            background: linear-gradient(120deg, #1e3a8a 0%, #0f172a 100%);
            border: 1px solid rgba(255,255,255,0.08);
            margin-bottom: 1.4rem;
        }
        .main-header h1 {
            color: #f8fafc;
            font-size: 2.1rem;
            margin-bottom: 0.2rem;
        }
        .main-header p {
            color: #cbd5e1;
            font-size: 1rem;
            margin: 0;
        }
        .metric-card {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 1.1rem 1.3rem;
        }
        section[data-testid="stSidebar"] {
            background-color: #0f172a;
        }
        .risk-badge {
            display: inline-block;
            padding: 0.5rem 1.2rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 1.05rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    """Load the best model saved by the training notebook."""
    if not os.path.exists(MODEL_NAME_FILE):
        return None, None

    with open(MODEL_NAME_FILE, "r", encoding="utf-8") as f:
        model_name = f.read().strip()

    if model_name == "XGBoost":
        import xgboost as xgb

        model = xgb.XGBRegressor()
        model.load_model(os.path.join(MODEL_DIR, "best_model.json"))
        return model, model_name

    if model_name == "Random Forest":
        import joblib

        model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
        return model, model_name

    if model_name == "LSTM":
        import tensorflow as tf

        model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "best_model.keras"))
        return model, model_name

    return None, model_name


def get_expected_columns(model):
    """Best-effort retrieval of the feature names the model was trained on."""
    try:
        booster = model.get_booster()
        if booster.feature_names:
            return list(booster.feature_names)
    except Exception:
        pass
    try:
        return list(model.feature_names_in_)
    except Exception:
        pass
    return None


# ------------------------------------------------------------------
# Feature engineering — mirrors the notebook exactly
# ------------------------------------------------------------------
def get_season(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Autumn"


def build_feature_row(inputs: dict) -> pd.DataFrame:
    date_val = inputs["date"]
    hour_val = inputs["hour"]
    timestamp = dt.datetime.combine(date_val, dt.time(hour=hour_val))

    row = {
        "year": timestamp.year,
        "month": timestamp.month,
        "hour": timestamp.hour,
        "day_of_week": timestamp.weekday(),
        "is_weekend": int(timestamp.weekday() in (5, 6)),
        "site": inputs["site"],
        "season": get_season(timestamp.month),
        "air_co": inputs["air_co"],
        "air_nox": inputs["air_nox"],
        "air_no2": inputs["air_no2"],
        "air_no": inputs["air_no"],
        "air_o3": inputs["air_o3"],
        "air_air_temp": inputs["air_air_temp"],
        "weather_tavg": inputs["weather_tavg"],
        "weather_tmin": inputs["weather_tmin"],
        "weather_tmax": inputs["weather_tmax"],
        "weather_prcp": inputs["weather_prcp"],
        "weather_wdir": inputs["weather_wdir"],
        "weather_wspd": inputs["weather_wspd"],
        "weather_pres": inputs["weather_pres"],
        "bike_cnt": inputs["bike_cnt"],
        "bike_t1": inputs["bike_t1"],
        "bike_t2": inputs["bike_t2"],
        "bike_hum": inputs["bike_hum"],
        "bike_wind_speed": inputs["bike_wind_speed"],
        "bike_weather_code": inputs["bike_weather_code"],
        "bike_is_holiday": inputs["bike_is_holiday"],
    }

    df = pd.DataFrame([row])

    # One-hot encode categorical columns exactly like pd.get_dummies did in training
    df = pd.get_dummies(df, columns=["site", "season"], dtype=int)
    return df


def align_to_model(df: pd.DataFrame, expected_columns) -> pd.DataFrame:
    if not expected_columns:
        return df
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    return df[expected_columns]


def risk_bucket(prediction: float):
    if prediction < 1:
        return "Low", "#22c55e"
    if prediction < 3:
        return "Moderate", "#f59e0b"
    return "High", "#ef4444"


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
def render_header():
    st.markdown(
        """
        <div class="main-header">
            <h1>🚦 Smart London Urban Risk Prediction</h1>
            <p>Estimating hourly road collision risk in London from air quality,
            weather, and bike-sharing activity — Tuwaiq Academy project.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(model_name):
    with st.sidebar:
        st.markdown("### ℹ️ About this project")
        st.write(
            "This app estimates the number of road collisions expected in a "
            "given hour in London, using environmental and transportation "
            "signals: air quality, weather conditions, and bike-sharing "
            "activity, combined with time-based patterns."
        )
        st.markdown("---")
        st.markdown("### 🧠 Active model")
        if model_name:
            st.success(f"**{model_name}**")
        else:
            st.error("No model found in `saved_models/`.")
        st.markdown("---")
        st.markdown("### 📚 Data sources")
        st.markdown(
            "- UK DEFRA AURN air quality (2015–2023)\n"
            "- London bike-sharing dataset\n"
            "- London weather (2000–2023)\n"
            "- UK road safety accidents & vehicles"
        )
        st.markdown("---")
        st.caption("Built with Streamlit · Tuwaiq Academy")


def render_predict_tab(model, model_name):
    st.subheader("Configure conditions")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            date_val = st.date_input("Date", value=dt.date.today())
        with col2:
            hour_val = st.slider("Hour of day", 0, 23, 8)

        site = st.selectbox("Monitoring site / area", LONDON_SITES)

        st.markdown("#### 🌫️ Air quality (µg/m³, CO in mg/m³)")
        a1, a2, a3 = st.columns(3)
        with a1:
            air_co = st.number_input("CO", min_value=0.0, max_value=10.0, value=0.3, step=0.1)
            air_no = st.number_input("NO", min_value=0.0, max_value=400.0, value=20.0, step=1.0)
        with a2:
            air_nox = st.number_input("NOx", min_value=0.0, max_value=500.0, value=60.0, step=1.0)
            air_o3 = st.number_input("O₃", min_value=0.0, max_value=200.0, value=30.0, step=1.0)
        with a3:
            air_no2 = st.number_input("NO₂", min_value=0.0, max_value=250.0, value=35.0, step=1.0)
            air_air_temp = st.number_input("Air temperature (°C)", min_value=-10.0, max_value=40.0, value=15.0, step=0.5)

        st.markdown("#### 🌦️ Weather")
        w1, w2, w3 = st.columns(3)
        with w1:
            weather_tavg = st.number_input("Avg temperature (°C)", min_value=-10.0, max_value=40.0, value=12.0, step=0.5)
            weather_prcp = st.number_input("Precipitation (mm)", min_value=0.0, max_value=100.0, value=0.0, step=0.5)
        with w2:
            weather_tmin = st.number_input("Min temperature (°C)", min_value=-15.0, max_value=35.0, value=8.0, step=0.5)
            weather_wdir = st.slider("Wind direction (°)", 0, 360, 180)
        with w3:
            weather_tmax = st.number_input("Max temperature (°C)", min_value=-10.0, max_value=45.0, value=16.0, step=0.5)
            weather_wspd = st.number_input("Wind speed (km/h)", min_value=0.0, max_value=100.0, value=15.0, step=0.5)
        weather_pres = st.slider("Atmospheric pressure (hPa)", 970, 1045, 1013)

        st.markdown("#### 🚲 Bike-sharing activity")
        b1, b2, b3 = st.columns(3)
        with b1:
            bike_cnt = st.number_input("Bike rentals count", min_value=0, max_value=10000, value=800, step=10)
            bike_hum = st.slider("Humidity (%)", 0, 100, 70)
        with b2:
            bike_t1 = st.number_input("Actual temperature (°C)", min_value=-10.0, max_value=40.0, value=13.0, step=0.5)
            bike_wind_speed = st.number_input("Bike wind speed (km/h)", min_value=0.0, max_value=100.0, value=10.0, step=0.5)
        with b3:
            bike_t2 = st.number_input("Feels-like temperature (°C)", min_value=-10.0, max_value=40.0, value=12.0, step=0.5)
            bike_weather_code = st.selectbox(
                "Weather condition",
                options=list(WEATHER_CODE_LABELS.keys()),
                format_func=lambda x: WEATHER_CODE_LABELS[x],
                index=0,
            )
        bike_is_holiday = st.checkbox("Public holiday", value=False)

        submitted = st.form_submit_button("🔮 Predict collision risk", use_container_width=True)

    if submitted:
        if model is None:
            st.error("No trained model is available. Please add the model files to `saved_models/`.")
            return

        inputs = {
            "date": date_val,
            "hour": hour_val,
            "site": site,
            "air_co": air_co,
            "air_nox": air_nox,
            "air_no2": air_no2,
            "air_no": air_no,
            "air_o3": air_o3,
            "air_air_temp": air_air_temp,
            "weather_tavg": weather_tavg,
            "weather_tmin": weather_tmin,
            "weather_tmax": weather_tmax,
            "weather_prcp": weather_prcp,
            "weather_wdir": weather_wdir,
            "weather_wspd": weather_wspd,
            "weather_pres": weather_pres,
            "bike_cnt": bike_cnt,
            "bike_t1": bike_t1,
            "bike_t2": bike_t2,
            "bike_hum": bike_hum,
            "bike_wind_speed": bike_wind_speed,
            "bike_weather_code": float(bike_weather_code),
            "bike_is_holiday": int(bike_is_holiday),
        }

        features = build_feature_row(inputs)
        expected_cols = get_expected_columns(model)
        features_aligned = align_to_model(features, expected_cols)

        try:
            if model_name == "LSTM":
                arr = features_aligned.to_numpy().reshape(1, 1, -1)
                prediction = float(model.predict(arr, verbose=0)[0][0])
            else:
                prediction = float(model.predict(features_aligned)[0])
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

        prediction = max(prediction, 0.0)
        level, color = risk_bucket(prediction)

        st.markdown("---")
        r1, r2 = st.columns([1, 2])
        with r1:
            st.metric("Predicted collisions (this hour)", f"{prediction:.2f}")
        with r2:
            st.markdown(
                f'<span class="risk-badge" style="background:{color}20; color:{color}; '
                f'border:1px solid {color};">Risk level: {level}</span>',
                unsafe_allow_html=True,
            )
        st.caption(
            "This estimate reflects the expected number of collisions in Greater London "
            "for the selected hour, given the specified environmental and traffic conditions."
        )


def render_performance_tab():
    st.subheader("Model comparison (test set)")
    st.dataframe(
        MODEL_METRICS.style.format(
            {"MAE": "{:.4f}", "MSE": "{:.4f}", "RMSE": "{:.4f}", "R2 Score": "{:.4f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.bar_chart(MODEL_METRICS.set_index("Model")[["RMSE"]])
    st.markdown(
        """
        **Takeaways from training**
        - XGBoost achieved the lowest error (RMSE ≈ 1.29) and the highest R² (≈ 0.73),
          and was selected as the final model.
        - Random Forest was competitive but less accurate than XGBoost.
        - The LSTM sequence model performed worst among the three, suggesting the
          hourly signal here is better captured by tree-based models than by a
          recurrent architecture.
        """
    )


def render_about_tab():
    st.subheader("About the project")
    st.markdown(
        """
        **Smart London Urban Risk Prediction** investigates how environmental and
        transportation-related factors influence road collision risk in London.

        Four datasets were combined at an hourly resolution:
        - UK DEFRA AURN air quality data (2015–2023)
        - London bike-sharing activity
        - London weather data (2000–2023)
        - UK road safety accidents & vehicles

        After cleaning, merging, and feature engineering (time-based features,
        weekend flag, season), three models were trained and compared: an LSTM
        neural network, a Random Forest regressor, and XGBoost. XGBoost was
        selected as the final model based on RMSE and R² on a held-out test set.

        **Disclaimer:** predictions are statistical estimates derived from
        historical data and should not be used as the sole basis for real-time
        safety decisions.
        """
    )


def main():
    inject_css()
    render_header()

    model, model_name = load_model()
    render_sidebar(model_name)

    tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Model Performance", "ℹ️ About"])
    with tab1:
        render_predict_tab(model, model_name)
    with tab2:
        render_performance_tab()
    with tab3:
        render_about_tab()


if __name__ == "__main__":
    main()
