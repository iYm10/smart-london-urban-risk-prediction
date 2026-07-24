import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="London Urban Risk Intelligence",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Premium visual system
# -----------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Manrope:wght@600;700;800&display=swap');

    :root {
        --navy: #071A2E;
        --navy-2: #0C2745;
        --cyan: #25C2D1;
        --mint: #69E0C5;
        --ice: #F3F8FB;
        --card: #FFFFFF;
        --text: #10243A;
        --muted: #687D90;
        --line: #DCE7EE;
        --danger: #ED6A5A;
        --warning: #F3B562;
        --success: #39B98A;
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(180deg, #F8FBFD 0%, #EEF5F8 100%); }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1.2rem; padding-bottom: 2.5rem; max-width: 1500px; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--navy) 0%, #09243D 100%);
        border-right: 1px solid rgba(255,255,255,.08);
    }
    [data-testid="stSidebar"] * { color: #F4FBFF; }
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stNumberInput label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stDateInput label { color: #D7E8F2 !important; font-weight: 600; }

    .brand-wrap { padding: 8px 4px 22px; }
    .brand-kicker { color: #6FE4DC; font-size: .72rem; font-weight: 800; letter-spacing: .17em; text-transform: uppercase; }
    .brand-title { color: white; font-family: 'Manrope'; font-size: 1.45rem; line-height: 1.18; font-weight: 800; margin-top: 6px; }
    .brand-sub { color: #A9C1D1; font-size: .83rem; margin-top: 8px; line-height: 1.55; }

    .hero {
        position: relative; overflow: hidden;
        background: linear-gradient(120deg, #071A2E 0%, #0D3655 62%, #12627A 100%);
        border-radius: 24px; padding: 30px 34px; color: white;
        box-shadow: 0 20px 55px rgba(15, 43, 67, .17);
        margin-bottom: 22px;
    }
    .hero:after {
        content: ''; position: absolute; width: 320px; height: 320px; border-radius: 50%;
        right: -80px; top: -150px; background: radial-gradient(circle, rgba(105,224,197,.32), rgba(105,224,197,0));
    }
    .hero-kicker { font-size: .72rem; letter-spacing: .16em; text-transform: uppercase; color: #7CE8DE; font-weight: 800; }
    .hero h1 { font-family: 'Manrope'; font-size: 2.05rem; margin: 7px 0 8px; line-height: 1.15; }
    .hero p { color: #D3E5ED; max-width: 820px; margin: 0; font-size: .95rem; line-height: 1.65; }
    .live-pill { display:inline-flex; align-items:center; gap:8px; margin-top:17px; padding:7px 12px; border:1px solid rgba(255,255,255,.18); border-radius:999px; background:rgba(255,255,255,.08); font-size:.78rem; }
    .live-dot { width:8px; height:8px; border-radius:50%; background:#67E2C0; box-shadow:0 0 0 5px rgba(103,226,192,.13); }

    .section-title { font-family:'Manrope'; font-size:1.03rem; font-weight:800; color:var(--text); margin:4px 0 12px; }
    .card {
        background: rgba(255,255,255,.94); border: 1px solid var(--line); border-radius: 18px;
        padding: 20px; box-shadow: 0 8px 30px rgba(29, 58, 78, .06); height: 100%;
    }
    .metric-label { color:var(--muted); font-size:.76rem; font-weight:700; text-transform:uppercase; letter-spacing:.08em; }
    .metric-value { color:var(--text); font-family:'Manrope'; font-size:1.75rem; font-weight:800; margin-top:6px; }
    .metric-note { color:var(--muted); font-size:.78rem; margin-top:5px; }
    .risk-low { color:var(--success); } .risk-moderate { color:var(--warning); } .risk-high { color:var(--danger); }

    .insight { border-left: 4px solid var(--cyan); padding: 13px 15px; background:#F3FAFC; border-radius: 4px 12px 12px 4px; color:#36546A; font-size:.89rem; line-height:1.55; margin-bottom:10px; }
    .factor-row { display:flex; justify-content:space-between; align-items:center; padding:10px 0; border-bottom:1px solid #E8EFF3; }
    .factor-row:last-child { border-bottom:0; }
    .factor-name { color:#29475C; font-size:.85rem; font-weight:600; }
    .factor-tag { font-size:.72rem; padding:5px 9px; border-radius:999px; background:#EAF6F7; color:#177783; font-weight:700; }

    .footer-note { text-align:center; color:#7B8D9C; font-size:.72rem; margin-top:24px; }
    .stButton > button { width:100%; border:0; border-radius:12px; padding:.72rem 1rem; background:linear-gradient(90deg,#25C2D1,#69E0C5); color:#06233A; font-weight:800; box-shadow:0 8px 22px rgba(37,194,209,.25); }
    .stButton > button:hover { color:#06233A; border:0; transform:translateY(-1px); }
    div[data-testid="stMetric"] { background:white; border:1px solid var(--line); padding:14px 16px; border-radius:15px; }
    </style>
    """,
    unsafe_allow_html=True,
)

BUNDLE_PATH = os.getenv("MODEL_BUNDLE_PATH", "saved_models/model_bundle.pkl")

@st.cache_resource
def load_bundle(path: str):
    if not os.path.exists(path):
        return None
    data = joblib.load(path)
    if data.get("model_name") == "LSTM" and data.get("model") is None:
        from tensorflow.keras.models import load_model
        keras_name = data.get("keras_model_path", "best_model.keras")
        keras_path = os.path.join(os.path.dirname(path), keras_name)
        data["model"] = load_model(keras_path)
    return data

bundle = load_bundle(BUNDLE_PATH)

# -----------------------------
# Helpers
# -----------------------------
def season_from_month(month: int) -> str:
    if month in (12, 1, 2): return "Winter"
    if month in (3, 4, 5): return "Spring"
    if month in (6, 7, 8): return "Summer"
    return "Autumn"


def risk_profile(prediction: float, thresholds: dict):
    low_max = float(thresholds.get("low_max", 0.25))
    high_min = float(thresholds.get("high_min", 0.75))
    if prediction <= low_max:
        return "Low", "risk-low", "Normal operating conditions. Continue routine monitoring."
    if prediction < high_min:
        return "Moderate", "risk-moderate", "Elevated conditions detected. Consider targeted operational readiness."
    return "High", "risk-high", "High-risk period predicted. Prioritise alerts, response coverage, and traffic monitoring."


def gauge(value: float, max_value: float, level: str):
    bar_color = {"Low": "#39B98A", "Moderate": "#F3B562", "High": "#ED6A5A"}[level]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"font": {"size": 38, "color": "#10243A"}, "suffix": " collisions"},
        gauge={
            "axis": {"range": [0, max(max_value, 1)], "tickcolor": "#9EB0BC"},
            "bar": {"color": bar_color, "thickness": .28},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, max_value * .33], "color": "#EAF7F2"},
                {"range": [max_value * .33, max_value * .66], "color": "#FFF4E4"},
                {"range": [max_value * .66, max_value], "color": "#FCEBE8"},
            ],
        },
    ))
    fig.update_layout(height=280, margin=dict(l=25, r=25, t=25, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def build_feature_frame(raw: dict, feature_columns: list[str], defaults: dict) -> pd.DataFrame:
    row = dict(defaults)
    row.update(raw)
    df = pd.DataFrame([row])
    df = pd.get_dummies(df, dtype=int)
    return df.reindex(columns=feature_columns, fill_value=0)


def make_prediction(bundle_data: dict, raw: dict) -> float:
    model = bundle_data["model"]
    name = bundle_data["model_name"]
    X = build_feature_frame(raw, bundle_data["feature_columns"], bundle_data.get("feature_defaults", {}))

    if name == "LSTM":
        scaler = bundle_data["scaler"]
        pca = bundle_data["pca"]
        arr = pca.transform(scaler.transform(X))
        arr = arr.reshape(arr.shape[0], 1, arr.shape[1])
        pred = model.predict(arr, verbose=0)
    else:
        pred = model.predict(X)
    return max(0.0, float(np.asarray(pred).reshape(-1)[0]))

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.markdown("""
    <div class="brand-wrap">
      <div class="brand-kicker">Urban Intelligence Platform</div>
      <div class="brand-title">London Risk<br>Command Centre</div>
      <div class="brand-sub">Scenario-based collision forecasting using weather, air quality, mobility and temporal signals.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Scenario inputs")
    selected_date = st.date_input("Date", value=datetime(2017, 7, 15).date())
    hour = st.slider("Hour of day", 0, 23, 17)
    site = st.selectbox("Monitoring site", [
        "London Bexley", "London Bloomsbury", "London Eltham",
        "London Haringey Priory Park South", "London Harlington",
        "London Hillingdon", "London Marylebone Road",
        "London N. Kensington", "London Westminster"
    ], index=6)

    st.markdown("### Weather")
    weather_tavg = st.slider("Average temperature (°C)", -10.0, 35.0, 16.0, .5)
    weather_tmin = st.slider("Minimum temperature (°C)", -15.0, 30.0, 11.0, .5)
    weather_tmax = st.slider("Maximum temperature (°C)", -5.0, 42.0, 21.0, .5)
    weather_prcp = st.slider("Precipitation (mm)", 0.0, 60.0, 1.0, .5)
    weather_wspd = st.slider("Wind speed (km/h)", 0.0, 80.0, 14.0, .5)
    weather_pres = st.slider("Pressure (hPa)", 950.0, 1050.0, 1014.0, 1.0)
    weather_wdir = st.slider("Wind direction (°)", 0, 360, 220)

    with st.expander("Air quality & mobility", expanded=False):
        air_no2 = st.number_input("NO₂", min_value=0.0, value=42.0, step=1.0)
        air_nox = st.number_input("NOx", min_value=0.0, value=65.0, step=1.0)
        air_no = st.number_input("NO", min_value=0.0, value=18.0, step=1.0)
        air_o3 = st.number_input("O₃", min_value=0.0, value=36.0, step=1.0)
        air_co = st.number_input("CO", min_value=0.0, value=0.4, step=0.1)
        air_air_temp = st.number_input("Air monitor temperature (°C)", value=16.0, step=.5)
        bike_cnt = st.number_input("Bike activity count", min_value=0, value=1200, step=50)
        bike_t1 = st.number_input("Bike temperature t1", value=16.0, step=.5)
        bike_t2 = st.number_input("Bike feels-like t2", value=15.0, step=.5)
        bike_hum = st.slider("Bike humidity", 0.0, 100.0, 68.0, 1.0)
        bike_wind_speed = st.number_input("Bike wind speed", min_value=0.0, value=14.0, step=.5)
        bike_weather_code = st.number_input("Bike weather code", min_value=1, value=2, step=1)

    predict_clicked = st.button("Run risk prediction", type="primary")

# Feature dictionary aligned to notebook feature engineering
month = selected_date.month
day_of_week = selected_date.weekday()
raw_features = {
    "site": site,
    "year": selected_date.year,
    "month": month,
    "hour": hour,
    "day_of_week": day_of_week,
    "is_weekend": int(day_of_week >= 5),
    "season": season_from_month(month),
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
    "bike_weather_code": bike_weather_code,
}

# -----------------------------
# Main dashboard
# -----------------------------
st.markdown("""
<div class="hero">
  <div class="hero-kicker">Smart London Urban Risk Prediction</div>
  <h1>Urban Collision Risk Intelligence</h1>
  <p>Decision-support dashboard for identifying potentially high-risk London hours from combined temporal, weather, air-quality and mobility conditions.</p>
  <div class="live-pill"><span class="live-dot"></span> Scenario engine ready</div>
</div>
""", unsafe_allow_html=True)

if bundle is None:
    st.error("Model bundle not found. Place `model_bundle.pkl` inside the `saved_models` folder. Use the export cell supplied with this app.")
    st.stop()

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if predict_clicked:
    try:
        st.session_state.prediction = make_prediction(bundle, raw_features)
    except Exception as exc:
        st.error(f"Prediction could not be generated: {exc}")

prediction = st.session_state.prediction
metrics = bundle.get("metrics", {})
thresholds = bundle.get("risk_thresholds", {"low_max": .25, "high_min": .75})

if prediction is None:
    st.info("Set the scenario in the left panel, then select **Run risk prediction**.")
    prediction = float(bundle.get("target_mean", 0.0))

level, level_class, action_text = risk_profile(prediction, thresholds)
max_gauge = max(float(bundle.get("target_p99", 2.0)), prediction * 1.25, 1.0)

st.markdown('<div class="section-title">Executive risk overview</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="card"><div class="metric-label">Predicted collisions</div><div class="metric-value">{prediction:.2f}</div><div class="metric-note">Expected count for the selected hour</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="card"><div class="metric-label">Risk classification</div><div class="metric-value {level_class}">{level}</div><div class="metric-note">Relative to training-data thresholds</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="card"><div class="metric-label">Active model</div><div class="metric-value" style="font-size:1.35rem">{bundle.get("model_name", "—")}</div><div class="metric-note">Lowest RMSE model selected</div></div>', unsafe_allow_html=True)
with c4:
    r2 = metrics.get("r2", metrics.get("R2 Score", None))
    display_r2 = f"{r2:.3f}" if isinstance(r2, (int, float)) else "—"
    st.markdown(f'<div class="card"><div class="metric-label">Model R²</div><div class="metric-value">{display_r2}</div><div class="metric-note">Held-out test performance</div></div>', unsafe_allow_html=True)

st.write("")
left, right = st.columns([1.35, 1])
with left:
    st.markdown('<div class="section-title">Predicted risk intensity</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.plotly_chart(gauge(prediction, max_gauge, level), use_container_width=True, config={"displayModeBar": False})
    st.markdown(f'<div class="insight"><b>Operational interpretation:</b> {action_text}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-title">Scenario profile</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    rush = 7 <= hour <= 9 or 16 <= hour <= 19
    night = hour < 6 or hour >= 22
    factors = [
        ("Time window", "Rush hour" if rush else "Off-peak"),
        ("Day type", "Weekend" if day_of_week >= 5 else "Weekday"),
        ("Precipitation", "Adverse" if weather_prcp >= 2 else "Low / dry"),
        ("Wind conditions", "Strong" if weather_wspd >= 30 else "Normal"),
        ("Mobility activity", "High" if bike_cnt >= 2000 else "Moderate / low"),
        ("Lighting proxy", "Nighttime" if night else "Daytime"),
    ]
    for name, tag in factors:
        st.markdown(f'<div class="factor-row"><span class="factor-name">{name}</span><span class="factor-tag">{tag}</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.write("")
st.markdown('<div class="section-title">Combined-condition intelligence</div>', unsafe_allow_html=True)
a, b, c = st.columns(3)
combined = []
if weather_prcp >= 2 and rush: combined.append("Rainfall and rush-hour activity overlap in this scenario.")
if night and day_of_week >= 5: combined.append("Nighttime and weekend conditions occur together.")
if bike_cnt >= 2000 and weather_prcp >= 2: combined.append("High mobility activity coincides with adverse weather.")
if air_no2 >= 60 and weather_wspd < 10: combined.append("Higher NO₂ and low wind may indicate stagnant urban conditions.")
if weather_tavg <= 3 and rush: combined.append("Extreme cold overlaps with rush-hour demand.")
if not combined: combined.append("No predefined high-risk interaction is strongly activated by the selected inputs.")

with a:
    st.markdown(f'<div class="card"><div class="metric-label">Primary interaction</div><div class="metric-value" style="font-size:1.05rem;line-height:1.45">{combined[0]}</div></div>', unsafe_allow_html=True)
with b:
    st.markdown(f'<div class="card"><div class="metric-label">Weather context</div><div class="metric-value" style="font-size:1.1rem">{weather_tavg:.1f}°C · {weather_prcp:.1f} mm</div><div class="metric-note">Average temperature and precipitation</div></div>', unsafe_allow_html=True)
with c:
    st.markdown(f'<div class="card"><div class="metric-label">Air & activity context</div><div class="metric-value" style="font-size:1.1rem">NO₂ {air_no2:.0f} · Bikes {bike_cnt:,}</div><div class="metric-note">Supporting indicators, not causal claims</div></div>', unsafe_allow_html=True)

with st.expander("Model governance & interpretation", expanded=False):
    st.write(
        "This dashboard predicts the expected hourly collision count from patterns learned in the merged London dataset. "
        "Air-quality and bike-sharing variables are supporting indicators. The output should not be interpreted as evidence that pollution or cycling activity causes collisions."
    )
    st.json({
        "model": bundle.get("model_name"),
        "rmse": metrics.get("rmse", metrics.get("RMSE")),
        "mae": metrics.get("mae", metrics.get("MAE")),
        "r2": metrics.get("r2", metrics.get("R2 Score")),
        "selected_site": site,
        "scenario_timestamp": f"{selected_date} {hour:02d}:00",
    })

st.markdown('<div class="footer-note">Smart London Urban Risk Prediction · Decision-support prototype · Predictions are estimates and should be reviewed alongside operational data.</div>', unsafe_allow_html=True)
