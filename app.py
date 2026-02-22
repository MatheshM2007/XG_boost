import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🏡 California House Price Predictor",
    page_icon="🏡",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---- Google Fonts ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ---- Global ---- */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}

/* ---- Hide default streamlit header/menu ---- */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ---- Hero Banner ---- */
.hero-banner {
    background: linear-gradient(135deg, rgba(102,126,234,0.25) 0%, rgba(118,75,162,0.25) 100%);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 24px;
    padding: 48px 40px 36px 40px;
    text-align: center;
    margin-bottom: 32px;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    box-shadow: 0 8px 32px rgba(31,38,135,0.37);
    animation: fadeInDown 0.7s ease;
}

.hero-title {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 12px 0;
    letter-spacing: -1px;
}

.hero-subtitle {
    color: rgba(255,255,255,0.65);
    font-size: 1.1rem;
    font-weight: 400;
    margin: 0;
}

/* ---- Glass Card ---- */
.glass-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 28px 28px 24px 28px;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    margin-bottom: 24px;
    animation: fadeInUp 0.6s ease;
}

.card-title {
    color: #a78bfa;
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ---- Inputs: labels and number inputs ---- */
label, .stSlider label, .stNumberInput label {
    color: rgba(255,255,255,0.85) !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
}

/* ---- Input boxes ---- */
input[type="number"], .stTextInput input {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(167,139,250,0.35) !important;
    border-radius: 10px !important;
    color: #fff !important;
    font-size: 0.95rem !important;
    padding: 8px 12px !important;
    transition: border-color 0.25s ease, box-shadow 0.25s ease;
}

input[type="number"]:focus {
    border-color: #a78bfa !important;
    box-shadow: 0 0 0 3px rgba(167,139,250,0.2) !important;
    outline: none !important;
}

/* ---- Slider ---- */
.stSlider > div > div > div {
    background: linear-gradient(90deg, #a78bfa, #60a5fa) !important;
}
.stSlider > div > div > div > div {
    background: #a78bfa !important;
    border: 2px solid #fff !important;
    box-shadow: 0 0 8px rgba(167,139,250,0.8) !important;
}

/* ---- Primary button ---- */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 14px !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    padding: 14px 32px !important;
    letter-spacing: 0.04em;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    box-shadow: 0 4px 20px rgba(102,126,234,0.5) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 8px 28px rgba(102,126,234,0.7) !important;
}
.stButton > button[kind="primary"]:active {
    transform: translateY(0) !important;
}

/* ---- Result Card ---- */
.result-card {
    background: linear-gradient(135deg, rgba(102,126,234,0.25) 0%, rgba(118,75,162,0.3) 100%);
    border: 1px solid rgba(167,139,250,0.4);
    border-radius: 20px;
    padding: 36px;
    text-align: center;
    animation: pop 0.5s cubic-bezier(0.34,1.56,0.64,1);
    box-shadow: 0 0 40px rgba(102,126,234,0.3);
    margin-top: 24px;
}

.result-label {
    color: rgba(255,255,255,0.65);
    font-size: 0.9rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 8px;
}

.result-value {
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
}

.result-note {
    color: rgba(255,255,255,0.45);
    font-size: 0.82rem;
    margin-top: 10px;
}

/* ---- Stats row ---- */
.stat-pill {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 14px 20px;
    text-align: center;
}

.stat-pill-label {
    color: rgba(255,255,255,0.5);
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}

.stat-pill-value {
    color: #a78bfa;
    font-size: 1.2rem;
    font-weight: 700;
    margin-top: 4px;
}

/* ---- Footer ---- */
.footer {
    text-align: center;
    color: rgba(255,255,255,0.3);
    font-size: 0.8rem;
    margin-top: 48px;
    padding-bottom: 24px;
}

/* ---- Divider ---- */
.custom-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(167,139,250,0.4), transparent);
    margin: 28px 0;
    border: none;
}

/* ---- Animations ---- */
@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-20px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes pop {
    0%   { transform: scale(0.8); opacity: 0; }
    100% { transform: scale(1);   opacity: 1; }
}

/* ---- Expander ---- */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 10px !important;
    color: rgba(255,255,255,0.75) !important;
    font-weight: 500 !important;
}

/* ---- Dataframe ---- */
.stDataFrame {
    border-radius: 12px;
    overflow: hidden;
}

/* ---- Success/Error messages ---- */
.stAlert {
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("xgb_california_model.pkl", "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"❌ Could not load model: {e}")

# ── Hero Banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🏡 California House Price Predictor</div>
    <p class="hero-subtitle">
        Powered by XGBoost · Enter property details below to get an instant market estimate
    </p>
</div>
""", unsafe_allow_html=True)

# ── Stats Row ─────────────────────────────────────────────────────────────────
s1, s2, s3, s4 = st.columns(4)
for col, label, value in zip(
    [s1, s2, s3, s4],
    ["Model", "Dataset", "Features", "Accuracy"],
    ["XGBoost", "CA Housing", "8 inputs", "~91%"],
):
    col.markdown(f"""
    <div class="stat-pill">
        <div class="stat-pill-label">{label}</div>
        <div class="stat-pill-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

# ── Input Form ────────────────────────────────────────────────────────────────
left, right = st.columns(2, gap="large")

with left:
    st.markdown('<div class="glass-card"><div class="card-title">💰 Financial & Property</div>', unsafe_allow_html=True)
    MedInc = st.number_input(
        "Median Income (tens of thousands $)",
        min_value=0.0, max_value=20.0, value=5.0, step=0.1,
        help="Median income of households in the block (in tens of thousands of USD)",
    )
    HouseAge = st.slider(
        "House Age (years)",
        min_value=1, max_value=52, value=20,
        help="Median age of houses in the block",
    )
    AveRooms = st.number_input(
        "Average Rooms per House",
        min_value=1.0, max_value=50.0, value=5.5, step=0.1,
    )
    AveBedrms = st.number_input(
        "Average Bedrooms per House",
        min_value=0.5, max_value=20.0, value=1.1, step=0.1,
    )
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="glass-card"><div class="card-title">📍 Location & Population</div>', unsafe_allow_html=True)
    Population = st.number_input(
        "Block Population",
        min_value=1, max_value=40000, value=1200, step=50,
    )
    AveOccup = st.number_input(
        "Average Occupancy (persons/house)",
        min_value=0.5, max_value=20.0, value=3.0, step=0.1,
    )
    Latitude = st.number_input(
        "Latitude",
        min_value=32.0, max_value=42.0, value=34.05, step=0.01,
        help="Geographic latitude of the block (California: ~32°N – 42°N)",
    )
    Longitude = st.number_input(
        "Longitude",
        min_value=-125.0, max_value=-114.0, value=-118.25, step=0.01,
        help="Geographic longitude of the block (California: ~-114° – -125°)",
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ── Predict Button ────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
btn_col = st.columns([1, 2, 1])[1]
with btn_col:
    predict = st.button("🔮  Predict House Price", type="primary", use_container_width=True)

# ── Prediction Result ─────────────────────────────────────────────────────────
if predict:
    if not model_loaded:
        st.error("⚠️ Model is not loaded. Please check the model file.")
    else:
        features = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                               Population, AveOccup, Latitude, Longitude]])
        feature_names = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
                         "Population", "AveOccup", "Latitude", "Longitude"]
        df_input = pd.DataFrame(features, columns=feature_names)

        try:
            prediction = model.predict(df_input)[0]
            price_usd = prediction * 100_000

            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">✅ Estimated Median House Value</div>
                <div class="result-value">${price_usd:,.0f}</div>
                <div class="result-note">
                    Based on XGBoost model trained on California Housing dataset
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("📊 View Input Summary"):
                st.dataframe(
                    df_input.T.rename(columns={0: "Your Input"}),
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <hr class="custom-divider">
    Model: <strong>XGBoost</strong> &nbsp;·&nbsp;
    Dataset: <strong>California Housing (sklearn)</strong> &nbsp;·&nbsp;
    Target: Median house value in USD (×100,000)
</div>
""", unsafe_allow_html=True)
