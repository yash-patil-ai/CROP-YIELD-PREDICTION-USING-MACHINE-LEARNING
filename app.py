import streamlit as st
import numpy as np
import pandas as pd
import warnings
import plotly.express as px
import plotly.graph_objects as go
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import time

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CropSense AI – Yield Prediction",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS  (professional dark-green theme)
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root variables ── */
:root {
    --green-900: #0d2b1a;
    --green-800: #133d26;
    --green-700: #1a5233;
    --green-500: #27ae60;
    --green-400: #2ecc71;
    --green-300: #58d68d;
    --amber:     #f39c12;
    --cream:     #f0ede6;
    --white:     #ffffff;
    --muted:     #a8b2aa;
    --card-bg:   rgba(19, 61, 38, 0.55);
    --border:    rgba(46, 204, 113, 0.18);
    --radius:    14px;
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: var(--green-900) !important;
    color: var(--cream) !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--green-800) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--cream) !important; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #133d26 0%, #0d2b1a 60%, #1a5233 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 2.5rem 2rem 2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "🌾";
    font-size: 9rem;
    position: absolute;
    right: 2rem; top: -1rem;
    opacity: 0.08;
    pointer-events: none;
}
.hero h1 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    font-size: 2.4rem !important;
    color: var(--green-400) !important;
    margin: 0 0 0.3rem !important;
    line-height: 1.15 !important;
}
.hero p { color: var(--muted); font-size: 1rem; margin: 0; }
.hero .badge {
    display: inline-block;
    background: rgba(46,204,113,0.12);
    border: 1px solid var(--border);
    border-radius: 99px;
    padding: 0.2rem 0.9rem;
    font-size: 0.78rem;
    color: var(--green-300);
    margin-bottom: 0.8rem;
    letter-spacing: 0.05em;
    font-weight: 500;
}

/* ── Section header ── */
.section-header {
    display: flex; align-items: center; gap: 0.6rem;
    font-family: 'Syne', sans-serif;
    font-size: 1.15rem; font-weight: 700;
    color: var(--green-300);
    margin: 1.8rem 0 0.9rem;
    border-left: 3px solid var(--green-500);
    padding-left: 0.75rem;
}

/* ── Cards ── */
.card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(8px);
}

/* ── Metric tiles ── */
.metric-row { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.2rem; }
.metric-tile {
    flex: 1; min-width: 130px;
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-tile .val {
    font-family: 'Syne', sans-serif;
    font-size: 2rem; font-weight: 800;
    color: var(--green-400);
    line-height: 1;
}
.metric-tile .lbl { font-size: 0.75rem; color: var(--muted); margin-top: 0.3rem; }

/* ── Result box ── */
.result-box {
    background: linear-gradient(135deg, rgba(39,174,96,0.18), rgba(46,204,113,0.08));
    border: 1.5px solid var(--green-500);
    border-radius: var(--radius);
    padding: 1.8rem 2rem;
    text-align: center;
    margin-top: 1.2rem;
}
.result-box .crop-name {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem; font-weight: 800;
    color: var(--green-400);
}
.result-box .crop-icon { font-size: 3rem; }
.result-box small { color: var(--muted); font-size: 0.85rem; }

/* ── Streamlit widgets ── */
div[data-testid="stNumberInput"] label,
div[data-testid="stSlider"] label,
div[data-testid="stSelectbox"] label { color: var(--cream) !important; font-size: 0.9rem; }

.stButton > button {
    background: var(--green-500) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 0.55rem 1.6rem !important;
    transition: background 0.2s, transform 0.1s !important;
}
.stButton > button:hover {
    background: var(--green-400) !important;
    transform: translateY(-1px) !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Upload area ── */
div[data-testid="stFileUploader"] {
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1rem !important;
}

/* ── Dataframe ── */
div[data-testid="stDataFrame"] { border-radius: var(--radius); overflow: hidden; }

/* ── Tabs ── */
button[data-baseweb="tab"] {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    color: var(--muted) !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--green-400) !important;
    border-bottom-color: var(--green-400) !important;
}

/* ── Success / warning ── */
div[data-testid="stAlert"] { border-radius: var(--radius) !important; }

/* ── Tooltip-like info text ── */
.info-text { color: var(--muted); font-size: 0.82rem; margin-top: 0.15rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
for key, val in {
    "model_trained": False,
    "model": None,
    "label_encoder": None,
    "df": None,
    "accuracy": None,
    "report_df": None,
    "crop_counts": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 0.5rem;'>
        <div style='font-family:Syne,sans-serif; font-size:1.3rem; font-weight:800;
                    color:#2ecc71; letter-spacing:0.04em;'>🌾 CropSense AI</div>
        <div style='color:#a8b2aa; font-size:0.78rem; margin-top:0.2rem;'>
            ML-Powered Crop Advisor
        </div>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    st.markdown("**📌 Workflow**")
    steps = [
        ("1", "Upload Dataset", st.session_state.df is not None),
        ("2", "Train SVM Model", st.session_state.model_trained),
        ("3", "Predict Crop", False),
    ]
    for num, label, done in steps:
        color = "#2ecc71" if done else "#a8b2aa"
        icon  = "✅" if done else "⏳"
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:0.5rem;"
            f"color:{color};font-size:0.88rem;margin:0.35rem 0;'>"
            f"{icon} <b>Step {num}</b> – {label}</div>",
            unsafe_allow_html=True
        )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("**ℹ️ About**")
    st.caption(
        "Uses a Support Vector Machine trained on Indian crop data "
        "(soil NPK, humidity, rainfall) to recommend the best crop."
    )
    st.caption("Developed by **TECH TITANS**")

# ─────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="badge">🤖 SVM · Machine Learning</div>
    <h1>CropSense AI</h1>
    <p>Precision crop recommendation powered by Support Vector Machine · Indian Agriculture Dataset</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  STEP 1 – DATASET UPLOAD
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">📂 Step 1 — Upload Dataset</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload **indian_crop_dataset.csv**",
    type="csv",
    help="CSV must include: N_SOIL, P_SOIL, K_SOIL, HUMIDITY, RAINFALL, CROP columns."
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state.df = df
    st.success(f"✅ Dataset loaded — **{len(df):,} rows** × **{len(df.columns)} columns**")

    tab1, tab2, tab3 = st.tabs(["📄 Preview", "📊 Statistics", "🌿 Crop Distribution"])

    with tab1:
        st.dataframe(df.head(10), use_container_width=True)

    with tab2:
        st.dataframe(df.describe().T.style.format("{:.2f}"), use_container_width=True)

    with tab3:
        if "CROP" in df.columns:
            crop_counts = df["CROP"].value_counts().reset_index()
            crop_counts.columns = ["Crop", "Count"]
            fig = px.bar(
                crop_counts, x="Crop", y="Count",
                color="Count",
                color_continuous_scale=["#133d26", "#27ae60", "#2ecc71"],
                template="plotly_dark",
                title="Samples per Crop"
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_family="DM Sans",
                showlegend=False,
                coloraxis_showscale=False,
                margin=dict(l=0, r=0, t=40, b=0),
            )
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)
            st.session_state.crop_counts = crop_counts
        else:
            st.warning("Column 'CROP' not found.")

# ─────────────────────────────────────────────
#  STEP 2 – MODEL TRAINING
# ─────────────────────────────────────────────
if st.session_state.df is not None:
    df = st.session_state.df

    st.markdown('<div class="section-header">⚙️ Step 2 — Train SVM Model</div>', unsafe_allow_html=True)

    with st.expander("🔧 Training Configuration", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("Test split (%)", 10, 40, 20, 5) / 100
            kernel    = st.selectbox("SVM Kernel", ["rbf", "linear", "poly", "sigmoid"])
        with col2:
            c_value  = st.select_slider("Regularisation (C)", [0.1, 0.5, 1.0, 5.0, 10.0], value=1.0)
            rand_seed = st.number_input("Random seed", 0, 999, 42)

    if st.button("🚀 Train Model", use_container_width=True):
        with st.spinner("Training SVM — please wait…"):
            progress = st.progress(0)

            cp = df.drop(
                [c for c in ['TEMPERATURE', 'ph'] if c in df.columns],
                axis=1
            )
            progress.progress(15)

            avg_cols = [c for c in ['P_SOIL', 'K_SOIL', 'HUMIDITY', 'RAINFALL', 'CROP_PRICE'] if c in cp.columns]
            cp[avg_cols] = cp[avg_cols].fillna(cp[avg_cols].mean())
            progress.progress(30)

            le = LabelEncoder()
            cp['target'] = le.fit_transform(cp['CROP'])
            progress.progress(50)

            feature_cols = [c for c in ['N_SOIL', 'P_SOIL', 'K_SOIL', 'HUMIDITY', 'RAINFALL'] if c in cp.columns]
            X = cp[feature_cols]
            y = cp['target']

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=int(rand_seed)
            )
            progress.progress(65)

            model = SVC(kernel=kernel, C=c_value, probability=True)
            model.fit(X_train, y_train)
            progress.progress(90)

            accuracy  = model.score(X_test, y_test)
            y_pred    = model.predict(X_test)
            report    = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report).T.iloc[:-3]

            st.session_state.model         = model
            st.session_state.label_encoder = le
            st.session_state.model_trained = True
            st.session_state.accuracy      = accuracy
            st.session_state.report_df     = report_df
            st.session_state.feature_cols  = feature_cols
            progress.progress(100)
            time.sleep(0.3)
            progress.empty()

        st.success("✅ Model trained successfully!")

    # ── Metrics (persistent after training) ──
    if st.session_state.model_trained:
        acc  = st.session_state.accuracy
        n_classes = len(st.session_state.label_encoder.classes_)
        n_train = len(st.session_state.df)

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-tile">
                <div class="val">{acc:.1%}</div>
                <div class="lbl">Model Accuracy</div>
            </div>
            <div class="metric-tile">
                <div class="val">{n_classes}</div>
                <div class="lbl">Crop Classes</div>
            </div>
            <div class="metric-tile">
                <div class="val">{n_train:,}</div>
                <div class="lbl">Training Samples</div>
            </div>
            <div class="metric-tile">
                <div class="val">SVM</div>
                <div class="lbl">Algorithm</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📈 Per-class Performance Report"):
            rd = st.session_state.report_df
            st.dataframe(
                rd[["precision", "recall", "f1-score", "support"]].style.format({
                    "precision": "{:.2f}", "recall": "{:.2f}",
                    "f1-score": "{:.2f}", "support": "{:.0f}"
                }).background_gradient(cmap="Greens", subset=["f1-score"]),
                use_container_width=True
            )

# ─────────────────────────────────────────────
#  STEP 3 – CROP PREDICTION
# ─────────────────────────────────────────────
if st.session_state.model_trained:
    st.markdown('<div class="section-header">🌱 Step 3 — Predict Crop</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**🪱 Soil Parameters**")
        n = st.number_input("Nitrogen (N_SOIL)", 0.0, 150.0, 90.0, help="Nitrogen content in soil (kg/ha)")
        p = st.number_input("Phosphorus (P_SOIL)", 0.0, 150.0, 42.0, help="Phosphorus content in soil (kg/ha)")
        k = st.number_input("Potassium (K_SOIL)", 0.0, 150.0, 43.0, help="Potassium content in soil (kg/ha)")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**🌦️ Climate Parameters**")
        humidity = st.slider("Humidity (%)", 0.0, 100.0, 60.0, 0.5)
        rainfall = st.slider("Rainfall (mm)", 0.0, 500.0, 120.0, 1.0)

        # mini gauges
        fig_g = go.Figure()
        fig_g.add_trace(go.Indicator(
            mode="gauge+number",
            value=humidity,
            domain={"row": 0, "column": 0},
            title={"text": "Humidity", "font": {"size": 11}},
            number={"suffix": "%", "font": {"size": 14}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar":  {"color": "#27ae60"},
                "bgcolor": "rgba(0,0,0,0)",
                "bordercolor": "rgba(255,255,255,0.1)",
                "steps": [{"range": [0, 100], "color": "rgba(255,255,255,0.04)"}],
            },
        ))
        fig_g.add_trace(go.Indicator(
            mode="gauge+number",
            value=rainfall,
            domain={"row": 0, "column": 1},
            title={"text": "Rainfall", "font": {"size": 11}},
            number={"suffix": " mm", "font": {"size": 14}},
            gauge={
                "axis": {"range": [0, 500], "tickwidth": 1},
                "bar":  {"color": "#f39c12"},
                "bgcolor": "rgba(0,0,0,0)",
                "bordercolor": "rgba(255,255,255,0.1)",
                "steps": [{"range": [0, 500], "color": "rgba(255,255,255,0.04)"}],
            },
        ))
        fig_g.update_layout(
            grid={"rows": 1, "columns": 2, "pattern": "independent"},
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#f0ede6", "family": "DM Sans"},
            height=160,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_g, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Predict button ──
    if st.button("🌾 Predict Best Crop", use_container_width=True):
        feature_cols = getattr(st.session_state, 'feature_cols',
                               ['N_SOIL', 'P_SOIL', 'K_SOIL', 'HUMIDITY', 'RAINFALL'])
        input_vals = {'N_SOIL': n, 'P_SOIL': p, 'K_SOIL': k, 'HUMIDITY': humidity, 'RAINFALL': rainfall}
        input_data = [[input_vals[c] for c in feature_cols]]

        prediction  = st.session_state.model.predict(input_data)
        crop_name   = st.session_state.label_encoder.inverse_transform(prediction)[0]

        # Confidence via probability (if supported)
        try:
            proba  = st.session_state.model.predict_proba(input_data)[0]
            top3_idx   = proba.argsort()[-3:][::-1]
            top3_crops = st.session_state.label_encoder.inverse_transform(top3_idx)
            top3_proba = proba[top3_idx]
            has_proba  = True
        except Exception:
            has_proba = False

        # crop emoji map
        EMOJI = {
            "rice": "🌾", "wheat": "🌾", "maize": "🌽", "mango": "🥭",
            "banana": "🍌", "apple": "🍎", "grapes": "🍇", "watermelon": "🍉",
            "muskmelon": "🍈", "papaya": "🍐", "orange": "🍊", "pomegranate": "🍎",
            "lentil": "🫘", "blackgram": "🫘", "mungbean": "🫘",
            "mothbeans": "🫘", "pigeonpeas": "🫘", "kidneybeans": "🫘",
            "chickpea": "🫘", "coffee": "☕", "jute": "🪢", "cotton": "🧶",
            "coconut": "🥥",
        }
        emoji = EMOJI.get(crop_name.lower(), "🌿")

        st.markdown(f"""
        <div class="result-box">
            <div class="crop-icon">{emoji}</div>
            <div style="color:#a8b2aa; font-size:0.85rem; margin:0.4rem 0 0.2rem;">Recommended Crop</div>
            <div class="crop-name">{crop_name.upper()}</div>
            <small>Based on N={n}, P={p}, K={k} | Humidity={humidity}% | Rainfall={rainfall} mm</small>
        </div>
        """, unsafe_allow_html=True)

        if has_proba:
            st.markdown("")
            st.markdown("**🏆 Top 3 Crop Recommendations**")
            fig_bar = px.bar(
                x=top3_proba * 100,
                y=top3_crops,
                orientation="h",
                text=[f"{v:.1f}%" for v in top3_proba * 100],
                color=top3_proba,
                color_continuous_scale=["#133d26", "#27ae60", "#2ecc71"],
                template="plotly_dark",
            )
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_family="DM Sans",
                coloraxis_showscale=False,
                showlegend=False,
                xaxis_title="Confidence (%)",
                yaxis_title="",
                margin=dict(l=0, r=0, t=10, b=0),
                height=200,
            )
            fig_bar.update_traces(textposition="outside", marker_line_width=0)
            st.plotly_chart(fig_bar, use_container_width=True)

else:
    st.info("⬆️ Complete Steps 1 & 2 to unlock Crop Prediction.")

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center; color:#a8b2aa; font-size:0.8rem; padding:0.5rem 0 1rem;'>"
    "🌾 CropSense AI &nbsp;·&nbsp; Developed by <b>TECH TITANS</b> &nbsp;·&nbsp; "
    "Powered by Scikit-learn · Streamlit · Plotly"
    "</div>",
    unsafe_allow_html=True
)
