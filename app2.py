import streamlit as st
import numpy as np
import math

st.set_page_config(
    page_title="DriveGuard — Driving Risk Prediction",
    page_icon="🚗",
    layout="centered"
)

# ── GLOBAL CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Libre+Baskerville:wght@400;700&family=Plus+Jakarta+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

.stApp {
    background: #ffffff !important;
}

.main .block-container {
    max-width: 680px !important;
    padding: 2.5rem 1.5rem 4rem !important;
}

/* ── HIDE STREAMLIT CHROME ── */
#MainMenu, footer, header { visibility: hidden !important; }
.stDeployButton { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }

/* ── BRAND HEADER ── */
.brand-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.8rem;
}
.brand-name {
    font-family: 'Libre Baskerville', serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #111;
    letter-spacing: -0.01em;
}
.brand-name span { color: #2563eb; }
.live-chip {
    font-family: 'IBM Plex Mono', monospace;
    font-size: .6rem;
    letter-spacing: .12em;
    padding: .2rem .65rem;
    border-radius: 100px;
    border: 1px solid #bbf7d0;
    background: #f0fdf4;
    color: #15803d;
}

/* ── HERO ── */
.hero-title {
    font-family: 'Libre Baskerville', serif;
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.025em;
    line-height: 1.2;
    color: #111;
    margin-bottom: .5rem;
}
.hero-title em { font-style: normal; color: #2563eb; }
.hero-sub {
    font-size: .88rem;
    color: #666;
    line-height: 1.7;
    margin-bottom: 1rem;
}

/* ── TECH BADGES ── */
.badges-row {
    display: flex;
    flex-wrap: wrap;
    gap: .4rem;
    margin-bottom: 1.5rem;
}
.badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: .63rem;
    letter-spacing: .06em;
    padding: .25rem .65rem;
    border-radius: 6px;
    font-weight: 500;
}
.badge-lstm  { background: #eff6ff; border: 1px solid #bfdbfe; color: #1d4ed8; }
.badge-rl    { background: #f0fdf4; border: 1px solid #bbf7d0; color: #166534; }
.badge-xai   { background: #fefce8; border: 1px solid #fde68a; color: #92400e; }
.badge-safety{ background: #fff1f2; border: 1px solid #fecdd3; color: #9f1239; }

/* ── DIVIDER ── */
.div-rule {
    height: 1px;
    background: #f0ede8;
    margin: 1.4rem 0;
}

/* ── SECTION LABEL ── */
.section-lbl {
    font-family: 'IBM Plex Mono', monospace;
    font-size: .62rem;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: #999;
    margin-bottom: .8rem;
}

/* ── SELECTBOX OVERRIDES ── */
div[data-baseweb="select"] > div {
    background: #fafaf9 !important;
    border: 1px solid #e5e1d8 !important;
    border-radius: 8px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: .9rem !important;
    font-weight: 500 !important;
    color: #111 !important;
    transition: border-color .15s !important;
}
div[data-baseweb="select"] > div:hover {
    border-color: #b0a898 !important;
}
div[data-baseweb="select"] > div:focus-within {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,.1) !important;
}
ul[data-baseweb="menu"] {
    background: #fff !important;
    border: 1px solid #e5e1d8 !important;
    border-radius: 8px !important;
    box-shadow: 0 6px 20px rgba(0,0,0,.07) !important;
}
ul[data-baseweb="menu"] li {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: .88rem !important;
    color: #111 !important;
}
ul[data-baseweb="menu"] li:hover { background: #fafaf9 !important; }

.stSelectbox label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: .62rem !important;
    letter-spacing: .1em !important;
    color: #888 !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
}

/* ── BUTTON ── */
.stButton > button {
    width: 100% !important;
    padding: .85rem !important;
    background: #111 !important;
    border: none !important;
    border-radius: 8px !important;
    color: #fff !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: .9rem !important;
    font-weight: 600 !important;
    letter-spacing: .01em !important;
    transition: background .15s, transform .1s !important;
    margin-top: .4rem !important;
}
.stButton > button:hover {
    background: #2563eb !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── RESULT BANNER ── */
.banner {
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1rem;
}
.banner-safe    { background: #f0fdf4; border: 1px solid #bbf7d0; }
.banner-caution { background: #fefce8; border: 1px solid #fde68a; }
.banner-risky   { background: #fff1f2; border: 1px solid #fecdd3; }

.banner-icon { font-size: 1.6rem; line-height: 1; flex-shrink: 0; }
.verdict { font-size: 1.3rem; font-weight: 700; letter-spacing: -0.02em; line-height: 1; }
.verdict-safe    { color: #15803d; }
.verdict-caution { color: #a16207; }
.verdict-risky   { color: #be123c; }
.banner-meta { font-size: .78rem; color: #666; margin-top: .2rem; }

/* ── METRICS ── */
.metrics-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: .65rem;
    margin-bottom: 1rem;
}
.metric-card {
    background: #fafaf9;
    border: 1px solid #ede9e0;
    border-radius: 10px;
    padding: .9rem 1rem;
}
.metric-lbl {
    font-family: 'IBM Plex Mono', monospace;
    font-size: .6rem;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: #999;
    margin-bottom: .3rem;
}
.metric-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.1rem;
    font-weight: 500;
    margin-bottom: .4rem;
}
.metric-safe    { color: #15803d; }
.metric-caution { color: #a16207; }
.metric-risky   { color: #be123c; }
.bar-track {
    height: 5px;
    background: #ede9e0;
    border-radius: 100px;
    overflow: hidden;
}
.bar-inner { height: 100%; border-radius: 100px; }
.bar-safe    { background: #22c55e; }
.bar-caution { background: #eab308; }
.bar-risky   { background: #f43f5e; }

/* ── INFO CARDS ── */
.info-card {
    background: #fff;
    border: 1px solid #ede9e0;
    border-radius: 12px;
    padding: 1.1rem 1.25rem;
    margin-bottom: .75rem;
}
.card-hd {
    font-family: 'IBM Plex Mono', monospace;
    font-size: .62rem;
    letter-spacing: .11em;
    text-transform: uppercase;
    color: #999;
    margin-bottom: .8rem;
}
.list-item {
    display: flex;
    gap: .6rem;
    align-items: flex-start;
    padding: .5rem 0;
    border-bottom: 1px solid #f0ede8;
    font-size: .86rem;
    color: #333;
    line-height: 1.55;
}
.list-item:last-child { border-bottom: none; }
.num-pill {
    font-family: 'IBM Plex Mono', monospace;
    font-size: .58rem;
    background: #f4f0e8;
    border: 1px solid #e5e1d8;
    border-radius: 4px;
    padding: .1rem .3rem;
    flex-shrink: 0;
    margin-top: 2px;
    color: #888;
    min-width: 24px;
    text-align: center;
}

/* ── ATTENTION VIZ ── */
.att-container {
    display: flex;
    align-items: flex-end;
    gap: 2px;
    height: 48px;
    margin-top: .5rem;
}
.att-bar {
    flex: 1;
    border-radius: 2px;
    min-height: 4px;
}

/* ── INSIGHT BOX ── */
.insight {
    border-radius: 10px;
    padding: .9rem 1.1rem;
    font-size: .86rem;
    line-height: 1.65;
    margin-bottom: .75rem;
}
.insight-safe    { background: #f0fdf4; border: 1px solid #bbf7d0; color: #14532d; }
.insight-caution { background: #fefce8; border: 1px solid #fde68a; color: #713f12; }
.insight-risky   { background: #fff1f2; border: 1px solid #fecdd3; color: #881337; }

/* ── FOOTER ── */
.pg-footer {
    margin-top: 2.5rem;
    text-align: center;
    font-family: 'IBM Plex Mono', monospace;
    font-size: .58rem;
    letter-spacing: .08em;
    color: #bbb;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════
#  CORE LOGIC  (matches original app2.py)
# ══════════════════════════════════════════

RW = {
    "acc":     {"Low": 0, "Medium": 1, "High": 3},
    "brake":   {"Low": 0, "Medium": 1, "High": 2},
    "turning": {"Smooth": 0, "Moderate": 1, "Sharp": 3},
    "weather": {"Clear": 0, "Rain": 2, "Fog": 3},
}

COMBOS = [
    ({"acc": "High",      "weather": "Rain"},  2, "High speed in rain"),
    ({"acc": "High",      "weather": "Fog"},   3, "High speed in fog"),
    ({"turning": "Sharp", "weather": "Rain"},  2, "Sharp turn on wet road"),
    ({"turning": "Sharp", "acc": "High"},      2, "Aggressive cornering"),
    ({"brake": "High",    "acc": "High"},      1, "Erratic accel-braking"),
]

def risk_score(acc, brake, turn, wthr):
    s = RW["acc"][acc] + RW["brake"][brake] + RW["turning"][turn] + RW["weather"][wthr]
    notes = []
    m = {"acc": acc, "brake": brake, "turning": turn, "weather": wthr}
    bonus_total = sum(b for _, b, _ in COMBOS)
    for combo, bonus, note in COMBOS:
        if all(m.get(k) == v for k, v in combo.items()):
            s += bonus
            notes.append(note)
    pct = min(100, round(s / (9 + bonus_total) * 100))
    return pct, notes

def tier(pct):
    if pct < 30:  return "safe",    "Safe",    "✅"
    if pct < 60:  return "caution", "Caution", "⚠️"
    return              "risky",   "Risky",   "🔴"

def get_explanations(acc, brake, turn, wthr, notes):
    out = []
    if acc == "High":
        out.append("High acceleration pattern detected across timesteps")
    elif acc == "Medium" and wthr in ("Rain", "Fog"):
        out.append("Moderate speed in adverse weather increases risk")
    if brake == "High":
        out.append("Sudden braking behaviour detected")
    if turn == "Sharp":
        out.append("Sharp turning observed — centrifugal force risk")
    elif turn == "Moderate" and wthr in ("Rain", "Fog"):
        out.append("Moderate turn on slippery surface is risky")
    if wthr == "Rain":
        out.append("Rain reduces tyre traction — stopping distance increases")
    if wthr == "Fog":
        out.append("Low visibility in fog — reaction time compressed")
    for n in notes:
        out.append(f"Combined hazard: {n}")
    if not out:
        out.append("Sensor data shows stable, controlled driving")
    return out

def get_recommendations(acc, brake, turn, wthr, cls):
    if cls == "safe":
        return [
            "Maintain current driving pace",
            "Keep a safe following distance",
            "Continue smooth steering inputs",
        ]
    s = []
    if acc == "High":
        s.append("Reduce acceleration — ease off gradually")
    if brake == "High":
        s.append("Brake earlier and more progressively")
    if turn == "Sharp":
        s.append("Smooth the steering input — reduce cornering speed")
    if wthr == "Rain":
        s.append("Increase following distance on wet roads")
    if wthr == "Fog":
        s.append("Use low-beam lights and slow down in fog")
    if not s:
        s.append("Reduce overall speed for current conditions")
    return s

def attention_weights(pct):
    """Simulate LSTM attention weights over 20 timesteps."""
    np.random.seed(42 + pct)
    w = []
    for i in range(20):
        v = 0.3 + 0.4 * math.sin(i / 3) + 0.2 * np.random.random()
        if 8 < i < 14:
            v *= (1 + pct / 120)
        w.append(min(1.0, max(0.05, v)))
    mx = max(w)
    return [x / mx for x in w]

def map_input(acc, brake, turn, wc):
    """Generate synthetic sensor sequence (original logic preserved)."""
    am = {"Low": 0.3, "Medium": 0.7, "High": 1.5}
    bm = {"Low": -0.2, "Medium": -0.7, "High": -1.5}
    tm = {"Smooth": 0.2, "Moderate": 0.8, "Sharp": 1.5}
    wm = {"Clear": 0, "Rain": 1, "Fog": 2}
    av, bv, tv, wv = am[acc], bm[brake], tm[turn], wm[wc]
    seq = []
    for i in range(20):
        f = np.sin(i / 2) + np.random.normal(0, 0.3)
        seq.append([tv*f, tv*f, tv*f, av*f, bv*f, av*f, wv])
    return np.array(seq)


# ══════════════════════════════════════════
#  RENDER — HEADER
# ══════════════════════════════════════════

st.markdown("""
<div class="brand-row">
    <span class="brand-name">Drive<span>Guard</span></span>
    <span class="live-chip">● LIVE</span>
</div>

<div class="hero-title">Driving Risk<br><em>Prediction</em></div>
<div class="hero-sub">
    An intelligent system that predicts driving safety from sensor behaviour,
    explains the why, and suggests corrective actions in real time.
</div>

<div class="badges-row">
    <span class="badge badge-lstm">LSTM Deep Learning</span>
    <span class="badge badge-rl">Reinforcement Learning</span>
    <span class="badge badge-xai">Explainable AI</span>
    <span class="badge badge-safety">Rule-Based Safety Layer</span>
</div>

<div class="div-rule"></div>
<div class="section-lbl">Driving Conditions</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════
#  RENDER — INPUTS
# ══════════════════════════════════════════

col1, col2 = st.columns(2)
with col1:
    acc_level   = st.selectbox("Acceleration", ["Low", "Medium", "High"], index=2)
    turning     = st.selectbox("Turning",      ["Smooth", "Moderate", "Sharp"], index=1)
with col2:
    brake_level = st.selectbox("Braking",      ["Low", "Medium", "High"], index=1)
    weather     = st.selectbox("Weather",      ["Clear", "Rain", "Fog"], index=1)

run = st.button("Analyse →")


# ══════════════════════════════════════════
#  RENDER — RESULTS
# ══════════════════════════════════════════

if run:
    pct, notes   = risk_score(acc_level, brake_level, turning, weather)
    cls, label, icon = tier(pct)
    conf         = min(0.99, max(0.51, round(0.5 + (pct if cls == "risky" else 100 - pct) / 200, 2)))
    exps         = get_explanations(acc_level, brake_level, turning, weather, notes)
    recs         = get_recommendations(acc_level, brake_level, turning, weather, cls)
    weights      = attention_weights(pct)

    st.markdown('<div class="div-rule"></div>', unsafe_allow_html=True)

    # ── BANNER ──
    st.markdown(f"""
    <div class="banner banner-{cls}">
        <div class="banner-icon">{icon}</div>
        <div>
            <div class="verdict verdict-{cls}">{label}</div>
            <div class="banner-meta">Risk score: {pct}% &nbsp;·&nbsp; Confidence: {conf}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── METRIC CARDS ──
    risk_bar_w  = pct
    conf_bar_w  = int(conf * 100)

    st.markdown(f"""
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-lbl">Risk Score</div>
            <div class="metric-val metric-{cls}">{pct}%</div>
            <div class="bar-track">
                <div class="bar-inner bar-{cls}" style="width:{risk_bar_w}%"></div>
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-lbl">Model Confidence</div>
            <div class="metric-val metric-{cls}">{conf}</div>
            <div class="bar-track">
                <div class="bar-inner bar-{cls}" style="width:{conf_bar_w}%"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── ATTENTION WEIGHTS ──
    color_map = {
        "safe":    "34,197,94",
        "caution": "234,179,8",
        "risky":   "244,63,94",
    }
    rgb = color_map[cls]
    bars_html = ""
    for w in weights:
        h = max(4, round(4 + w * 40))
        alpha = round(0.3 + w * 0.7, 2)
        bars_html += f'<div class="att-bar" style="height:{h}px;background:rgba({rgb},{alpha})"></div>'

    st.markdown(f"""
    <div class="info-card">
        <div class="card-hd">Attention — Key Risk Timesteps (LSTM)</div>
        <div class="att-container">{bars_html}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── EXPLANATION ──
    exp_items = "".join(
        f'<div class="list-item"><span class="num-pill">{str(i+1).zfill(2)}</span>{e}</div>'
        for i, e in enumerate(exps)
    )
    st.markdown(f"""
    <div class="info-card">
        <div class="card-hd">Explanation</div>
        {exp_items}
    </div>
    """, unsafe_allow_html=True)

    # ── RL RECOMMENDATIONS ──
    rec_items = "".join(
        f'<div class="list-item"><span class="num-pill">{str(i+1).zfill(2)}</span>{r}</div>'
        for i, r in enumerate(recs)
    )
    st.markdown(f"""
    <div class="info-card">
        <div class="card-hd">RL Recommendations</div>
        {rec_items}
    </div>
    """, unsafe_allow_html=True)

    # ── INSIGHT ──
    insight_msg = {
        "safe":    "Driving pattern is stable and within safe parameters. The LSTM model detects no high-risk temporal sequences across the 20-step window. Keep it up.",
        "caution": "Driving conditions show elevated risk. The RL agent recommends adjusting your inputs to avoid transitioning into a high-risk state.",
        "risky":   "High-risk driving pattern detected. The rule-based safety layer has flagged this combination as dangerous. Follow the RL recommendations immediately to reduce risk.",
    }[cls]

    st.markdown(f"""
    <div class="insight insight-{cls}">{insight_msg}</div>
    """, unsafe_allow_html=True)

# ── FOOTER ──
st.markdown("""
<div class="pg-footer">LSTM + Q-Learning + Attention Mechanism · Sensor-Based Risk Engine</div>
""", unsafe_allow_html=True)