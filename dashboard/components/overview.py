import streamlit as st
import pandas as pd


def render_overview(model_comparison: dict, training_summary: dict):
    # ── Hero Banner ──────────────────────────────────────────────────────────
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #0F6CBD 0%, #0E4A8A 50%, #0D3468 100%);
        border-radius: 16px;
        padding: 40px 48px;
        margin-bottom: 32px;
        position: relative;
        overflow: hidden;
    ">
        <div style="
            position: absolute; top: -40px; right: -40px;
            width: 200px; height: 200px;
            background: rgba(255,255,255,0.05);
            border-radius: 50%;
        "></div>
        <div style="
            position: absolute; bottom: -60px; right: 80px;
            width: 140px; height: 140px;
            background: rgba(255,255,255,0.04);
            border-radius: 50%;
        "></div>
        <div style="position: relative; z-index: 1;">
            <span style="
                background: rgba(255,255,255,0.15);
                color: #93C5FD;
                font-size: 12px;
                font-weight: 600;
                letter-spacing: 2px;
                text-transform: uppercase;
                padding: 4px 12px;
                border-radius: 20px;
                border: 1px solid rgba(147,197,253,0.3);
            ">Biotech Predictive Intelligence Platform</span>
            <h1 style="
                color: #FFFFFF;
                font-size: 38px;
                font-weight: 700;
                margin: 16px 0 8px 0;
                letter-spacing: -0.5px;
                line-height: 1.2;
            ">🧬 AG-BioSignals</h1>
            <p style="
                color: #BFDBFE;
                font-size: 16px;
                line-height: 1.7;
                max-width: 680px;
                margin: 0;
            ">
                An institutional-grade platform that fuses <b style="color:#fff;">FinBERT NLP sentiment</b> from FDA regulatory events 
                with <b style="color:#fff;">MACD, RSI &amp; Bollinger Bands</b>, powering an ensemble of 
                <b style="color:#fff;">XGBoost + LSTM</b> models to generate actionable BUY / HOLD / SELL signals 
                on biotech equities — priced in <b style="color:#fff;">₹ Indian Rupees</b>.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Live Performance KPIs ────────────────────────────────────────────────
    st.markdown("#### 📊 Live Model Performance")

    try:
        xgb_acc = training_summary["xgb_metrics"]["accuracy"]
        lstm_acc = training_summary["lstm_metrics"]["accuracy"]
        xgb_f1 = training_summary["xgb_metrics"].get("f1_score", 0)
        lstm_f1 = training_summary["lstm_metrics"].get("f1_score", 0)
        baseline_acc = model_comparison.get("Generic_Baseline", {}).get("Accuracy", 0)
        domain_acc = model_comparison.get("Domain_Specific", {}).get("Accuracy", 0)
        lift = (domain_acc - baseline_acc) * 100
    except (KeyError, TypeError):
        xgb_acc = lstm_acc = xgb_f1 = lstm_f1 = 0
        lift = 0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("XGBoost Accuracy", f"{xgb_acc*100:.1f}%", "NLP-enhanced tabular model")
    with k2:
        st.metric("LSTM Accuracy", f"{lstm_acc*100:.1f}%", "10-day sequence lookback")
    with k3:
        st.metric("XGBoost F1", f"{xgb_f1:.3f}", "Macro-averaged")
    with k4:
        st.metric("Domain Lift", f"+{lift:.2f}%", "vs price-only baseline")

    st.markdown("---")

    # ── 4-Step Processing Pipeline ───────────────────────────────────────────
    st.markdown("#### ⚙️ How It Works — 4-Stage Pipeline")

    steps = [
        {
            "num": "01",
            "color": "#0B2545",
            "icon": "🌐",
            "title": "Data Ingestion",
            "body": (
                "Pulls daily OHLCV prices for 6 biotech equities via <b>yfinance</b>. "
                "Simultaneously queries the <b>OpenFDA Enforcement API</b> for drug recalls, "
                "safety alerts, and approval notices — chronologically aligned by date."
            ),
            "tags": ["yfinance", "OpenFDA API", "6 Tickers"],
        },
        {
            "num": "02",
            "color": "#134074",
            "icon": "🧠",
            "title": "NLP Sentiment",
            "body": (
                "Event text is tokenized and passed through <b>ProsusAI/FinBERT</b>, "
                "a financial domain BERT model. Outputs positive/negative/neutral scores. "
                "Regulatory event types (FDA approval, recall, trial milestone) are labeled "
                "with severity weights."
            ),
            "tags": ["FinBERT", "transformers", "Event Labeling"],
        },
        {
            "num": "03",
            "color": "#0F6CBD",
            "icon": "📐",
            "title": "Feature Engineering",
            "body": (
                "Merges sentiment scores with <b>RSI</b>, <b>MACD</b>, and <b>Bollinger Bands</b>. "
                "Applies a 3-day exponential smoothing window to sentiment to capture lagged "
                "market reactions. Normalises all features via standard scaling."
            ),
            "tags": ["RSI", "MACD", "Bollinger Bands", "EMA Smoothing"],
        },
        {
            "num": "04",
            "color": "#0284C7",
            "icon": "🤖",
            "title": "Ensemble Prediction",
            "body": (
                "<b>XGBoost</b> evaluates tabular features; a <b>PyTorch LSTM</b> processes "
                "rolling 10-day sequences. Their softmax probabilities are combined 50/50 "
                "to emit BUY, HOLD, or SELL signals with confidence scores."
            ),
            "tags": ["XGBoost", "PyTorch LSTM", "50/50 Ensemble"],
        },
    ]

    cols = st.columns(4)
    for col, step in zip(cols, steps):
        tags_html = " ".join(
            f'<span style="background:{step["color"]}22; color:{step["color"]}; '
            f'font-size:11px; font-weight:600; padding:2px 8px; border-radius:20px; '
            f'margin-right:4px; border:1px solid {step["color"]}44;">{t}</span>'
            for t in step["tags"]
        )
        col.markdown(f"""
        <div style="
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-top: 4px solid {step['color']};
            border-radius: 10px;
            padding: 20px;
            height: 320px;
            display: flex;
            flex-direction: column;
        ">
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
                <span style="
                    font-size: 11px; font-weight: 700; color: {step['color']};
                    background: {step['color']}15; padding: 3px 8px; border-radius: 20px;
                ">STEP {step['num']}</span>
                <span style="font-size: 20px;">{step['icon']}</span>
            </div>
            <h4 style="margin: 0 0 10px 0; color: #1E293B; font-size: 15px;">{step['title']}</h4>
            <p style="font-size: 13px; color: #64748B; line-height: 1.6; flex: 1; margin: 0 0 14px 0;">
                {step['body']}
            </p>
            <div style="margin-top: auto; display: flex; flex-wrap: wrap; gap: 4px;">{tags_html}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Covered Stocks ───────────────────────────────────────────────────────
    st.markdown("#### 🏥 Covered Biotech Equities")

    stocks = [
        {"ticker": "MRNA", "name": "Moderna Inc.", "focus": "mRNA vaccines & oncology therapeutics", "color": "#0B2545"},
        {"ticker": "PFE",  "name": "Pfizer Inc.",  "focus": "Broad pharmaceutical & vaccine portfolio", "color": "#134074"},
        {"ticker": "BNTX", "name": "BioNTech SE",  "focus": "mRNA cancer immunotherapy pipeline", "color": "#0F6CBD"},
        {"ticker": "NVAX", "name": "Novavax Inc.", "focus": "Protein sub-unit vaccine technology", "color": "#0284C7"},
        {"ticker": "GILD", "name": "Gilead Sciences","focus": "Antiviral & oncology drug portfolio", "color": "#06B6D4"},
        {"ticker": "AMGN", "name": "Amgen Inc.",   "focus": "Biologics & biosimilar therapeutics",  "color": "#38BDF8"},
    ]

    s_cols = st.columns(6)
    for sc, stk in zip(s_cols, stocks):
        sc.markdown(f"""
        <div style="
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 16px 12px;
            text-align: center;
        ">
            <div style="
                font-size: 20px; font-weight: 700;
                color: {stk['color']};
                background: {stk['color']}15;
                border-radius: 8px;
                padding: 6px 0;
                margin-bottom: 8px;
                letter-spacing: 1px;
            ">{stk['ticker']}</div>
            <div style="font-size: 12px; font-weight: 600; color: #1E293B; margin-bottom: 4px;">{stk['name']}</div>
            <div style="font-size: 11px; color: #94A3B8; line-height: 1.4;">{stk['focus']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Tech Stack ───────────────────────────────────────────────────────────
    st.markdown("#### 🛠️ Technology Stack")

    tech = [
        ("🐍 Python 3.11", "Core runtime"),
        ("⚡ XGBoost", "Gradient boosted trees"),
        ("🔥 PyTorch", "LSTM sequence model"),
        ("🤗 FinBERT", "Financial NLP (transformers)"),
        ("📈 yfinance", "Market data feed"),
        ("💊 OpenFDA", "Regulatory event API"),
        ("📊 Plotly", "Interactive charts"),
        ("🖥️ Streamlit", "Dashboard framework"),
    ]

    t_cols = st.columns(8)
    for tc, (label, desc) in zip(t_cols, tech):
        tc.markdown(f"""
        <div style="
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 12px 8px;
            text-align: center;
        ">
            <div style="font-size: 13px; font-weight: 600; color: #1E293B; margin-bottom: 3px;">{label}</div>
            <div style="font-size: 11px; color: #94A3B8;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("⚠️ AG-BioSignals is for research and educational purposes only. Not financial advice. All prices in ₹ INR (1 USD = ₹83.5).")
