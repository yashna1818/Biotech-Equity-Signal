import streamlit as st

def render_landing(model_comparison: dict, training_summary: dict):
    # Custom Landing CSS styling
    st.markdown("""
    <style>
    .landing-body {
        margin: 0;
        padding: 0;
    }
    
    .hero-container {
        background: linear-gradient(135deg, #0A192F 0%, #0B2545 40%, #0F3A5F 100%);
        border-radius: 16px;
        padding: 60px 48px;
        margin-bottom: 40px;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(15, 108, 189, 0.25);
        box-shadow: 0 10px 30px -10px rgba(10, 25, 47, 0.5);
    }
    
    .hero-title {
        color: #FFFFFF;
        font-size: 42px;
        font-weight: 700;
        margin-top: 12px;
        margin-bottom: 12px;
        letter-spacing: -0.5px;
        line-height: 1.2;
    }
    
    .hero-subtitle {
        color: #94A3B8;
        font-size: 17px;
        line-height: 1.7;
        max-width: 720px;
        margin-bottom: 24px;
    }
    
    .pillar-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 24px;
        height: 280px;
        display: flex;
        flex-direction: column;
        transition: transform 0.3s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    
    .pillar-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 20px -8px rgba(15, 108, 189, 0.15);
        border-color: rgba(15, 108, 189, 0.3);
    }
    
    .pillar-icon {
        font-size: 32px;
        margin-bottom: 16px;
    }
    
    .pillar-title {
        font-size: 16px;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 10px;
    }
    
    .pillar-text {
        font-size: 13.5px;
        color: #475569;
        line-height: 1.6;
        flex: 1;
    }
    
    .cta-button-container {
        display: flex;
        justify-content: center;
        margin-top: 24px;
    }
    
    .metric-badge {
        background: rgba(15, 108, 189, 0.1);
        color: #0F6CBD;
        font-size: 11px;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 20px;
        border: 1px solid rgba(15, 108, 189, 0.2);
    }
    
    .stat-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 14px 20px;
        text-align: center;
    }
    
    .stat-val {
        font-size: 20px;
        font-weight: 700;
        color: #38BDF8;
    }
    
    .stat-lbl {
        font-size: 11px;
        color: #94A3B8;
        margin-top: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Hero Banner
    st.markdown("""
    <div class="hero-container">
        <div style="display: flex; gap: 8px; align-items: center;">
            <span class="metric-badge">Institutional Release v2.1</span>
        </div>
        <h1 class="hero-title">🧬 AG-BioSignals</h1>
        <p class="hero-subtitle">
            An institutional-grade predictive platform fusing clinical FinBERT sentiment analysis with 
            classical stock technical indicators. Powered by a regularized XGBoost classifier and a sequence-sensitive 
            PyTorch LSTM model to detect signal changes for biotech equities.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Capabilities Grid ────────────────────────────────────────────────────
    st.markdown("### 🔬 System Architecture & Modules")
    st.caption("Fusing qualitative bio-regulatory events with quantitative market models.")
    
    p1, p2, p3, p4 = st.columns(4)
    
    with p1:
        st.markdown("""
        <div class="pillar-card">
            <div class="pillar-icon">🧠</div>
            <div class="pillar-title">FinBERT NLP Sentiment</div>
            <div class="pillar-text">
                Tokenizes and normalizes biomedical terms (e.g., trial phases, approvals, warnings) and classifies financial sentiment scores using the FinBERT transformer model.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with p2:
        st.markdown("""
        <div class="pillar-card">
            <div class="pillar-icon">📐</div>
            <div class="pillar-title">Feature Engineering</div>
            <div class="pillar-text">
                Blends technical momentum markers (RSI, MACD, Bollinger Bands) with NLP features, smoothed with a 3-day exponential rolling window to model market digestion lag.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with p3:
        st.markdown("""
        <div class="pillar-card">
            <div class="pillar-icon">🤖</div>
            <div class="pillar-title">Ensemble Forecasting</div>
            <div class="pillar-text">
                Classifies market movement into BUY / HOLD / SELL directions. Aggregates XGBoost tabular probabilities and PyTorch LSTM sequence outcomes into a unified signal.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with p4:
        st.markdown("""
        <div class="pillar-card">
            <div class="pillar-icon">📊</div>
            <div class="pillar-title">Simulated Trading</div>
            <div class="pillar-text">
                Tracks simulated portfolio returns and alpha generation (vs Buy & Hold benchmark) inside an interactive transaction ledger priced in Indian Rupees (₹).
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # ── Live Statistics ──────────────────────────────────────────────────────
    st.markdown("### 📈 Live Model Confidence Summary")
    st.caption("Active metrics computed on out-of-sample validation runs.")

    try:
        xgb_acc = training_summary["xgb_metrics"]["accuracy"]
        lstm_acc = training_summary["lstm_metrics"]["accuracy"]
        baseline_acc = model_comparison.get("Generic_Baseline", {}).get("Accuracy", 0)
        domain_acc = model_comparison.get("Domain_Specific", {}).get("Accuracy", 0)
        lift = (domain_acc - baseline_acc) * 100
    except (KeyError, TypeError):
        xgb_acc = lstm_acc = baseline_acc = domain_acc = 0.50
        lift = 0.0

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(f"""
        <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px; padding:16px 20px; box-shadow:0 1px 3px rgba(0,0,0,0.05); text-align:center;">
            <div style="font-size:11px; font-weight:700; color:#64748B; letter-spacing:0.5px; text-transform:uppercase;">XGBoost Accuracy</div>
            <div style="font-size:28px; font-weight:700; color:#0F6CBD; margin-top:4px;">{xgb_acc*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with s2:
        st.markdown(f"""
        <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px; padding:16px 20px; box-shadow:0 1px 3px rgba(0,0,0,0.05); text-align:center;">
            <div style="font-size:11px; font-weight:700; color:#64748B; letter-spacing:0.5px; text-transform:uppercase;">LSTM Accuracy</div>
            <div style="font-size:28px; font-weight:700; color:#0F6CBD; margin-top:4px;">{lstm_acc*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with s3:
        st.markdown(f"""
        <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px; padding:16px 20px; box-shadow:0 1px 3px rgba(0,0,0,0.05); text-align:center;">
            <div style="font-size:11px; font-weight:700; color:#64748B; letter-spacing:0.5px; text-transform:uppercase;">Domain Performance Lift</div>
            <div style="font-size:28px; font-weight:700; color:#22C55E; margin-top:4px;">+{lift:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with s4:
        st.markdown(f"""
        <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px; padding:16px 20px; box-shadow:0 1px 3px rgba(0,0,0,0.05); text-align:center;">
            <div style="font-size:11px; font-weight:700; color:#64748B; letter-spacing:0.5px; text-transform:uppercase;">Active Coverage</div>
            <div style="font-size:28px; font-weight:700; color:#0F6CBD; margin-top:4px;">6 Equities</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    # CTA Launch Button
    btn_col1, btn_col2, btn_col3 = st.columns([1.5, 1, 1.5])
    with btn_col2:
        # Beautiful SaaS-style launch button
        if st.button("🔓 Launch Workspace", use_container_width=True, key="launch_workspace_btn"):
            st.session_state.page = "dashboard"
            st.rerun()
            
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.caption("<center>⚠️ Research and simulation sandbox. Not financial advice. Fuses OpenFDA and Yahoo Finance metrics.</center>", unsafe_allow_html=True)
