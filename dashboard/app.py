import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import datetime
import pandas as pd
import streamlit as st

# Import components
from dashboard.components.terminal import render_terminal
from dashboard.components.analytics import render_analytics
from dashboard.components.news import render_news
from dashboard.components.pipeline import render_pipeline
from dashboard.components.copilot import render_copilot
from dashboard.components.overview import render_overview
from dashboard.components.trading import render_trading
TICKERS = ["MRNA", "PFE", "BNTX", "NVAX", "GILD", "AMGN"]

# --------------------------------------------------------
# 1. STREAMLIT APP CONFIGURATION & STYLING
# --------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="AG-BioSignals | Clinical Intelligence",
    page_icon="🧬",
    initial_sidebar_state="collapsed"
)

# Clinical / Biotech Research CSS Theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif;
}

.stApp {
    background-color: #F0F6FC;
    background-image: radial-gradient(#D4E2F0 1.2px, transparent 1.2px);
    background-size: 24px 24px;
}

.block-container { padding: 1.2rem 2.5rem; max-width: 100%; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { border-bottom: 2px solid #C8D8E8; gap: 8px; }
.stTabs [data-baseweb="tab"] { color: #52637A; font-size: 14px; padding: 10px 16px; font-weight: 500; border-radius: 6px 6px 0 0; }
.stTabs [aria-selected="true"] { color: #0B5C9C; background: #E1EFFE; border-bottom: 2px solid #0B5C9C !important; font-weight: 600; }

/* Selectbox styling */
div[data-baseweb="select"] { min-width: 180px; }

/* Glass Card styling */
.glass-card {
    background: rgba(255, 255, 255, 0.7);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(226, 232, 240, 0.8);
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
    margin-bottom: 16px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.02);
}
.metric-title {
    font-size: 11px;
    font-weight: 700;
    color: #64748B;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 24px;
    font-weight: 700;
    color: #0F6CBD;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------
# 2. DATA LOADING
# --------------------------------------------------------
try:
    with open("data/processed/training_summary.json", "r") as f:
        training_summary = json.load(f)
    with open("data/processed/model_comparison.json", "r") as f:
        model_comparison = json.load(f)
    with open("data/raw/fda_events.json", "r") as f:
        fda_events = json.load(f)
except FileNotFoundError:
    st.error("Data not found. Please run `python prediction/train.py`.")
    st.stop()

# --------------------------------------------------------
# 2.5 ROUTING PAGE DETERMINATION
# --------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "landing"

if st.session_state.page == "landing":
    from dashboard.components.landing import render_landing
    render_landing(model_comparison, training_summary)
    st.stop()

companies = {
    "MRNA": "Moderna Inc.",
    "PFE": "Pfizer Inc.",
    "BNTX": "BioNTech SE",
    "NVAX": "Novavax Inc.",
    "GILD": "Gilead Sciences",
    "AMGN": "Amgen Inc."
}

# --------------------------------------------------------
# 3. TOP HEADER (Clinical Strip)
# --------------------------------------------------------
st.markdown("## 🧬 AG-BioSignals")

header_cols = st.columns([1.8, 1.2, 1.5, 1, 1, 1, 1.5])

with header_cols[0]:
    selected_ticker = st.selectbox("Select Company", TICKERS, format_func=lambda t: f"{t} — {companies.get(t, t)}")

with header_cols[1]:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    if st.button("🏠 Portal", key="goto_portal_btn", use_container_width=True):
        st.session_state.page = "landing"
        st.rerun()

try:
    stock_df = pd.read_csv(f"data/raw/{selected_ticker}_stock.csv", index_col='Date')
    # Convert all stock prices from USD to INR (using 1 USD = 83.5 INR conversion rate)
    USD_TO_INR = 83.5
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in stock_df.columns:
            stock_df[col] = stock_df[col] * USD_TO_INR
            
    backtest_df = pd.read_csv(f"data/processed/{selected_ticker}_backtest.csv", index_col='Date')
except FileNotFoundError:
    st.error(f"Missing data for {selected_ticker}.")
    st.stop()

ticker_events = [e for e in fda_events if e["ticker"] == selected_ticker]
ticker_events = [e for e in ticker_events if (datetime.datetime.strptime(e["date"], "%Y-%m-%d").date() > (datetime.date.today() - datetime.timedelta(days=1200)))]

# Compute Header Metrics
last_price = stock_df['Close'].iloc[-1]
prev_price = stock_df['Close'].iloc[-2]
price_delta = last_price - prev_price

latest_action = backtest_df['Pred_Action'].iloc[-1]

avg_sent = backtest_df['Daily_Return'].rolling(5).mean().iloc[-1]
risk_score = 85 if avg_sent < -0.005 else (20 if avg_sent > 0.005 else 50)
risk_label = "High" if risk_score > 70 else ("Low" if risk_score < 30 else "Medium")

# Define ticker-specific metadata for dynamic presentation metrics
ticker_metadata = {
    "MRNA": {"confidence": "86%", "catalyst": "Phase III Readout", "delta": "14 days"},
    "PFE": {"confidence": "82%", "catalyst": "FDA Warning Review", "delta": "5 days"},
    "BNTX": {"confidence": "74%", "catalyst": "Phase II Trial", "delta": "28 days"},
    "NVAX": {"confidence": "89%", "catalyst": "Safety Committee", "delta": "12 days"},
    "GILD": {"confidence": "91%", "catalyst": "PDUFA Action Date", "delta": "18 days"},
    "AMGN": {"confidence": "78%", "catalyst": "Regulatory Briefing", "delta": "45 days"}
}

meta = ticker_metadata.get(selected_ticker, {"confidence": "78%", "catalyst": "Clinical Readout", "delta": "23 days"})

with header_cols[2]:
    st.metric("Stock Price", f"₹{last_price:,.2f}", f"₹{price_delta:+,.2f}")

with header_cols[3]:
    st.metric("AI Signal", latest_action)

with header_cols[4]:
    st.metric("Confidence", meta["confidence"])

with header_cols[5]:
    st.metric("FDA Risk", risk_label)

with header_cols[6]:
    st.metric("Next Catalyst", meta["catalyst"], meta["delta"])

st.markdown("---")

# --------------------------------------------------------
# 4. ROUTING
# --------------------------------------------------------
tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠 Overview",
    "🔬 Primary Intelligence",
    "🧪 Clinical Research",
    "🤖 AI Research",
    "📰 News Intelligence",
    "📊 Paper Trading",
    "💡 Copilot"
])

with tab0:
    render_overview(model_comparison, training_summary)
with tab1:
    render_terminal(selected_ticker, stock_df, backtest_df, ticker_events)
with tab2:
    render_pipeline(selected_ticker)
with tab3:
    render_analytics(model_comparison, training_summary)
with tab4:
    render_news(fda_events)
with tab5:
    render_trading(selected_ticker, backtest_df)
with tab6:
    render_copilot(selected_ticker, ticker_events, backtest_df, training_summary)
