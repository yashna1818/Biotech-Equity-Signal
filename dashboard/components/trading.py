import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from data.loader import TICKERS

def render_trading(selected_ticker: str, backtest_df: pd.DataFrame):
    st.markdown(f"### 📊 Simulated Paper Trading: {selected_ticker}")
    st.caption("Tracking simulated portfolio performance based on FinBERT + Lexicon NLP-Enhanced signals.")
    
    # Conversion rate: 1 USD = 83.5 INR
    USD_TO_INR = 83.5
    initial_capital_inr = 100000.0 * USD_TO_INR # ₹83,50,000.00
    
    # Calculate single asset portfolio values in INR
    backtest_df['Portfolio_Value_INR'] = initial_capital_inr * (1 + backtest_df['Cum_Strategy'])
    backtest_df['BH_Value_INR'] = initial_capital_inr * (1 + backtest_df['Cum_Buy_Hold'])
    
    current_val = backtest_df['Portfolio_Value_INR'].iloc[-1]
    bh_val = backtest_df['BH_Value_INR'].iloc[-1]
    
    # ── Multi-Asset Catalyst-Weighted Allocation Optimizer ────────────────────
    backtests = {}
    for t in TICKERS:
        path = f"data/processed/{t}_backtest.csv"
        if os.path.exists(path):
            backtests[t] = pd.read_csv(path, index_col='Date')
            
    current_multi_val = None
    if len(backtests) == len(TICKERS):
        pos_dict = {}
        ret_dict = {}
        common_idx = backtest_df.index
        for t in TICKERS:
            # Shift single asset position and align
            df = backtests[t]
            # Handle possible missing column index alignments safely
            pos_col = df['Position'] if 'Position' in df.columns else (df['Pred_Action'] == 'BUY').astype(float)
            pos_dict[t] = pos_col.reindex(common_idx).fillna(0.0)
            ret_dict[t] = df['Daily_Return'].reindex(common_idx).fillna(0.0)
            
        pos_df = pd.DataFrame(pos_dict)
        ret_df = pd.DataFrame(ret_dict)
        
        # Calculate capital weights decided at day t-1 and held during day t
        pos_sum_shifted = pos_df.shift(1).sum(axis=1)
        # Allocate capital equally among all active BUY recommendations (or hold Cash if sum = 0)
        weights = pos_df.shift(1).div(pos_sum_shifted.replace(0, np.nan), axis=0).fillna(0.0)
        
        multi_daily_ret = (weights * ret_df).sum(axis=1)
        multi_cum_ret = (1 + multi_daily_ret).cumprod() - 1.0
        
        backtest_df['Multi_Asset_Value_INR'] = initial_capital_inr * (1 + multi_cum_ret)
        current_multi_val = backtest_df['Multi_Asset_Value_INR'].iloc[-1]
        
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-title">{selected_ticker} Portfolio</div>
            <div class="metric-value">₹{current_val:,.2f}</div>
            <span style="color: {'#22C55E' if current_val >= initial_capital_inr else '#EF4444'}; font-size: 13px; font-weight: 600;">
                {((current_val/initial_capital_inr)-1)*100:+.2f}% Return
            </span>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        if current_multi_val is not None:
            st.markdown(f"""
            <div class="glass-card">
                <div class="metric-title">Ensemble Multi-Asset</div>
                <div class="metric-value" style="color: #0B5C9C;">₹{current_multi_val:,.2f}</div>
                <span style="color: {'#22C55E' if current_multi_val >= initial_capital_inr else '#EF4444'}; font-size: 13px; font-weight: 600;">
                    {((current_multi_val/initial_capital_inr)-1)*100:+.2f}% Return
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="glass-card">
                <div class="metric-title">Ensemble Multi-Asset</div>
                <div class="metric-value">N/A</div>
                <span style="color: #64748B; font-size: 13px;">No data loaded</span>
            </div>
            """, unsafe_allow_html=True)
            
    with col3:
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-title">Buy & Hold Benchmark</div>
            <div class="metric-value">₹{bh_val:,.2f}</div>
            <span style="color: #64748B; font-size: 13px; font-weight: 500;">
                {((bh_val/initial_capital_inr)-1)*100:+.2f}% Return
            </span>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        if current_multi_val is not None:
            multi_outperformance = current_multi_val - bh_val
            pct_multi_outperformance = ((current_multi_val - bh_val) / initial_capital_inr) * 100
            st.markdown(f"""
            <div class="glass-card">
                <div class="metric-title">Ensemble Multi-Alpha</div>
                <div class="metric-value" style="color: {'#22C55E' if multi_outperformance >= 0 else '#EF4444'};">₹{multi_outperformance:+,.2f}</div>
                <span style="color: {'#22C55E' if multi_outperformance >= 0 else '#EF4444'}; font-size: 13px; font-weight: 600;">
                    {pct_multi_outperformance:+.2f}% vs B&H
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            outperformance = current_val - bh_val
            st.markdown(f"""
            <div class="glass-card">
                <div class="metric-title">Alpha Generated</div>
                <div class="metric-value" style="color: {'#22C55E' if outperformance >= 0 else '#EF4444'};">₹{outperformance:+,.2f}</div>
                <span style="color: #64748B; font-size: 13px;">Single Asset</span>
            </div>
            """, unsafe_allow_html=True)

    # Plot
    st.markdown("#### Profit & Loss Curve (₹ INR)")
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=backtest_df.index, y=backtest_df['Portfolio_Value_INR'], 
        name=f'{selected_ticker} Strategy', 
        line=dict(color='#0F6CBD', width=2.5)
    ))
    
    if current_multi_val is not None:
        fig.add_trace(go.Scatter(
            x=backtest_df.index, y=backtest_df['Multi_Asset_Value_INR'], 
            name='Ensemble Multi-Asset Portfolio', 
            line=dict(color='#008080', width=3)
        ))
        
    fig.add_trace(go.Scatter(
        x=backtest_df.index, y=backtest_df['BH_Value_INR'], 
        name=f'{selected_ticker} Buy & Hold Benchmark', 
        line=dict(color='#94A3B8', width=1.8, dash='dash')
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=0, r=0, t=20, b=0),
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.01),
        xaxis=dict(showgrid=False, tickfont=dict(color="#1E293B")),
        yaxis=dict(showgrid=True, gridcolor="#F1F5F9", tickprefix="₹", tickfont=dict(color="#1E293B"))
    )
    st.plotly_chart(fig, use_container_width=True, key="pnl_chart")
    
    # Trade History Table
    st.markdown("#### Recent Signal Ledger")
    ledger_df = backtest_df[['Close', 'Daily_Return', 'Pred_Action']].tail(10)[::-1].copy()
    
    # Convert Close to INR
    ledger_df['Close'] = ledger_df['Close'] * USD_TO_INR
    ledger_df.rename(columns={'Close': 'Close (₹)', 'Daily_Return': 'Daily Return'}, inplace=True)
    
    def color_action(val):
        if val == 'BUY':
            return 'background-color: rgba(34, 197, 94, 0.15); color: #22C55E; font-weight: 600;'
        elif val == 'SELL':
            return 'background-color: rgba(239, 68, 68, 0.15); color: #EF4444; font-weight: 600;'
        return 'background-color: rgba(148, 163, 184, 0.15); color: #94A3B8;'
        
    styled = ledger_df.style.format({
        'Close (₹)': '₹{:,.2f}',
        'Daily Return': '{:+.2%}'
    }).map(color_action, subset=['Pred_Action'])
    st.dataframe(styled, use_container_width=True, key="signal_ledger_table")
