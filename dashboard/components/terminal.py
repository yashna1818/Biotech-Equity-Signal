import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Pipeline data mapping ticker to its drug details for rendering phase bars
pipeline_indicators = {
    "MRNA": [
        {"drug": "mRNA-1273.815", "target": "COVID-19 Booster", "phase": 3, "prob": "85%"},
        {"drug": "mRNA-1345", "target": "RSV Vaccine", "phase": 4, "prob": "90%"},
        {"drug": "mRNA-4157", "target": "Melanoma (Oncology)", "phase": 2, "prob": "45%"},
        {"drug": "mRNA-3927", "target": "Propionic Acidemia", "phase": 1, "prob": "15%"}
    ],
    "PFE": [
        {"drug": "Comirnaty Booster", "target": "COVID-19 Vaccine", "phase": 3, "prob": "75%"},
        {"drug": "Paxlovid IV", "target": "COVID-19 Treatment", "phase": 2, "prob": "60%"},
        {"drug": "OncoShield", "target": "Oncology", "phase": 1, "prob": "15%"}
    ],
    "BNTX": [
        {"drug": "BNT162b2", "target": "mRNA Vaccine", "phase": 3, "prob": "85%"},
        {"drug": "CAR-T-900", "target": "Solid Tumors", "phase": 2, "prob": "35%"},
        {"drug": "LipoVacc", "target": "Cancer Immunotherapy", "phase": 1, "prob": "20%"}
    ],
    "NVAX": [
        {"drug": "NVX-CoV2373", "target": "COVID-19 Vaccine", "phase": 3, "prob": "80%"},
        {"drug": "FluNano", "target": "Nano-flu Vaccine", "phase": 2, "prob": "50%"},
        {"drug": "ComboVax", "target": "COVID + Flu Combo", "phase": 1, "prob": "25%"}
    ],
    "GILD": [
        {"drug": "Veklury (IV)", "target": "Antiviral COVID-19", "phase": 3, "prob": "95%"},
        {"drug": "Lenacapavir", "target": "HIV-1 Prevention", "phase": 3, "prob": "70%"},
        {"drug": "Trodelvy", "target": "Triple-Negative Breast Cancer", "phase": 2, "prob": "55%"}
    ],
    "AMGN": [
        {"drug": "Repatha", "target": "Cardiovascular", "phase": 3, "prob": "90%"},
        {"drug": "Tepezza", "target": "Thyroid Eye Disease", "phase": 3, "prob": "80%"},
        {"drug": "Lumakras", "target": "NSCLC (Oncology)", "phase": 2, "prob": "40%"}
    ]
}

def render_terminal(selected_ticker: str, stock_df: pd.DataFrame, backtest_df: pd.DataFrame, ticker_events: list):
    
    col_left, col_center, col_right = st.columns([1, 2.5, 1])
    
    # -----------------------------------------------------------
    # LEFT PANEL: Clinical Pipeline Overview
    # -----------------------------------------------------------
    with col_left:
        st.markdown("#### 🧬 Clinical Pipeline")
        st.caption("Active target indications and trial phases.")
        
        pipelines = pipeline_indicators.get(selected_ticker, [
            {"drug": "Asset-Alpha", "target": "Oncology", "phase": 3, "prob": "75%"},
            {"drug": "Asset-Beta", "target": "Immunology", "phase": 2, "prob": "40%"}
        ])
        
        for p in pipelines:
            p_val = p['phase']
            # Build phase indicator using Streamlit columns instead of raw HTML
            st.markdown(f"**{p['drug']}** — *{p['target']}*")
            
            phase_labels = ["Phase I", "Phase II", "Phase III", "FDA Review"]
            phase_cols = st.columns(4)
            for i, pc in enumerate(phase_cols):
                filled = (i + 1) <= p_val
                color = "#14B8A6" if (i == 3 and filled) else ("#0F6CBD" if filled else "#E2E8F0")
                pc.markdown(
                    f'<div style="height:6px;background:{color};border-radius:3px;"></div>'
                    f'<div style="font-size:9px;color:#94A3B8;text-align:center;margin-top:2px;">{phase_labels[i]}</div>',
                    unsafe_allow_html=True
                )
            
            st.caption(f"Probability of Approval: **{p['prob']}**")
            st.markdown("---")

    # -----------------------------------------------------------
    # CENTER PANEL: Primary Intelligence Workspace
    # -----------------------------------------------------------
    with col_center:
        st.markdown("#### 📈 Price Action & Clinical Events")
        
        chart_df = stock_df.reset_index()
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.75, 0.25])
        
        # Area chart
        fig.add_trace(go.Scatter(
            x=chart_df['Date'], y=chart_df['Close'],
            fill='tozeroy', mode='lines',
            line=dict(color='#0F6CBD', width=2),
            fillcolor='rgba(15, 108, 189, 0.08)',
            name='Close Price'
        ), row=1, col=1)
        
        # Volume bars
        vol_colors = ['#EF4444' if row['Open'] > row['Close'] else '#22C55E' for _, row in chart_df.iterrows()]
        fig.add_trace(go.Bar(
            x=chart_df['Date'], y=chart_df['Volume'],
            marker_color=vol_colors, name='Volume', opacity=0.5
        ), row=2, col=1)
                        
        # Overlay events
        event_x, event_y, event_text, event_color = [], [], [], []
        for event in ticker_events:
            date_str = event["date"]
            if date_str in stock_df.index:
                event_x.append(date_str)
                event_y.append(stock_df.loc[date_str, 'Close'])
                event_text.append(event["text"])
                sent = str(event.get("sentiment", "neutral")).lower()
                c = '#22C55E' if sent == "positive" else ('#EF4444' if sent == "negative" else '#F59E0B')
                event_color.append(c)

        if event_x:
            fig.add_trace(go.Scatter(
                x=event_x, y=event_y, mode='markers', name='Clinical Events',
                marker=dict(color=event_color, size=9, symbol='diamond',
                            line=dict(color='#FFFFFF', width=1.5)),
                text=event_text,
                hovertemplate="<b>%{x}</b><br>%{text}<extra></extra>"
            ), row=1, col=1)
            
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF",
            margin=dict(l=10, r=10, t=10, b=10), height=380,
            xaxis=dict(showgrid=False, type='category', nticks=8, tickfont=dict(size=10)),
            xaxis2=dict(showgrid=False, type='category', nticks=8, tickfont=dict(size=10)),
            yaxis=dict(showgrid=True, gridcolor="#F1F5F9", side="right", tickfont=dict(size=10)),
            yaxis2=dict(showgrid=False, showticklabels=False),
            showlegend=False
        )
        st.plotly_chart(fig, key="main_chart", use_container_width=True)
        
        # Prediction Timeline
        st.markdown("#### 🤖 AI Prediction Timeline")
        pred_df = backtest_df[['Pred_Action', 'Daily_Return']].tail(25)
        
        timeline_cols = st.columns(len(pred_df))
        for i, (_, row) in enumerate(pred_df.iterrows()):
            act = row['Pred_Action']
            color = "#22C55E" if act == "BUY" else ("#EF4444" if act == "SELL" else "#CBD5E1")
            timeline_cols[i].markdown(
                f'<div style="height:20px;background:{color};border-radius:2px;" title="{act}"></div>',
                unsafe_allow_html=True
            )

    # -----------------------------------------------------------
    # RIGHT PANEL: AI Research Insights
    # -----------------------------------------------------------
    with col_right:
        st.markdown("#### 🔬 AI Research Insights")
        
        # Extract dynamic insights
        pos_factors = [e["text"] for e in ticker_events if e.get("sentiment") == "positive"]
        neg_factors = [e["text"] for e in ticker_events if e.get("sentiment") == "negative"]
        
        # Subsample to keep it compact
        pos_list = [f"- {text}" for text in pos_factors[-2:]] if pos_factors else ["- Stable base assets and strong market penetration."]
        neg_list = [f"- {text}" for text in neg_factors[-2:]] if neg_factors else ["- Generic competitor entry risk.", "- General market volatility."]
        
        # Catalysts from pipeline
        ticker_pipeline = pipeline_indicators.get(selected_ticker, [])
        catalysts = []
        for p in ticker_pipeline:
            if p["phase"] == 4:
                catalysts.append(f"- PDUFA target date for {p['drug']}.")
            elif p["phase"] == 3:
                catalysts.append(f"- Phase III readout for {p['drug']} ({p['target']}).")
        if not catalysts:
            catalysts = [f"- Upcoming clinical trial readout.", f"- Sector regulatory guidance review."]
        else:
            catalysts = catalysts[:2]
            
        st.markdown("**:green[Key Positive Factors]**")
        for item in pos_list:
            st.markdown(item)
        
        st.markdown("**:red[Key Negative Factors]**")
        for item in neg_list:
            st.markdown(item)
        
        st.markdown("**:blue[Upcoming Catalysts]**")
        for item in catalysts:
            st.markdown(item)
        
        st.markdown("---")
        st.markdown("**Model Explanation**")
        st.caption("The ensemble model converges the tabular XGBoost classifier (weight: 0.50) and the sequential LSTM time-series model (weight: 0.50), smoothing transient technical movements with structured clinical NLP sentiment.")
