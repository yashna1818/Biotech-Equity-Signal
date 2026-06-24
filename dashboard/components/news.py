import streamlit as st
import pandas as pd
import plotly.graph_objects as go


def render_news(fda_events: list):
    st.markdown("### 📰 Regulatory Intelligence Feed")
    st.caption("Real-time FDA regulatory event tracking with FinBERT sentiment classification and impact scoring.")

    if not fda_events:
        st.info("No regulatory events found.")
        return

    events_df = pd.DataFrame(fda_events)

    # Ensure required columns
    if 'event_type' not in events_df.columns:
        events_df['event_type'] = 'Regulatory Action'
    if 'sentiment' not in events_df.columns:
        events_df['sentiment'] = 'neutral'

    events_df['sentiment'] = events_df['sentiment'].fillna('neutral').astype(str).str.lower()
    events_df['date'] = pd.to_datetime(events_df['date'])
    events_df = events_df.sort_values('date', ascending=False)

    # ── KPI Row ──────────────────────────────────────────────────────────────
    total = len(events_df)
    pos_count = (events_df['sentiment'] == 'positive').sum()
    neg_count = (events_df['sentiment'] == 'negative').sum()
    neu_count = total - pos_count - neg_count
    tickers_covered = events_df['ticker'].nunique() if 'ticker' in events_df.columns else '—'

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Events", total)
    k2.metric("🟢 Positive", pos_count, f"{pos_count/total*100:.0f}%")
    k3.metric("🔴 Negative", neg_count, f"{neg_count/total*100:.0f}%")
    k4.metric("🟡 Neutral", neu_count, f"{neu_count/total*100:.0f}%")
    k5.metric("Tickers Covered", tickers_covered)

    st.markdown("---")

    # ── Filters ──────────────────────────────────────────────────────────────
    fc1, fc2, fc3 = st.columns([2, 2, 2])
    with fc1:
        ticker_options = ["All"] + sorted(events_df['ticker'].unique().tolist()) if 'ticker' in events_df.columns else ["All"]
        sel_ticker = st.selectbox("Filter by Ticker", ticker_options, key="news_ticker_filter")
    with fc2:
        sel_sentiment = st.selectbox("Filter by Sentiment", ["All", "Positive", "Negative", "Neutral"], key="news_sent_filter")
    with fc3:
        max_show = st.slider("Events to display", min_value=5, max_value=min(50, total), value=min(20, total), key="news_count")

    filtered = events_df.copy()
    if sel_ticker != "All":
        filtered = filtered[filtered['ticker'] == sel_ticker]
    if sel_sentiment != "All":
        filtered = filtered[filtered['sentiment'] == sel_sentiment.lower()]

    st.markdown(f"**Showing {min(max_show, len(filtered))} of {len(filtered)} events**")
    st.markdown("---")

    # ── Sentiment Timeline Mini-Chart ─────────────────────────────────────────
    with st.expander("📊 Sentiment Timeline", expanded=False):
        chart_df = events_df.copy()
        chart_df['date_str'] = chart_df['date'].dt.strftime('%Y-%m-%d')
        chart_df['impact'] = chart_df['sentiment'].map({'positive': 1, 'neutral': 0, 'negative': -1})
        daily = chart_df.groupby('date_str')['impact'].sum().reset_index()
        colors = ['#22C55E' if v > 0 else ('#EF4444' if v < 0 else '#94A3B8') for v in daily['impact']]
        fig = go.Figure(go.Bar(
            x=daily['date_str'], y=daily['impact'],
            marker_color=colors,
            hovertemplate='<b>%{x}</b><br>Net Sentiment: %{y}<extra></extra>'
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#FFFFFF',
            height=200, margin=dict(l=0, r=0, t=10, b=10),
            yaxis=dict(showgrid=True, gridcolor='#F1F5F9', zeroline=True, zerolinecolor='#CBD5E1'),
            xaxis=dict(showgrid=False, nticks=10, tickfont=dict(size=9)),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True, key="news_chart")

    # ── Event Cards ───────────────────────────────────────────────────────────
    for _, row in filtered.head(max_show).iterrows():
        sent = str(row.get('sentiment', 'neutral')).lower()

        if sent == "positive":
            icon = "✅"
            badge_color = "#22C55E"
            badge_bg = "#F0FDF4"
            badge_text = "POSITIVE"
            impact = 25
            border_color = "#86EFAC"
        elif sent == "negative":
            icon = "🔴"
            badge_color = "#EF4444"
            badge_bg = "#FEF2F2"
            badge_text = "NEGATIVE"
            impact = 85
            border_color = "#FCA5A5"
        else:
            icon = "🟡"
            badge_color = "#F59E0B"
            badge_bg = "#FFFBEB"
            badge_text = "NEUTRAL"
            impact = 50
            border_color = "#FCD34D"

        event_type = str(row.get('event_type', 'General'))
        ticker = str(row.get('ticker', '—'))
        date_str = row['date'].strftime('%d %b %Y')
        text = str(row.get('text', ''))

        # Impact bar fill
        bar_pct = impact
        bar_color = badge_color

        st.markdown(f"""
        <div style="
            background: {badge_bg};
            border: 1px solid {border_color};
            border-left: 5px solid {badge_color};
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 12px;
        ">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:16px;">
                <div style="flex:1;">
                    <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px; flex-wrap:wrap;">
                        <span style="
                            background:{badge_color}; color:#fff;
                            font-size:10px; font-weight:700;
                            padding:2px 10px; border-radius:20px;
                            letter-spacing:0.5px;
                        ">{badge_text}</span>
                        <span style="
                            background:#E2E8F0; color:#475569;
                            font-size:10px; font-weight:600;
                            padding:2px 10px; border-radius:20px;
                        ">{ticker}</span>
                        <span style="
                            background:#EFF6FF; color:#0F6CBD;
                            font-size:10px; font-weight:600;
                            padding:2px 10px; border-radius:20px;
                        ">{event_type}</span>
                        <span style="font-size:11px; color:#94A3B8;">📅 {date_str}</span>
                    </div>
                    <div style="font-size:14px; color:#1E293B; font-weight:500; line-height:1.5;">
                        {icon} {text}
                    </div>
                </div>
                <div style="text-align:center; min-width:80px;">
                    <div style="font-size:11px; color:#64748B; font-weight:600; margin-bottom:4px;">Impact</div>
                    <div style="font-size:22px; font-weight:700; color:{badge_color};">{impact}</div>
                    <div style="font-size:9px; color:#94A3B8;">/100</div>
                    <div style="
                        height:4px; background:#E2E8F0; border-radius:2px; margin-top:6px;
                        width:60px; position:relative; overflow:hidden;
                    ">
                        <div style="
                            height:100%; width:{bar_pct}%; background:{bar_color};
                            border-radius:2px; position:absolute; left:0;
                        "></div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
