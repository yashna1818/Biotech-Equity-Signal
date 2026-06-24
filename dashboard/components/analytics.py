import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def plot_confusion_matrix(conf_matrix, title="Confusion Matrix"):
    labels = ["SELL", "HOLD", "BUY"]
    z = conf_matrix
    z_reversed = list(reversed(z))
    y_labels = list(reversed(labels))
    
    fig = go.Figure(data=go.Heatmap(
        z=z_reversed,
        x=labels,
        y=y_labels,
        colorscale="Blues",
        showscale=False,
        text=z_reversed,
        texttemplate="%{text}",
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#1E293B"), x=0.5, xanchor="center"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=40, r=40, t=40, b=40),
        height=280,
        xaxis=dict(title="Predicted", tickfont=dict(color="#1E293B")),
        yaxis=dict(title="True", tickfont=dict(color="#1E293B")),
    )
    return fig

def render_analytics(model_comparison: dict, training_summary: dict):
    st.markdown("### 🤖 AI Research & Model Confidence")
    st.caption("Scientific evaluation of prediction history, feature weighting, and FinBERT/Lexicon NLP analysis.")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("#### Scientific Model Validation")
        xgb_f1 = training_summary["xgb_metrics"]["f1_score"]
        lstm_f1 = training_summary["lstm_metrics"]["f1_score"]
        xgb_acc = training_summary["xgb_metrics"]["accuracy"]
        lstm_acc = training_summary["lstm_metrics"]["accuracy"]
        
        metrics_data = {
            "Algorithm": ["XGBoost (FinBERT/Lexicon + Tech)", "LSTM (Sequential Array)", "Baseline (No NLP)"],
            "Validation Accuracy": [f"{xgb_acc:.4f}", f"{lstm_acc:.4f}", f"{model_comparison['Generic_Baseline']['Accuracy']:.4f}"],
            "F1 Confidence": [f"{xgb_f1:.4f}", f"{lstm_f1:.4f}", f"{model_comparison['Generic_Baseline']['F1_Score']:.4f}"]
        }
        st.dataframe(pd.DataFrame(metrics_data), hide_index=True, key="model_metrics")

    with col_right:
        st.markdown("#### Backtest Results (vs Buy & Hold)")
        comparison_summary = training_summary["backtest_summary"]
        
        ticker_compare_list = []
        for tick, val in comparison_summary.items():
            ticker_compare_list.append({
                "Asset": tick,
                "Buy & Hold": f"{val['Buy_Hold_Return'] * 100:.2f}%",
                "AI Strategy": f"{val['Strategy_Return'] * 100:.2f}%",
                "Alpha Generated": f"{val['Outperformance'] * 100:.2f}%"
            })
        st.dataframe(pd.DataFrame(ticker_compare_list), hide_index=True, key="backtest_results")

    st.markdown("---")
    st.markdown("#### Scientific Confusion Matrices")
    st.caption("Detailed true positive, false positive, and false negative splits for XGBoost and LSTM classifiers.")
    
    # Render heatmaps side-by-side
    col_xgb, col_lstm = st.columns(2)
    with col_xgb:
        fig_xgb = plot_confusion_matrix(training_summary["xgb_metrics"]["confusion_matrix"], "XGBoost Confusion Matrix")
        st.plotly_chart(fig_xgb, key="xgb_conf_chart", use_container_width=True)
        
    with col_lstm:
        fig_lstm = plot_confusion_matrix(training_summary["lstm_metrics"]["confusion_matrix"], "LSTM Confusion Matrix")
        st.plotly_chart(fig_lstm, key="lstm_conf_chart", use_container_width=True)

    st.markdown("---")
    st.markdown("#### SHAP Feature Importance")
    st.caption("Attribution of scientific, regulatory, and technical features to the ensemble decision boundary.")
    
    features = training_summary["feature_names"]
    importances = training_summary["feature_importances"]
    
    feat_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values("Importance", ascending=True)
    
    fig3 = go.Figure(go.Bar(
        x=feat_df['Importance'],
        y=feat_df['Feature'],
        orientation='h',
        marker_color='#14B8A6'
    ))
    fig3.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=0, r=0, t=10, b=0),
        height=400,
        xaxis=dict(showgrid=True, gridcolor="#F1F5F9"),
        yaxis=dict(tickfont=dict(size=11, color="#1E293B"))
    )
    st.plotly_chart(fig3, key="shap_chart", use_container_width=True)
