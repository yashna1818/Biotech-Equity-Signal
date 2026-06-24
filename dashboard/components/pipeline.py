import streamlit as st
import pandas as pd

def render_pipeline(selected_ticker: str):
    st.markdown(f"### 🧪 Clinical Pipeline: {selected_ticker}")
    st.caption("Tracking clinical trial progress, enrollment metrics, and regulatory milestones.")
    
    pipelines = {
        "MRNA": [
            {"Drug": "mRNA-1273.815", "Indication": "COVID-19 Booster", "Phase": "Phase III", "Enrollment": "3,400 / 3,400", "Status": "Active", "Prob_Approval": "85%", "Target_Date": "2026-11-15"},
            {"Drug": "mRNA-1345", "Indication": "RSV Vaccine", "Phase": "FDA Review", "Enrollment": "37,000 / 37,000", "Status": "PDUFA", "Prob_Approval": "90%", "Target_Date": "2026-05-12"},
            {"Drug": "mRNA-4157", "Indication": "Melanoma (Oncology)", "Phase": "Phase IIb", "Enrollment": "157 / 200", "Status": "Recruiting", "Prob_Approval": "45%", "Target_Date": "2027-02-28"},
            {"Drug": "mRNA-3927", "Indication": "Propionic Acidemia", "Phase": "Phase I", "Enrollment": "15 / 50", "Status": "Recruiting", "Prob_Approval": "15%", "Target_Date": "2028-09-01"}
        ],
        "PFE": [
            {"Drug": "Comirnaty Booster", "Indication": "COVID-19 Vaccine", "Phase": "Phase III", "Enrollment": "14,000 / 20,000", "Status": "Recruiting", "Prob_Approval": "75%", "Target_Date": "2027-04-15"},
            {"Drug": "Paxlovid IV", "Indication": "COVID-19 Treatment", "Phase": "Phase II", "Enrollment": "500 / 500", "Status": "Active", "Prob_Approval": "60%", "Target_Date": "2026-08-10"},
            {"Drug": "OncoShield", "Indication": "Breast Cancer / Oncology", "Phase": "Phase I", "Enrollment": "45 / 50", "Status": "Active", "Prob_Approval": "15%", "Target_Date": "2028-05-01"}
        ],
        "BNTX": [
            {"Drug": "BNT162b2", "Indication": "mRNA Vaccine", "Phase": "Phase III", "Enrollment": "15,000 / 15,000", "Status": "Active", "Prob_Approval": "85%", "Target_Date": "2026-12-01"},
            {"Drug": "CAR-T-900", "Indication": "Solid Tumors", "Phase": "Phase II", "Enrollment": "120 / 150", "Status": "Recruiting", "Prob_Approval": "35%", "Target_Date": "2027-06-30"},
            {"Drug": "LipoVacc", "Indication": "Cancer Immunotherapy", "Phase": "Phase I", "Enrollment": "30 / 60", "Status": "Active", "Prob_Approval": "20%", "Target_Date": "2028-03-15"}
        ],
        "NVAX": [
            {"Drug": "NVX-CoV2373", "Indication": "COVID-19 Vaccine", "Phase": "Phase III", "Enrollment": "8,000 / 8,000", "Status": "Active", "Prob_Approval": "80%", "Target_Date": "2026-10-10"},
            {"Drug": "FluNano", "Indication": "Nano-flu Vaccine", "Phase": "Phase IIb", "Enrollment": "800 / 1,000", "Status": "Recruiting", "Prob_Approval": "50%", "Target_Date": "2027-04-20"},
            {"Drug": "ComboVax", "Indication": "COVID + Flu Combo", "Phase": "Phase I", "Enrollment": "60 / 100", "Status": "Recruiting", "Prob_Approval": "25%", "Target_Date": "2028-01-15"}
        ],
        "GILD": [
            {"Drug": "Veklury (IV)", "Indication": "Antiviral COVID-19", "Phase": "Phase III", "Enrollment": "1,200 / 1,200", "Status": "Active", "Prob_Approval": "95%", "Target_Date": "2026-09-01"},
            {"Drug": "Lenacapavir", "Indication": "HIV-1 Prevention", "Phase": "Phase III", "Enrollment": "4,000 / 5,000", "Status": "Recruiting", "Prob_Approval": "70%", "Target_Date": "2027-03-10"},
            {"Drug": "Trodelvy", "Indication": "Triple-Negative Breast Cancer", "Phase": "Phase IIb", "Enrollment": "250 / 300", "Status": "Recruiting", "Prob_Approval": "55%", "Target_Date": "2027-11-20"}
        ],
        "AMGN": [
            {"Drug": "Repatha", "Indication": "Cardiovascular / Cholesterol", "Phase": "Phase III", "Enrollment": "2,500 / 2,500", "Status": "Active", "Prob_Approval": "90%", "Target_Date": "2026-07-15"},
            {"Drug": "Tepezza", "Indication": "Thyroid Eye Disease", "Phase": "Phase III", "Enrollment": "1,800 / 2,000", "Status": "Recruiting", "Prob_Approval": "80%", "Target_Date": "2027-01-30"},
            {"Drug": "Lumakras", "Indication": "NSCLC / Oncology", "Phase": "Phase IIb", "Enrollment": "180 / 200", "Status": "Recruiting", "Prob_Approval": "40%", "Target_Date": "2027-09-15"}
        ],
        "DEFAULT": [
            {"Drug": "Asset-A", "Indication": "Oncology", "Phase": "Phase II", "Enrollment": "120 / 150", "Status": "Recruiting", "Prob_Approval": "30%", "Target_Date": "2027-01-01"},
            {"Drug": "Asset-B", "Indication": "Immunology", "Phase": "Phase I", "Enrollment": "45 / 50", "Status": "Active", "Prob_Approval": "10%", "Target_Date": "2028-05-01"}
        ]
    }
    
    data = pipelines.get(selected_ticker, pipelines["DEFAULT"])
    df = pd.DataFrame(data)
    
    def color_prob(val):
        try:
            num = int(val.replace('%', ''))
            if num >= 70: return 'color: #22C55E; font-weight: 600;'
            if num <= 30: return 'color: #EF4444; font-weight: 600;'
            return 'color: #F59E0B; font-weight: 600;'
        except:
            return ''
            
    styled = df.style.map(color_prob, subset=['Prob_Approval'])
    
    st.dataframe(styled, hide_index=True, key="pipeline_table")
