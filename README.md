# Biotech Equity Signal Detection Using Domain-Specific NLP and Regulatory Event Analysis

An end-to-end machine learning and NLP platform designed to forecast short-term stock price movements for biotechnology equities. The platform leverages Yahoo Finance stock metrics, financial sentiment (via FinBERT/local lexicons), and regulatory announcements parsed from OpenFDA API (or custom simulated logs) to classify signals into **BUY**, **SELL**, or **HOLD**.

---

## 🧬 Project Overview
Biotech stock prices are famously sensitive to binary regulatory outcomes, such as FDA drug approvals, clinical trial phase achievements, and safety warnings. This project implements a system to show that:
> **"Domain-specific NLP and FDA regulatory event analysis improve biotech stock prediction accuracy compared to generic financial sentiment analysis systems."**

### Core Components
1. **Data Collection Module**: Retrieves historical stock metrics via `yfinance` and parses drug recall/approval records using the OpenFDA API.
2. **Text Preprocessor**: Normalizes biomedical terms (e.g., standardizing trial phases: "Phase III", "p3", "Phase 3" -> `phase_3`) and cleans text.
3. **Sentiment & Event Engine**: Classifies financial text sentiment and detects specific events (FDA approvals, trial success/failures, drug recalls, safety warnings) with their severity.
4. **Feature Engineer**: Merges raw stock metrics (momentum, volatility, moving averages) with smoothed NLP event indicators.
5. **Predictive Models**: Trains and evaluates an **XGBoost Classifier** (for tabular feature vectors) and a **PyTorch LSTM** (for sequential 10-day history tracking).
6. **Streamlit Dashboard**: A high-end interactive UI with custom glassmorphic styling, live stock charts, regulatory event overlays, and a comparative metrics visualizer.

---

## 📂 Project Structure
```
/Users/yashna/FINANCEEL/
│
├── requirements.txt                   # Project package dependencies
├── README.md                          # Set up and running instructions
│
├── data/                              # Data directory
│   ├── raw/                           # CSV files from yfinance & raw FDA json logs
│   └── processed/                     # Merged features and backtesting metrics
│
├── preprocessing/                     # NLP preprocessing module
│   ├── __init__.py
│   └── text_preprocessor.py           # Text cleaners and term standardizer
│
├── sentiment/                         # Sentiment and event parsing
│   ├── __init__.py
│   └── analyzer.py                    # FinBERT model loader & lexicon fallbacks
│
├── prediction/                        # Machine learning & pipelines
│   ├── __init__.py
│   ├── features.py                    # Multi-source feature merge & labeling
│   ├── models.py                      # XGBoost and LSTM architectures
│   ├── train.py                       # Trainer, backtester, and weight exporter
│   └── compare.py                     # Performance test (Generic vs NLP-enhanced)
│
└── dashboard/                         # Streamlit dashboard interface
    ├── __init__.py
    └── app.py                         # Interactive dashboard
```

---

## ⚙️ Setup and Installation

### 1. Prerequisites
Make sure you have Python 3.8+ installed. We use `uv` for lightning-fast package management.

### 2. Set Up Virtual Environment & Install Dependencies
Run the following commands in the project root:
```bash
# Create the virtual environment
uv venv

# Install dependencies
uv pip install -r requirements.txt
```

---

## 🚀 Running the Project

### 1. Run the Data & Model Pipeline
This script downloads stock data, generates/fetches regulatory events, merges features, trains models, runs backtests, and saves all outputs:
```bash
PYTHONPATH=. .venv/bin/python prediction/train.py
```
This will train the models and save the evaluation metrics under `data/processed/training_summary.json`.

### 2. Run the Comparison Experiment
Run the baseline comparative test to verify the value of domain-specific NLP:
```bash
PYTHONPATH=. .venv/bin/python prediction/compare.py
```
This saves comparison results to `data/processed/model_comparison.json`.

### 3. Launch the Interactive Dashboard
Launch the Streamlit app to view the charts and test live inputs:
```bash
.venv/bin/streamlit run dashboard/app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🔬 Model Definitions & Methodology

### Labeling
- **BUY (2)**: Close price increases by $>2\%$ over the next 5 days.
- **SELL (0)**: Close price decreases by $<-2\%$ over the next 5 days.
- **HOLD (1)**: Close price remains stable within the $[-2\%, +2\%]$ range.

### Feature Vectors
- **Technicals**: Open, High, Low, Close, Volume, Daily Return, SMA_10, SMA_30, Volatility, Momentum, Volume Ratio.
- **NLP Events**: Sentiment polarity, sentiment confidence, count of FDA approvals, trial successes, trial failures, recalls, and warnings.

---

## 💡 Notes on Offline Execution & Model Fallbacks
To ensure the system works locally and under any network restrictions:
- **Lexicon Fallback**: If `transformers` or the `ProsusAI/finbert` model weights cannot be downloaded, a customized biotech-financial rule-based dictionary analyzer automatically runs.
- **Mock Generators**: If Yahoo Finance or the OpenFDA API are blocked or slow, the loader generates realistic mock stock trends and events. Approvals and trial outcomes are synced to match corresponding stock price movements for demonstrative integrity.
