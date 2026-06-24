import os
import pandas as pd
import numpy as np
from preprocessing.text_preprocessor import TextPreprocessor
from sentiment.analyzer import BiotechSentimentAnalyzer

class FeatureEngineer:
    """
    Combines stock technical metrics with NLP sentiments and FDA event indicators
    to build training and testing matrices for XGBoost and LSTM.
    """
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.preprocessor = TextPreprocessor()
        # Enable domain-specific transformers pipeline for rich sentiment features
        self.analyzer = BiotechSentimentAnalyzer(use_transformers=True)

    def compute_technical_indicators(self, stock_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes financial technical indicators from raw stock price:
        - SMA_10, SMA_30 (Simple Moving Averages)
        - Volatility (10-day rolling standard deviation of close returns)
        - Momentum (Close price divided by close price 5 days ago)
        - Volume_Ratio (Daily volume divided by 10-day average volume)
        - RSI (14-day Relative Strength Index)
        - MACD, MACD_Signal, MACD_Hist (Moving Average Convergence Divergence)
        - BB_High, BB_Low (Bollinger Bands)
        """
        df = stock_df.copy()
        
        # Sort index just in case
        df.sort_index(inplace=True)
        
        # Calculate daily returns
        df['Daily_Return'] = df['Close'].pct_change()
        
        # Moving averages
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_30'] = df['Close'].rolling(window=30).mean()
        
        # Rolling Volatility
        df['Volatility'] = df['Daily_Return'].rolling(window=10).std()
        
        # Momentum (5-day return)
        df['Momentum'] = df['Close'] / df['Close'].shift(5) - 1.0
        
        # Volume relative ratio
        df['Vol_MA_10'] = df['Volume'].rolling(window=10).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Vol_MA_10']
        
        # RSI (14-day)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD (12, 26, 9)
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands (20-day, 2 std)
        sma_20 = df['Close'].rolling(window=20).mean()
        std_20 = df['Close'].rolling(window=20).std()
        df['BB_High'] = sma_20 + 2 * std_20
        df['BB_Low'] = sma_20 - 2 * std_20
        
        # Forward fill and backward fill any NaNs from rolling windows
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        
        return df

    def aggregate_nlp_features(self, events_list: list, dates_index: pd.Index) -> pd.DataFrame:
        """
        Processes text events, runs sentiment analysis, and aggregates features for each date.
        Returns a DataFrame indexed by date with aggregated sentiment scores and event flags.
        """
        # Create empty DataFrame with same dates as stock data
        nlp_df = pd.DataFrame(index=dates_index)
        nlp_df['Avg_Sentiment'] = 0.0
        nlp_df['Sentiment_Confidence'] = 0.0
        nlp_df['FDA_Approval_Count'] = 0.0
        nlp_df['Trial_Success_Count'] = 0.0
        nlp_df['Trial_Failure_Count'] = 0.0
        nlp_df['Drug_Recall_Count'] = 0.0
        nlp_df['Safety_Warning_Count'] = 0.0
        nlp_df['Event_Severity_Sum'] = 0.0
        
        # Group events by date
        date_groups = {}
        for event in events_list:
            date = event["date"]
            if date not in date_groups:
                date_groups[date] = []
            date_groups[date].append(event)
            
        # Analyze and aggregate per date
        for date in dates_index:
            if date not in date_groups:
                continue
                
            day_events = date_groups[date]
            sentiments = []
            confidences = []
            approvals = 0
            successes = 0
            failures = 0
            recalls = 0
            warnings = 0
            severity_sum = 0.0
            
            for event in day_events:
                # Preprocess text
                cleaned_text = self.preprocessor.preprocess(event["text"])
                
                # Analyze sentiment
                sentiment = self.analyzer.analyze_sentiment(cleaned_text)
                
                # Convert sentiment label to score (-1 for negative, 0 for neutral, 1 for positive)
                sent_score = 0.0
                if sentiment["label"] == "positive":
                    sent_score = 1.0
                elif sentiment["label"] == "negative":
                    sent_score = -1.0
                    
                sentiments.append(sent_score)
                confidences.append(sentiment["score"])
                
                # Detect events
                event_flags = self.analyzer.detect_events(cleaned_text)
                approvals += event_flags["fda_approval"]
                successes += event_flags["trial_success"]
                failures += event_flags["trial_failure"]
                recalls += event_flags["drug_recall"]
                warnings += event_flags["safety_warning"]
                severity_sum += event_flags["event_severity"]
                
            nlp_df.loc[date, 'Avg_Sentiment'] = np.mean(sentiments) if sentiments else 0.0
            nlp_df.loc[date, 'Sentiment_Confidence'] = np.mean(confidences) if confidences else 0.0
            nlp_df.loc[date, 'FDA_Approval_Count'] = approvals
            nlp_df.loc[date, 'Trial_Success_Count'] = successes
            nlp_df.loc[date, 'Trial_Failure_Count'] = failures
            nlp_df.loc[date, 'Drug_Recall_Count'] = recalls
            nlp_df.loc[date, 'Safety_Warning_Count'] = warnings
            nlp_df.loc[date, 'Event_Severity_Sum'] = severity_sum
            
        # Apply exponential rolling window to NLP features to allow news effects to persist over a few days
        # A clinical trial success or FDA approval doesn't just affect 1 day; it lingers.
        rolling_cols = [
            'Avg_Sentiment', 'FDA_Approval_Count', 'Trial_Success_Count', 
            'Trial_Failure_Count', 'Drug_Recall_Count', 'Safety_Warning_Count', 
            'Event_Severity_Sum'
        ]
        
        # Smooth using a rolling 3-day exponential moving window
        nlp_df[rolling_cols] = nlp_df[rolling_cols].ewm(span=3, adjust=False).mean()
        
        return nlp_df

    def create_labels(self, stock_df: pd.DataFrame, horizon: int = 5) -> pd.Series:
        """
        Creates target labels based on stock returns over 'horizon' days.
        Label definitions:
        - BUY (1): Return > 2%
        - SELL (-1): Return < -2%
        - HOLD (0): Return between -2% and 2%
        """
        # Calculate percent return over the next 'horizon' days
        future_return = stock_df['Close'].shift(-horizon) / stock_df['Close'] - 1.0
        
        # Map to classes
        labels = pd.Series(0, index=stock_df.index)
        labels[future_return > 0.02] = 1   # BUY
        labels[future_return < -0.02] = -1 # SELL
        
        # Fill final 'horizon' days (where we don't have future prices) with NaNs so they can be dropped
        labels.iloc[-horizon:] = np.nan
        
        return labels

    def build_merged_features(self, ticker: str, events_list: list) -> pd.DataFrame:
        """
        Builds a complete, labeled, merged feature matrix for a specific ticker.
        """
        stock_file = os.path.join(self.raw_dir, f"{ticker}_stock.csv")
        if not os.path.exists(stock_file):
            raise FileNotFoundError(f"Stock data for {ticker} not found. Please run loader first.")
            
        stock_df = pd.read_csv(stock_file, index_col='Date')
        
        # 1. Technical indicators
        tech_df = self.compute_technical_indicators(stock_df)
        
        # Filter events for this ticker
        ticker_events = [e for e in events_list if e["ticker"] == ticker]
        
        # 2. Aggregated NLP features
        nlp_df = self.aggregate_nlp_features(ticker_events, tech_df.index)
        
        # 3. Combine Technical and NLP Features
        merged = pd.concat([tech_df, nlp_df], axis=1)
        
        # Add lag features (returns, sentiment, severity)
        for lag in [1, 2, 3]:
            merged[f'Daily_Return_Lag_{lag}'] = merged['Daily_Return'].shift(lag)
            merged[f'Avg_Sentiment_Lag_{lag}'] = merged['Avg_Sentiment'].shift(lag)
            merged[f'Event_Severity_Lag_{lag}'] = merged['Event_Severity_Sum'].shift(lag)
            
        merged.ffill(inplace=True)
        merged.bfill(inplace=True)
        
        # 4. Generate Target labels
        merged['Target'] = self.create_labels(merged, horizon=5)
        
        # Drop rows with NaNs in Target (the last few days)
        merged = merged.dropna(subset=['Target'])
        
        return merged

    def prepare_data_for_xgboost(self, merged_df: pd.DataFrame, use_nlp: bool = True):
        """
        Prepares training arrays (X, y) for XGBoost.
        Optionally excludes NLP features to train a 'Generic stock-only' comparison model.
        """
        df = merged_df.copy()
        
        # Define features
        tech_features = [
            'Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return', 
            'SMA_10', 'SMA_30', 'Volatility', 'Momentum', 'Volume_Ratio',
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'BB_High', 'BB_Low',
            'Daily_Return_Lag_1', 'Daily_Return_Lag_2', 'Daily_Return_Lag_3'
        ]
        nlp_features = [
            'Avg_Sentiment', 'Sentiment_Confidence', 'FDA_Approval_Count', 'Trial_Success_Count', 
            'Trial_Failure_Count', 'Drug_Recall_Count', 'Safety_Warning_Count', 'Event_Severity_Sum',
            'Avg_Sentiment_Lag_1', 'Avg_Sentiment_Lag_2', 'Avg_Sentiment_Lag_3',
            'Event_Severity_Lag_1', 'Event_Severity_Lag_2', 'Event_Severity_Lag_3'
        ]
        
        features = tech_features + nlp_features if use_nlp else tech_features
        
        # Convert Target from [-1, 0, 1] to [0, 1, 2] since XGBoost multi-classifier requires 0-indexed non-negative integers
        y = df['Target'].map({-1: 0, 0: 1, 1: 2}).values
        X = df[features].values
        
        return X, y, features

    def prepare_data_for_lstm(self, merged_df: pd.DataFrame, time_steps: int = 10, use_nlp: bool = True):
        """
        Creates sequences of length 'time_steps' for LSTM inputs.
        Returns shapes: X: (samples, time_steps, features), y: (samples,)
        """
        df = merged_df.copy()
        
        tech_features = [
            'Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return', 
            'SMA_10', 'SMA_30', 'Volatility', 'Momentum', 'Volume_Ratio',
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'BB_High', 'BB_Low',
            'Daily_Return_Lag_1', 'Daily_Return_Lag_2', 'Daily_Return_Lag_3'
        ]
        nlp_features = [
            'Avg_Sentiment', 'Sentiment_Confidence', 'FDA_Approval_Count', 'Trial_Success_Count', 
            'Trial_Failure_Count', 'Drug_Recall_Count', 'Safety_Warning_Count', 'Event_Severity_Sum',
            'Avg_Sentiment_Lag_1', 'Avg_Sentiment_Lag_2', 'Avg_Sentiment_Lag_3',
            'Event_Severity_Lag_1', 'Event_Severity_Lag_2', 'Event_Severity_Lag_3'
        ]
        
        features = tech_features + nlp_features if use_nlp else tech_features
        
        # Standardize features using simple z-score (LSTM is sensitive to feature scale)
        for col in features:
            mean = df[col].mean()
            std = df[col].std() if df[col].std() > 0 else 1.0
            df[col] = (df[col] - mean) / std
            
        X_seq, y_seq = [], []
        
        # Convert labels from [-1, 0, 1] to [0, 1, 2]
        y_vals = df['Target'].map({-1: 0, 0: 1, 1: 2}).values
        X_vals = df[features].values
        
        for i in range(len(df) - time_steps):
            X_seq.append(X_vals[i : i + time_steps])
            y_seq.append(y_vals[i + time_steps])
            
        return np.array(X_seq), np.array(y_seq), features

if __name__ == "__main__":
    import json
    from data.loader import BiotechDataLoader
    
    # Run loader to verify we have raw files
    loader = BiotechDataLoader()
    events = loader.fetch_openfda_data()
    stocks = loader.fetch_stock_data()
    
    # Test feature engineer
    fe = FeatureEngineer()
    merged = fe.build_merged_features("MRNA", events)
    print("\nFeature Engineering Verification:")
    print(f"Merged features shape for MRNA: {merged.shape}")
    print(f"Columns: {list(merged.columns)}")
    
    # XGBoost shape
    X_xgb, y_xgb, feat_xgb = fe.prepare_data_for_xgboost(merged, use_nlp=True)
    print(f"XGBoost X: {X_xgb.shape}, y: {y_xgb.shape}")
    
    # LSTM shape
    X_lstm, y_lstm, feat_lstm = fe.prepare_data_for_lstm(merged, time_steps=10, use_nlp=True)
    print(f"LSTM X: {X_lstm.shape}, y: {y_lstm.shape}")
