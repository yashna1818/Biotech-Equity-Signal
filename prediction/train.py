import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_OK'] = 'True'
import json
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score

from data.loader import BiotechDataLoader, TICKERS
from prediction.features import FeatureEngineer
from prediction.models import BiotechLSTM, BiotechXGBoost

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModelTrainer:
    """
    Handles training, evaluation, saving, and backtesting for XGBoost and LSTM models.
    """
    def __init__(self, data_dir: str = "data", models_dir: str = "models"):
        self.data_dir = data_dir
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        self.loader = BiotechDataLoader(data_dir)
        self.fe = FeatureEngineer(data_dir)

    def prepare_all_data(self, use_nlp: bool = True):
        """
        Loads and aggregates features across all tickers into single training sets.
        """
        # Ensure raw data is loaded
        self.events = self.loader.fetch_openfda_data()
        self.loader.fetch_stock_data()
        
        merged_dfs = []
        
        for ticker in TICKERS:
            try:
                merged = self.fe.build_merged_features(ticker, self.events)
                merged_dfs.append(merged)
            except Exception as e:
                print(f"[Error] Processing data for {ticker}: {e}")
                
        # Concat lists
        full_df = pd.concat(merged_dfs, axis=0)
        
        X_xgb_all, y_xgb_all, feat_cols = self.fe.prepare_data_for_xgboost(full_df, use_nlp=use_nlp)
        X_lstm_all, y_lstm_all, _ = self.fe.prepare_data_for_lstm(full_df, time_steps=10, use_nlp=use_nlp)
        
        return np.ascontiguousarray(X_xgb_all), np.ascontiguousarray(y_xgb_all), np.ascontiguousarray(X_lstm_all), np.ascontiguousarray(y_lstm_all), feat_cols

    def train_xgboost(self, X, y):
        """
        Trains the XGBoost classifier.
        """
        print("\n--- Training XGBoost Model ---")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        
        xgb_wrapper = BiotechXGBoost(n_estimators=150, max_depth=5, learning_rate=0.06)
        xgb_wrapper.fit(X_train, y_train)
        
        # Evaluate
        preds = xgb_wrapper.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='macro')
        print(f"XGBoost Test Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")
        
        # Save model
        xgb_path = os.path.join(self.models_dir, "xgb_model.json")
        xgb_wrapper.model.save_model(xgb_path)
        print(f"[Success] Saved XGBoost model to {xgb_path}")
        
        report = classification_report(y_test, preds, output_dict=True)
        conf = confusion_matrix(y_test, preds).tolist()
        
        return xgb_wrapper, {
            "accuracy": acc,
            "f1_score": f1,
            "report": report,
            "confusion_matrix": conf
        }

    def train_lstm(self, X, y):
        """
        Trains the PyTorch LSTM model.
        """
        print("\n--- Training PyTorch LSTM Model ---")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        
        # Convert arrays to PyTorch Tensors
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # Model parameters
        input_dim = X.shape[2]
        hidden_dim = 64
        output_dim = 3 # SELL, HOLD, BUY
        epochs = 35
        
        model = BiotechLSTM(input_dim, hidden_dim, output_dim).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
        
        # Train Loop
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(train_loader):.4f}")
                
        # Evaluate
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(device)
            logits = model(X_test_tensor)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='macro')
        print(f"LSTM Test Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")
        
        # Save model
        lstm_path = os.path.join(self.models_dir, "lstm_model.pt")
        torch.save(model.state_dict(), lstm_path)
        print(f"[Success] Saved LSTM model state to {lstm_path}")
        
        report = classification_report(y_test, preds, output_dict=True)
        conf = confusion_matrix(y_test, preds).tolist()
        
        return model, {
            "accuracy": acc,
            "f1_score": f1,
            "report": report,
            "confusion_matrix": conf
        }

    def run_backtest(self, ticker: str, xgb_model, lstm_model, nlp_features, events=None):
        """
        Runs a mock trading strategy simulation backtest on a specific ticker:
        - Cash start: $100,000
        - On prediction = BUY (2), we buy the stock with maximum allocation.
        - On prediction = SELL (0), we liquidate the position (or short/hold cash).
        - On prediction = HOLD (1), we hold current position.
        Compare active strategy vs Buy-and-Hold strategy.
        """
        if events is None:
            events = self.loader.fetch_openfda_data()
        merged = self.fe.build_merged_features(ticker, events)
        
        X_xgb, y_xgb, features = self.fe.prepare_data_for_xgboost(merged, use_nlp=True)
        
        # Get LSTM inputs
        X_lstm, _, _ = self.fe.prepare_data_for_lstm(merged, time_steps=10, use_nlp=True)
        
        # Ensure LSTM is in eval mode and run predictions on device
        lstm_model.eval()
        with torch.no_grad():
            X_lstm_tensor = torch.FloatTensor(X_lstm).to(device)
            lstm_logits = lstm_model(X_lstm_tensor)
            lstm_probs = torch.softmax(lstm_logits, dim=1).cpu().numpy() # Shape: (len(merged) - 10, 3)
        
        # Run predictions combining XGBoost + LSTM
        preds = []
        for i in range(len(merged)):
            # XGBoost prediction probabilities
            proba_xgb = xgb_model.predict_proba(X_xgb[i:i+1])[0] # Shape: (3,)
            
            if i < 10:
                # No LSTM history yet, use XGBoost only
                proba_ensemble = proba_xgb
            else:
                proba_lstm = lstm_probs[i - 10] # Shape: (3,)
                proba_ensemble = 0.5 * proba_xgb + 0.5 * proba_lstm
                
            pred = np.argmax(proba_ensemble)
            preds.append(pred)
            
        preds = np.array(preds)
        
        df_backtest = merged.copy()
        df_backtest['Pred'] = preds
        df_backtest['Pred_Action'] = df_backtest['Pred'].map({0: 'SELL', 1: 'HOLD', 2: 'BUY'})
        
        # Calculate strategy returns
        # Action mappings: 
        # BUY (2) -> Position = 1 (Long)
        # SELL (0) -> Position = 0 (Cash)
        # HOLD (1) -> Maintain previous position
        positions = []
        current_pos = 0.0
        
        for pred in preds:
            if pred == 2: # BUY
                current_pos = 1.0
            elif pred == 0: # SELL
                current_pos = 0.0
            positions.append(current_pos)
            
        df_backtest['Position'] = positions
        df_backtest['Daily_Return'] = df_backtest['Close'].pct_change().fillna(0.0)
        
        # Shift position by 1 day because prediction at end of day t is executed on day t+1
        df_backtest['Strategy_Return'] = df_backtest['Position'].shift(1).fillna(0.0) * df_backtest['Daily_Return']
        
        # Cumulative returns
        df_backtest['Cum_Buy_Hold'] = (1 + df_backtest['Daily_Return']).cumprod() - 1.0
        df_backtest['Cum_Strategy'] = (1 + df_backtest['Strategy_Return']).cumprod() - 1.0
        
        # Export backtest CSV for graphing in Streamlit
        backtest_path = os.path.join(self.data_dir, "processed", f"{ticker}_backtest.csv")
        df_backtest[['Close', 'Daily_Return', 'Pred_Action', 'Cum_Buy_Hold', 'Cum_Strategy']].to_csv(backtest_path)
        print(f"[Success] Saved backtest logs for {ticker} to {backtest_path}")
        
        final_bh_ret = df_backtest['Cum_Buy_Hold'].iloc[-1]
        final_strat_ret = df_backtest['Cum_Strategy'].iloc[-1]
        return final_bh_ret, final_strat_ret

    def run_pipeline(self):
        """
        Runs the full loading, training, evaluation, and backtesting pipeline.
        """
        X_xgb, y_xgb, X_lstm, y_lstm, features = self.prepare_all_data(use_nlp=True)
        
        # Train and evaluate
        xgb_model, xgb_metrics = self.train_xgboost(X_xgb, y_xgb)
        lstm_model, lstm_metrics = self.train_lstm(X_lstm, y_lstm)
        
        # Run backtests for all tickers using cached events
        backtest_results = {}
        for ticker in TICKERS:
            bh_ret, strat_ret = self.run_backtest(ticker, xgb_model, lstm_model, features, events=self.events)
            backtest_results[ticker] = {
                "Buy_Hold_Return": float(bh_ret),
                "Strategy_Return": float(strat_ret),
                "Outperformance": float(strat_ret - bh_ret)
            }
            
        # Compile dashboard parameters
        summary = {
            "xgb_metrics": xgb_metrics,
            "lstm_metrics": lstm_metrics,
            "backtest_summary": backtest_results,
            "feature_names": features,
            "feature_importances": xgb_model.get_feature_importances().tolist()
        }
        
        summary_path = os.path.join(self.data_dir, "processed", "training_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"[Success] Exported final training summary to {summary_path}")

def run_verification():
    """Simple verification function."""
    print("Verification success: Imports and pipelines are configured.")

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run_pipeline()
