import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_OK'] = 'True'
import json
import argparse
import xgboost as xgb
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from data.loader import BiotechDataLoader
from prediction.features import FeatureEngineer
from prediction.models import BiotechXGBoost

class ModelComparator:
    """
    Compares prediction performance between:
    - Generic Baseline Model (Stock technicals only)
    - Domain-Specific NLP Model (Stock technicals + FDA events + sentiment)
    """
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.fe = FeatureEngineer(data_dir)
        self.loader = BiotechDataLoader(data_dir)

    def run_comparison(self, tickers=["MRNA", "PFE", "BNTX", "NVAX", "GILD", "AMGN"]):
        print("\n=== STARTING COMPARATIVE NLP PERFORMANCE TEST ===")
        events = self.loader.fetch_openfda_data()
        
        # Lists to store data for all tickers
        merged_dfs = []
        for ticker in tickers:
            try:
                merged = self.fe.build_merged_features(ticker, events)
                merged_dfs.append(merged)
            except Exception as e:
                print(f"[Error] Merging features for {ticker}: {e}")
                
        if not merged_dfs:
            raise ValueError("No data could be merged. Please run loader first.")
            
        full_df = pd.concat(merged_dfs, axis=0)
        
        # 1. Prepare data for GENERIC baseline (NLP features excluded)
        X_generic, y_generic, _ = self.fe.prepare_data_for_xgboost(full_df, use_nlp=False)
        X_g_train, X_g_test, y_g_train, y_g_test = train_test_split(
            X_generic, y_generic, test_size=0.30, random_state=42, stratify=y_generic
        )
        
        # 2. Prepare data for DOMAIN-SPECIFIC model (NLP features included)
        X_domain, y_domain, _ = self.fe.prepare_data_for_xgboost(full_df, use_nlp=True)
        X_d_train, X_d_test, y_d_train, y_d_test = train_test_split(
            X_domain, y_domain, test_size=0.30, random_state=42, stratify=y_domain
        )
        
        # Train Generic Baseline
        generic_model = BiotechXGBoost(n_estimators=150, max_depth=5, learning_rate=0.06)
        generic_model.fit(X_g_train, y_g_train)
        g_preds = generic_model.predict(X_g_test)
        
        # Train Domain-Specific
        domain_model = BiotechXGBoost(n_estimators=150, max_depth=5, learning_rate=0.06)
        domain_model.fit(X_d_train, y_d_train)
        d_preds = domain_model.predict(X_d_test)
        
        # Compute metrics
        metrics = {
            "Generic_Baseline": {
                "Accuracy": float(accuracy_score(y_g_test, g_preds)),
                "Precision": float(precision_score(y_g_test, g_preds, average='macro', zero_division=0)),
                "Recall": float(recall_score(y_g_test, g_preds, average='macro', zero_division=0)),
                "F1_Score": float(f1_score(y_g_test, g_preds, average='macro', zero_division=0))
            },
            "Domain_Specific": {
                "Accuracy": float(accuracy_score(y_d_test, d_preds)),
                "Precision": float(precision_score(y_d_test, d_preds, average='macro', zero_division=0)),
                "Recall": float(recall_score(y_d_test, d_preds, average='macro', zero_division=0)),
                "F1_Score": float(f1_score(y_d_test, d_preds, average='macro', zero_division=0))
            }
        }
        
        # Print results
        print("\nResults summary:")
        for name, data in metrics.items():
            print(f"\n{name} Model:")
            for metric, val in data.items():
                print(f"  - {metric}: {val:.4f}")
                
        # Save results
        out_path = os.path.join(self.data_dir, "processed", "model_comparison.json")
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"\n[Success] Comparison metrics exported to {out_path}")
        
        return metrics

if __name__ == "__main__":
    comparator = ModelComparator()
    comparator.run_comparison()
