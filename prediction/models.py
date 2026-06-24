import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_OK'] = 'True'
import xgboost as xgb
import torch
import torch.nn as nn
import numpy as np

# --------------------------------------------------------
# 1. PYTORCH LSTM TIME-SERIES MODEL
# --------------------------------------------------------
class BiotechLSTM(nn.Module):
    """
    LSTM (Long Short-Term Memory) network for stock direction classification.
    Inputs are sequential stock features (length = time_steps, dimensions = num_features).
    Outputs are probability logits for 3 classes: SELL (0), HOLD (1), BUY (2).
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 3, num_layers: int = 2, dropout: float = 0.2):
        super(BiotechLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM Layer
        # batch_first=True means input tensor shape is (batch, seq, feature)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Fully Connected Output Layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate through LSTM
        # out shape: (batch_size, seq_len, hidden_dim)
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        # out[:, -1, :] takes the final output of the sequence
        out = self.fc(out[:, -1, :])
        return out


# --------------------------------------------------------
# 2. XGBOOST TABULAR CLASSIFIER
# --------------------------------------------------------
class BiotechXGBoost:
    """
    Wrapper for XGBoost Classifier to simplify training, prediction,
    and evaluation procedures.
    """
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.05, random_state=42):
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            eval_metric="mlogloss",
            use_label_encoder=False,
            reg_alpha=0.1,
            reg_lambda=1.0,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=1
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Trains the XGBoost model on the tabular features.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts classes: 0 (SELL), 1 (HOLD), 2 (BUY).
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts probability scores for each class.
        """
        return self.model.predict_proba(X)
        
    def get_feature_importances(self):
        """
        Returns feature importances.
        """
        return self.model.feature_importances_
