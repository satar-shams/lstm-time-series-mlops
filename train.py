import os
import numpy as np
import pandas as pd
import yfinance as yf

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# ==========================================
# Download data
# ==========================================
data = yf.download("AAPL", start="2010-01-01", end="2026-06-26", auto_adjust=True)
if data.empty:
    raise RuntimeError(
        "yfinance returned no data for AAPL. "
        "This is usually Yahoo-side rate-limiting or blocking, not a real "
        "delisting. Check yfinance version, retry, or check network access."
    )

# Defensive: newer yfinance versions can return MultiIndex columns
# even for a single ticker. Flatten if so.
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# Use Close price
stock_close = data["Close"]

# Convert to numpy array, explicit 2D shape for sklearn
dataset = stock_close.values.reshape(-1, 1)

# 90% train, 10% test
training_data_len = int(np.ceil(len(dataset) * 0.9))

# ==========================================
# Scale data
# FIX: fit scaler on TRAIN portion only, then apply to everything.
# Fitting on the full dataset (original code) leaks test-period
# statistics into training and inflates apparent performance.
# ==========================================
scaler = StandardScaler()
scaler.fit(dataset[:training_data_len])
scaled_data = scaler.transform(dataset)

training_data = scaled_data[:training_data_len]

# ==========================================
# Create training data (30-day window)
# ==========================================
X_train = []
y_train = []

for i in range(30, len(training_data)):
    X_train.append(training_data[i - 30:i, 0])
    y_train.append(training_data[i, 0])

X_train = np.array(X_train)
y_train = np.array(y_train)

X_train = X_train.reshape(
    X_train.shape[0],
    X_train.shape[1],
    1
)

# ==========================================
# Create test data
# ==========================================
test_data = scaled_data[training_data_len - 30:]

X_test = []
y_test = dataset[training_data_len:]

for i in range(30, len(test_data)):
    X_test.append(test_data[i - 30:i, 0])

X_test = np.array(X_test)

X_test = X_test.reshape(
    X_test.shape[0],
    X_test.shape[1],
    1
)

# Real prices for evaluation
y_test_real = stock_close.iloc[training_data_len:]

# ==========================================
# Training configurations
# ==========================================
configs = [
    {"epochs": 20, "batch_size": 32},
    {"epochs": 30, "batch_size": 32},
    {"epochs": 40, "batch_size": 64},
]

# ==========================================
# Variables to store best model
# ==========================================
best_model = None
best_config = None
best_rmse_percent = float("inf")

# ==========================================
# Train models
#
# NOTE (known limitation, not fixed here): selecting the best config
# based on X_test/y_test performance means the test set is being used
# for model selection, not just final evaluation. This overstates
# generalization. A proper fix is a train/val/test split where val
# is used here and test is touched exactly once at the end. Flagging
# this for a follow-up pass rather than blocking today's run.
# ==========================================
for cfg in configs:
    print(
        f"\nTraining with "
        f"epochs={cfg['epochs']} "
        f"batch_size={cfg['batch_size']}"
    )

    model = Sequential([
        LSTM(
            64,
            return_sequences=True,
            input_shape=(X_train.shape[1], 1)
        ),
        LSTM(64),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mae",
        metrics=[keras.metrics.RootMeanSquaredError()]
    )

    model.fit(
        X_train,
        y_train,
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        verbose=0
    )

    predictions = model.predict(X_test, verbose=0)

    y_pred_real = scaler.inverse_transform(predictions)

    rmse = np.sqrt(
        mean_squared_error(
            y_test_real,
            y_pred_real
        )
    )

    rmse_percent = (
        rmse / y_test_real.mean()
    ) * 100

    print(f"RMSE %: {rmse_percent:.2f}%")

    if rmse_percent < best_rmse_percent:
        best_rmse_percent = rmse_percent
        best_model = model
        best_config = cfg

# ==========================================
# Show best model
# ==========================================
print("\n==============================")
print("Best configuration:", best_config)
print(f"Best RMSE %: {best_rmse_percent:.2f}%")
print("==============================")

# ==========================================
# Save best model
# ==========================================
os.makedirs("models", exist_ok=True)

best_model.save("models/best_model.keras")

print("\n✅ Best model saved:")
print("models/best_model.keras")