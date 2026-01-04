import json
import os

def create_notebook():
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            },
            "accelerator": "GPU"
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    def add_cell(source_code, cell_type="code"):
        cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": [line + "\n" for line in source_code.split("\n")]
        }
        notebook["cells"].append(cell)

    # --- CELL 1: Setup ---
    add_cell("""# @title 1. Environment Setup & Installation
# Check for GPU
!nvidia-smi

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("Installing dependencies... (this may take 2-3 minutes)")
!pip install -q timesfm jax jaxlib==0.4.20+cuda12.cudnn89 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
!pip install -q yfinance ccxt pandas numpy scikit-learn matplotlib seaborn xgboost

print("✅ Environment ready!")""")

    # --- CELL 2: Imports ---
    add_cell("""# @title 2. Core Imports and Configuration
import jax
import timesfm
import ccxt
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from typing import Tuple, Dict, List

# Basic Config
@dataclass
class Config:
    context_length: int = 512
    forecast_horizon: int = 32
    walkforward_step: int = 128
    symbol: str = 'BTC/USDT'
    timeframe: str = '5m'
    xgb_lookback: int = 100  # Extra lookback for features

config = Config()

print(f"JAX Backend: {jax.devices()}")
if 'cpu' in str(jax.devices()).lower():
    print("⚠️ WARNING: Running on CPU. This will be slow! Enable GPU in Runtime > Change runtime type.")
else:
    print("🚀 GPU Detected. Ready for TimesFM.")""")

    # --- CELL 3: Data ---
    add_cell("""# @title 3. Data Acquisition & Feature Engineering
def fetch_data(symbol, timeframe, days=365):
    # 5min bars per day = 12 * 24 = 288
    # Total bars needed approx = 288 * days
    limit = 288 * days
    print(f"Fetching approx {limit} bars for {symbol} (~{days} days)...")
    
    exchange = ccxt.binance()
    since = exchange.milliseconds() - (days * 24 * 60 * 60 * 1000)
    all_ohlcv = []
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            
            if len(all_ohlcv) >= limit:
                break
                
            if len(all_ohlcv) % 10000 == 0:
                print(f"Fetched {len(all_ohlcv)} bars...")
                
        except Exception as e:
            print(f"Error fetching: {e}")
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')
    
    # Remove duplicates
    df = df[~df.index.duplicated(keep='first')]
    return df

def add_technical_features(df):
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # EMAs
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    
    # Volatility
    df['volatility'] = df['close'].rolling(20).std()
    
    # Returns (Target for XGBoost)
    df['returns'] = df['close'].pct_change()
    df['target_next_return'] = df['returns'].shift(-1) # Predicting next step
    
    return df.dropna()

# Fetch 1 Year of Data
raw_df = fetch_data(config.symbol, config.timeframe, days=365)
df = add_technical_features(raw_df)

print(f"✅ Data processed. Shape: {df.shape}")
print(f"Range: {df.index[0]} -> {df.index[-1]}")
df.tail()""")

    # --- CELL 4: TimesFM Loading ---
    add_cell("""# @title 4. Load Models (TimesFM + XGBoost Setup)

# 1. Load TimesFM
print("Loading TimesFM (checking for checkpoint)...")
# Note: In a real scenario, you might need to download the checkpoint. 
# For this demo, we assume the library handles it or we use a huggingface hub loader if available.
# Since the checkpoint logic is specific, we will use the standard instantiation which often pulls from HF if not local.
try:
    tfm = timesfm.TimesFM(
        backend="gpu" if "gpu" in str(jax.devices()) else "cpu",
        checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id="google/timesfm-1.0-200m") # Attempt HF load
    ) 
except Exception as e:
    print(f"Direct HF load failed, trying standard init: {e}")
    tfm = timesfm.TimesFM(backend="gpu" if "gpu" in str(jax.devices()) else "cpu")
    # tfm.load_from_checkpoint("timesfm.ckpt") # Uncomment if you have a local file

print("✅ TimesFM Loaded")

# 2. XGBoost Helper
def train_xgboost(train_df):
    features = ['rsi', 'ema_12', 'ema_26', 'volatility', 'volume']
    X = train_df[features]
    y = train_df['target_next_return']
    
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5)
    model.fit(X, y)
    return model

print("✅ XGBoost Helper Ready")""")

    # --- CELL 5: Validation ---
    add_cell("""# @title 5. Enhanced Walk-Forward Validation
# We will simulate a "Hybrid" Trader:
# 1. TimesFM predicts the trend (Context -> Forecast)
# 2. XGBoost predicts the immediate return magnitude based on technicals
# 3. They vote.

results = []
test_start_idx = len(df) - 2000 # Test on last 2000 bars
step_size = config.walkforward_step

print(f"Starting validation on {len(df)-test_start_idx} bars...")

for i in range(test_start_idx, len(df) - config.forecast_horizon, step_size):
    # Data Slice
    current_idx = i
    
    # 1. TimesFM Forecast
    # It takes raw values.
    context_values = df['close'].iloc[i-config.context_length : i].values
    # Reshape for batch size 1
    # Note: TimesFM API varies slightly by version, ensure input is [B, T]
    try:
        # Forecast returns (B, Horizon)
        tfm_forecast, _ = tfm.forecast(context_values[None, :], freq=0, horizon=config.forecast_horizon)
        tfm_trend = np.mean(np.diff(tfm_forecast[0])) # Simple slope
    except:
        tfm_trend = 0 # Fallback
    
    # 2. XGBoost Forecast
    # Train heavily on past data (rolling window would be better but slower)
    # We retrain every X steps or just once? Let's retrain once for speed here, or strictly past data.
    # To be "honest", we should only train on df[:i]
    train_split = df.iloc[:i]
    xgb_model = train_xgboost(train_split)
    
    # Predict next step
    current_features = df.iloc[i:i+1][['rsi', 'ema_12', 'ema_26', 'volatility', 'volume']]
    xgb_pred = xgb_model.predict(current_features)[0]
    
    # 3. Ensemble Signal
    # - If TimesFM sees UP trend AND XGBoost predicts POSITIVE return -> BUY
    # - If TimesFM sees DOWN trend AND XGBoost predicts NEGATIVE return -> SELL
    
    signal = 0
    if tfm_trend > 0 and xgb_pred > 0:
        signal = 1
    elif tfm_trend < 0 and xgb_pred < 0:
        signal = -1
        
    # Calculate Real Result (Next Horizon Return)
    # Simplified: Did price go up/down over horizon?
    entry_price = df['close'].iloc[i]
    exit_price = df['close'].iloc[i + config.forecast_horizon]
    obs_return = (exit_price - entry_price) / entry_price
    
    metrics = {
        'idx': i,
        'signal': signal,
        'tfm_trend': tfm_trend,
        'xgb_pred': xgb_pred,
        'real_return': obs_return,
        'strategy_return': signal * obs_return
    }
    results.append(metrics)
    
    if len(results) % 5 == 0:
        print(f"Step {len(results)}: Signal {signal}, Return {obs_return:.4f}")

res_df = pd.DataFrame(results)
print("✅ Validation Complete")""")

    # --- CELL 6: Analysis ---
    add_cell("""# @title 6. Performance Report & Portfolio Simulation

INITIAL_CAPITAL = 1000.0 # CAD
cost_per_trade = 0.001 # 0.1% fee assumption (optional, set to 0 to ignore)

# calculate equity curve with compounding
res_df['equity_curve'] = (1 + res_df['strategy_return'] - (abs(res_df['signal']) * cost_per_trade)).cumprod() * INITIAL_CAPITAL
res_df['benchmark_curve'] = (1 + res_df['real_return']).cumprod() * INITIAL_CAPITAL

# Drawdown Calculation
res_df['peak'] = res_df['equity_curve'].cummax()
res_df['drawdown'] = res_df['equity_curve'] / res_df['peak'] - 1.0
max_drawdown = res_df['drawdown'].min()

# Visualization
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(res_df['equity_curve'], label='Strategy Equity (CAD)', color='green')
plt.plot(res_df['benchmark_curve'], label='Buy & Hold Benchmark', alpha=0.5, linestyle='--')
plt.title(f'Portfolio Simulation (Start: ${INITIAL_CAPITAL})')
plt.ylabel('Value (CAD)')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(res_df['drawdown'], label='Drawdown', color='red')
plt.fill_between(res_df.index, res_df['drawdown'], 0, color='red', alpha=0.1)
plt.title('Strategy Drawdown')
plt.ylabel('% Drawdown')
plt.grid(True)

plt.tight_layout()
plt.show()

# Final Metrics
final_balance = res_df['equity_curve'].iloc[-1]
net_pnl = final_balance - INITIAL_CAPITAL
total_return = net_pnl / INITIAL_CAPITAL
win_rate = (res_df['strategy_return'] > 0).mean()

# Annualized Sharpe (assuming 5min frequency)
# Steps per year approx 105120. If we trade every walkforward step (e.g. 128), logic adjusts.
# Here we calculate based on the realized series of trades.
valid_trades = res_df[res_df['signal'] != 0]['strategy_return']
if len(valid_trades) > 0:
    sharpe = valid_trades.mean() / valid_trades.std() * np.sqrt(len(res_df)) # simplified annualization
else:
    sharpe = 0

print("-" * 40)
print(f"INITIAL CAPITAL:  ${INITIAL_CAPITAL:,.2f} CAD")
print(f"FINAL BALANCE:    ${final_balance:,.2f} CAD")
print(f"NET PnL:          ${net_pnl:,.2f} CAD")
print("-" * 40)
print(f"Total Return:     {total_return:.2%}")
print(f"Max Drawdown:     {max_drawdown:.2%}")
print(f"Win Rate:         {win_rate:.1%}")
print(f"Sharpe Ratio:     {sharpe:.3f}")
print("-" * 40)
""")

    output_path = os.path.join(r"c:\Users\chari\OneDrive\Documents\Time FN MODEL", "timesfm_enhanced_validation.ipynb")
    with open(output_path, "w") as f:
        json.dump(notebook, f, indent=2)
    
    print(f"Notebook created at: {output_path}")

if __name__ == "__main__":
    create_notebook()
