import json
import os

def create_notebook():
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.0"},
            "accelerator": "GPU"
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    def add_code_cell(source_code):
        notebook["cells"].append({
            "cell_type": "code",
            "metadata": {},
            "source": [line + "\n" for line in source_code.split("\n")],
            "outputs": [],
            "execution_count": None
        })

    def add_markdown_cell(source_text):
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source_text.split("\n")]
        })

    # --- Instructions ---
    add_markdown_cell("""# TimesFM Validation - GitHub Install
## Colab Python 3.12 Compatibility Fix
Since PyPI versions don't support Python 3.12, we install from GitHub.

1. Run Cell 1 (Installation from GitHub)
2. **RESTART RUNTIME**: `Runtime` → `Restart session`
3. Run Cell 2 to verify
""")

    # --- CELL 1: Installation from GitHub ---
    add_code_cell("""# @title 1. Install TimesFM from GitHub (Python 3.12 fix)
!nvidia-smi
!python --version

print("=" * 60)
print("PyPI versions require Python <3.12")
print("Installing directly from GitHub repository...")
print("=" * 60)

# Uninstall any broken attempts
!pip uninstall -y timesfm lingvo paxml praxis 2>/dev/null || true

# Install from GitHub main branch (may have Python 3.12 fixes)
print()
print("Step 1: Installing TimesFM from GitHub...")
!pip install git+https://github.com/google-research/timesfm.git --no-deps

# Install minimal dependencies manually (avoiding lingvo)
print()
print("Step 2: Installing minimal dependencies...")
!pip install jax jaxlib numpy pandas scikit-learn huggingface_hub einshape

# Install other tools
print()
print("Step 3: Installing analysis tools...")
!pip install -q ccxt matplotlib seaborn xgboost

print()
print("=" * 60)
print("✅ Installation attempted!")
print("=" * 60)
print()
print("⚠️ RESTART: Runtime → Restart session")
print("⚠️ Then run Cell 2 to verify")
print("=" * 60)""")

    # --- CELL 2: Verify ---
    add_code_cell("""# @title 2. Verify Installation
print("Checking installation...")

try:
    import timesfm
    print(f"✅ timesfm imported!")
except ImportError as e:
    print(f"❌ timesfm failed: {e}")
    print()
    print("=" * 50)
    print("ALTERNATIVE: TimesFM does NOT support Python 3.12")
    print("Your options:")
    print("1. Use Kaggle notebooks (Python 3.10)")
    print("2. Use local machine with Python 3.10")
    print("3. Wait for Google to update TimesFM")
    print("=" * 50)
    raise

try:
    import jax
    print(f"✅ JAX: {jax.devices()}")
except:
    print("⚠️ JAX import issue")

print()
print("✅ Ready to proceed!")""")

    # --- CELL 3: Config ---
    add_code_cell("""# @title 3. Configuration
import timesfm
import jax
import ccxt
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class Config:
    context_length: int = 512
    forecast_horizon: int = 128
    walkforward_step: int = 128
    # Using Kraken (Binance is geo-blocked from Colab)
    symbol: str = 'BTC/USD'
    timeframe: str = '5m'
    model_repo: str = "google/timesfm-1.0-200m"

config = Config()
print(f"Config: {config.model_repo}")
print(f"Exchange: Kraken (Binance blocked from Colab)")""")

    # --- CELL 4: Data ---
    add_code_cell("""# @title 4. Load BTC Data from CSV
# Upload btc_5min_1year.csv to Colab first!
# (Use the file icon on the left sidebar, or drag & drop)

from google.colab import files
import os

# Check if file exists, if not prompt upload
csv_file = 'btc_5min_1year.csv'

if not os.path.exists(csv_file):
    print("Please upload btc_5min_1year.csv...")
    uploaded = files.upload()
    csv_file = list(uploaded.keys())[0]

print(f"Loading {csv_file}...")
df = pd.read_csv(csv_file, index_col='timestamp', parse_dates=True)
print(f"✅ Loaded {len(df)} bars")
print(f"Date range: {df.index[0]} to {df.index[-1]}")

# Add features
def add_features(df):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain/loss))
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    df['volatility'] = df['close'].rolling(20).std()
    df['returns'] = df['close'].pct_change()
    df['target'] = df['returns'].shift(-1)
    return df.dropna()

df = add_features(df)
print(f"✅ {len(df)} bars with features ready for backtest")""")

    # --- CELL 5: Model ---
    add_code_cell("""# @title 5. Load Model
print(f"Loading {config.model_repo}...")

backend = "gpu" if "gpu" in str(jax.devices()).lower() else "cpu"

tfm = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(
        backend=backend,
        per_core_batch_size=32,
        horizon_len=config.forecast_horizon,
    ),
    checkpoint=timesfm.TimesFmCheckpoint(
        huggingface_repo_id=config.model_repo
    )
)
print(f"✅ Loaded on {backend}")

def train_xgb(train_df):
    X = train_df[['rsi', 'ema_12', 'ema_26', 'volatility', 'volume']]
    y = train_df['target']
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, n_jobs=-1, verbosity=0)
    model.fit(X, y)
    return model

print("✅ Ready")""")

    # --- CELL 6: Backtest ---
    add_code_cell("""# @title 6. Backtest
results = []
test_start = len(df) - 2000

print("Backtesting...")

for i in range(test_start, len(df) - config.forecast_horizon, config.walkforward_step):
    ctx = df['close'].iloc[i-config.context_length:i].values.astype(np.float32)
    try:
        fc, _ = tfm.forecast([ctx], freq=[0])
        tfm_trend = np.mean(np.diff(fc[0]))
    except:
        tfm_trend = 0
    
    xgb_model = train_xgb(df.iloc[:i])
    xgb_pred = xgb_model.predict(df.iloc[i:i+1][['rsi', 'ema_12', 'ema_26', 'volatility', 'volume']])[0]
    
    if tfm_trend > 0 and xgb_pred > 0:
        sig = 1
    elif tfm_trend < 0 and xgb_pred < 0:
        sig = -1
    else:
        sig = 0
    
    ret = (df['close'].iloc[i + config.forecast_horizon] - df['close'].iloc[i]) / df['close'].iloc[i]
    results.append({'signal': sig, 'real_return': ret, 'strategy_return': sig * ret})
    
    if len(results) % 5 == 0:
        print(f"Step {len(results)}: {sig}, {ret:.4f}")

res_df = pd.DataFrame(results)
print("✅ Done")""")

    # --- CELL 7: Results ---
    add_code_cell("""# @title 7. Results - $1000 CAD
CAPITAL = 1000.0
FEE = 0.001

res_df['equity'] = (1 + res_df['strategy_return'] - abs(res_df['signal'])*FEE).cumprod() * CAPITAL
res_df['benchmark'] = (1 + res_df['real_return']).cumprod() * CAPITAL
res_df['peak'] = res_df['equity'].cummax()
res_df['dd'] = res_df['equity'] / res_df['peak'] - 1

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
ax1.plot(res_df['equity'], label='Strategy', color='green')
ax1.plot(res_df['benchmark'], label='BTC', linestyle='--', alpha=0.7)
ax1.set_title(f'Portfolio (${CAPITAL} CAD)')
ax1.legend()
ax1.grid(True)

ax2.fill_between(range(len(res_df)), res_df['dd'], 0, color='red', alpha=0.3)
ax2.set_title('Drawdown')
ax2.grid(True)
plt.tight_layout()
plt.show()

final = res_df['equity'].iloc[-1]
pnl = final - CAPITAL
wr = (res_df['strategy_return'] > 0).mean()
trades = res_df[res_df['signal'] != 0]['strategy_return']
sharpe = trades.mean() / trades.std() * np.sqrt(len(res_df)) if len(trades) > 0 else 0

print("=" * 40)
print(f"INITIAL:      ${CAPITAL:,.2f} CAD")
print(f"FINAL:        ${final:,.2f} CAD")
print(f"NET PnL:      ${pnl:,.2f} CAD")
print("=" * 40)
print(f"Max Drawdown: {res_df['dd'].min():.2%}")
print(f"Win Rate:     {wr:.1%}")
print(f"Sharpe:       {sharpe:.3f}")
print("=" * 40)""")

    output_path = os.path.join(r"c:\Users\chari\OneDrive\Documents\Time FN MODEL", "timesfm_enhanced_validation.ipynb")
    with open(output_path, "w") as f:
        json.dump(notebook, f, indent=2)
    print(f"Notebook created: {output_path}")

if __name__ == "__main__":
    create_notebook()
