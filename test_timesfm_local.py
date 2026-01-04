"""
Test TimesFM locally with BTC data - Using correct API
"""
import pandas as pd
import numpy as np

print("Loading TimesFM...")
import timesfm

print(f"Available classes: {[a for a in dir(timesfm) if not a.startswith('_')]}")

# Load BTC data
print("\nLoading BTC data...")
df = pd.read_csv('btc_5min_1year.csv', index_col='timestamp', parse_dates=True)
print(f"Loaded {len(df)} bars")

# Get context
context = df['close'].iloc[-512:].values.astype(np.float32)
print(f"Context: {len(context)} values, last price: {context[-1]:.2f}")

# Initialize model (PyTorch version on CPU)
print("\nInitializing TimesFM (this downloads ~814MB checkpoint on first run)...")

tfm = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(
        backend="cpu",  # Use CPU
        per_core_batch_size=32,
        horizon_len=128,
    ),
    checkpoint=timesfm.TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
    )
)

print("✅ Model loaded!")

# Run forecast
print("\nRunning forecast...")
forecast, quantiles = tfm.forecast([context], freq=[0])

print(f"\n📈 Forecast Results:")
print(f"   Current price: ${context[-1]:,.2f}")
print(f"   Forecast (next 128 bars):")
print(f"     Step 1:   ${forecast[0][0]:,.2f}")
print(f"     Step 32:  ${forecast[0][31]:,.2f}")
print(f"     Step 64:  ${forecast[0][63]:,.2f}")
print(f"     Step 128: ${forecast[0][127]:,.2f}")

# Trend
trend = np.mean(np.diff(forecast[0]))
if trend > 0:
    print(f"\n   📈 BULLISH trend predicted (avg change: +${trend:.2f})")
else:
    print(f"\n   📉 BEARISH trend predicted (avg change: ${trend:.2f})")

print("\n✅ TimesFM working locally!")
