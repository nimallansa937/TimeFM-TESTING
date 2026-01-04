"""
Proper Train/Test Split Backtest
- TRAIN: BTC 1hr from 2021-01-01 to 2024-12-31
- TEST:  BTC 1hr from 2025-01-01 to 2025-12-31

This is a proper out-of-sample test!
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import requests
from datetime import datetime
import time

print("=" * 70)
print("TimesFM + XGBoost - Proper Train/Test Split")
print("  Training: 2021-01-01 to 2024-12-31 (4 years)")
print("  Testing:  2025-01-01 to 2025-12-31 (1 year)")
print("=" * 70)

# ============================================================
# 1. LOAD TIMESFM
# ============================================================
print("\n[1/7] Loading TimesFM...")
import timesfm

tfm = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(
        backend="cpu",
        per_core_batch_size=32,
        horizon_len=24,  # 24 hours forecast
    ),
    checkpoint=timesfm.TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
    )
)
print("✅ TimesFM loaded!")

# ============================================================
# 2. DOWNLOAD HISTORICAL DATA (CryptoCompare Hourly)
# ============================================================
print("\n[2/7] Downloading BTC hourly data...")

def fetch_hourly_data(start_date, end_date):
    """Fetch BTC/USD hourly data from CryptoCompare"""
    all_data = []
    to_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    
    while to_ts > start_ts:
        url = "https://min-api.cryptocompare.com/data/v2/histohour"
        params = {
            'fsym': 'BTC',
            'tsym': 'USD',
            'limit': 2000,
            'toTs': to_ts
        }
        
        try:
            resp = requests.get(url, params=params, timeout=30)
            data = resp.json()
            
            if data.get('Response') == 'Error':
                print(f"   API Error: {data.get('Message')}")
                break
                
            bars = data.get('Data', {}).get('Data', [])
            if not bars:
                break
                
            all_data = bars + all_data
            to_ts = bars[0]['time'] - 1
            
            print(f"   Downloaded {len(all_data)} bars... (back to {datetime.fromtimestamp(bars[0]['time']).strftime('%Y-%m-%d')})")
            time.sleep(0.3)
            
        except Exception as e:
            print(f"   Error: {e}")
            break
    
    df = pd.DataFrame(all_data)
    df['timestamp'] = pd.to_datetime(df['time'], unit='s')
    df = df.rename(columns={'volumefrom': 'volume'})
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df = df.set_index('timestamp')
    df = df[~df.index.duplicated(keep='first')]
    return df.sort_index()

# Download TRAINING data (2021-2024)
print("\n   Fetching TRAINING data (2021-2024)...")
train_df = fetch_hourly_data("2021-01-01", "2024-12-31")
print(f"   ✅ Training: {len(train_df)} bars ({train_df.index[0]} to {train_df.index[-1]})")

# Download TEST data (2025)
print("\n   Fetching TEST data (2025)...")
test_df = fetch_hourly_data("2025-01-01", "2025-12-31")
print(f"   ✅ Testing: {len(test_df)} bars ({test_df.index[0]} to {test_df.index[-1]})")

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
print("\n[3/7] Engineering features...")

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

train_df = add_features(train_df)
test_df = add_features(test_df)
print(f"   Training: {len(train_df)} bars with features")
print(f"   Testing:  {len(test_df)} bars with features")

# ============================================================
# 4. TRAIN XGBOOST ON 2021-2024 DATA
# ============================================================
print("\n[4/7] Training XGBoost on 2021-2024 data...")

feature_cols = ['rsi', 'ema_12', 'ema_26', 'volatility', 'volume']
X_train = train_df[feature_cols]
y_train = train_df['target']

xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    n_jobs=-1,
    verbosity=1
)
xgb_model.fit(X_train, y_train)
print("✅ XGBoost trained on 4 years of data!")

# ============================================================
# 5. TEST ON 2025 DATA (OUT OF SAMPLE)
# ============================================================
print("\n[5/7] Testing on 2025 data (OUT-OF-SAMPLE)...")

CONTEXT_LENGTH = 168  # 1 week of hourly data
FORECAST_HORIZON = 24  # 24 hour trades
STEP_SIZE = 24  # Trade every 24 hours

results = []
test_start = CONTEXT_LENGTH  # Need context from training data

# Combine data for context
combined_df = pd.concat([train_df, test_df])

for i in range(len(train_df), len(combined_df) - FORECAST_HORIZON, STEP_SIZE):
    # TimesFM forecast
    context = combined_df['close'].iloc[i-CONTEXT_LENGTH:i].values.astype(np.float32)
    
    try:
        forecast, _ = tfm.forecast([context], freq=[0])
        tfm_trend = np.mean(np.diff(forecast[0]))
    except:
        tfm_trend = 0
    
    # XGBoost prediction (using pre-trained model!)
    features = combined_df.iloc[i:i+1][feature_cols]
    xgb_pred = xgb_model.predict(features)[0]
    
    # Signal (selective: both must agree)
    if tfm_trend > 0 and xgb_pred > 0:
        signal = 1
    elif tfm_trend < 0 and xgb_pred < 0:
        signal = -1
    else:
        signal = 0
    
    # Returns
    entry_price = combined_df['close'].iloc[i]
    exit_price = combined_df['close'].iloc[i + FORECAST_HORIZON]
    actual_return = (exit_price - entry_price) / entry_price
    strategy_return = signal * actual_return
    
    results.append({
        'timestamp': combined_df.index[i],
        'entry_price': entry_price,
        'exit_price': exit_price,
        'signal': signal,
        'actual_return': actual_return,
        'strategy_return': strategy_return
    })
    
    if len(results) % 30 == 0:
        signal_str = {1: "LONG", -1: "SHORT", 0: "HOLD"}[signal]
        print(f"   Trade {len(results)}: {signal_str} @ ${entry_price:,.0f}, ret={actual_return:.2%}")

res_df = pd.DataFrame(results)
print(f"\n✅ Out-of-sample test complete! {len(results)} trading opportunities")

# ============================================================
# 6. CALCULATE PERFORMANCE
# ============================================================
print("\n[6/7] Calculating performance...")

INITIAL_CAPITAL = 1000.0
FEE = 0.001

res_df['equity'] = (1 + res_df['strategy_return'] - abs(res_df['signal']) * FEE).cumprod() * INITIAL_CAPITAL
res_df['benchmark'] = (1 + res_df['actual_return']).cumprod() * INITIAL_CAPITAL
res_df['peak'] = res_df['equity'].cummax()
res_df['drawdown'] = res_df['equity'] / res_df['peak'] - 1

# Stats
final_balance = res_df['equity'].iloc[-1]
benchmark_final = res_df['benchmark'].iloc[-1]
net_pnl = final_balance - INITIAL_CAPITAL
max_drawdown = res_df['drawdown'].min() * 100

active_trades = res_df[res_df['signal'] != 0]['strategy_return']
win_rate = (active_trades > 0).mean() * 100 if len(active_trades) > 0 else 0
sharpe = active_trades.mean() / active_trades.std() * np.sqrt(365) if len(active_trades) > 1 else 0

long_count = (res_df['signal'] == 1).sum()
short_count = (res_df['signal'] == -1).sum()
hold_count = (res_df['signal'] == 0).sum()

# ============================================================
# 7. VISUALIZE & REPORT
# ============================================================
print("\n[7/7] Generating report...")

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Equity curve
axes[0].plot(res_df['timestamp'], res_df['equity'], label='Strategy', color='green', linewidth=2)
axes[0].plot(res_df['timestamp'], res_df['benchmark'], label='Buy & Hold BTC', color='orange', linestyle='--', alpha=0.7)
axes[0].axhline(y=INITIAL_CAPITAL, color='gray', linestyle=':', alpha=0.5)
axes[0].set_title(f'2025 OUT-OF-SAMPLE Performance (Trained on 2021-2024)')
axes[0].set_ylabel('Portfolio Value (CAD)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Drawdown
axes[1].fill_between(res_df['timestamp'], res_df['drawdown'] * 100, 0, color='red', alpha=0.3)
axes[1].set_title('Drawdown')
axes[1].set_ylabel('%')
axes[1].grid(True, alpha=0.3)

# BTC Price
axes[2].plot(res_df['timestamp'], res_df['entry_price'], color='blue', alpha=0.7)
axes[2].set_title('BTC Price During Test Period')
axes[2].set_ylabel('USD')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('train_test_results.png', dpi=150)
print("   Chart saved: train_test_results.png")

# Final Report
print("\n" + "=" * 70)
print("📊 OUT-OF-SAMPLE BACKTEST RESULTS (2025)")
print("   Model trained on: 2021-01-01 to 2024-12-31")
print("   Model tested on:  2025-01-01 to 2025-12-31")
print("=" * 70)

print(f"\n💰 PORTFOLIO PERFORMANCE")
print(f"   Initial Capital:   ${INITIAL_CAPITAL:,.2f} CAD")
print(f"   Final Balance:     ${final_balance:,.2f} CAD")
print(f"   Net PnL:           ${net_pnl:+,.2f} CAD ({(net_pnl/INITIAL_CAPITAL)*100:+.1f}%)")
print(f"   Buy & Hold BTC:    ${benchmark_final:,.2f} CAD ({(benchmark_final/INITIAL_CAPITAL-1)*100:+.1f}%)")

print(f"\n📈 RISK METRICS")
print(f"   Max Drawdown:      {max_drawdown:.2f}%")
print(f"   Sharpe Ratio:      {sharpe:.3f}")
print(f"   Win Rate:          {win_rate:.1f}%")

print(f"\n🔄 TRADING ACTIVITY")
print(f"   Total Periods:     {len(res_df)}")
print(f"   Long Trades:       {long_count}")
print(f"   Short Trades:      {short_count}")
print(f"   Hold (No Trade):   {hold_count}")

print(f"\n📅 TEST PERIOD")
print(f"   Start: {res_df['timestamp'].iloc[0]}")
print(f"   End:   {res_df['timestamp'].iloc[-1]}")
print(f"   BTC Start: ${res_df['entry_price'].iloc[0]:,.0f}")
print(f"   BTC End:   ${res_df['entry_price'].iloc[-1]:,.0f}")

print("\n" + "=" * 70)
if net_pnl > 0:
    print("✅ PROFITABLE on out-of-sample data!")
else:
    print("❌ Loss on out-of-sample data")
    
if final_balance > benchmark_final:
    print("🏆 Strategy OUTPERFORMED Buy & Hold!")
else:
    print("📉 Buy & Hold outperformed strategy")
print("=" * 70)

# Save
res_df.to_csv('train_test_trades.csv')
print(f"\n📁 Files saved: train_test_results.png, train_test_trades.csv")
