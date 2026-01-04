"""
TimesFM + XGBoost Ensemble Backtest
Full local validation on 1 year of 5-min BTC data
Starting capital: $1000 CAD
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from datetime import datetime

print("=" * 60)
print("TimesFM + XGBoost Ensemble Backtest")
print("=" * 60)

# ============================================================
# 1. LOAD TIMESFM
# ============================================================
print("\n[1/6] Loading TimesFM model...")
import timesfm

tfm = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(
        backend="cpu",
        per_core_batch_size=32,
        horizon_len=128,
    ),
    checkpoint=timesfm.TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
    )
)
print("✅ TimesFM loaded!")

# ============================================================
# 2. LOAD DATA
# ============================================================
print("\n[2/6] Loading BTC data...")
df = pd.read_csv('btc_5min_1year.csv', index_col='timestamp', parse_dates=True)
print(f"✅ Loaded {len(df)} bars")
print(f"   Date range: {df.index[0]} to {df.index[-1]}")

# ============================================================
# 3. ADD FEATURES
# ============================================================
print("\n[3/6] Engineering features...")

# RSI
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
df['rsi'] = 100 - (100 / (1 + gain/loss))

# EMAs
df['ema_12'] = df['close'].ewm(span=12).mean()
df['ema_26'] = df['close'].ewm(span=26).mean()

# Volatility
df['volatility'] = df['close'].rolling(20).std()

# Returns
df['returns'] = df['close'].pct_change()
df['target'] = df['returns'].shift(-1)

# Clean
df = df.dropna()
print(f"✅ Features added. {len(df)} bars ready.")

# ============================================================
# 4. XGBOOST HELPER
# ============================================================
def train_xgboost(train_df):
    X = train_df[['rsi', 'ema_12', 'ema_26', 'volatility', 'volume']]
    y = train_df['target']
    model = xgb.XGBRegressor(
        n_estimators=100, 
        learning_rate=0.05, 
        max_depth=5, 
        n_jobs=-1, 
        verbosity=0
    )
    model.fit(X, y)
    return model

# ============================================================
# 5. WALK-FORWARD BACKTEST
# ============================================================
print("\n[4/6] Running walk-forward backtest...")

CONTEXT_LENGTH = 512
FORECAST_HORIZON = 128  # Original optimal setting
STEP_SIZE = 128         # Original optimal setting
TEST_BARS = 20000       # ~35 days of test data

results = []
test_start = len(df) - TEST_BARS
total_steps = (TEST_BARS - FORECAST_HORIZON) // STEP_SIZE

print(f"   Testing last {TEST_BARS} bars")
print(f"   Expected steps: {total_steps}")
print()

for step_num, i in enumerate(range(test_start, len(df) - FORECAST_HORIZON, STEP_SIZE)):
    # --- TimesFM Forecast ---
    context = df['close'].iloc[i-CONTEXT_LENGTH:i].values.astype(np.float32)
    
    try:
        forecast, _ = tfm.forecast([context], freq=[0])
        tfm_trend = np.mean(np.diff(forecast[0]))
    except Exception as e:
        print(f"   TimesFM error: {e}")
        tfm_trend = 0
    
    # --- XGBoost Forecast ---
    xgb_model = train_xgboost(df.iloc[:i])
    features = df.iloc[i:i+1][['rsi', 'ema_12', 'ema_26', 'volatility', 'volume']]
    xgb_pred = xgb_model.predict(features)[0]
    
    # --- Ensemble Signal (ALWAYS LONG OR SHORT - no hold) ---
    # Combine both signals: positive = long, negative = short
    combined_signal = tfm_trend + (xgb_pred * 1000)  # Scale xgb_pred to similar magnitude
    
    if combined_signal > 0:
        signal = 1  # Long
    else:
        signal = -1  # Short
    
    # --- Calculate Return ---
    entry_price = df['close'].iloc[i]
    exit_price = df['close'].iloc[i + FORECAST_HORIZON]
    actual_return = (exit_price - entry_price) / entry_price
    strategy_return = signal * actual_return
    
    results.append({
        'timestamp': df.index[i],
        'entry_price': entry_price,
        'exit_price': exit_price,
        'tfm_trend': tfm_trend,
        'xgb_pred': xgb_pred,
        'signal': signal,
        'actual_return': actual_return,
        'strategy_return': strategy_return
    })
    
    # Progress
    if (step_num + 1) % 5 == 0:
        signal_str = {1: "LONG", -1: "SHORT", 0: "HOLD"}[signal]
        print(f"   Step {step_num+1}/{total_steps}: {signal_str}, ret={actual_return:.4f}")

res_df = pd.DataFrame(results)
print(f"\n✅ Backtest complete! {len(results)} trades evaluated.")

# ============================================================
# 6. PERFORMANCE REPORT
# ============================================================
print("\n[5/6] Calculating performance...")

INITIAL_CAPITAL = 1000.0  # CAD
FEE = 0.001  # 0.1% per trade

# Equity curve
res_df['equity'] = (1 + res_df['strategy_return'] - abs(res_df['signal']) * FEE).cumprod() * INITIAL_CAPITAL
res_df['benchmark'] = (1 + res_df['actual_return']).cumprod() * INITIAL_CAPITAL

# Drawdown
res_df['peak'] = res_df['equity'].cummax()
res_df['drawdown'] = res_df['equity'] / res_df['peak'] - 1

# Stats
final_balance = res_df['equity'].iloc[-1]
net_pnl = final_balance - INITIAL_CAPITAL
total_return = (final_balance / INITIAL_CAPITAL - 1) * 100
max_drawdown = res_df['drawdown'].min() * 100
win_rate = (res_df['strategy_return'] > 0).mean() * 100

# Sharpe (annualized for 5-min bars)
active_trades = res_df[res_df['signal'] != 0]['strategy_return']
if len(active_trades) > 1:
    sharpe = active_trades.mean() / active_trades.std() * np.sqrt(len(res_df))
else:
    sharpe = 0

# Signal breakdown
long_count = (res_df['signal'] == 1).sum()
short_count = (res_df['signal'] == -1).sum()
hold_count = (res_df['signal'] == 0).sum()

# ============================================================
# 7. VISUALIZATION
# ============================================================
print("\n[6/6] Generating charts...")

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Equity curve
axes[0].plot(res_df['equity'], label='Strategy', color='green', linewidth=2)
axes[0].plot(res_df['benchmark'], label='Buy & Hold BTC', color='orange', linestyle='--', alpha=0.7)
axes[0].axhline(y=INITIAL_CAPITAL, color='gray', linestyle=':', alpha=0.5)
axes[0].set_title(f'Portfolio Value (Start: ${INITIAL_CAPITAL:,.0f} CAD)')
axes[0].set_ylabel('CAD')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Drawdown
axes[1].fill_between(range(len(res_df)), res_df['drawdown'] * 100, 0, color='red', alpha=0.3)
axes[1].plot(res_df['drawdown'] * 100, color='red', linewidth=1)
axes[1].set_title('Drawdown')
axes[1].set_ylabel('%')
axes[1].grid(True, alpha=0.3)

# Signals
colors = {1: 'green', -1: 'red', 0: 'gray'}
for i, row in res_df.iterrows():
    axes[2].bar(i, row['strategy_return'] * 100, color=colors[row['signal']], alpha=0.7)
axes[2].set_title('Trade Returns')
axes[2].set_ylabel('Return %')
axes[2].set_xlabel('Trade #')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('backtest_results.png', dpi=150)
print("   Chart saved: backtest_results.png")

# ============================================================
# FINAL REPORT
# ============================================================
print("\n" + "=" * 60)
print("📊 TIMESFM + XGBOOST ENSEMBLE BACKTEST RESULTS")
print("=" * 60)
print(f"\n💰 PORTFOLIO PERFORMANCE")
print(f"   Initial Capital:  ${INITIAL_CAPITAL:,.2f} CAD")
print(f"   Final Balance:    ${final_balance:,.2f} CAD")
print(f"   Net PnL:          ${net_pnl:+,.2f} CAD")
print(f"   Total Return:     {total_return:+.2f}%")

print(f"\n📈 RISK METRICS")
print(f"   Max Drawdown:     {max_drawdown:.2f}%")
print(f"   Sharpe Ratio:     {sharpe:.3f}")
print(f"   Win Rate:         {win_rate:.1f}%")

print(f"\n🔄 TRADING ACTIVITY")
print(f"   Total Signals:    {len(res_df)}")
print(f"   Long Trades:      {long_count}")
print(f"   Short Trades:     {short_count}")
print(f"   Hold (No Trade):  {hold_count}")

print("\n" + "=" * 60)

# Interpretation
if sharpe > 2:
    print("🏆 EXCELLENT: Sharpe > 2.0 indicates outstanding risk-adjusted returns!")
elif sharpe > 1:
    print("✅ GOOD: Sharpe > 1.0 indicates solid risk-adjusted returns.")
else:
    print("⚠️ CAUTION: Sharpe < 1.0 suggests poor risk-adjusted returns.")

if net_pnl > 0:
    print(f"✅ PROFITABLE: Strategy made ${net_pnl:.2f} CAD")
else:
    print(f"❌ LOSS: Strategy lost ${-net_pnl:.2f} CAD")

print("=" * 60)

# Save results
res_df.to_csv('backtest_trades.csv')
print(f"\n📁 Files saved:")
print(f"   - backtest_results.png (chart)")
print(f"   - backtest_trades.csv (trade log)")
print("\n✅ Backtest complete!")
