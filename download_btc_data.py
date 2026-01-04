"""
Download 1 year of 5-min BTC/USDT data from Binance
Run this LOCALLY (on your PC), then upload btc_5min_1year.csv to Colab
"""
import ccxt
import pandas as pd
from datetime import datetime
import time

def download_btc_data(days=365):
    print(f"Downloading {days} days of 5-min BTC/USDT data from Binance...")
    
    exchange = ccxt.binance()
    symbol = 'BTC/USDT'
    timeframe = '5m'
    
    # Calculate timestamps
    limit_per_request = 1000
    bars_needed = 288 * days  # 288 5-min bars per day
    
    since = exchange.milliseconds() - (days * 24 * 60 * 60 * 1000)
    all_ohlcv = []
    
    while len(all_ohlcv) < bars_needed:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit_per_request)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            
            if len(all_ohlcv) % 10000 == 0:
                print(f"  Downloaded {len(all_ohlcv)} bars...")
                
            time.sleep(0.1)  # Rate limit
            
        except Exception as e:
            print(f"Error: {e}")
            break
    
    # Create DataFrame
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')
    df = df[~df.index.duplicated(keep='first')]
    
    # Save to CSV
    filename = 'btc_5min_1year.csv'
    df.to_csv(filename)
    print(f"\n✅ Saved {len(df)} bars to {filename}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"\nNow upload '{filename}' to Google Colab!")
    
    return df

if __name__ == "__main__":
    df = download_btc_data(days=365)
