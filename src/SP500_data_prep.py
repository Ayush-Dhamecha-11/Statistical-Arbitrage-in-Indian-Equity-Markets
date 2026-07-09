import pandas as pd
from pathlib import Path
from datetime import datetime
import yfinance as yf

sp500 = pd.read_csv("https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv")
tickers = sp500["Symbol"].tolist()
end_date   = datetime(2026, 1, 1)
start_date = datetime(2007, 12, 1)

historical_data = {}

for sym in tickers[0:500]:
    ticker = yf.Ticker(sym)
    df = ticker.history(start=start_date, end=end_date, interval="1d", auto_adjust=True)
    df.columns = [col.lower() for col in df.columns]
    df = df.ffill()
    print(f"  {sym}: {len(df)} trading days")

    if len(df) == 4549:
        historical_data[sym] = df

for sym, df in historical_data.items():
    historical_data[sym] = df.drop(columns=["dividends", "stock splits"], errors="ignore")

# get sector info
metadata = []
for sym, df in historical_data.items():
    ticker = yf.Ticker(sym)
    info = ticker.info
    data = {
        'symbol': sym,
        'name': info.get('longName', 'N/A'),
        'sector': info.get('sector', 'N/A'),
        'industry': info.get('industry', 'N/A'), 
    }
    metadata.append(data)
    #print(data['sector'])

metadata_df = pd.DataFrame(metadata)

# Store Historical data
historical_dir = Path('../data/historical_SP500')
historical_dir.mkdir(parents=True, exist_ok=True)

for symbol, df in historical_data.items():
    try:
        filepath = historical_dir / f"{symbol}_historical.csv"
        df.to_csv(filepath)
    except Exception as e:
        print(f"Error saving price history for {symbol}: {e}")

print(f"Saved price history for {len(historical_data)} stocks")

# Store metadata
try:
    meta_path = Path('../data/SP500_metadata.csv')
    metadata_df.to_csv(meta_path)
except Exception as e:
    print(f"{e}: Error saving metadata")