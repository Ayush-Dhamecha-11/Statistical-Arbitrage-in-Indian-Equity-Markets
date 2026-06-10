import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path
from typing import Dict, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NSEDataFetcher:

    def __init__(self, data_dir='../data', top_n=100):

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.top_n = top_n

        # top 78 major liquid NSE stocks
        self.nse_stocks = [
            "ABB.NS", "ADANIENT.NS", "ADANIPORTS.NS", "AMBUJACEM.NS", "APOLLOHOSP.NS", 
            "ASIANPAINT.NS", "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", 
            "BAJAJHLDNG.NS", "BANKBARODA.NS", "BEL.NS", "BPCL.NS", "BHARTIARTL.NS", 
            "BOSCHLTD.NS", "BRITANNIA.NS", "CGPOWER.NS", "CANBK.NS", "CHOLAFIN.NS", 
            "CIPLA.NS", "CUMMINSIND.NS", "DLF.NS", "DIVISLAB.NS", "DRREDDY.NS",
            "EICHERMOT.NS", "GAIL.NS", "GODREJCP.NS", "GRASIM.NS", "HCLTECH.NS", 
            "HDFCBANK.NS", "HINDALCO.NS", "HINDUNILVR.NS", "HINDZINC.NS", "ICICIBANK.NS",
            "ITC.NS", "INDHOTEL.NS", "IOC.NS", "INFY.NS", "JSWSTEEL.NS", 
            "JINDALSTEL.NS", "KOTAKBANK.NS", "LT.NS", "M&M.NS", "MARUTI.NS", 
            "NTPC.NS", "NESTLEIND.NS", "ONGC.NS", "PIDILITIND.NS", "PFC.NS", 
            "POWERGRID.NS", "PNB.NS", "RELIANCE.NS", "MOTHERSON.NS", "SHREECEM.NS", 
            "SHRIRAMFIN.NS", "SIEMENS.NS", "SOLARINDS.NS", "SBIN.NS", "SUNPHARMA.NS",
            "TVSMOTOR.NS", "TCS.NS", "TATACONSUM.NS", "TATAPOWER.NS", "TATASTEEL.NS", 
            "TECHM.NS", "TITAN.NS", "TORNTPHARM.NS", "TRENT.NS", "ULTRACEMCO.NS", 
            "UNIONBANK.NS", "UNITDSPR.NS", "VEDL.NS", "WIPRO.NS", "ZYDUSLIFE.NS",
            "BANKINDIA.NS", "BHEL.NS", "CONCOR.NS",
        ][:top_n]

    def fetch_historical_data(self, start_date=None, end_date=None, interval='1d') -> Tuple[Dict[str, pd.DataFrame], Dict[str, dict], Dict[str, dict]]:
        """
        Fetch daily OHLCV price history + fundamental info for all stocks.

        Returns
            historical_data : dict {symbol -> DataFrame}
        """

        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=5 * 365)

        logger.info(f"Fetching historical data for {len(self.nse_stocks)} stocks from {start_date.date()} to {end_date.date()}")

        historical_data = {}

        for i, symbol in enumerate(self.nse_stocks):

            # Historical price data
            try:
                logger.info(f"[{i+1}/{len(self.nse_stocks)}] Fetching prices: {symbol}")
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date, interval=interval, auto_adjust=True)

                if df.empty:
                    logger.warning(f"  No price data for {symbol} — skipping")
                    continue

                df.columns = [col.lower() for col in df.columns]
                df.index = df.index.strftime('%Y-%m-%d')
                
                if symbol == "HCLTECH.NS":
                    df = df.drop(index = "2010-02-06") # Has one extra day than others, which causes misalignment in characteristics computation.

                df = df.ffill()
                historical_data[symbol] = df
                logger.info(f"  {symbol}: {len(df)} trading days")

            except Exception as e:
                logger.error(f"  Price fetch failed for {symbol}: {e}")
                continue

        logger.info(f"Successfully fetched prices for {len(historical_data)} stocks")

        return historical_data

    def save_data(self, historical_data: Dict[str, pd.DataFrame]) -> None:
        
        """Save all fetched and computed data to disk."""
        logger.info("Saving data to disk...")
 
        # Price history
        historical_dir = self.data_dir / 'historical'
        historical_dir.mkdir(parents=True, exist_ok=True)
 
        for symbol, df in historical_data.items():
            try:
                safe_name = symbol.replace('.NS', '').replace('&', '_')
                filepath = historical_dir / f"{safe_name}_historical.csv"
                df.to_csv(filepath)
            except Exception as e:
                logger.error(f"Error saving price history for {symbol}: {e}")
 
        logger.info(f"Saved price history for {len(historical_data)} stocks")
 
        # Run info JSON
        try:
            info = {
                'total_stocks':       len(historical_data),
                'stocks':             list(historical_data.keys()),
                'date_fetched':       datetime.now().isoformat(),
            }
            info_path = self.data_dir / 'data_info.json'
            with open(info_path, 'w') as f:
                json.dump(info, f, indent=2)
            logger.info(f"Saved data info --> {info_path}")
        except Exception as e:
            logger.error(f"Error saving data info: {e}")

    def run_and_fetch(self, start_date=None, end_date=None):
        """
        Returns
            historical_data : dict {symbol -> OHLCV DataFrame}
        """

        logger.info("-" * 60)
        logger.info("NSE DATA FETCHING")
        logger.info("-" * 60)

        historical_data = self.fetch_historical_data(start_date, end_date)
        self.save_data(historical_data)
 
        logger.info("=" * 60)
        logger.info("Data fetch and preparation completed successfully!")
        logger.info("=" * 60)
 
        return historical_data


def main():

    # Initialise and run
    fetcher = NSEDataFetcher(data_dir='../data', top_n=100)

    end_date   = datetime(2026, 1, 1)
    start_date = datetime(2007, 12, 1)

    historical_data = fetcher.run_and_fetch(
        start_date=start_date,
        end_date=end_date,
    )

    # Summary
    print("\n" + "-" * 60)
    print("Data Fetched Successfully!")
    print("-" * 60)
    print(f"Stocks with price data : {len(historical_data)}") 
    print("-" * 60)


if __name__ == '__main__':
    main()