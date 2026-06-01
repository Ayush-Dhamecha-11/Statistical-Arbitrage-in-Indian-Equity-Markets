"""
nse_data_fetcher.py

Pipeline:

1. Fetch OHLCV history for all Nifty 100 stocks
2. Fetch fundamental info
3. Fetch Nifty 50 index history (for Beta and Resid_Var computation)
4. Compute all firm characteristics per stock
5. Stack into a panel, cross-sectionally rank-normalise

Output files:

  data/historical/<SYMBOL>_historical.csv  - OHLCV price data
  data/characteristics/<SYMBOL>_chars.csv  - raw characteristics per stock
  data/nse_metadata.csv                    - fundamental metadata
  data/panel_characteristics_raw.parquet   - stacked panel (raw values)
  data/panel_characteristics_norm.parquet  - stacked panel (rank-normalised)
  data/data_info.json                      - run metadata

"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

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

        # Nifty 100 constituents (major liquid NSE stocks)
        self.nse_stocks = [
            "ABB.NS", "ADANIENSOL.NS", "ADANIENT.NS", "ADANIGREEN.NS", "ADANIPORTS.NS",
            "ADANIPOWER.NS", "AMBUJACEM.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "DMART.NS",
            "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BAJAJHLDNG.NS",
            "BANKBARODA.NS", "BEL.NS", "BPCL.NS", "BHARTIARTL.NS", "BOSCHLTD.NS",
            "BRITANNIA.NS", "CGPOWER.NS", "CANBK.NS", "CHOLAFIN.NS", "CIPLA.NS",
            "COALINDIA.NS", "CUMMINSIND.NS", "DLF.NS", "DIVISLAB.NS", "DRREDDY.NS",
            "EICHERMOT.NS", "ETERNAL.NS", "GAIL.NS", "GODREJCP.NS", "GRASIM.NS",
            "HCLTECH.NS", "HDFCAMC.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HINDALCO.NS",
            "HAL.NS", "HINDUNILVR.NS", "HINDZINC.NS", "HYUNDAI.NS", "ICICIBANK.NS",
            "ITC.NS", "INDHOTEL.NS", "IOC.NS", "IRFC.NS", "INFY.NS",
            "INDIGO.NS", "JSWSTEEL.NS", "JINDALSTEL.NS", "JIOFIN.NS", "KOTAKBANK.NS",
            "LT.NS", "LODHA.NS", "M&M.NS", "MARUTI.NS", "MAXHEALTH.NS",
            "MAZDOCK.NS", "MUTHOOTFIN.NS", "NTPC.NS", "NESTLEIND.NS", "ONGC.NS",
            "PIDILITIND.NS", "PFC.NS", "POWERGRID.NS", "PNB.NS", "RECLTD.NS",
            "RELIANCE.NS", "SBILIFE.NS", "MOTHERSON.NS", "SHREECEM.NS", "SHRIRAMFIN.NS",
            "SIEMENS.NS", "SOLARINDS.NS", "SBIN.NS", "SUNPHARMA.NS",
            "TVSMOTOR.NS", "TATACAP.NS", "TCS.NS", "TATACONSUM.NS",
            "TATAPOWER.NS", "TATASTEEL.NS", "TECHM.NS", "TITAN.NS",
            "TORNTPHARM.NS", "TRENT.NS", "ULTRACEMCO.NS", "UNIONBANK.NS", "UNITDSPR.NS",
            "VBL.NS", "VEDL.NS", "WIPRO.NS", "ZYDUSLIFE.NS",
            "BAJAJHFL.NS", "BANKINDIA.NS", "BHEL.NS", "CONCOR.NS", "GMRINFRA.NS",
        ][:top_n]

        # Nifty 50 index ticker (used as market benchmark)
        self.market_ticker = "^NSEI"

    def fetch_historical_meta_data(self, start_date=None, end_date=None, interval='1d') -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, Dict[str, dict]]:
        """
        Fetch daily OHLCV price history + fundamental info for all stocks.

        Returns
   
            historical_data : dict {symbol -> DataFrame}
            metadata_df     : DataFrame of fundamental summary (one row per stock)
            info_dict       : dict {symbol -> raw yfinance .info dict}
        """
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=5 * 365)

        logger.info(f"Fetching historical data for {len(self.nse_stocks)} stocks from {start_date.date()} to {end_date.date()}")

        historical_data: Dict[str, pd.DataFrame] = {}
        metadata_list = []
        info_dict: Dict[str, dict] = {}

        for i, symbol in enumerate(self.nse_stocks):

            # Historical price data
            try:
                logger.info(f"[{i+1}/{len(self.nse_stocks)}] Fetching prices: {symbol}")
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date, interval=interval)

                if df.empty:
                    logger.warning(f"  No price data for {symbol} — skipping")
                    continue

                df.columns = [col.lower() for col in df.columns]
                historical_data[symbol] = df
                logger.info(f"  {symbol}: {len(df)} trading days")

            except Exception as e:
                logger.error(f"  Price fetch failed for {symbol}: {e}")
                continue

            # Fundamental info
            try:
                logger.info(f"[{i+1}/{len(self.nse_stocks)}] Fetching info:   {symbol}")
                info = ticker.info
                metadata = {
                    'symbol': symbol,
                    'name': info.get('longName', 'N/A'),
                    'sector': info.get('sector', 'N/A'),
                    'industry': info.get('industry', 'N/A'),
                    'market_cap': info.get('marketCap', np.nan),
                    'current_price': info.get('currentPrice', np.nan),
                    'pe_ratio': info.get('trailingPE', np.nan),
                    'pb_ratio': info.get('priceToBook', np.nan),
                    'dividend_yield': info.get('dividendYield', np.nan),
                    '52_week_high': info.get('fiftyTwoWeekHigh', np.nan),
                    '52_week_low': info.get('fiftyTwoWeekLow', np.nan),
                    'beta': info.get('beta', np.nan),
                    'total_assets': info.get('totalAssets', np.nan),
                    'total_debt': info.get('totalDebt', np.nan),
                    'total_revenue': info.get('totalRevenue', np.nan),
                    'gross_profits': info.get('grossProfits', np.nan),
                    'ebit': info.get('ebit', np.nan),
                    'operating_cf': info.get('operatingCashflow', np.nan),
                    'free_cashflow': info.get('freeCashflow', np.nan),
                    'shares_outstanding': info.get('sharesOutstanding', np.nan),
                    'book_value_ps': info.get('bookValue', np.nan),
                }
                metadata_list.append(metadata)

            except Exception as e:
                logger.error(f"  Info fetch failed for {symbol}: {e}")
                info_dict[symbol] = {}

        metadata_df = pd.DataFrame(metadata_list)
        logger.info(f"Successfully fetched prices for {len(historical_data)} stocks")

        return historical_data, metadata_df

    def save_data(
        self,
        historical_data: Dict[str, pd.DataFrame],
        metadata_df: pd.DataFrame,
    ) -> None:
        
        # Save all fetched and computed data to disk
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

        # Metadata
        try:
            metadata_path = self.data_dir / 'nse_metadata.csv'
            metadata_df.to_csv(metadata_path, index=False)
            logger.info(f"Saved metadata → {metadata_path}")
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")

        # Run info JSON
        try:
            info = {
                'total_stocks': len(historical_data),
                'stocks': list(historical_data.keys()),
                'date_fetched': datetime.now().isoformat(),
            }

            info_path = self.data_dir / 'data_info.json'
            with open(info_path, 'w') as f:
                json.dump(info, f, indent=2)
            logger.info(f"Saved data info → {info_path}")
        except Exception as e:
            logger.error(f"Error saving data info: {e}")


    def run_and_fetch(self, start_date=None, end_date=None):
        """
        Parameters
            compute_characteristics : bool
                Set False to skip characteristic computation

        Returns
            historical_data : dict {symbol -> OHLCV DataFrame}
            metadata_df     : DataFrame with fundamental info
        """
        logger.info("-" * 60)
        logger.info("NSE STATISTICAL ARBITRAGE DATA PIPELINE")
        logger.info("-" * 60)

        historical_data, metadata_df = self.fetch_historical_meta_data(start_date, end_date)
        self.save_data(historical_data, metadata_df)

        logger.info("-" * 60)
        logger.info("Data saved successfully!")
        logger.info("-" * 60)

        return historical_data, metadata_df


def main():

    # Initialise and run
    fetcher = NSEDataFetcher(data_dir='../data', top_n=100)

    end_date   = datetime.now()
    start_date = end_date - timedelta(days=5 * 365)

    historical_data, metadata_df = fetcher.run_and_fetch(
        start_date=start_date,
        end_date=end_date,
    )

    # Summary
    print("\n" + "-" * 60)
    print("PIPELINE SUMMARY")
    print("-" * 60)
    print(f"Stocks with price data : {len(historical_data)}")
    print(f"Metadata rows          : {len(metadata_df)}")

    print("-" * 60)


if __name__ == '__main__':
    main()