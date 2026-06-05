"""
nse_data_fetcher.py

Pipeline:

1. Fetch OHLCV history for top 78 stocks with sufficient time series history for proper training
2. Fetch Nifty 50 index history and use as market benchmark for risk characteristics
3. Compute all firm characteristics per stock
4. Stack into a panel, cross-sectionally rank-normalise

Output files:

  data/historical/<SYMBOL>_historical.csv  - OHLCV price data
  data/characteristics/<SYMBOL>_chars.csv  - raw characteristics per stock
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

        self.tot_characteristics = [
            "Ret_D1", "Ret_W1", "STD_W1", "ST_Rev", "r12_2", "r12_7", "r36_13",
            "Variance", "HighLowVol", "Resid_Var", "Beta",
            "Vol", "SUV", "LTurnover", "Amihud", "Rel2High",
            "MktCap_Proxy", "PriceMom_Acc", "VolMom", "RSI14", "MACD_Sig", "DistMA50", "DistMA200",
        ]

        # Nifty 50 index ticker (used as market benchmark)
        self.market_ticker = "^NSEI"

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
    
    def compute_past_returns(self, hist: pd.DataFrame) -> pd.DataFrame:
        """
        Compute past-return characteristics from price history.

        Returns
            pd.DataFrame - one row per trading day, one column per characteristic.
    
        Characteristics
            r12_2 : Medium-term momentum: return from last month compared to 12 months ago
                    (approx. trading days 21 to 252 lagged)
            r12_7 : Intermediate momentum: return of 7 months ago compared to 12 months ago
                    (approx. trading days 126 to 252 lagged)
            r36_13 : Long-term momentum: return of 13 months ago compared to 36 months ago
                    (approx. trading days 252 to 756 lagged)
            ST_Rev : Short-term reversal: prior 1-month return (t-21 to t-1)
            Ret_D1 : Most recent daily return (yesterday's return)
            Ret_W1 : Most recent weekly return (past 5 trading days)
            STD_W1 : Weekly return volatility: std of daily returns over past 5 days
        """

        df = pd.DataFrame(index=hist.index)
    
        ret = hist["close"].pct_change() # daily simple return

        df["Ret_D1"] = ret.shift(1)
        df["Ret_W1"] = hist["close"].pct_change(5).shift(1)
        df["STD_W1"] = ret.rolling(window=5).std().shift(1)
        df["ST_Rev"] = hist["close"].pct_change(21).shift(1)
        df["r12_2"] = (hist["close"].shift(21) / hist["close"].shift(252) - 1)
        df["r12_7"] = (hist["close"].shift(126) / hist["close"].shift(252) - 1)
        df["r36_13"] = (hist["close"].shift(252) / hist["close"].shift(756) - 1)
    
        return df
    
    def compute_volatility(self, hist: pd.DataFrame, market_ind: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Compute risk characteristics.

        Returns
            pd.DataFrame - one row per trading day, one column per characteristic.
    
        Characteristics
            Variance : Rolling 21-day variance of daily returns.
            HighLowVol: range-based volatility estimator.
                        Uses daily High and Low prices to estimate the daily variance:
                            σ² = (1 / (4 ln2)) × (ln(H/L))²
                        Summed over a 21-day rolling window and annualised.
            Resid_Var : Idiosyncratic variance from a rolling 252-day OLS regression
                        of the stock's daily returns on the Nifty 50 returns:
            Beta : Slope coefficient β from the same OLS regression.
                Measures co-movement with the market.
        """    

        df = pd.DataFrame(index=hist.index)
        market_hist = market_ind.reindex(hist.index, method="ffill")
        ret = hist["close"].pct_change()
    
        # Variance
        df["Variance"] = ret.rolling(21).var().shift(1)
    
        # HighLowVol
        ln_hl_sq = np.log(hist["high"] / hist["low"]) ** 2
        park_var = ln_hl_sq.rolling(21).sum() / (4 * np.log(2) * 21)
        df["HighLowVol"] = park_var.shift(1)
    
        # Beta and Resid_Var
        if market_hist is not None and not market_hist.empty:
            mkt_ret = market_hist["close"].pct_change().rename("mkt")
    
            # Align on date — inner join so only trading days present in both
            combined = pd.concat([ret.rename("stock"), mkt_ret], axis=1).dropna()
    
            n = len(hist.index)
            resid_var_arr = np.full(n, np.nan)
            beta_arr = np.full(n, np.nan)
    
            hist_dates = hist.index
    
            for i in range(252, n):
                # Window: the 252 calendar-aligned trading days before position i
                window_end_date = hist_dates[i]
                # Select combined rows where date < window_end_date, take last 252
                mask = combined.index < window_end_date
                window = combined.loc[mask].iloc[-252:]
    
                if len(window) < 60:
                    continue
    
                X = np.column_stack([np.ones(len(window)), window["mkt"].values])
                y = window["stock"].values
    
                try:
                    # Least-Square regression to get Beta and Residuals
                    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                    beta_arr[i] = coeffs[1]
                    resid = y - X @ coeffs
                    resid_var_arr[i] = np.var(resid, ddof=2)

                except np.linalg.LinAlgError:
                    pass  # leave as NaN if singular
    
            df["Resid_Var"] = pd.Series(resid_var_arr, index=hist.index).shift(1)
            df["Beta"] = pd.Series(beta_arr, index=hist.index).shift(1)

        else:
            logger.warning("market_hist not provided — Beta=NaN, Resid_Var≈0.7×Variance")
            df["Beta"] = np.nan
            df["Resid_Var"] = df["Variance"] * 0.7
    
        return df

    def compute_liquidity(self, hist: pd.DataFrame) -> pd.DataFrame:
        """
        Compute volume, liquidity, and trading-friction characteristics.
    
        Returns
            pd.DataFrame - one row per trading day, one column per characteristic.

        Characteristics
            Vol : Log of 5-day cumulative trading volume.
            SUV : Standard Unexplained Volume (abnormal volume).
                    SUV_t = (log(Vol_t) - μ̂) / σ̂
                    where μ̂ and σ̂ are the rolling 252-day mean and std of
                    log volume.
            LTurnover : Log of 21-day average daily volume divided by its own
                        long-run (252-day) mean. 
            Amihud : Amihud illiquidity ratio.
                        ILLIQ_t = |R_t| / (P_t × Vol_t)
                    where P × Vol approximates daily INR traded value and R_t is return. 
            Rel2High : Price relative to 52-week high.
                        Rel2High_t = Close_t / max(High_{t-252:t-1})
        """

        df = pd.DataFrame(index=hist.index)
    
        close  = hist["close"]
        volume = hist["volume"].clip(lower=1)
        ret    = close.pct_change()
    
        # Vol
        df["Vol"] = np.log(volume.rolling(5).sum()).shift(1)
    
        # SUV
        log_vol      = np.log(volume)
        roll_mean    = log_vol.rolling(252, min_periods=60).mean()
        roll_std     = log_vol.rolling(252, min_periods=60).std().clip(lower=1e-8)
        df["SUV"]    = ((log_vol - roll_mean) / roll_std).shift(1)
    
        # LTurnover
        avg_21  = volume.rolling(21,  min_periods=5).mean()
        avg_252 = volume.rolling(252, min_periods=60).mean().clip(lower=1)
        df["LTurnover"] = np.log((avg_21 / avg_252).clip(lower=1e-6)).shift(1)
    
        # Amihud
        rupee_vol = close * volume
        amihud_daily = ret.abs() / rupee_vol.clip(lower=1)
        # Scale by 1e7 to bring to a numerically comfortable range
        df["Amihud"] = (amihud_daily.rolling(21, min_periods=5).mean() * 1e7).shift(1)
    
        # Rel2High
        high_252 = hist["high"].rolling(252, min_periods=60).max().shift(1)
        df["Rel2High"] = (close / high_252.clip(lower=1e-8)).shift(1)
    
        return df

    def compute_price_level_signal(self, hist: pd.DataFrame) -> pd.DataFrame:
        """
        Compute price-level characteristics.

        Returns
            pd.DataFrame - one row per trading day, one column per characteristic.

        Characteristics
            MktCap_Proxy : Log of (close × 21-day avg volume).
            PriceMom_Acc : Price momentum acceleration = r2_1 minus r12_2.
            VolMom : Volume momentum --> recent 5-day avg volume relative to
                    21-day avg volume.
            RSI14 : Relative Strength Index over 14 days.
                        RSI = 100 - 100 / (1 + RS)
                        where RS = average gain / average loss
            MACD_Sig : MACD signal line = 9-day EMA of (12-day EMA - 26-day EMA).
            DistMA50 : Percentage distance of current close from 50-day SMA.
                            DistMA50 = (Close - MA50) / MA50
            DistMA200 : Percentage distance of current close from 200-day SMA.
        """

        df  = pd.DataFrame(index=hist.index)
        c   = hist["close"]
        ret = c.pct_change()
        vol = hist["volume"].clip(lower=1)
    
        # MktCap_Proxy
        avg_vol_21 = vol.rolling(21, min_periods=5).mean()
        df["MktCap_Proxy"] = np.log((c * avg_vol_21).clip(lower=1)).shift(1)
    
        # PriceMom_Acc
        r2_1  = c.shift(1) / c.shift(2) - 1
        r12_2 = c.shift(21) / c.shift(252) - 1
        df["PriceMom_Acc"] = (r2_1 - r12_2).shift(1)
    
        # VolMom
        avg_vol_5 = vol.rolling(5,  min_periods=2).mean()
        avg_vol_21 = vol.rolling(21, min_periods=5).mean().clip(lower=1)
        df["VolMom"] = (avg_vol_5 / avg_vol_21).shift(1)
    
        # RSI14
        delta    = c.diff()
        gain     = delta.clip(lower=0)
        loss     = (-delta).clip(lower=0)
        # We use simple rolling mean for the first value then Wilder's method
        avg_gain = gain.rolling(14, min_periods=14).mean()
        avg_loss = loss.rolling(14, min_periods=14).mean().clip(lower=1e-10)
        rs       = avg_gain / avg_loss
        rsi      = 100.0 - (100.0 / (1.0 + rs))
        df["RSI14"] = rsi.shift(1)
    
        # MACD_Sig
        ema12    = c.ewm(span=12, adjust=False).mean()
        ema26    = c.ewm(span=26, adjust=False).mean()
        macd     = ema12 - ema26
        signal   = macd.ewm(span=9, adjust=False).mean()
        # Normalise by rolling 252-day std of close to make cross-sectionally comparable
        price_std = c.rolling(252, min_periods=60).std().clip(lower=1e-8)
        df["MACD_Sig"] = (signal / price_std).shift(1)
    
        # DistMA50
        ma50 = c.rolling(50, min_periods=25).mean().clip(lower=1e-8)
        df["DistMA50"] = ((c - ma50) / ma50).shift(1)
    
        # DistMA200 
        ma200 = c.rolling(200, min_periods=100).mean().clip(lower=1e-8)
        df["DistMA200"] = ((c - ma200) / ma200).shift(1)
    
        return df
    
    def fetch_market_index(self, start_date=None, end_date=None) -> Optional[pd.DataFrame]:
        
        """Fetch Nifty 50 index history for market-model computations."""
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=5 * 365)
 
        try:
            logger.info(f"Fetching market index ({self.market_ticker})...")
            ticker = yf.Ticker(self.market_ticker)
            df = ticker.history(start=start_date, end=end_date, interval='1d')
 
            if df.empty:
                logger.warning("No market index data retrieved — Beta/Resid_Var will use fallback")
                return None
 
            df.columns = [c.lower() for c in df.columns]
            logger.info(f"Market index: {len(df)} trading days")
            return df
 
        except Exception as e:
            logger.error(f"Market index fetch failed: {e}")
            return None


    def compute_all_characteristics(
        self,
        historical_data: dict,
        market_hist: Optional[pd.DataFrame] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute all characteristics for all stocks.

        Returns
            dict {symbol -> DataFrame of characteristics with date index}
        """
        
        logger.info("-"*60)
        logger.info("Computing characteristics for all stocks...")
        logger.info("-"*60)

        chars_d = {}

        for i, (symbol, hist) in enumerate(historical_data.items()):

            logger.info(f"Computing characteristics for {symbol}...")

            frames = []

            try:
                frames.append(self.compute_past_returns(hist))
            except Exception as e:
                logger.warning(f"  compute_past_returns failed: {e}")
        
            try:
                frames.append(self.compute_volatility(hist, market_hist))
            except Exception as e:
                logger.warning(f"  compute_volatility failed: {e}")
        
            try:
                frames.append(self.compute_liquidity(hist))
            except Exception as e:
                logger.warning(f"  compute_liquidity failed: {e}")
        
            try:
                frames.append(self.compute_price_level_signal(hist))
            except Exception as e:
                logger.warning(f"  compute_price_level_signal failed: {e}")
        
            if not frames:
                logger.error("  All characteristic groups failed — returning empty DataFrame")
                return pd.DataFrame(index=hist.index)
    
            result = pd.concat(frames, axis=1)

            # Add risk-free-return as another feature
            rfr1 = pd.read_csv(self.data_dir / 'RFR.csv')
            rfr = rfr1[::-1]

            # Filling missing values with last observed value
            result = result.ffill()

            # To bypass initial null values in characteristics, we crop price history to start from 2011-02-01
            result = result["2011-02-01" :]
            rfr = rfr.reindex(result.index, method=None)
            result["RFR"] = rfr["Price"] / 100 # Assign risk-free-return

            chars_d[symbol] = result

        logger.info(f"Characteristics computed for {len(chars_d)} / {len(historical_data)} stocks")
        return chars_d

    def build_panel(self, chars_dict: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        
        """
        Stack per-stock characteristics into a MultiIndex panel and
        apply cross-sectional rank normalisation.
 
        The 47 = 23 raw chars × 2 (raw + median) + 1 (RFR).
        We output both the raw panel and the rank-normalised panel.
 
        Returns
            panel_raw : pd.DataFrame, MultiIndex (date, symbol), raw values
            panel_norm : pd.DataFrame, MultiIndex (date, symbol), rank-normalised
        """

        logger.info("Building cross-sectional panel...")
        if not chars_dict:
            logger.error("chars_dict is empty — cannot build panel")
            return pd.DataFrame(), pd.DataFrame()
 
        # Stack all stocks into (date, symbol) MultiIndex
        frames = []
        for symbol, df in chars_dict.items():
            if df.empty:
                continue
            tmp1 = df.copy()
            # To bypass initial null values in characteristics, we crop price history to start from 2011-02-01
            tmp = tmp1["2011-02-01" :]
            tmp.index.name = "date"
            tmp["symbol"] = symbol
            tmp = tmp.reset_index().set_index(["date", "symbol"])
            frames.append(tmp)
 
        if not frames:
            logger.error("No characteristics data to stack")
            return pd.DataFrame(), pd.DataFrame()
 
        panel_raw = pd.concat(frames).sort_index()
        logger.info(
            f"Stacked panel: {panel_raw.shape[0]:,} rows × {panel_raw.shape[1]} cols "
            f"| {panel_raw.index.get_level_values('symbol').nunique()} stocks"
        )
 
        # Cross-sectional median features to preserve level information.
        logger.info("Computing cross-sectional medians...")
        char_cols = [c for c in panel_raw.columns if c != "RFR"]
        median_cols = {}
 
        for col in char_cols:
            # Compute cross-sectional median for each date
            cs_median = panel_raw[col].groupby(level="date").transform("median")
            #logger.info(f"  {col}_csmedian: {cs_median.notna().sum()} non-null values, {cs_median.shape} ")
            median_cols[f"{col}_csmedian"] = cs_median

        logger.info(f"Computed {len(median_cols)} median features")
         
        median_df = pd.DataFrame(median_cols, index=panel_raw.index)
        panel_raw_extended = pd.concat([panel_raw, median_df], axis=1)

        logger.info("Imputing leading NaNs with cross-sectional medians...")
        all_feature_cols = char_cols + list(median_cols.keys())

        for col in all_feature_cols:
        # Find rows where this column is NaN
            is_nan = panel_raw_extended[col].isna()
            if is_nan.any():
                # Fill each NaN with that date's cross-sectional median
                cs_med = (
                    panel_raw_extended[col]
                    .groupby(level="date")
                    .transform("median")
                )
                panel_raw_extended[col] = panel_raw_extended[col].fillna(cs_med)
 
        # Cross-sectional rank normalisation
        logger.info("Cross-sectionally rank-normalising characteristics...")
        panel_norm = panel_raw_extended.copy()
 
        for col in char_cols:
            # Rank within each date's cross-section
            panel_norm[col] = (panel_raw_extended[col]
                .groupby(level="date")
                .transform(lambda x: x.rank(pct=True))
            )
            # Also normalise the median features
            med_col = f"{col}_csmedian"
            if med_col in panel_norm.columns:
                panel_norm[med_col] = (
                    panel_raw_extended[med_col]
                    .groupby(level="date")
                    .transform(lambda x: x.rank(pct=True)))
 
        dates = panel_norm.index.get_level_values("date")
        logger.info(
            f"Final panel: {panel_norm.shape[0]:,} rows × {panel_norm.shape[1]} cols | "
            f"{dates.min()} --> {dates.max()}"
        )
        return panel_raw_extended, panel_norm

    def save_data(
        self,
        historical_data: Dict[str, pd.DataFrame],
        chars_dict: Optional[Dict[str, pd.DataFrame]] = None,
        panel_raw: Optional[pd.DataFrame] = None,
        panel_norm: Optional[pd.DataFrame] = None,
    ) -> None:
        
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
 
        # Per-stock raw characteristics
        if chars_dict:
            chars_dir = self.data_dir / 'characteristics'
            chars_dir.mkdir(parents=True, exist_ok=True)
 
            for symbol, df in chars_dict.items():
                try:
                    safe_name = symbol.replace('.NS', '').replace('&', '_')
                    filepath = chars_dir / f"{safe_name}_chars.csv"
                    df.to_csv(filepath)
                except Exception as e:
                    logger.error(f"Error saving characteristics for {symbol}: {e}")
 
            logger.info(f"Saved per-stock characteristics for {len(chars_dict)} stocks")
 
        # Panel (parquet for efficiency)
        if panel_raw is not None and not panel_raw.empty:
            try:
                panel_raw_path = self.data_dir / 'panel_characteristics_raw.parquet'
                panel_raw.to_parquet(panel_raw_path)
                logger.info(f"Saved raw panel --> {panel_raw_path}  {panel_raw.shape}")
            except Exception as e:
                logger.warning(f"Parquet save failed, trying CSV: {e}")
                panel_raw.to_csv(self.data_dir / 'panel_characteristics_raw.csv')
 
        if panel_norm is not None and not panel_norm.empty:
            try:
                panel_norm_path = self.data_dir / 'panel_characteristics_norm.parquet'
                panel_norm.to_parquet(panel_norm_path)
                logger.info(f"Saved normalised panel --> {panel_norm_path}  {panel_norm.shape}")
            except Exception as e:
                logger.warning(f"Parquet save failed, trying CSV: {e}")
                panel_norm.to_csv(self.data_dir / 'panel_characteristics_norm.csv')
 
        # Run info JSON
        try:
            info = {
                'total_stocks':       len(historical_data),
                'stocks':             list(historical_data.keys()),
                'date_fetched':       datetime.now().isoformat(),
                'characteristics':    self.tot_characteristics,
                'n_characteristics':  len(self.tot_characteristics),
                'panel_shape':        list(panel_raw.shape) if panel_raw is not None else None,
            }
            info_path = self.data_dir / 'data_info.json'
            with open(info_path, 'w') as f:
                json.dump(info, f, indent=2)
            logger.info(f"Saved data info --> {info_path}")
        except Exception as e:
            logger.error(f"Error saving data info: {e}")

    def run_and_fetch(self, start_date=None, end_date=None):
        """
        Parameters
            compute_characteristics : bool
                Set False to skip characteristic computation

        Returns
            historical_data : dict {symbol -> OHLCV DataFrame}
            metadata_dict : dict {symbol -> metadata}
            panel_norm : pd.DataFrame, MultiIndex (date, symbol), rank-normalised characteristics
        """

        logger.info("-" * 60)
        logger.info("NSE STATISTICAL ARBITRAGE DATA PIPELINE")
        logger.info("-" * 60)

        historical_data = self.fetch_historical_data(start_date, end_date)
        panel_raw = panel_norm = None
        chars_dict = {}
 
        if historical_data:

            market_hist = self.fetch_market_index(start_date, end_date)
            chars_dict = self.compute_all_characteristics(
                historical_data, market_hist
            )

            if chars_dict is not None:
                panel_raw, panel_norm = self.build_panel(chars_dict)

        cropped_historical_data = {}
        for symbol, df in historical_data.items():
            # To bypass initial null values in characteristics, we crop price history to start from 2011-02-01
            cropped_historical_data[symbol] = df.loc["2011-02-01" :]
 
        self.save_data(cropped_historical_data, chars_dict, panel_raw, panel_norm)
 
        logger.info("=" * 60)
        logger.info("Data fetch and preparation completed successfully!")
        logger.info("=" * 60)
 
        return historical_data, panel_norm


def main():

    # Initialise and run
    fetcher = NSEDataFetcher(data_dir='../data', top_n=100)

    end_date   = datetime(2026, 1, 1)
    start_date = datetime(2007, 12, 1)

    historical_data, panel_norm = fetcher.run_and_fetch(
        start_date=start_date,
        end_date=end_date,
    )

    # Summary
    print("\n" + "-" * 60)
    print("Data Summary")
    print("-" * 60)
    print(f"Stocks with price data : {len(historical_data)}")
 
    if panel_norm is not None and not panel_norm.empty:
        dates = panel_norm.index.get_level_values("date")
        syms  = panel_norm.index.get_level_values("symbol")
        print(f"Panel shape            : {panel_norm.shape}")
        print(f"Panel date range       : {dates.min()} --> {dates.max()}")
        print(f"Unique stocks in panel : {syms.nunique()}")
        print(f"Feature columns        : {len(panel_norm.columns)}")
        print(f"\nSample (first 5 rows × 8 cols):")
        print(panel_norm.iloc[:5, :8].to_string())
 
    print("-" * 60)


if __name__ == '__main__':
    main()