"""
data_loader.py
-------------

Read the data that is already saved to disk and convert it into
clean, aligned PyTorch tensors that can be futher used in process.

It takes
-------------
  data/panel_characteristics_norm.parquet - MultiIndex (date, symbol), 47 cols
  data/historical/<SYMBOL>_historical.csv - daily OHLCV per stock
 
Computes
----------------
  X : torch.Tensor, shape (T, N, M)
        T = number of trading days
        N = number of stocks
        M = number of features 

  R : torch.Tensor, shape (T, N)
        Daily returns, aligned to the exact same T dates and N stocks as X.
        Returns at position [t, i] are the return of stock i on day t.
        These are the R_t values the model trades.

  dates : list of T pandas Timestamps
            Lets every other file map tensor row t back to a calendar date.

  symbols : list of N strings
              Lets every other file map tensor column i back to a stock ticker.

  splits : list of dicts
              Rolling window index boundaries for training, validation and test.
              The paper uses 8-year train / 2-year val / 1-year test windows,
              rolling forward by 1 year at a time.

"""
import logging
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Loads the saved pipeline outputs and exposes clean tensors to the model.

    Parameters
    ----------
    panel_path : Path to panel_characteristics_norm.parquet.
                 This file has MultiIndex (date, symbol) and ~47 feature columns.
                 It is the X_{t-1} matrix described in the paper.
        
    hist_dir : Directory containing per-stock price CSVs:
                    <SYMBOL>_historical.csv
                Each CSV has a DatetimeIndex (or a 'Date' column) and lowercase
                OHLCV columns: open, high, low, close, volume.
        
    train_years : Length of each training window in calendar years.

    val_years : Length of validation sub-window at the end of training.
                Validation is taken from the training window - the model trains
                on (train_years - val_years) years, then validates on the last
                val_years years of the same window.

    test_years : Length of out-of-sample test window following each training window.

    """

    def __init__(
        self,
        panel_path: str = "../data/panel_characteristics_norm.parquet",
        hist_dir: str = "../data/historical/",
        train_years: int = 8,
        val_years: int = 2,
        test_years: int = 1,
    ):
        self.panel_path = Path(panel_path)
        self.hist_dir = Path(hist_dir)
        self.train_years = train_years
        self.val_years = val_years
        self.test_years = test_years

        # These are populated by _load() and cached for repeated access
        self._X: Optional[torch.Tensor] = None   # (T, N, M)
        self._R: Optional[torch.Tensor] = None   # (T, N)
        self._dates: Optional[List] = None   # length T
        self._symbols: Optional[List[str]] = None   # length N
        self._feature_names: Optional[List[str]] = None   # length M
        self._splits: Optional[List[dict]] = None   # rolling windows

        logger.info("-" * 60)
        logger.info("DataLoader - initialising")
        logger.info("-" * 60)

        self.master_load()

    def get_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, List, List[str]]:
        """
        Return tensors.

        Returns
        -------
        X : torch.Tensor - shape (T, N, M) characteristics
        R : torch.Tensor - shape (T, N) daily returns
        dates : list of T pd.Timestamp objects
        symbols : list of N ticker strings
        """
        return self._X, self._R, self._dates, self._symbols

    def get_rolling_splits(self) -> List[dict]:
        """
        Return the list of rolling window split dicts.

        Each dict contains:
            window : int - window number starting from 1
            train_start : date - first date of training data
            train_end : date - last date of training data
                        (includes validation sub-window)
            val_start : date - first date of validation sub-window
            val_end : date - last date of validation sub-window
            test_start : date - first date of out-of-sample test
            test_end : date - last date of out-of-sample test
            train_idx_start : int - integer index into X / R tensors
            train_idx_end : int - (exclusive) end index for training
            val_idx_start : int
            val_idx_end : int
            test_idx_start : int
            test_idx_end : int
        """
        return self._splits

    def get_window_tensors(
        self,
        split: dict,
        X: Optional[torch.Tensor] = None,
        R: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor]:
        """
        Slice X and R tensors for one rolling window.

        Parameters
        ----------
        split : dict - one element from get_rolling_splits()
        X : tensor

        Returns
        -------
        X_train, R_train : tensors for the full training window
        X_val, R_val : tensors for the validation sub-window
        X_test, R_test : tensors for the out-of-sample test window
        """
        if X is None:
            X = self._X
        if R is None:
            R = self._R

        X_train = X[split["train_idx_start"] : split["train_idx_end"]]
        R_train = R[split["train_idx_start"] : split["train_idx_end"]]

        X_val   = X[split["val_idx_start"]   : split["val_idx_end"]]
        R_val   = R[split["val_idx_start"]   : split["val_idx_end"]]

        X_test  = X[split["test_idx_start"]  : split["test_idx_end"]]
        R_test  = R[split["test_idx_start"]  : split["test_idx_end"]]

        return X_train, R_train, X_val, R_val, X_test, R_test

    def describe(self) -> None:

        print("\n" + "-" * 65)
        print("  DataLoader - Data Summary")
        print("-" * 65)
        print(f"  Panel file    : {self.panel_path}")
        print(f"  History dir   : {self.hist_dir}")
        print()
        print(f"  Tensor X      : {tuple(self._X.shape)} (T, N, M)")
        print(f"  Tensor R      : {tuple(self._R.shape)} (T, N)")
        print(f"  Trading days  : {len(self._dates)}")
        print(f"  Stocks        : {len(self._symbols)}")
        print(f"  Features      : {len(self._feature_names)}")
        print(f"  Date range    : {self._dates[0].date()} --> {self._dates[-1].date()}")
        print()
        print(f"  Rolling windows  : {len(self._splits)}")
        print(f"  Train window     : {self.train_years} years")
        print(f"  Val sub-window   : {self.val_years} years  (last {self.val_years}yr of train)")
        print(f"  Test window      : {self.test_years} year(s) per roll")
        print()

        for sp in self._splits:
            n_tr = sp["train_idx_end"] - sp["train_idx_start"]
            n_val = sp["val_idx_end"] - sp["val_idx_start"]
            n_te = sp["test_idx_end"] - sp["test_idx_start"]

            print(
                f"  Window {sp['window']:2d}  "
                f"train {sp['train_start'].year}-{sp['train_end'].year} ({n_tr:4d}d)  "
                f"val {sp['val_start'].year}-{sp['val_end'].year} ({n_val:3d}d)  "
                f"test {sp['test_start'].year}-{sp['test_end'].year} ({n_te:3d}d)"
            )

        print()
        print(f"  NaN in X      : {torch.isnan(self._X).sum().item()}")
        print(f"  NaN in R      : {torch.isnan(self._R).sum().item()}")
        print(f"  Inf in X      : {torch.isinf(self._X).sum().item()}")
        print(f"  Inf in R      : {torch.isinf(self._R).sum().item()}")
        print("-" * 65 + "\n")

    def master_load(self) -> None:
        """
        Master load method called once in __init__.
        Runs all private steps in order and populates self._X, self._R, etc.
        """
        # Get symbols, dates, raw feature DataFrame
        panel_df, panel_dates, panel_symbols, feature_names = self.load_panel()

        # Load returns for all panel symbols
        returns_df = self.load_returns(panel_symbols)

        # Build X tensor (T, N, M)
        X = self.build_X(panel_df, panel_dates, panel_symbols, feature_names)

        # Build R tensor (T, N)
        R = self.build_R(returns_df, panel_dates, panel_symbols)

        nan_X = torch.isnan(X).sum().item()
        nan_R = torch.isnan(R).sum().item()

        if nan_X > 0:
            logger.warning(f"  {nan_X} NaN values in X - replacing with 0.5 (median rank)")
            X = torch.nan_to_num(X, nan=0.5)
        if nan_R > 0:
            logger.warning(f"  {nan_R} NaN values in R - replacing with 0.0")
            R = torch.nan_to_num(R, nan=0.0)

        inf_X = torch.isinf(X).sum().item()
        inf_R = torch.isinf(R).sum().item()
        if inf_X > 0:
            logger.warning(f"  {inf_X} Inf values in X - clipping")
            X = torch.clamp(X, -10.0, 10.0)
        if inf_R > 0:
            logger.warning(f"  {inf_R} Inf values in R - clipping")
            R = torch.clamp(R, -0.5, 0.5)   # ±50% daily return cap

        # Compute rolling window splits
        splits = self.make_rolling_splits(panel_dates)

        self._X = X
        self._R = R
        self._dates = panel_dates
        self._symbols = panel_symbols
        self._feature_names = feature_names
        self._splits = splits

        logger.info("-" * 60)
        logger.info(f"Load complete.")
        logger.info(f"  X shape    : {tuple(X.shape)}")
        logger.info(f"  R shape    : {tuple(R.shape)}")
        logger.info(f"  Date range : {panel_dates[0].date()} --> {panel_dates[-1].date()}")
        logger.info(f"  Stocks     : {len(panel_symbols)}")
        logger.info(f"  Features   : {len(feature_names)}")
        logger.info(f"  Windows    : {len(splits)}")
        logger.info("-" * 60)

    def load_panel(self):
        """
        Load panel_characteristics_norm.parquet.
        The parquet file has a MultiIndex with levels (date, symbol).
        The columns are the ~47 normalised feature values.

        Returns
        -------
        panel_df : pd.DataFrame, MultiIndex (date, symbol), all feature cols
        panel_dates : pd.DatetimeIndex - all unique dates in the panel
        panel_symbols : list of str - all unique symbols in the panel
        feature_names : list of str - ordered feature column names
        """
        logger.info("-" * 60)
        logger.info(f"Loading panel from parquet: {self.panel_path}")
        logger.info("-" * 60)

        if not self.panel_path.exists():
            raise FileNotFoundError(
                f"Panel file not found: {self.panel_path}\n"
                f"Run nse_price_pipeline.py first to generate it."
            )

        panel = pd.read_parquet(self.panel_path)

        if not isinstance(panel.index, pd.MultiIndex):
            raise ValueError(
                "Panel parquet must have a MultiIndex. "
                f"Got index type: {type(panel.index)}"
            )

        # Extract metadata
        panel_dates = panel.index.get_level_values("date").unique().sort_values()
        panel_symbols = sorted(panel.index.get_level_values("symbol").unique().tolist())
        feature_names = list(panel.columns)

        logger.info(f"  Panel shape    : {panel.shape}")
        logger.info(f"  Trading days   : {len(panel_dates)}")
        logger.info(f"  Unique symbols : {len(panel_symbols)}")
        logger.info(f"  Feature cols   : {len(feature_names)}")
        logger.info(
            f"  Date range     : {panel_dates.min().date()} --> {panel_dates.max().date()}"
        )

        return panel, panel_dates, panel_symbols, feature_names

    def load_returns(self, symbols: List[str]) -> pd.DataFrame:
        """
        Load daily price CSVs for all symbols and compute daily returns.

        Returns
        -------
        returns_df : pd.DataFrame
            rows = dates (DatetimeIndex)
            cols = symbols (same strings as input)
        """
        logger.info("-" * 60)
        logger.info(f"Loading returns from: {self.hist_dir}")
        logger.info("-" * 60)

        if not self.hist_dir.exists():
            raise FileNotFoundError(
                f"Historical directory not found: {self.hist_dir}\n"
                f"Run nse_price_pipeline.py first to generate price CSVs."
            )

        price_series = {}
        missing_files = []

        for symbol in symbols:
            #print(symbol)
            fname = f"{symbol.replace(".NS", "").replace("&", "_").replace("_chars", "")}.csv"
            fpath = self.hist_dir / fname

            if not fpath.exists():
                missing_files.append(symbol)
                continue

            try:
                df = pd.read_csv(fpath, index_col=0, parse_dates=True)
                df = df["2011-02-01":]
                price_series[symbol] = df['close']

            except Exception as e:
                logger.error(f"  {symbol}: failed to load CSV — {e}")

        if missing_files:
            logger.warning(
                f"  {len(missing_files)} symbols had no CSV: "
                f"{missing_files[:10]}{'...' if len(missing_files) > 10 else ''}"
            )

        if not price_series:
            raise RuntimeError(
                "No price data loaded. "
                "Check that hist_dir contains *_historical.csv files."
            )

        prices_df = pd.DataFrame(price_series)   # (T_max, N_loaded)
        prices_df.index.name = "date"
        prices_df = prices_df.sort_index()

        # Compute daily simple returns
        returns_df = prices_df.pct_change()

        logger.info(f"  Price CSVs loaded : {len(price_series)} / {len(symbols)}")
        logger.info(
            f"  Returns shape     : {returns_df.shape}"
            f"  ({returns_df.shape[0]} days x {returns_df.shape[1]} stocks)"
        )
        logger.info(
            f"  Returns date range: {returns_df.index.min().date()} "
            f"--> {returns_df.index.max().date()}"
        )

        return returns_df

    def build_X(
        self,
        panel_df: pd.DataFrame,
        dates: List,
        symbols: List[str],
        feature_names: List[str],
    ) -> torch.Tensor:
        """
        Convert panel DataFrame to a 3D tensor.

        The panel has MultiIndex (date, symbol). We need to pivot it into
        shape (T, N, M):
            axis 0 (T) = time - one entry per trading day
            axis 1 (N) = stocks - one entry per stock
            axis 2 (M) = features - one entry per characteristic

        The operation:
        1. unstack(level='symbol') converts (T*N, M) --> (T, N*M) with
           MultiIndex columns (feature, symbol).
        2. Reorder columns so the inner dimension is features for
           a fixed stock - i.e. shape (T, N, M) after reshaping.

        Concretely for each time step t, X[t] is a matrix of shape (N, M)
        where X[t, i, :] is the feature vector for stock i on day t.
  
        Parameters
        ----------
        panel_df : MultiIndex DataFrame (date, symbol), M feature columns
        dates : list of T pd.Timestamps - the T axis ordering
        symbols : list of N strings - the N axis ordering
        feature_names : list of M strings - the M axis ordering

        Returns
        -------
        torch.Tensor shape (T, N, M)
        """
        logger.info("-" * 60)
        logger.info("Building X tensor (T, N, M)")
        logger.info("-" * 60)

        T = len(dates)
        N = len(symbols)
        M = len(feature_names)

        # Unstack symbol level to wide format
        # After unstacking --> index = date, cols = MultiIndex (feature, symbol)
        wide = panel_df.unstack(level="symbol")

        # Reorder columns: for each symbol, all features in order
        # We want X[t, i, :] to be the feature vector for symbol i.
        wide.columns = wide.columns.swaplevel(0, 1) # now (symbol, feature)
        wide = wide.sort_index(axis=1) # sort so symbols are grouped

        # Reindexing
        # Build the expected MultiIndex column order
        expected_cols = pd.MultiIndex.from_product(
            [symbols, feature_names], names=["symbol", "feature"]
        )
        wide = wide.reindex(index=pd.DatetimeIndex(dates), columns=expected_cols)

        # Convert to numpy --> reshape --> tensor
        arr = wide.values.astype(np.float32)   # (T, N*M)

        if arr.shape != (T, N * M):
            raise RuntimeError(
                f"Unexpected array shape after unstack: {arr.shape}. "
                f"Expected ({T}, {N * M}). "
                "This usually means date or symbol alignment failed."
            )

        arr_3d = arr.reshape(T, N, M) # (T, N, M)
        X = torch.tensor(arr_3d, dtype=torch.float32)

        logger.info(f"  X shape : {tuple(X.shape)}   (T={T}, N={N}, M={M})")
        logger.info(f"  X range : [{X.nan_to_num().min():.4f}, {X.nan_to_num().max():.4f}]")

        return X

    def build_R(
        self,
        returns_df: pd.DataFrame,
        dates: List,
        symbols: List[str],
    ) -> torch.Tensor:
        """
        Convert aligned returns DataFrame to a 2D float32 tensor.

        For the model:
            R[t] has shape (N,1) — the return vector on day t

        Returns
        -------
        torch.Tensor shape (T, N)
        """
        logger.info("-" * 60)
        logger.info("Building R tensor (T, N)")
        logger.info("-" * 60)

        arr = returns_df.values.astype(np.float32)   # (T, N)
        R = torch.tensor(arr, dtype=torch.float32)

        T, N = R.shape
        logger.info(f"  R shape : {tuple(R.shape)}   (T={T}, N={N})")
        logger.info(
            f"  R Range : [{R.nan_to_num().min():.4f}, {R.nan_to_num().max():.4f}]"
        )

        return R

    def make_rolling_splits(self, dates: List) -> List[dict]:
        """
        Compute rolling window index boundaries.

        Window structure for train_years=8, val_years=2, test_years=1:

        |-------8 years-------|--2 val years--|--1 test--|

        We work with year boundaries rather than exact day counts to match
        the paper's description. A "year" = all trading days in that
        calendar year.

        Returns
        -------
        list of dicts, one per rolling window

        """
        logger.info("-" * 60)
        logger.info("Computing rolling window splits")
        logger.info("-" * 60)

        dates_series = pd.Series(dates)
        years = sorted(dates_series.dt.year.unique().tolist())

        if len(years) < self.train_years + self.test_years:
            raise RuntimeError(
                f"Not enough years of data for rolling windows. "
                f"Have {len(years)} years, need at least "
                f"{self.train_years + self.test_years}."
            )

        # Compute the integer index of the first and
        # last trading day in that year within the dates list.
        year_to_idx = {}
        for yr in years:
            mask = dates_series.dt.year == yr
            indices = dates_series[mask].index.tolist()
            if indices:
                year_to_idx[yr] = (indices[0], indices[-1])

        splits = []
        window_num = 1

        # First training window starts at the first year in dates
        train_start_year = years[0]

        while True:
            train_end_year = train_start_year + self.train_years - 1
            val_start_year = train_end_year + 1
            test_start_year = train_end_year + self.val_years + 1
            test_end_year = test_start_year + self.test_years - 1

            # Check all required years exist in the data
            if test_end_year not in year_to_idx:
                break
            if train_start_year not in year_to_idx:
                break

            # Integer indices
            # train
            train_idx_start = year_to_idx[train_start_year][0]
            train_idx_end = year_to_idx[train_end_year][1] + 1  # exclusive

            # validation
            val_idx_start = year_to_idx[val_start_year][0]
            val_idx_end = year_to_idx[train_end_year + self.val_years][1] + 1 # exclusive

            # test
            test_idx_start = year_to_idx[test_start_year][0]
            test_idx_end = year_to_idx[test_end_year][1] + 1  # exclusive

            splits.append({
                "window": window_num,
                # calendar dates for human readability
                "train_start": dates[train_idx_start],
                "train_end": dates[train_idx_end - 1],
                "val_start": dates[val_idx_start],
                "val_end": dates[val_idx_end - 1],
                "test_start": dates[test_idx_start],
                "test_end": dates[test_idx_end - 1],
                # integer indices for tensor slicing
                "train_idx_start": train_idx_start,
                "train_idx_end": train_idx_end,
                "val_idx_start": val_idx_start,
                "val_idx_end": val_idx_end,
                "test_idx_start": test_idx_start,
                "test_idx_end": test_idx_end,
            })

            logger.info(
                f"  Window {window_num:2d}: "
                f"train {train_start_year}-{train_end_year}  "
                f"val {val_start_year}-{train_end_year + self.val_years}  "
                f"test {test_start_year}-{test_end_year}  "
                f"(train_idx {train_idx_start}:{train_idx_end}, "
                f"test_idx {test_idx_start}:{test_idx_end})"
            )

            # Roll forward by test_years
            train_start_year += self.test_years
            window_num += 1

        if not splits:
            raise RuntimeError(
                "No rolling windows could be constructed. "
                "Check that your date range covers at least "
                f"{self.train_years + self.test_years} years."
            )

        logger.info(f"  Total rolling windows: {len(splits)}")
        return splits


if __name__ == "__main__":

    data_dir = "../data"

    loader = DataLoader(
        panel_path = f"{data_dir}/panel_characteristics_norm.parquet",
        hist_dir = f"{data_dir}/historical/",
        train_years = 8,
        val_years = 2,
        test_years = 1,
    )

    # Print full summary
    loader.describe()

    # Verify tensor shapes
    X, R, dates, symbols = loader.get_tensors()
    print(f"X dtype : {X.dtype}   R dtype : {R.dtype}")

    # Verify a single window slice works correctly
    splits = loader.get_rolling_splits()
    if splits:
        sp = splits[0]
        X_tr, R_tr, X_val, R_val, X_te, R_te = loader.get_window_tensors(sp)
        print(f"\nWindow 1 tensor shapes:")
        print(f"  X_train : {tuple(X_tr.shape)}")
        print(f"  R_train : {tuple(R_tr.shape)}")
        print(f"  X_val   : {tuple(X_val.shape)}")
        print(f"  R_val   : {tuple(R_val.shape)}")
        print(f"  X_test  : {tuple(X_te.shape)}")
        print(f"  R_test  : {tuple(R_te.shape)}")

        # Verify the time dimension adds up correctly
        T_train = sp["train_idx_end"]  - sp["train_idx_start"]
        T_val   = sp["val_idx_end"]    - sp["val_idx_start"]
        T_test  = sp["test_idx_end"]   - sp["test_idx_start"]
        print(f"\n  Train days : {T_train}")
        print(f"  Val days   : {T_val}  (sub-window of train)")
        print(f"  Test days  : {T_test}")
        print(f"\n  X[t=0, stock=0-1, feature=0-4] = {X_tr[0, 0:2, :5]}")
        print(f"  R[t=1, stock=0-4] = {R_tr[1, :5]}")