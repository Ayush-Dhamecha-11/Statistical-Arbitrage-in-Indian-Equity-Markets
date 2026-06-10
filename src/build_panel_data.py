
import os
import pandas as pd
import numpy as np
from datetime import datetime
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

class DataPanelBuilder:

    def __init__(self, data_dir='../data'):

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        char_data_dir = Path("../data/characteristics")
        self.chars_dict = {}

        for file_path in char_data_dir.glob("*.csv"):

            stock_symbol = file_path.stem
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            self.chars_dict[stock_symbol] = df

            
    def build_panel(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
            
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
        if not self.chars_dict:
            logger.error("chars_dict is empty — cannot build panel")
            return pd.DataFrame(), pd.DataFrame()

        # Stack all stocks into (date, symbol) MultiIndex
        frames = []
        for symbol, df in self.chars_dict.items():
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
        rolling_days = 30 # monthly window

        daily_rfr = panel_raw["RFR"].groupby(level="date").first()
        rfr_z_val = (
            daily_rfr - daily_rfr.rolling(rolling_days, min_periods=5).mean()
        ) / daily_rfr.rolling(rolling_days, min_periods=5).std()

        rfr_z_val = rfr_z_val.replace([np.inf, -np.inf], np.nan).fillna(0)
        panel_raw["RFR"] = rfr_z_val

        for col in char_cols:
            # Rank within each date's cross-section
            panel_norm[col] = (panel_raw_extended[col]
                .groupby(level="date")
                .transform(lambda x: x.rank(pct=True))
            )
            # Normalise the median features over time-horizon
            med_col = f"{col}_csmedian"
            if med_col in panel_norm.columns:
                extract_col = (
                    panel_raw_extended[med_col]
                    .groupby(level="date")
                    .first())
                rolling_mean = extract_col.rolling(window=rolling_days, min_periods=5).mean()
                rolling_std = extract_col.rolling(window=rolling_days, min_periods=5).std()

                z_val_col = (extract_col - rolling_mean) / rolling_std
                z_val_col = z_val_col.replace([np.inf, -np.inf], np.nan).fillna(0)
                panel_norm[f"{med_col}"] = z_val_col

        dates = panel_norm.index.get_level_values("date")
        logger.info(
            f"Final panel: {panel_norm.shape[0]:,} rows × {panel_norm.shape[1]} cols | "
            f"{dates.min()} --> {dates.max()}"
        )

        return panel_raw_extended, panel_norm

    def save_data(
        self,
        panel_raw: Optional[pd.DataFrame] = None,
        panel_norm: Optional[pd.DataFrame] = None,
    ) -> None:
        
        """Save all computed data to disk."""
        logger.info("Saving data to disk...")

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

        try:
            info_path = self.data_dir / "data_info.json"

            # Read the existing JSON data that Step 1 created
            if info_path.exists():
                with open(info_path, "r") as f:
                    current_info = json.load(f)
            else:
                # Fallback if the file doesn't exist for some reason
                current_info = {}

            # Add new Characteristics info to the existing dictionary
            current_info["panel_shape"] = list(panel_raw.shape) if panel_raw is not None else None

            # Write the combined dictionary back to the same file
            with open(info_path, "w") as f:
                json.dump(current_info, f, indent=2)

            logger.info(f"Updated data info --> {info_path}")

        except Exception as e:
            logger.error(f"Error updating data info: {e}")


    def run_panel_build(self):
        """
        Returns
            panel_norm : pd.DataFrame, MultiIndex (date, symbol), rank-normalised
        """

        if self.chars_dict is not None:
            panel_raw, panel_norm = self.build_panel()
        else:
            logger.info("Characteristic data empty!")

        self.save_data(panel_raw, panel_norm)

        logger.info("-" * 60)
        logger.info("Data Panel preparation completed successfully!")
        logger.info("-" * 60)

        return panel_norm


def main():

    # Initialise and run
    builder = DataPanelBuilder(data_dir='../data')
    panel_norm = builder.run_panel_build()

    # Summary
    print("\n" + "-" * 60)
    print("Data Summary")
    print("-" * 60)
 
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