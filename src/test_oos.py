"""
test_oos.py

Implements the training scheme:

For each rolling window:

    Test - after training converges, evaluate on the 1-year test slice
            with no gradient updates, collecting out-of-sample returns

    After all rolling windows:
    - Concatenate out-of-sample test returns into one full out-of-sample series
    - Compute final performance metrics (annualised Sharpe, return, volatility,
      market beta, net Sharpe)

Structure
    WindowMetrics - stores results for one rolling window
    OOSTester - main class for testing
        .run_test() - main function to test of out-of-sample window splits
        .test_window() - test on each split and get results
        .compute_metrics() - computes annualised Sharpe, return, beta, etc.
        .load_checkpoints() - restore best model weights for testing
    
"""
import json
import logging
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
from data_loader import DataLoader
from train import ModelConfig, RollingTrainer
from attention_factor_model import AttentionFactorModel
from compute_firm_characteristics import ComputeCharacteristics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

@dataclass
class WindowMetrics:
    """
    Stores all results for one rolling window.

    Metrics
        SR : Annualised Sharpe ratio (gross, before transaction costs)
        mu : Annualised mean return in % (gross)
        sigma : Annualised volatility in % (gross)
        SR_net : Annualised net Sharpe ratio (after transaction costs)
        mu_net : Annualised mean net return in %
        sigma_net : Annualised net volatility in %
        beta : Market beta of the portfolio (should be near 0 for stat arb)
        mean_cost : Mean daily transaction cost
    """

    window: int = 0
    train_start: str = ""
    train_end: str = ""
    test_start: str = ""
    test_end: str = ""

    SR: float = 0.0
    mu: float = 0.0
    sigma: float = 0.0
    SR_net: float = 0.0
    mu_net: float = 0.0
    sigma_net: float = 0.0
    beta: float = 0.0
    mean_cost: float = 0.0
    best_val_loss: float = 0.0
    best_epoch: int = 0

    # Raw return series (stored for concatenating full out-of-sample series)
    returns_gross: List[float] = field(default_factory=list)
    returns_net: List[float] = field(default_factory=list)
    test_dates: List[str] = field(default_factory=list)

class OOSTester:

    def __init__(self, config: ModelConfig):
        self.cfg = config
        self.save_dir = Path(config.save_dir)
        self.save_dir1 = self.save_dir / f"model_K{self.cfg.K}"
        self.save_dir1 = Path(self.save_dir1)
        
        # Set random seed for reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        logger.info(f"OOSTester initialised | device={config.device} | K={config.K}")

    
    def run_test(
        self,
        X: torch.Tensor,
        R: torch.Tensor,
        splits: List[dict],
        dates: List,
        symbols: List[str],
    ) -> Dict:
        """
        Parameters
            X : full characteristics tensor from data_loader - (T, N, M)
            R : full returns tensor from data_loader - (T, N)
            splits : list of split dicts from data_loader.get_rolling_splits()
            dates : list of T pd.Timestamps
            symbols : list of N ticker strings
        """
        
        N = X.shape[1]
        M = X.shape[2]

        logger.info("-" * 65)
        logger.info(f"Testing Out-of-Sample Years for K={self.cfg.K}")
        logger.info(f"  Stocks (N)     : {N}")
        logger.info(f"  Features (M)   : {M}")
        logger.info(f"  Factors (K)    : {self.cfg.K}")
        logger.info("-" * 65)

        all_window_metrics: List[WindowMetrics] = []
        oos_returns_gross = []
        oos_returns_net = []
        oos_dates = []
        oos_R_f = []
        oos_att_w = []

        X = X.to(self.cfg.device)
        R = R.to(self.cfg.device)
        oos_market_ret = [] 

        for split in splits:
            window_num = split["window"]
            logger.info("")
            logger.info("-" * 65)
            logger.info(
                f"WINDOW {window_num}/{len(splits)}  |  "
                f"Test:  {split['test_start'].date()} --> {split['test_end'].date()}"
            )
            logger.info("-" * 65)

            X_test = X[split["test_idx_start"] : split["test_idx_end"]]
            R_test = R[split["test_idx_start"] : split["test_idx_end"]]

            test_dates = dates[split["test_idx_start"] : split["test_idx_end"]]

            # Create fresh model for this window
            model = AttentionFactorModel(
                N = N,
                M = M,
                K = self.cfg.K,
                d = self.cfg.d,
                s = self.cfg.s,
                lambda_ridge = self.cfg.lambda_ridge,
                d_hidden_conv = self.cfg.d_hidden_conv,
                lambda_squash = self.cfg.lambda_squash,
                dropout = self.cfg.dropout,
            ).to(self.cfg.device)

            # Load best checkpoint and run final test
            self.load_checkpoints(model, window_num)
            test_metrics = self.test_window(model, X_test, R_test, test_dates)

            # Assign window results
            wm = WindowMetrics(
                window = window_num,
                train_start = str(split["train_start"].date()),
                train_end = str(split["train_end"].date()),
                test_start = str(split["test_start"].date()),
                test_end = str(split["test_end"].date()),
                SR = test_metrics["SR"],
                mu = test_metrics["mu"],
                sigma = test_metrics["sigma"],
                SR_net = test_metrics["SR_net"],
                mu_net = test_metrics["mu_net"],
                sigma_net = test_metrics["sigma_net"],
                beta = test_metrics["beta"],
                mean_cost = test_metrics["mean_cost"],
                returns_gross = test_metrics["returns_gross"],
                returns_net = test_metrics["returns_net"],
                test_dates = [str(d.date()) for d in test_dates],
            )
            all_window_metrics.append(wm)

            # Accumulate OOS series
            dates_aligned = test_dates[model.s:]
            oos_returns_gross.extend(test_metrics["returns_gross"])
            oos_returns_net.extend(test_metrics["returns_net"])
            oos_dates.extend([str(d.date()) for d in dates_aligned])
            oos_R_f.extend(test_metrics["R_f_daily"])
            oos_att_w.append(test_metrics["att_w"])

            self.log_window_results(wm)

            comp_char = ComputeCharacteristics()
            start_str = test_dates[0].strftime('%Y-%m-%d')
            end_str = test_dates[-1].strftime('%Y-%m-%d')
            market_returns = comp_char.fetch_market_index(start_str,end_str)
            mkt_ret = market_returns['close'].pct_change()

            oos_market_ret.append(mkt_ret)

        # Compute aggregate metrics over full OOS period
        oos_att_w = np.concatenate(oos_att_w, axis=0)
        oos_gross_arr = np.array(oos_returns_gross)
        oos_net_arr = np.array(oos_returns_net)
        oos_R_f_arr = np.array(oos_R_f)

        mkt_tot = pd.concat(oos_market_ret)
        mkt_tot = mkt_tot.sort_index()
        aligned_market_returns = None
        if market_returns is not None:
            aligned_oos_ret, aligned_market_returns = self.align_market_returns(
                mkt_tot, oos_gross_arr, oos_dates
            )

        final_metrics = self.compute_metrics(oos_gross_arr, oos_net_arr, oos_R_f_arr, True, aligned_market_returns, aligned_oos_ret)

        results = {
            "window_metrics": all_window_metrics,
            "oos_att_weights": oos_att_w,
            "oos_returns_gross": oos_returns_gross,
            "oos_returns_net": oos_returns_net,
            "oos_dates": oos_dates,
            "final_metrics": final_metrics,
            "config": asdict(self.cfg),
        }

        self.log_final_results(final_metrics, len(splits))
        self.save_results(results)

        return results

    
    def align_market_returns(
        self,
        market_returns: pd.Series,
        oos_gross: np.ndarray,
        oos_dates: List[str],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Slice the full market_returns array down to just the OOS dates used across
        all rolling windows.
        """
        mkt_ret = market_returns.copy()
        mkt_ret.index = mkt_ret.index.strftime('%Y-%m-%d')

        port = pd.Series(oos_gross, index=oos_dates)
        common_dates = sorted(set(mkt_ret.index).intersection(set(port.index)))
        #print("comm: ", len(common_dates))

        if not common_dates:
            logger.warning(
                "  No common dates between market_returns and oos_dates — "
                "skipping beta"
            )
            return None, None
 
        n_dropped_mkt = len(mkt_ret) - len(common_dates)
        n_dropped_port = len(port) - len(common_dates)
 
        r_mkt_common  = mkt_ret.loc[common_dates].values
        r_port_common = port.loc[common_dates].values
 
        return r_port_common, r_mkt_common
    
    
    def test_window(
        self,
        model: AttentionFactorModel,
        X_test: torch.Tensor,
        R_test: torch.Tensor,
        test_dates: List,
    ) -> Dict:
        """
        Run the best-checkpoint model on the out-of-sample test window.
        The model uses the weights from the best validation epoch.

        Parameters
            model : loaded with best checkpoint weights
            X_test : (T_test, N, M)
            R_test : (T_test, N)
            test_dates : list of T_test pd.Timestamps

        Returns
            dict with performance metrics + raw return series
        """
    
        model.eval()
        model.reset_omega_prev()
        model.omega_prev = model.omega_prev.to(self.cfg.device)

        with torch.no_grad():
            seq_output = model.forward_sequence(X_test, R_test)

        returns_gross = seq_output["returns_gross"].cpu().numpy()
        returns_net = seq_output["returns_net"].cpu().numpy()
        costs = seq_output["costs"].cpu().numpy()
        att_weight = seq_output["att_w"].cpu().numpy()

        R_f_daily = X_test[model.s:, 0, 23]
        R_f_daily = R_f_daily.cpu().numpy()

        # compute performance metrics
        metrics = self.compute_metrics(returns_gross, returns_net, R_f_daily, oos=False)
        metrics["mean_cost"] = float(costs.mean())
        metrics["returns_gross"] = returns_gross.tolist()
        metrics["returns_net"] = returns_net.tolist()
        metrics["R_f_daily"] = R_f_daily.tolist()
        metrics["att_w"] = att_weight

        return metrics
    

    def compute_metrics(
        self,
        returns_gross: np.ndarray,
        returns_net: np.ndarray,
        R_f_daily: np.ndarray,
        oos: bool = False,
        market_returns: Optional[np.ndarray] = None,
        oos_gross: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Compute the performance metrics.

        Parameters
            returns_gross : gross returns obtained from the model generated portfolio
            returns_net : net returns obtained from the model generated portfolio
            R_f_daily : risk-free daily return
            oos : boolean variable to identify the computation is for one window or all window jointly
            market_returns : market's daily return to compute beta
            oos_gross : out-of-sample date aligned gross returns to compute beta 

        Metrics
            SR : Annualised Sharpe ratio (gross)
            mu : Annualised mean return in %
            sigma : Annualised volatility in %
            SR_net : Annualised net Sharpe ratio
            mu_net : Annualised mean net return in %
            sigma_net : Annualised net volatility in %
            beta : Market beta of the portfolio.
                    Estimated as Cov(R_port, R_market) / Var(R_market).
                    For statistical arbitrage this should be near zero - the
                    strategy should be market-neutral.
                    We approximate using the mean return as a proxy here
                    since we do not have the market returns in this function.
                    Full market beta is computed in the results summary.
        """
        # Gross returns
        excess_gross = returns_gross - R_f_daily

        mu1 = excess_gross.mean()
        std1 = excess_gross.std(ddof=1)

        if std1 < 1e-10:
            SR = 0.0
        else:
            SR = float(np.sqrt(252) * mu1 / std1)

        # Report raw annual return/volatility
        compounded1 = np.prod(1 + returns_gross)
        mu1_ann = float(((compounded1 ** (252/returns_gross.shape[0])) - 1) * 100)
        sig1_ann = float(returns_gross.std(ddof=1) * np.sqrt(252) * 100)

        # Net returns
        excess_net = returns_net - R_f_daily

        mu2 = excess_net.mean()
        std2 = excess_net.std(ddof=1)

        if std2 < 1e-10:
            SR_net = 0.0
        else:
            SR_net = float(np.sqrt(252) * mu2 / std2)

        compounded2 = np.prod(1 + returns_net)
        mu2_ann = float(((compounded2 ** (252/returns_net.shape[0])) - 1) * 100)
        sig2_ann = float(returns_net.std(ddof=1) * np.sqrt(252) * 100)

        beta = 0.0
        if oos:
            if market_returns is None:
                logger.warning(
                    "  oos=True but market_returns not provided - beta=0.0"
                )
            elif len(market_returns) != len(oos_gross):
                logger.warning(
                    f"  market_returns length ({len(market_returns)}) != "
                    f"returns_gross length ({len(oos_gross)}) - beta=0.0"
                )
            else:
                var_mkt = market_returns.var(ddof=1)
                if var_mkt < 1e-12:
                    beta = 0.0
                
                cov = np.cov(oos_gross, market_returns, ddof=1)[0, 1]
                beta = float(cov / var_mkt)

        return {
            "SR": SR,
            "mu": mu1_ann,
            "sigma": sig1_ann,
            "SR_net": SR_net,
            "mu_net": mu2_ann,
            "sigma_net": sig2_ann,
            "beta": beta,  # placeholder
        }

    
    def load_checkpoints(self, model: AttentionFactorModel, window_num: int) -> None:
        """
        Load the best checkpoint for the window into the model.

        strict=True (default) means every key in the checkpoint must match
        the model's state_dict exactly
        """
        path = self.save_dir1 / f"K{self.cfg.K}_checkpoint_window_{window_num:02d}.pt"
        if not path.exists():
            logger.warning(
                f"  Checkpoint not found at {path} - using current weights for testing"
            )
            return
        state = torch.load(path, map_location=self.cfg.device)
        model.load_state_dict(state, strict=True)
        logger.info(f"  Loaded best checkpoint from {path}")


    def log_window_results(self, wm: WindowMetrics) -> None:
        """Log one window's test results"""
        print()
        logger.info(
            f"Window {wm.window} Test Results "
            f"({wm.test_start} --> {wm.test_end}):"
        )
        logger.info(f"    {'Metric':<12} {'Gross':>10} {'Net':>10}")
        logger.info(f"    {'─'*34}")
        logger.info(f"    {'SR':<12} {wm.SR:>10.3f} {wm.SR_net:>10.3f}")
        logger.info(f"    {'Mean Ret %':<12} {wm.mu:>10.2f} {wm.mu_net:>10.2f}")
        logger.info(f"    {'Vol %':<12} {wm.sigma:>10.2f} {wm.sigma_net:>10.2f}")
        logger.info(f"    {'Avg Cost':<12} {wm.mean_cost:>10.6f}")

    def log_final_results(self, metrics: Dict, n_windows: int) -> None:
        """Log aggregate out-of-sample results."""
        print()
        logger.info("-" * 65)
        logger.info(f"FULL OUT-OF-SAMPLE RESULTS  ({n_windows} windows)")
        logger.info("-" * 65)
        logger.info(f"  {'Metric':<20} {'Gross':>12} {'Net':>12}")
        logger.info(f"  {'-'*50}")
        logger.info(
            f"  {'Sharpe (ann.)':<20} "
            f"{metrics['SR']:>12.3f} {metrics['SR_net']:>12.3f}"
        )
        logger.info(
            f"  {'Mean Return %':<20} "
            f"{metrics['mu']:>12.2f} {metrics['mu_net']:>12.2f}"
        )
        logger.info(
            f"  {'Volatility %':<20} "
            f"{metrics['sigma']:>12.2f} {metrics['sigma_net']:>12.2f}"
        )
        logger.info("-" * 65)

    def save_results(self, results: Dict) -> None:
        """
        Save the complete results dictionary to disk as JSON.
        WindowMetrics objects are converted to plain dicts for JSON serialisation.
        """
        serialisable = {
            "window_metrics": [asdict(wm) for wm in results["window_metrics"]],
            "oos_returns_gross": results["oos_returns_gross"],
            "oos_returns_net": results["oos_returns_net"],
            "oos_dates": results["oos_dates"],
            "final_metrics": results["final_metrics"],
            "config": results["config"],
        }

        path = Path(self.save_dir / "results_metadata") 
        path.mkdir(parents=True, exist_ok=True)
        path1 = path / f"results_K{self.cfg.K}.json"

        with open(path1, "w") as f:
            json.dump(serialisable, f, indent=2)
        logger.info(f"  Results saved --> {path1}")


if __name__ == "__main__":

    data_dir = "../data"

    # Load the data
    loader = DataLoader(
        panel_path = f"{data_dir}/panel_characteristics_norm.parquet",
        hist_dir = f"{data_dir}/historical/",
        train_years = 8,
        val_years = 2,
        test_years = 1,
    )
    loader.describe()

    X, R, dates, symbols = loader.get_tensors()
    splits = loader.get_rolling_splits()

    logger.info(f"X shape: {tuple(X.shape)}  R shape: {tuple(R.shape)}")
    logger.info(f"Rolling windows: {len(splits)}")

    K_list = [5]
    final_content = {}

    for k in K_list:
        # Configure and run trainer
        config = ModelConfig(
            K = k,
            d = 32,
            s = 30,
            epochs = 30,
            lr = 0.003,
            weight_decay = 0.05,
            lambda_var = 50.0,
            patience = 7,
            save_dir = f"{data_dir}/../model_results_temp",
            device="cpu"
        )

        # Train-validation-test
        trainer = RollingTrainer(config)
        trainer.run_train(X, R, splits, dates, symbols)
        tester = OOSTester(config)
        results = tester.run_test(X, R, splits, dates, symbols)
        final_content[k] = results

        # Print summary table
        print("\n" + "-" * 65)
        print(f"RESULTS SUMMARY   K={config.K}")
        print("-" * 65)
        print(f"{'Window':<8} {'Test Period':<22} {'SR':>6} {'SR_net':>8} {'mu%':>7} {'mu_net%':>9}")
        print("-" * 65)
        for wm in results["window_metrics"]:
            print(
                f"{wm.window:<8} "
                f"{wm.test_start[:7]}-->{wm.test_end[:7]:<12} "
                f"{wm.SR:>6.3f} "
                f"{wm.SR_net:>8.3f} "
                f"{wm.mu:>7.2f} "
                f"{wm.mu_net:>9.2f}"
            )
        print("-" * 65)
        fm = results["final_metrics"]
        print(
            f"{'FULL OOS':<8} {'':22} "
            f"{fm['SR']:>6.3f} "
            f"{fm['SR_net']:>8.3f} "
            f"{fm['mu']:>7.2f} "
            f"{fm['mu_net']:>9.2f}"
        )
        print("-" * 65)

    print("Factor number-wise Out-of-Sample Annualized Performance")
    print("-" * 65)