import logging
from pathlib import Path
import pandas as pd
from data_loader import DataLoader
from train import RollingTrainer, ModelConfig
from test_oos import OOSTester
from show_results import show_all_oos_final_results, plot_factor_portfolio_weights, Color_maps

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


data_dir = "../data"

years_to_train = 8
years_to_validate = 2
years_to_test = 1

# Load the data
loader = DataLoader(
    panel_path = f"{data_dir}/panel_characteristics_norm.parquet",
    hist_dir = f"{data_dir}/historical/",
    train_years = years_to_train,
    val_years = years_to_validate,
    test_years = years_to_test,
)
loader.describe()

# get tensors on which the model will be trained
X, R, dates, symbols = loader.get_tensors()
splits = loader.get_rolling_splits()

logger.info(f"X shape: {tuple(X.shape)}  R shape: {tuple(R.shape)}")
logger.info(f"Rolling windows: {len(splits)}")

# load the csv file having company sector information
metadata = pd.read_csv('../data/stock_metadata.csv')
symbols = metadata['symbol'].tolist()
sectors = metadata['sector'].unique().tolist()
long_names = metadata['name'].tolist()
sector_map = metadata.set_index('symbol')['sector'].to_dict()

# Model parameters to configure
K_list = [3,5]
hidden_projection_dim = 16
lookback_length = 30
epoch = 30
learning_rate = 0.003
weight_decay = 0.005
lambda_var_coeff = 5.0
early_stop_patience = 7

# store final metrics of each tested model
final_content = {}
out_path = Path(f"../result_figures")
out_path.mkdir(parents=True, exist_ok=True)

# Training loop

for k in K_list:
    # Configure and run trainer
    config = ModelConfig(
        K = k,
        d = hidden_projection_dim,
        s = lookback_length,
        epochs = epoch,
        lr = learning_rate,
        weight_decay = weight_decay,
        lambda_var = lambda_var_coeff,
        patience = early_stop_patience,
        save_dir = f"{data_dir}/../model_results",
    )
    
    #Training
    trainer = RollingTrainer(config)
    trainer.run_train(X, R, splits, dates, symbols)

# Testing loop
for k in K_list:
    
    config = ModelConfig(
        K = k,
        d = hidden_projection_dim,
        s = lookback_length,
        epochs = epoch,
        lr = learning_rate,
        weight_decay = weight_decay,
        lambda_var = lambda_var_coeff,
        patience = early_stop_patience,
        save_dir = f"{data_dir}/../model_results",
    )
    
    tester = OOSTester(config)

    # Testing
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
    print()

# Result Plottings and final table

for k in K_list:
    results = final_content[k]
    fig = plot_factor_portfolio_weights(
        symbols = symbols,
        sector_map = sector_map,
        n_factors_to_show = min(k,6),
        top_n = 10,
        metrics = results,
        color_map = Color_maps,
    )

    out_path1 = out_path / f"Attention_weights_K={k}_test.png"
    fig.savefig(out_path1, dpi=120, bbox_inches="tight")
    print(f"Saved test figure (K={k}) to {out_path}")


show_all_oos_final_results(final_content)