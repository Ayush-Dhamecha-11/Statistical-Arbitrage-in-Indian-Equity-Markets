import logging
from pathlib import Path
import pandas as pd
from data_loader import DataLoader
from train import RollingTrainer, ModelConfig
from test_oos import OOSTester
from signature_prep import build_signature_features, plot_signature_similarity
from show_results import show_all_oos_final_results, plot_factor_portfolio_weights, plot_beta_loading, Color_maps

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Model parameters to configure
data_dir = "../data"
years_to_train = 8
years_to_validate = 2
years_to_test = 1
K_list = [1,3,5,8,15,20,30,50]
md = ["learnable"]  # ["learnable", "pca"]
hidden_projection_dim = 32
lookback_length = 40
epoch = 30
learning_rate = 0.005
weight_decay = 0.002
lambda_var_coeff = 0.0
lambda_ridge = 0.01
lambda_squash = 0.001
early_stop_patience = 6
dev = "cpu"

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

# Replace firm characteristics with signature features
syms = []
for ele in symbols:
    sym = ele.replace("_", "").replace("historical", "").replace("chars", "")
    syms.append(sym)

#X_sig = build_signature_features(R, depth=3, window=250)
#dist_df = plot_signature_similarity(X_sig, syms, figsize=(14, 14))

# load the csv file having company sector information
metadata = pd.read_csv('../data/stock_metadata.csv')
symbols = metadata['symbol'].tolist()
sectors = metadata['sector'].unique().tolist()
long_names = metadata['name'].tolist()
sector_map = metadata.set_index('symbol')['sector'].to_dict()

# store final metrics of each tested model
final_content = {}
out_path = Path(f"../result_figures")
out_path.mkdir(parents=True, exist_ok=True)

# Training loop

for k in K_list:
    for mdd in md:
        # Configure and run trainer
        config = ModelConfig(
            K = k,
            d = hidden_projection_dim,
            s = lookback_length,
            epochs = epoch,
            lr = learning_rate,
            weight_decay = weight_decay,
            d_hidden_conv = hidden_projection_dim,
            lambda_var = lambda_var_coeff,
            lambda_ridge = lambda_ridge,
            lambda_squash = lambda_squash,
            patience = early_stop_patience,
            save_dir = f"{data_dir}/../model_results_{mdd}",
            device=dev,
            mode=mdd,
        )
        
        #Training
        #trainer = RollingTrainer(config)
        #trainer.run_train(X, R, splits, dates, symbols)
        tester = OOSTester(config)

        # Testing
        results = tester.run_test(X, R, splits, dates, symbols)
        final_content[f"{k}_{mdd}"] = results

        # Print summary table
        """print("\n" + "-" * 65)
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
        print()"""

# Result Plottings and final table
"""for k in K_list:
    for mdd in md:
        results = final_content[f"{k}_{mdd}"]
        fig1 = plot_factor_portfolio_weights(
            symbols = symbols,
            sector_map = sector_map,
            n_factors_to_show = min(k,6),
            top_n = 10,
            metrics = results,
            color_map = Color_maps,
        )

        out_path1 = out_path / f"Attention_weights_K={k}_{mdd}_test.png"
        fig1.savefig(out_path1, dpi=120, bbox_inches="tight")
        print(f"Saved test figure (K={k}) to {out_path}")

        if k == 1:
            continue

        fig2 = plot_beta_loading(
            symbols = symbols,
            sector_map = sector_map,
            n_comp = 2,
            pxty = 20,
            metrics = results,
            color_map = Color_maps,
        )

        out_path2 = out_path / f"t-SNE_plot_K={k}_{mdd}_test.png"
        fig2.savefig(out_path2, dpi=120, bbox_inches="tight")
        print(f"Saved test figure (K={k}) to {out_path}")"""


show_all_oos_final_results(final_content)