"""show_results.py

This file displays final metrics (SR, mu, sigma etc) table jointly
for all tested factor's results
It plots cumulative returns altogather in a lineplot
It plots Attention factor composition of each stocks for first n factors
"""
from sklearn.manifold import TSNE
from adjustText import adjust_text
import umap
import logging
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates

logger = logging.getLogger(__name__)

def plot_beta_loading(
    symbols: list,
    sector_map: dict,
    n_comp=2,
    pxty=20,
    metrics: dict = None,
    color_map: dict = None,
):

    tsne = TSNE(
        n_components=n_comp,
        perplexity=pxty,
        learning_rate="auto",
        init="pca",
        random_state=42
    )

    """reducer = umap.UMAP(
        n_neighbors=10,
        random_state=42
    )"""

    embedding = tsne.fit_transform(metrics["oos_betas"][0])

    colors = []
    texts = []

    for stock, sec in sector_map.items():
        colors.append(get_color_for_sector(sec, color_map))

    fig = plt.figure(figsize=(12,8))
    plt.scatter(embedding[:,0], embedding[:,1], c=colors)

    for i in range(1,12):
        texts.append(
            plt.text(
                embedding[i,0],
                embedding[i,1],
                symbols[i*7],
                fontsize=8
            )
        )   
    adjust_text(texts)
    plt.xlabel("t-SNE Dimension 1", fontsize=14)
    plt.ylabel("t-SNE Dimension 2", fontsize=14)
    plt.tight_layout()

    return fig

def show_all_oos_final_results(final_metrics) -> None:
    """
    This function displays final combined matrics as a table for all tested models
    and also plots lineplot of cumulative returns

    Parameter
        final_metrics : dictionary containing each tested models' metrics

    """
    print()
    print("-"*68)
    print("-"*15, "Out Of Sample Annualized Performance", "-"*15)
    print("-"*68)
    print(f"{'K':<10} {'SR':>4} {'mu':>6} {'sigma':>7} {'SR_net':>8} {'mu_net':>8} {'sigma_net':>10} {'Beta':>5}")
    print("-"*68)
    # Dictionary to store results for all K (factors)
    returns_dict = {}
    
    for k, metric in final_metrics.items():
        print(
            f"{k:<10}"
            f"{metric["final_metrics"]["SR"]:>6.2f}"
            f"{metric["final_metrics"]["mu"]:>7.2f}"
            f"{metric["final_metrics"]["sigma"]:>7.2f}"
            f"{metric["final_metrics"]["SR_net"]:>8.2f}"
            f"{metric["final_metrics"]["mu_net"]:>9.2f}"
            f"{metric["final_metrics"]["sigma_net"]:>10.2f}"
            f"{metric["final_metrics"]["beta"]:>8.2f}"
        )

        returns_dict["RF_returns"] = metric["oos_R_f_daily"]
        returns_dict["Nifty100_returns"] = metric["oos_nifty100"]
        returns_dict[f"K={k}"] = metric["oos_returns_net"]
        
        dates_str = metric["oos_dates"]
        dates = pd.to_datetime(dates_str)
    
    print()
    # Convert to dataframe having dates as index and K (=1,3,etc) as column with returns
    df_returns = pd.DataFrame(returns_dict, index=dates)

    # Cumulative returns
    df_cumulative = ((df_returns + 1).cumprod() - 1)*100

    fig = plt.figure(figsize=(11, 7))

    # Loop through each column and draw them
    for model_name in df_cumulative.columns:
        plt.plot(
            df_cumulative.index, 
            df_cumulative[model_name], 
            label=model_name, 
            linewidth=2
        )

    plt.xlabel("Year", fontsize=18, labelpad=10)
    plt.ylabel("Cumulative Return", fontsize=18, labelpad=10)

    ax = plt.gca() # Get current active axis to modify ticks
    ax.xaxis.set_major_locator(mdates.YearLocator(1)) # Force a tick every 1 year
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y')) # Display format as YYYY

    # Rotate X-ticks by 45 degrees so they don't overlap
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.yticks(fontsize=14)

    plt.grid(True, which='both', linestyle='-', color='#e0e0e0', linewidth=0.8)
    plt.legend(loc="upper left", fontsize=13, frameon=True)
    plt.xlim(df_cumulative.index.min(), df_cumulative.index.max())
    plt.tight_layout()

    out_path = Path("../result_figures/Cumulative_returns_test.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print("Saved test figure of cumulative returns")

    plt.show()
    

# Fixed colour mapping so the legend is consistent across multiple figures.
# actually contains (yfinance sector strings for NSE stocks).
Color_maps = {
    "Technology": "#9B7FC7",
    "Financial Services":"#3360DD",
    "Energy":"#7B4B3A",
    "Healthcare":"#5911DF",
    "Consumer Defensive":"#E55B20",
    "Consumer Cyclical":"#E07BC0",
    "Industrials":"#C0392B",
    "Basic Materials":"#D91A89",
    "Utilities":"#E7B32E",
    "Real Estate":"#2D833B",
    "Communication Services":"#5DC8D8",
    "Unknown":"#999999",
}

def get_color_for_sector(sector: str, color_map: Dict[str, str]) -> str:
    """Return the colour for a sector, with a safe fallback."""
    return color_map.get(sector, color_map.get("Unknown", "#999999"))

def plot_factor_portfolio_weights(
    symbols: list,
    sector_map: dict,
    n_factors_to_show=6,
    top_n=10,
    metrics: dict = None,
    color_map: dict = None,
    figsize_per_panel: tuple = (4.2, 3.6),
) -> plt.Figure:

    """
    We display the top company weights for some primary factors.
    This describes the composition of each factor's portfolio.
    Which stocks make it up and how much weight each one carries 
    within that single factor.

    omega_F has shape (K, N). Row k is a portfolio: N weights that sum to 1
    (because of the row-wise Softmax). We will do slicing of row and sorting 
    in descending order to get top contributors.

    Parameters
        symbols : list of N stock strings
        sector_map : dict {symbol -> sector name}
        n_factors_to_show : how many factor panels to draw
        top_n : how many top stocks per factor
        color_map : the sector->colour dictionary
        figsize_per_panel : (width, height) inches for each subplot

    Returns
        matplotlib.figure.Figure

    Layout
        2 rows x 3 columns.
        If n_factors_to_show is not divisible by 3, falls back to a
        ceil(n/3) x 3 grid with empty panels hidden.
    """

    if color_map is None:
        color_map = Color_maps

    # Access omega_F for this date
    omega_F = np.abs(metrics["oos_att_weights"][246]) # (K, N)

    # Determine grid layout
    n_cols = 3
    n_rows = int(np.ceil(n_factors_to_show / n_cols))

    fig_w = figsize_per_panel[0] * n_cols
    fig_h = figsize_per_panel[1] * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    axes = np.atleast_2d(axes).reshape(n_rows, n_cols)

    fig.suptitle(
        "Attention Weights for Factor Portfolio Weights",
        fontsize=15, fontweight="bold", y=1.02,
    )

    sectors_used = set()   # track which sectors appear, for the legend

    # one panel per factor
    for k in range(n_factors_to_show):
        row, col = divmod(k, n_cols)
        ax = axes[row, col]

        weights_k = omega_F[k, :] # (N,)

        # Top-N stocks by weight, descending
        top_idx = np.argsort(weights_k)[::-1][:top_n] # indices, largest first
        top_weights = weights_k[top_idx]
        top_symbols = [symbols[i] for i in top_idx]
        top_sectors = [sector_map.get(s, "Unknown") for s in top_symbols]
        top_colors = [get_color_for_sector(sec, color_map) for sec in top_sectors]

        sectors_used.update(top_sectors)

        # Percentage of full universe held by these top N stocks
        pct_top_n = top_weights.sum() * 100.0

        # Horizontal bar chart, largest on top
        y_pos = np.arange(top_n)[::-1]
        ax.barh(y_pos, top_weights, color=top_colors, edgecolor="none")

        clean_labels = top_symbols
        ax.set_yticks(y_pos)
        ax.set_yticklabels(clean_labels, fontsize=9)
        ax.set_title(f"Factor {k + 1} ({pct_top_n:.1f}%)", fontsize=12)
        ax.set_xlim(left=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelsize=8)

    # Hide unused panels if n_factors_to_show isn't a multiple of n_cols
    for k in range(n_factors_to_show, n_rows * n_cols):
        row, col = divmod(k, n_cols)
        axes[row, col].axis("off")

    legend_handles = [
        mpatches.Patch(color=get_color_for_sector(sec, color_map), label=sec)
        for sec in sorted(sectors_used)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(len(legend_handles), 5),
        bbox_to_anchor=(0.5, -0.02 - 0.02 * n_rows),
        frameon=False,
        fontsize=9,
    )

    plt.tight_layout(rect=[0, 0.04 + 0.02 * n_rows, 1, 0.97])

    return fig
