"""
Compute time-augmented path signatures from stock return data.

Replaces the firm characteristics tensor X : (T, N, M) with
signature features S : (T, N, sig_dim) computed from the 
return series R : (T, N).
"""

import math
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, leaves_list
import signatory

def time_augment(returns_window: torch.Tensor) -> torch.Tensor:
    """
    Convert a 1D return series into a 2D time-augmented path.

    Parameters
        returns_window : W daily returns for one stock

    Returns
        path : (W, 2)
                Column 0: time index normalised to [0, 1]
                Column 1: raw return values
    """
    W = returns_window.shape[0]
    t = torch.linspace(0.0, 1.0, W, device=returns_window.device, dtype=returns_window.dtype) # (W,)
    path = torch.stack([t, returns_window], dim=1) # (W, 2)
    return path


def sig_dim(path_dim: int, depth: int) -> int:
    """
    Number of terms in a truncated signature at given depth.
    """
    return sum(path_dim ** k for k in range(1, depth + 1))

def build_signatures_signat(
    R:      torch.Tensor,
    depth:  int,
    window: int,
) -> torch.Tensor:
    """
    Build signature feature tensor using the signatory library.

    Parameters
        R : full return tensor from data_loader - (T, N)
        depth : signature truncation depth
        window : lookback window in trading days

    Returns
        X_sig : (T, N, sig_dim)
                X_sig[t, i, :] is the signature of stock i's return path
                over the window [t-window, t-1]. Zero-filled for t < window.
    """
    T, N = R.shape
    sdim  = sig_dim(2, depth) # 2 channels: time + return
    X_sig = torch.zeros(T, N, sdim, device=R.device, dtype=R.dtype)

    for t in range(window, T):

        # Slice the trailing window for ALL N stocks at once
        R_win = R[t - window : t].T # (N, window)

        # Build time-augmented paths for all N stocks
        # time_vec is the same for every stock
        time_vec = torch.linspace(0.0, 1.0, window, device=R.device, dtype=R.dtype) # (window,)

        # Expand time_vec to match N stocks
        time_exp = time_vec.unsqueeze(0).expand(N, -1) # (N, window)

        # Stack to shape (N, window, 2)
        paths = torch.stack([time_exp, R_win], dim=2)   # (N, window, 2)

        # Compute signatures for all N stocks at once using signatory
        sigs = signatory.signature(paths, depth, basepoint=False) # (N, sdim)

        X_sig[t] = sigs

    return X_sig

def build_signature_features(
    R:             torch.Tensor,
    depth:         int  = 3,
    window:        int  = 252,
) -> torch.Tensor:
    """
    Builds the signature feature tensor.

    Parameters

        R : torch.Tensor - (T, N)
            Daily return tensor from NSEDataLoader.get_tensors().

        depth : int, default 3
                Signature truncation depth.
                depth=2 →  3 features  (coarse, fast)
                depth=3 → 14 features  (recommended starting point)
                depth=4 → 30 features  (richer, slower)
                depth=5 → 62 features  (approaches original 53-feature X)

        window : int, default 252
                Lookback window in trading days.

    Returns
        X_sig : torch.Tensor - (T, N, sig_dim)
                X_sig[t, i, :] is the signature of stock i's time-augmented
                return path over days [t-window, t-1].
    """
    X_sig = build_signatures_signat(R, depth, window)

    # Cross-sectional z-score normalisation across N stocks on each day.
    # X_sig: (T, N, sig_dim)
    mean = X_sig.mean(dim=1, keepdim=True) # (T, 1, sig_dim)
    std  = X_sig.std(dim=1, keepdim=True).clamp(min=1e-8)
    X_sig = (X_sig - mean) / std
    # Re-zero the warm-up period
    X_sig[:window] = 0.0

    return X_sig

def plot_signature_similarity(
    X_sig,
    stock_names,
    figsize=(14, 12),
):
    """
    Visualise similarity between assets using signature vectors.

    Parameters
        X_sig : torch.Tensor or ndarray - (T,N,M)
        stock_names : list[str]

    Returns
        distance_matrix : ndarray
    """

    signatures = X_sig.mean(axis=0)
    condensed = pdist(signatures, metric="euclidean")  # condensed distance matrix
    distance_matrix = squareform(condensed)
    Z = linkage(condensed, method="average")
    order = leaves_list(Z)

    # pairwise Euclidean distances
    distance_matrix = distance_matrix[order][:, order]

    ordered_names = [stock_names[i] for i in order]

    distance_df = pd.DataFrame(
        distance_matrix,
        index=ordered_names,
        columns=ordered_names,
    )

    fig = plt.figure(figsize=figsize)

    ax = sns.heatmap(
        distance_df,
        annot=False,
        square=True,
        fmt=".2f",
        cmap="viridis",
        linewidths=0,
        xticklabels=True,
        yticklabels=True,
        cbar_kws={"label": "Signature Distance"},
    )

    ax.tick_params(axis='both', labelsize=8) 

    plt.suptitle(
        "Heatmap of Assets Closeness using Signature Features",
        fontsize=18,
        fontweight="bold",
        y=0.95,
        x=0.5,
    )

    out_path = Path("../sig_result_figures/clustermap_stock_signatures.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print("Saved test figure of cumulative returns")
    print("Saved heatmap of signature distances")

    return distance_df

if __name__ == "__main__":
    import torch

    torch.manual_seed(0)
    T, N = 3500, 78
    R = torch.randn(T, N) * 0.01   # synthetic daily returns

    print("Testing with signatory backend...")
    X_sig = build_signature_features(R, depth=3, window=40)

    print(f"R shape:     {tuple(R.shape)}")
    print(f"X_sig shape: {tuple(X_sig.shape)}")
    print(f"Expected:    ({T}, {N}, {sig_dim(2, 3)})")
