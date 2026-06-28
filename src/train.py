"""
train.py

Implements the training scheme:

For each rolling window:
    1. Train - run Adam for 30 epochs on the 8-year training slice
                 minimising the combined net Sharpe + explained variance loss
    2. Validate - after each epoch, evaluate on the 2-year validation
                  slice (sub-window of training, no gradient updates)
                  Use validation loss for early stopping and best-model selection

Structure
    ModelConfig - dataclass holding all hyperparameters
    EarlyStopping - helper class, stops training when val loss stagnates
    RollingTrainer - main class, containes central work
        .run_train() - entry point: loops over all rolling windows
        .train_window() - one full training run for one window
        .run_epoch() - one pass over the training data
        .evaluate() - evaluation pass (no gradients)
        .save_checkpoints() - save best model weights per window
"""

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import DataLoader
from attention_factor_model import AttentionFactorModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """
    All hyperparameters for the Attention Factor Model and training loop.

    Model hyperparameters
        K : Number of latent attention factors
        d : Attention hidden dimension
        s : LongConv lookback window (days)
        d_hidden_conv : LongConv hidden channels
        lambda_ridge : Ridge penalty for beta computation, Default: 1e-4
        lambda_squash : LongConv squash strength, Default: 0.001
        dropout : Dropout rate, Default: 0.1

    Training hyperparameters
        epochs : Passes over training data per window, Default: 30
        lr : Adam learning rate, Default: 0.003
        weight_decay : Adam weight decay (applied to LongConv), Default: 0.05
        lambda_var : Weight on explained variance term in loss, Default: 100
        R_f_annual : Annual risk-free rate for Sharpe computation
        patience : Early stopping patience (epochs without val improvement)
        min_delta : Minimum improvement in val loss to count as improvement
    """

    # Model
    K: int = 8
    d: int = 32
    s: int = 30
    d_hidden_conv: int = 32
    lambda_ridge: float = 1e-2
    lambda_squash: float = 0.001
    dropout: float = 0.1

    # Training
    epochs: int = 30
    lr: float = 0.003
    weight_decay: float = 0.05
    lambda_var: float = 100.0
    patience: int = 7
    min_delta: float = 1e-4
    mode: str = "learnable"

    device: str = "auto"
    save_dir: str = f"../model_results_{mode}"
    seed: int = 0

    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

class EarlyStopping:
    """
    Stops training when the validation loss has not improved for `patience`
    consecutive epochs.
    We use validation data to select tuning parameters. Early stopping
    is a way to prevent overfitting on the training data while using 
    validation loss as the stopping criterion.

    Parameters
        patience : how many epochs to wait after last improvement
        min_delta : minimum change to count as improvement
    """

    def __init__(self, patience=7, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        """
        Call after each epoch with the current validation loss.
        Returns True if training should stop.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def reset(self):
        """Reset state for a new rolling window."""
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False


class RollingTrainer:
    """
    Organize the full rolling window training, validation, and testing work.

    Parameters
        config : ModelConfig
    """

    def __init__(self, config: ModelConfig):
        self.cfg = config
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir1 = self.save_dir / f"model_K{self.cfg.K}"
        self.save_dir1 = Path(self.save_dir1)
        self.save_dir1.mkdir(parents=True, exist_ok=True)

        # Set random seed for reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        logger.info(f"RollingTrainer initialised | device={config.device} | K={config.K}")

    def run_train(
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
        logger.info(f"Rolling Window Training Pipeline for K={self.cfg.K}")
        logger.info(f"  Stocks (N)     : {N}")
        logger.info(f"  Features (M)   : {M}")
        logger.info(f"  Factors (K)    : {self.cfg.K}")
        logger.info(f"  Windows        : {len(splits)}")
        logger.info(f"  Epochs/window  : {self.cfg.epochs}")
        logger.info(f"  Device         : {self.cfg.device}")
        logger.info("-" * 65)

        # Move full tensors to device once for efficient slicing
        X = X.to(self.cfg.device)
        R = R.to(self.cfg.device)

        if self.cfg.mode == "pca":
            latent_pca_factors = self.compute_pca_factor(R, self.cfg.K, 126)

        for split in splits:
            window_num = split["window"]
            logger.info("")
            logger.info("-" * 65)
            logger.info(
                f"WINDOW {window_num}/{len(splits)}  |  "
                f"Train: {split['train_start'].date()} --> {split['train_end'].date()}  |  "
                f"Test:  {split['test_start'].date()} --> {split['test_end'].date()}"
            )
            logger.info("-" * 65)

            # Slice tensors for this window
            X_train = X[split["train_idx_start"] : split["train_idx_end"]]
            R_train = R[split["train_idx_start"] : split["train_idx_end"]]

            X_val = X[split["val_idx_start"] : split["val_idx_end"]]
            R_val = R[split["val_idx_start"] : split["val_idx_end"]]

            X_test = X[split["test_idx_start"] : split["test_idx_end"]]
            R_test = R[split["test_idx_start"] : split["test_idx_end"]]

            test_dates = dates[split["test_idx_start"] : split["test_idx_end"]]

            logger.info(
                f"  Shapes - train: {tuple(X_train.shape)}  "
                f"val: {tuple(X_val.shape)}  "
                f"test: {tuple(X_test.shape)}"
            )
            
            if self.cfg.mode == "pca":
                factor_train = latent_pca_factors[split["train_idx_start"] : split["train_idx_end"]]
                factor_val = latent_pca_factors[split["val_idx_start"] : split["val_idx_end"]]
            else:
                factor_train = None
                factor_val = None
            
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

            # Train on this window
            best_val_loss, best_epoch = self.train_window(
                model, X_train, R_train, X_val, R_val, factor_train, factor_val, window_num
            )

            logger.info(
                f"  Shapes - train: {tuple(X_train.shape)}  "
                f"val: {tuple(X_val.shape)}  "
                f"test: {tuple(X_test.shape)}"
            )
            
            logger.info(
                f"window {window_num} - Best Validation Loss - {best_val_loss} | Best Epoch - {best_epoch}"
            )

        logger.info("-" * 65)
        logger.info(f"Training completed for K={self.cfg.K}")
        logger.info("-" * 65)


    def train_window(
        self,
        model: AttentionFactorModel,
        X_train: torch.Tensor,
        R_train: torch.Tensor,
        X_val: torch.Tensor,
        R_val: torch.Tensor,
        factor_train: torch.Tensor,
        factor_val: torch.Tensor,
        window_num: int,
    ) -> Tuple[float, int]:
        """
        Train the model for up to `epochs` passes over the training data,
        evaluate on the validation set after each epoch.

        Saves the best model checkpoint (lowest validation loss).

        Parameters
            model : newly initialised AttentionFactorModel on device
            X_train : (T_train, N, M)
            R_train : (T_train, N)
            X_val : (T_val, N, M)
            R_val : (T_val, N)
            window_num : int, for checkpoint naming

        Returns
            best_val_loss : float
            best_epoch : int
        """

        # Optimiser
        # Adam, lr=0.003, weight_decay=0.05 on LongConv params.
        # weight decay applied specifically to the LongConv model.
        # W_K and Q do not get weight decay - they are the attention parameters
        longconv_params = list(model.longconv.parameters())
        attention_params = [model.W_K.weight, model.Q]
        optimizer = optim.Adam([
            {"params": attention_params, "weight_decay": 0.0},
            {"params": longconv_params, "weight_decay": self.cfg.weight_decay},
        ], lr=self.cfg.lr)

        # Learning rate scheduler
        # halve the LR if validation loss doesn't improve for 3 epochs.
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.8, patience=3
        )

        # Early stopping
        early_stop = EarlyStopping(
            patience=self.cfg.patience,
            min_delta=self.cfg.min_delta,
        )

        best_val_loss = float("inf")
        best_train_loss = float("inf")
        best_epoch = 0

        logger.info(
            f"  Training  T={X_train.shape[0]} days | "
            f"Validation T={X_val.shape[0]} days | "
            f"Max epochs={self.cfg.epochs}"
        )

        for epoch in range(1, self.cfg.epochs + 1):

            # One training epoch
            train_loss, train_info = self.run_epoch(
                model, X_train, R_train, factor_train, optimizer, train=True
            )

            # Validation
            val_loss, val_info = self.run_epoch(
                model, X_val, R_val, factor_val, optimizer=None, train=False
            )

            # LR scheduler step
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            # Save best checkpoint
            if val_loss < best_val_loss - self.cfg.min_delta:
                best_val_loss = val_loss
                best_epoch = epoch
                self.save_checkpoints(model, window_num)

            # Logging every epochs
            logger.info(
                f"  Epoch {epoch:3d}/{self.cfg.epochs} | "
                f"train_loss={train_loss:.4f} "
                f"(SR={train_info['sharpe_daily']:.3f}) | "
                f"val_loss={val_loss:.4f} "
                f"(SR={val_info['sharpe_daily']:.3f}) | "
                f"best_val={best_val_loss:.4f} @ ep{best_epoch} | "
                f"lr={current_lr:.5f}"
            )

            # Early stopping check
            if early_stop.step(val_loss):
                logger.info(
                    f"  Early stopping at epoch {epoch} "
                    f"(no improvement for {self.cfg.patience} epochs)"
                )
                break

        logger.info(
            f"  Training complete | best_val_loss={best_val_loss:.4f} "
            f"at epoch {best_epoch}"
        )

        return best_val_loss, best_epoch

    def run_epoch(
        self,
        model: AttentionFactorModel,
        X: torch.Tensor,
        R: torch.Tensor,
        factor_mat: torch.Tensor,
        optimizer: Optional[optim.Optimizer],
        train: bool,
    ) -> Tuple[float, Dict]:
        """
        Run one complete pass over the data

        For Training (train=True):
            - model.train() mode (dropout active)
            - gradients computed and parameters updated via optimizer.step()

        For Validation / Test (train=False):
            - model.eval() mode (dropout disabled)
            - torch.no_grad() - no computation graph
            - optimizer is None (no parameter updates)

        The model's transaction cost computation depends on omega_{t-1},
        Thus, the entire time series must be processed in order.
        Thus, one "batch" = one full rolling window.
        The gradient is computed once per epoch over the entire window.

        Parameters
            X : characteristics for this split
            R : returns for this split
            optimizer : Adam optimizer (None for evaluation)
            train : bool

        Returns
            loss_val : scalar loss value for this epoch - float
            info : metrics breakdown for logging - dict
        """

        if train:
            model.train()
        else:
            model.eval()

        # Reset portfolio state at the start of each epoch
        # we always start from zero position, same as at the start of live trading
        model.reset_omega_prev()
        model.omega_prev = model.omega_prev.to(self.cfg.device)

        context = torch.enable_grad() if train else torch.no_grad()

        with context:
            # Full forward pass over the entire time slice
            seq_output = model.forward_sequence(X, R, factor_mat, self.cfg.mode)

            # Compute loss
            loss, info = model.attention_factor_loss(
                returns_net = seq_output["returns_net"],
                omegas = seq_output["omegas"],
                eps_t = seq_output["eps_t"],
                R_t = seq_output["R_t"],
                R_f = X[model.s:, 0, 23],
                lambda_var = self.cfg.lambda_var,
            )

            if train:
                # Backpropagation

                optimizer.zero_grad()

                # Check for NaN loss before backprop to prevent corruption of parameterrs
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(
                        f"    NaN/Inf loss detected — skipping this epoch's update"
                    )
                    return float("inf"), info

                loss.backward()

                # Gradient clipping: prevents exploding gradients
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

        return loss.item(), info
    
    def compute_pca_factor(
        self,
        R: torch.Tensor,
        K: int,
        window: int = 252,
    ) -> torch.Tensor:
        """
        omega_F[t] = top-K eigenvectors of Cov(R[t-window:t]), shape (K, N).
        Uses only R[t-window:t], strictly excluding day t (no look-ahead).
        Rows before `window` are left as zero (insufficient history).
        """
        T, N = R.shape
        omega_F = torch.zeros(T, K, N, dtype=R.dtype, device=R.device)

        for t in range(window, T):
            R_win = R[t - window : t]
            cov = torch.cov(R_win.T)
            eigvals, eigvecs = torch.linalg.eigh(cov)
            omega_F[t] = eigvecs[:, -K:].T
        
        omega_F = omega_F / omega_F.abs().sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return omega_F
    

    def save_checkpoints(self, model: AttentionFactorModel, window_num: int) -> None:
        """
        Save model weights for the window.

        We save only model.state_dict() - the parameter tensors 
        The checkpoint is named by window number so each rolling window
        has its own independent best-model file
        """
        path_f = self.save_dir1 / f"K{self.cfg.K}_checkpoint_window_{window_num:02d}.pt" 
        torch.save(model.state_dict(), path_f)


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
            save_dir = f"{data_dir}/../model_results",
            device="cpu",
        )

        # Train-validation
        trainer = RollingTrainer(config)
        trainer.run_train(X, R, splits, dates, symbols)

        logger.info("-" * 20, "Training Completed", "-" * 20)
