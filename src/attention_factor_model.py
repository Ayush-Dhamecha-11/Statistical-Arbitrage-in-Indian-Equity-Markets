"""
attention_factor_model.py

This file contains two classes:
    LongConv - sequence model (Appendix A of the attention_factors_stats_arb paper)
    AttentionFactorModel - the full model formation method

Notations:
    N = number of stocks in the universe
    M = number of firm characteristics (features per stock)
    K = number of latent attention factors
    d = embedding hidden dimension
    s = lookback window for LongConv residual history
    T = number of trading days in a training window
 
Learnable parameters:
    W_K : embedding matrix - projects characteristics into d-dim space - (M, d)
    Q : query matrix - K query vectors, one per factor - (K, d)
    theta : LongConv parameters - kernel K_conv (d, s) and skip D (d,1)
 
Quantities computed during the forward pass:
    X_tilde_{t-1} : embedded characteristics - (N, d)
    omega_F_{t-1} : factor portfolio weights - (K, N)
    F_t : factor returns - (K, 1)
    beta_{t-1} : factor loadings - (N, K)
    omega_eps_{t-1} : residual projection matrix - (N, N)
    eps_t : stock residuals - (N, 1)
    omega_port : arbitrage weights in residual space (LongConv output) - (N, 1)
    omega_t : final portfolio weights in asset space - (N, 1)
    R_port_t : gross portfolio return - scalar
    R_port_net_t : net portfolio return after transaction costs - scalar
 
Training Objective:

    max  SR_net  +  lambda_var * (1/N) * sum_i [1 - Var(eps_i)/Var(R_i)]
   - Implemented as a minimisation loss (negative of the above).
 
LongConvolution formation procedure:

        y = K_conv * u  +  D ⊙ u
                                
    where:
    u : given input - (N, d, s)
    K_conv : learnable convolution kernel - (d, s) 
    D : learnable skip connection - (d, 1)
    *  = convolution along the time (s) dimension, via FFT:
            K_conv * u = iFFT(FFT(u) ⊙ FFT(K_conv))
    ⊙  = element-wise multiplication
 
    Kernel initialisation (geometric decay):
        K[h, t] = x * exp( -t/s * (s/2)^(h/d) )  ; x ~ N(0,1)
 
    Squash regulariser applied each forward pass:
        K_squashed = sign(K) * max(|K| - lambda_squash, 0)

"""
import math
import logging
from typing import Tuple, Dict
from data_loader import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

class LongConv(nn.Module):
    """
    This will take a window of past residuals for each stock and outputs 
    a scalar portfolio weight for each stock in the residual space.
    
    The convolution is global - the kernel spans the entire input length s.
    A global kernel can learn patterns at any lag within the lookback window.
    """

    def __init__(
        self,
        seq_len = 30,
        d_h = 32,
        lambda_squash = 0.001,
        dropout = 0.1,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.d_h = d_h
        self.lambda_squash = lambda_squash

        # Convolution kernal with decay - gives filter attending long and short lags 
        K_init = self.geometric_decay(d_h, seq_len)
        self.K_conv = nn.Parameter(K_init)

        # Skip parameter
        self.D = nn.Parameter(torch.randn(d_h) * 0.01)

        # Input projection - 1 channel (residuals) --> d_h channels
        # Each stock has a scalar residual at each timestep.
        # We project to d_hidden channels so the convolution can learn
        self.input_proj = nn.Linear(1, d_h)

        # Output projection - d_h channels --> 1 channel
        # After convolution we have d_hidden values per stock.
        # We project back to a single portfolio weight.
        self.output_proj = nn.Linear(d_h, 1)

        # Dropout
        self.dropout = nn.Dropout(p=dropout)

        # Layer norm for training stability
        self.layer_norm = nn.LayerNorm(d_h)

    @staticmethod
    def geometric_decay(d_hidden, seq_len) -> torch.Tensor:
        """
        Initialise K_conv with geometric decay.

        K[h, t] = x_h * exp( -t/T * (d/2)^(h/d) )

        where:
            h -> [0, d_hidden-1] is the hidden dimension index
            t -> [0, seq_len-1] is the time index (lag)
            x_h ~ N(0,1) is a random scale per channel
        """
    
        h_idx = torch.arange(d_hidden, dtype=torch.float32) # (d_hidden,)
        t_idx = torch.arange(seq_len, dtype=torch.float32) #(seq_len,)

        decay_rate = (d_hidden / 2.0) ** (h_idx / d_hidden)
        decay_rate = decay_rate.unsqueeze(1) # (d_hidden, 1)

        t_norm = t_idx / seq_len
        t_norm = t_norm.unsqueeze(0) # (1, seq_len)

        K_init = torch.exp(-t_norm * decay_rate) # (d_hidden, seq_len)
        scale = torch.randn(d_hidden, 1)
        K_init = K_init * scale

        return K_init
    
    def squash(self, K: torch.Tensor) -> torch.Tensor:
        """
        Applying element-wise squash operator to the kernal

        K_squashed = sign(K) * max(|K| - lambda_squash, 0)

        Proximal step for an L1 penalty. It shrinks all kernel
        and the result is sparse kernel
        """

        return K.sign() * ((K.abs() - self.lambda_squash).clamp(min=0.0))
    
    def fft_conv(self, u, K) -> torch.Tensor:
        """
        Using FFT convolution theorem

            K * u = iFFT(FFT(u) ⊙ FFT(K))

        Parameters
            u : (N, d_hidden, s) - input sequence
            K : (d_hidden, s) - convolution kernel

        Returns
            (N, d_hidden, s) - convolved output, causal
        
        We zero-pad both u and K to length 2s before FFT, then take
        only the first s elements of the output. This implements
        causal (one-sided) convolution - output at position t only
        depends on inputs at positions <= t. No look-ahead.
        """

        s = u.shape[-1]
        fft_len = 2*s

        # FFT
        U_f = torch.fft.rfft(u, n=fft_len, dim=-1) # (N, d_hidden, 1 + fft_len//2)
        K_f = torch.fft.rfft(K, n=fft_len, dim=-1) # (d_hidden, 1 + fft_len//2)

        # Elementwise multiplication of FFTs
        Y_f = U_f * (K_f.unsqueeze(0))

        # iFFT
        y = torch.fft.irfft(Y_f, n=fft_len, dim=-1) # (N, d_hidden, fft_len)

        # Take first s elements --> (N, d_hidden, s)
        return y[..., :s] 
    
    def forward(self, eps_hist) -> torch.Tensor:
        """
        Process residual history and output per-stock weights

        Parameters
        eps_hist : torch.Tensor - (N, s) 
                   Last s daily residuals for each of the N stocks.
                   eps_history[i, t] = residual of stock i at lag t
                   (t=0 is oldest, t=s-1 is most recent)

        Returns
        omega_port : torch.Tensor - (N,)
                     Portfolio weight for each stock in residual space
                     +ve value --> long position
                     -ve value --> short position
        """

        N, s = eps_hist.shape

        # Project input of each timestep into d_hidden channel
        u = eps_hist.unsqueeze(-1) # (N, s, 1)
        u = self.input_proj(u)
        u = u.transpose(1, 2) # (N, s, d_hidden) --> (N, d_hidden, s) (for conv)

        # Apply squash operation
        K_squash = self.squash(self.K_conv) # (d_hidden, s)

        # LongConv operation
        y_conv = self.fft_conv(u, K_squash) # (N, d_hidden, s)

        # Skip connection path
        y_skip = self.D.view(1, self.d_h, 1) * u # (1, d_hidden, 1) ⊙ (N, d_hidden, s)

        y = y_conv + y_skip # (N, d_hidden, s)

        # Take last timestep
        y_last = y[:,:,-1] # (N, d_hidden)

        # Do norm + dropout --> (N, d_hidden)
        y_last = self.layer_norm(y_last)
        y_last = self.dropout(y_last) 

        # Project Output
        omega_port = self.output_proj(y_last) # (N, 1)
        omega_port = omega_port.squeeze(-1) # (N,)

        return omega_port


class AttentionFactorModel(nn.Module):
    """
    Learns:
        1. Attention factors, embeddings, query-vectors (W_K, Q)
        2. Arbitrage trading policy (LongConv theta parameter)
    
    All parameters are optimized as per training objective which is
    maximizing the net sharpe ratio after transection costs plus an
    explained variance regularization term.

    Parameters

    N : Number of stocks
    M : Number of characteristics
    K : Number of latent attention factors
    d : Dimension of the embedded characteristic space
    s : lookback window for LongConv
    lambda_ridge : Inverse stabilizer and ridge panelty for factor loadings
    d_hidden_conv : LongConv hidden dimension (convolution channels)
    lambda_squash : LongConv squash regularizer
    dropout : dropout rate for neuron efficiency

    """

    def __init__(
        self,
        N, M, K=8, d=32, s=30,
        lambda_ridge=1e-4,
        d_hidden_conv=32,
        lambda_squash=0.001,
        dropout=0.1,
    ):
        
        super().__init__()

        self.N = N
        self.M = M
        self.K = K
        self.d = d
        self.s = s
        self.lambda_ridge = lambda_ridge

        # Learnable parameter 1 -- Embedding matrix W_K:
        # maps M-dimensional characteristics into d-dimensional space
        self.W_K = nn.Linear(M, d, bias=False) # (M, d)

        # Learnable parameter 2 -- Query matrix Q:
        # K query vectors, each query has dimesion d
        # multiplication with it gives contribution of each stock to each factor
        self.Q = nn.Parameter(torch.randn(K, d) * (1.0 / math.sqrt(d)))

        # Learnable parameter 3 -- LongConv sequence model
        # Takes (N, s) residual history, gives (N,) portfolio weights
        # it's internal parameters (K_conv, D, input_proj, output_proj) are
        # optimized jointly with above two
        self.longconv = LongConv(
            seq_len=s,
            d_h=d_hidden_conv,
            lambda_squash=lambda_squash,
            dropout=dropout,
        )

        # buffer to store previous portfolio weights
        # It is needed to compute turnover cost at current time 
        self.register_buffer("omega_prev", torch.zeros(N))

        self.init_weights()

        logger.info(
            f"AttentionFactorModel initialised | "
            f"N={N} M={M} K={K} d={d} s={s} | "
            f"params={sum(p.numel() for p in self.parameters()):,}"
        )

    def init_weights(self) -> None:
        # Xavier Uniform initialization for W_K, Q is already initialized
        nn.init.xavier_uniform_(self.W_K.weight)
    
    def reset_omega_prev(self) -> None:
        # Reset the previous portfolio weights to zero
        # Used when new rolling window of training/testing is started
        self.omega_prev.zero_()

    def embed_characteristics(self, X_prev) -> torch.Tensor:
        # Embed with W_K
        # X_tilde = X_prev @ W_K.T
        return self.W_K(X_prev) # (N, d)
    
    def compute_factor_weights(self, X_tilde) -> torch.Tensor:
        # Compute factor weight matrix - omega_F
        # omega_F = softmax(Q @ X_tilde.T / sqrt(d))
        # Each row k contains the attention score of each stock for factor k
        # Softmax is applied along dim=1 (across N stocks for each factor)

        scores = torch.mm(self.Q, X_tilde.T) / math.sqrt(self.d) # (K, N)
        omega_F = F.softmax(scores, dim=1)
        return omega_F
    
    def compute_factor_returns(
        self,
        omega_F,
        R_curr,
    ) -> torch.Tensor:
        
        # Compute factor returns F_t = omega_F @ R_t
        # each factor return is weighted average return of its portfolio
        return torch.mv(omega_F, R_curr) # (K,)
    
    def compute_factor_loadings(self, omega_F) -> torch.Tensor:
        """
        Compute factor loadings - beta

        parameter - omega_F : (K, N)

        Closed form solution:
        
        beta.T = omega_F.T @ inv(omega_F @ omega_F.T + lambda_ridge * I_K)
        beta = inv(omega_F @ omega_F.T + lambda_ridge * I_K).T @ omega_F

        beta loadings tell how much each stock is exposed to each factor.
        Here the factors are tradable portfolios (omega_F rows), 
        so the loadings are derived analytically from the portfolio 
        weights rather than estimated by OLS.

        """ 
        A = torch.mm(omega_F, omega_F.T) # (K, K)
        A = A + self.lambda_ridge * torch.eye(self.K, device=omega_F.device, dtype=omega_F.dtype)

        beta_T = torch.linalg.solve(A, omega_F) # (K, N)
        beta = beta_T.T # (N, K)
        return beta
    
    def compute_residuals(
            self,
            R_curr,
            beta,
            F_t,
    ) -> torch.Tensor:
        
        """
        Compute idiosyncratic residuals.
        R_curr : (N,)
        beta : (N, K)
        F_t : (K,)
        eps_t : (N,)

        eps_t = R_t - beta @ F_t = (I_N - beta @ omega_F) @ R_t
        """
        return R_curr - torch.mv(beta, F_t) # (N,)
    
    def compute_residual_projection_mat(
        self,
        beta,
        omega_F,
    ) -> torch.Tensor:
        
        """
        beta : (N, K)
        omega_F : (K, N)
        omega_eps : (N, N)

        Compute residual projection matrix
        omega_eps = I_N - beta @ omega_F

        This matrix projects any return vector into the residual (idiosyncratic)
        space. It is used to map portfolio weights from residual space back to
        asset space
        """

        I_N = torch.eye(self.N, device=beta.device, dtype=beta.dtype)
        return I_N - torch.mm(beta, omega_F) # (N, N)
    
    def compute_portfolio_weights(
        self, 
        omega_eps,
        omega_port,
    ) -> torch.Tensor:
        
        """
        Map residual-space weights back to asset space.

        omega_eps : residual projection matrix - (N, N)
        omega_port : weights in residual space from LongConv - (N,)
        omega_t : weights in asset space - (N,)

        omega_t = omega_eps.T @ omega_port

        this omega_t is the vector of actual stock positions.
        omega_t[i] > 0 --> long stock i, omega_t[i] 0 --> short stock i.
        """

        return torch.mv(omega_eps.T, omega_port) # (N,)
    
    def compute_transection_costs(
        self, 
        omega_t,
        omega_prev,
    ) -> torch.Tensor:
        
        """
        Compute transaction costs as per mentioned in paper.

        omega_t : current portfolio weights - (N,)
        omega_prev : previous portfolio weights - (N,)

        cost = 0.0005 * ||omega_t - omega_prev||_1   (5 bps turnover cost)
             + 0.0001 * ||max(-omega_t, 0)||_1        (1 bp shorting cost)

        The L1 norm of (omega_t - omega_prev) is the total absolute
        rebalancing across all N stocks - the "turnover". Multiplying by
        0.0005 (5 basis points = 0.05%) converts this to a return cost.

        The second term penalises short positions. max(-omega_t, 0) isolates
        the negative weights (short positions) and the L1 norm sums their
        absolute values. Multiplying by 0.0001 (1 bp) is the daily cost
        of borrowing shares to short.
        """

        turnover_cost = 0.0005 * torch.abs(omega_t - omega_prev).sum()
        shorting_cost = 0.0001 * torch.clamp(-omega_t, min=0.0).sum()
        return turnover_cost + shorting_cost
    
    def forward(
        self,
        X_prev,
        R_curr,
        eps_hist
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for one timestep t.

        Parameters

        X_prev(X_t-1): firm-characterisitcs at t-1 - (N, M)
        R_curr(R_t) : Daily return at time t - (N,)
        eps_hist : last s daily residuals for each stock - (N, s)

        Returns

        'R_port_gross' : gross portfolio return before costs - scalar
        'R_port_net' : net portfolio return after costs - scalar
        'cost' : transaction cost for this step - scalar
        'omega_t' : final asset-space portfolio weights - (N,)
        'omega_F' : factor portfolio weights - (K, N)
        'omega_port' : residual-space weights from LongConv - (N,)
        'eps_t' : current step residuals - (N,)
        'beta' : factor loadings - (N, K)
        'F_t' : factor returns - (K,)
        'explained_var' : fraction of variance explained by factors - scalar
                                    (used in the loss function)

        """

        # Embed characteristics
        X_tilde = self.embed_characteristics(X_prev) # (N, d)
        
        # Attention factor weights
        omega_F = self.compute_factor_weights(X_tilde) # (K, N)

        # Factor returns
        F_t = self.compute_factor_returns(omega_F, R_curr) # (K,)

        # Factor loadings
        beta = self.compute_factor_loadings(omega_F) # (N, K)

        # Residuals
        eps_t = self.compute_residuals(R_curr, beta, F_t) # (N,)

        # LongConv -> residual-space portfolio weights
        omega_port = self.longconv(eps_hist) # (N,)

        # Mapping to asset space
        omega_eps = self.compute_residual_projection_mat(beta, omega_F) # (N, N)
        omega_t = self.compute_portfolio_weights(omega_eps, omega_port) # (N,)

        # Gross profit return
        R_port_gross = torch.dot(R_curr, omega_t) # scalar

        # Transaction cost and net return
        cost = self.compute_transection_costs(omega_t, self.omega_prev) # scalar
        R_port_net = R_port_gross - cost # scalar

        # Update omega_prev buffer for next step
        # Do not let gradient flow to next timestep
        self.omega_prev.copy_(omega_t.detach())

        # Compute explained variance
        var_R = (R_curr ** 2).mean().clamp(min=1e-8)
        var_eps = (eps_t ** 2).mean()
        explained_var = 1.0 - var_eps / var_R  # scalar

        return {
            "R_port_gross": R_port_gross,
            "R_port_net": R_port_net,
            "cost": cost,
            "omega_t": omega_t,
            "omega_F": omega_F,
            "omega_port": omega_port,
            "eps_t": eps_t,
            "beta": beta,
            "F_t": F_t,
            "explained_var": explained_var,
        }
    
    def forward_sequence(self, X, R) -> Dict[str, torch.Tensor]:
        """
        Run the forward pass across an entire sequence of T time steps.
        It passes a full window of T days and gets back a sequence of portfolio returns.

        Parameters

        X : Characteristics tensor for T days - (T, N, M)
            The model uses X[t-1] as X_prev when computing the portfolio for day t

        R : Returns tensor for T days - (T, N)
            
        Returns
    
        dict with keys:
            'returns_gross' : gross portfolio return each day - (T_eff,)
            'returns_net' : net portfolio return each day - (T_eff,)
            'costs' : transaction cost each day - (T_eff, )
            'omegas' : portfolio weights each day - (T_eff, N)
            'eps_history' : residual histories - (T_eff, N, s)
            'explained_vars' : per-step explained variance - (T_eff,)

        where T_eff = T - s  (first s steps are used to fill the residual buffer)
            The LongConv needs s days of past residuals.
            For the first s days we accumulate residuals but do not trade.
            Trading starts from day s onwards (T_eff = T - s effective steps).
        """
        dev = X.device
        Xdt = X.dtype
        T, N, M = X.shape
        # Residual buffer
        eps_buffer = torch.zeros(N, self.s, device=dev, dtype=Xdt)

        # reset portfolio weight state
        self.reset_omega_prev()
        self.omega_prev = self.omega_prev.to(dev)

        # Storages for outputs
        returns_gross = []
        returns_net = []
        costs = []
        omegas = []
        explained_vars = []

        # main loop over timesteps
        for t in range(1, T):

            X_prev = X[t-1] # (N, M)
            R_curr = R[t] # (N,)

            if t < self.s:
                # Accumulate residuals
                with torch.no_grad():
                    X_tilde = self.embed_characteristics(X_prev)
                    omega_F = self.compute_factor_weights(X_tilde)
                    F_t = self.compute_factor_returns(omega_F, R_curr)
                    beta = self.compute_factor_loadings(omega_F)
                    eps_t = self.compute_residuals(R_curr, beta, F_t)

                # Append new residual at right
                eps_buffer = torch.roll(eps_buffer, shifts=-1, dims=1)
                eps_buffer[:, -1] = eps_t.detach()
                continue

            # Forward pass
            output = self.forward(X_prev, R_curr, eps_buffer)

            # Update residual buffer with the newly computed eps_t
            eps_buffer = torch.roll(eps_buffer, shifts=-1, dims=1)
            eps_buffer[:, -1] = output["eps_t"].detach()

            # collect outputs
            returns_gross.append(output["R_port_gross"])
            returns_net.append(output["R_port_net"])
            costs.append(output["cost"])
            omegas.append(output["omega_t"])
            explained_vars.append(output["explained_var"])

        if not returns_net:
            raise RuntimeError(
                f"No trading steps executed. T={T}, s={self.s}. "
                f"Need T > s. Got T-s={T - self.s} effective steps."
            )
        
        return {
            "returns_gross": torch.stack(returns_gross), # (T_eff,)
            "returns_net": torch.stack(returns_net), # (T_eff,)
            "costs": torch.stack(costs), # (T_eff,)
            "omegas": torch.stack(omegas), # (T_eff, N)
            "explained_vars": torch.stack(explained_vars), # (T_eff,)
        }


    def attention_factor_loss(
        self,
        returns_net,
        explained_vars,
        R_f, 
        lambda_var=100.0
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        """
        Combined loss function:
        minimize the negative of -
            SR_net + lambda_var * mean_explained_variance

        here:
            SR_net = (mean(R_net) - R_f) / std(R_net)
            mean_explained_variance = mean over time of [1 - Var(eps_t) / Var(R_t)]
        
        Parameters

        returns_net : net portfolio returns for the window - (T_eff,)
        explained_vars : per-step explained variance proxies - (T_eff, )
        R_f : daily risk-free rate (Gov 10-year bond rate used) - (T_eff, )
        lambda_var: weight on variance explanation term - float

        Returns

        loss : value to call .backward() on - scalar
        info : breakdown for logging - dict
        """

        T = returns_net.shape[0]

        # Net Sharpe ratio
        mean_ret = returns_net.mean()
        R_f_mean = R_f.mean()
        std_ret = returns_net.std(unbiased=True).clamp(min=1e-8)
        sharpe = (mean_ret - R_f_mean) / std_ret
        print(f"Sharpe - {sharpe.shape}")

        # Explained variance term
        mean_exp_var = explained_vars.mean().clamp(min=0.0, max=1.0)
        print(f"mean_exp_var - {mean_exp_var.shape}")

        # Combined loss (negative because we minimise)
        loss = -(sharpe + lambda_var * mean_exp_var)
        print(f"loss - {loss.shape}")

        # Diagnostics dict for logging
        info = {
            "loss": loss.detach(),
            "mean_ret": mean_ret.detach(),
            "mean_exp_var": mean_exp_var.detach(),
        }

        return loss, info
    
# Check
if __name__ == "__main__":

    torch.manual_seed(42)

    data_dir = "../data"
    loader = DataLoader(
        panel_path = f"{data_dir}/panel_characteristics_norm.parquet",
        hist_dir = f"{data_dir}/historical/",
        train_years = 8,
        val_years = 2,
        test_years = 1,
    )

    X, R, dates, symbols = loader.get_tensors()
    splits = loader.get_rolling_splits()
    X_tr, R_tr, X_val, R_val, X_te, R_te = loader.get_window_tensors(splits[0])
    R_f = X_tr[:, 0, 23]
    print(f"R_f: {R_f.shape}")

    # Hyperparameters
    K = 30 # factors (start small for testing)
    d = 32 # attention hidden dim
    s = 30 # LongConv lookback
    T, N, M = X.shape

    model = AttentionFactorModel(N, M, K, d, s)

    print()
    print("-" * 60)
    print("Testing AttentionFactorModel.forward_sequence()...")
    print("-" * 60)

    seq_output = model.forward_sequence(X_tr, R_tr)

    T_tr, _, _ = X_tr.shape
    T_eff = T_tr - s - 1   # warm-up uses first s steps, loop starts at t=1
    print(f"  T={T_tr}, s={s}, expected T_eff ≈ {T_eff}")
    print(f"  returns_net shape   : {seq_output['returns_net'].shape}")
    print(f"  omegas shape        : {seq_output['omegas'].shape}")
    print(f"  explained_vars shape: {seq_output['explained_vars'].shape}")

    # Test loss function
    print()
    print("-" * 60)
    print("Testing attention_factor_loss()...")
    print("-" * 60)

    loss, info = model.attention_factor_loss(
        seq_output["returns_net"],
        seq_output["explained_vars"],
        R_f,
        lambda_var=100.0,
    )
    print(f"  loss             : {info['loss'].item():.4f}")
    print(f"  mean_exp_var     : {info['mean_exp_var'].item():.4f}")

    # Test backward pass
    print()
    print("-" * 60)
    print("Testing backward pass (gradient flow)...")
    print("-" * 60)

    loss.backward()

    # Check all parameters received gradients
    no_grad = []
    for name, param in model.named_parameters():
        if param.grad is None:
            no_grad.append(name)
        else:
            print(f"  {name:<35} grad norm = {param.grad.norm().item():.6f}")

    if no_grad:
        print(f"  WARNING: no gradient for: {no_grad}")
    else:
        print("  All parameters received gradients")

    print()
    print(f"Total model parameters: {sum(p.numel() for p in model.parameters()):,}")