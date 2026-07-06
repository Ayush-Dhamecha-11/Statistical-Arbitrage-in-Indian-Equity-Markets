# Project Outcomes

This repository presents an implementation and empirical evaluation of a deep learning framework for statistical arbitrage in the Indian equity market based on the **Attention Factor Model**. The implementation closely follows the methodology proposed in the original research while adapting it to Indian market data and extending it with additional experiments involving **path signatures** from rough path theory.

The objective is to learn dynamic market-neutral trading portfolios directly from historical firm information and residual return dynamics, rather than relying on manually designed factor models.

---

# Overview

The complete framework consists of two different factor generation approaches sharing the same portfolio construction network:

1. **Learnable Attention Factors + LongConv**
2. **PCA Factors + LongConv**

Both approaches estimate latent factors, construct daily long-short portfolios, and evaluate their out-of-sample performance under rolling expanding-window backtesting.

---

# 1. Learnable Attention Factor + LongConv

The primary model follows the architecture proposed in the research paper "attention_factors_stat_arb.pdf".

Instead of assuming predefined market factors, the model learns latent attention factors directly from firm characteristics.

For every trading day:

- firm characteristics are embedded into a latent representation,
- attention weights determine each stock's contribution to every factor,
- latent factor returns are estimated,
- residual returns are computed,
- historical residual sequences are processed using **LongConv**,
- LongConv produces dynamic portfolio weights based on temporal residual patterns,
- characteristic-based portfolio weights and LongConv outputs are combined to construct the final trading portfolio.

The objective function jointly maximizes portfolio Sharpe Ratio while encouraging the learned attention factors to explain cross-sectional return variation.

This architecture enables the model to simultaneously learn:

- cross-sectional relationships between firms,
- latent market structure,
- temporal dynamics in residual returns.

---

## Attention Factor Interpretation

The learned attention weights provide an interpretable view of which stocks contribute most strongly to each latent factor.

<img width="1500" height="949" alt="Attention_weights_K=20_learnable_test" src="https://github.com/user-attachments/assets/3b7e2faa-9660-477c-8c54-953087ccb7b2" />


Rather than producing arbitrary hidden features, the attention mechanism consistently allocates larger weights to economically meaningful firms across multiple sectors such as

- Financial Services
- Consumer Defensive
- Technology
- Industrials
- Utilities

The factor portfolios therefore exhibit meaningful sector exposure.

---

# 2. PCA + LongConv

The same LongConv portfolio construction network was trained using **Principal Component Analysis (PCA)** factors instead of learnable attention factors.

The workflow is identical except that latent factors are obtained by applying PCA to firm characteristics.

Unlike the attention model, PCA factors are fixed linear combinations determined solely by variance maximization and cannot adapt during training.

---

# Incorporating Path Signatures

In addition to reproducing the original methodology, this project explores **path signatures** as an alternative representation of asset behaviour.

Instead of using raw firm characteristics, each stock's historical return trajectory is represented by its truncated path signature.

The signature transform captures higher-order interactions of the return path and provides a mathematically rich representation of temporal dynamics.

Cross-sectional normalization is then applied before the signature vectors are used as model inputs.

---

## Signature Similarity Analysis

To investigate whether signatures capture meaningful relationships between assets, pairwise Euclidean distances between stock signature vectors were computed.

For every pair of stocks,

```
Distance(i,j) = || Signature_i − Signature_j ||₂
```

The resulting distance matrix was reordered using hierarchical clustering and visualized as a heatmap.

<img width="1363" height="1446" alt="clustermap_stock_signatures" src="https://github.com/user-attachments/assets/9e12fb9a-edb4-4f97-88a6-24b20f2a7abf" />


The heatmap reveals several important observations:

- clusters of stocks exhibit highly similar signature representations,
- similar companies naturally appear adjacent after clustering,
- certain assets consistently remain distant from the majority of the market,

This experiment suggests that path signatures provide an informative similarity measure between assets and may serve as a useful foundation for future clustering or statistical arbitrage strategies.

---

# Out-of-Sample Performance

The models were evaluated using rolling expanding-window backtesting.

Each experiment follows the procedure:

- train on historical years,
- validate on subsequent years,
- test on unseen future data,
- roll the window forward,
- concatenate all out-of-sample returns.

Performance metrics include:

- Annualized Sharpe Ratio
- Annualized Mean Return
- Annualized Volatility
- Net Sharpe Ratio
- Net Mean Return
- Portfolio Beta

Transaction costs and short-selling costs are incorporated when computing net performance.

---

## Learnable Attention Factors

<img width="1308" height="828" alt="Cumulative_returns_test_learnable" src="https://github.com/user-attachments/assets/ba250ba9-d309-43ef-9a9d-556740474184" />

The learnable attention factor model consistently generates positive cumulative returns throughout the testing period.

The strategy substantially outperforms the benchmark Nifty100 index while maintaining relatively stable behaviour across different numbers of attention factors.

Increasing the number of attention factors generally produces richer latent representations, although the performance improvement eventually saturates, indicating that additional factors contribute diminishing marginal information.

---

## PCA Factors

<img width="1308" height="828" alt="Cumulative_returns_test_pca" src="https://github.com/user-attachments/assets/8d4baf9f-b0fb-42b1-8f7c-202e8ecd43e1" />

The PCA approach also produces profitable trading strategies.

However, because PCA factors remain fixed throughout training, their ability to adapt to changing market structure is inherently limited.

The comparison demonstrates that the LongConv architecture itself contributes significantly to portfolio construction, while learnable attention factors provide additional flexibility by jointly optimizing factor extraction and trading decisions.

---

# Performance Summary

| Metric | Best Observed Value |
|----------|--------------------|
| Annualized Sharpe Ratio | ~1.6 |
| Annualized Return | ~32% |
| Annualized Volatility | ~15% |
| Portfolio Beta | ~1.02 |

The portfolio consistently achieved positive out-of-sample returns across multiple model configurations.

---

# Discussion

Several observations emerge from the experiments.

- Learnable attention factors successfully discover economically meaningful latent factors directly from firm characteristics.
- LongConv effectively captures temporal dependencies in residual returns for dynamic portfolio construction.
- PCA provides a strong statistical baseline but lacks the adaptability of learnable attention factors.
- Path signatures encode rich information about historical return trajectories and naturally induce meaningful similarity structures among stocks.
- Signature-based similarity analysis opens an additional research direction for clustering-based statistical arbitrage and relative-value trading.

Although the implementation closely follows the original methodology, it was adapted to Indian equity data, which contains fewer firm characteristics than the original U.S. dataset. Consequently, absolute performance differs from the reported values in the paper while preserving the overall behaviour of the proposed framework.

---

# Conclusion

This project successfully reproduces the core ideas of the Attention Factor Model for statistical arbitrage and evaluates them on Indian equity markets.

Beyond reproducing the original architecture, the project introduces path-signature representations for financial time series and demonstrates their ability to reveal meaningful similarity structures among assets.

Overall, the implementation shows that combining latent factor learning with temporal sequence modelling provides a powerful framework for data-driven statistical arbitrage and establishes a foundation for further research into representation learning for quantitative finance.
