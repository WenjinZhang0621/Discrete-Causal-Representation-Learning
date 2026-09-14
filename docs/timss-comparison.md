# TIMSS latent-law comparison

The default ratio is 0.2, selected by the lowest observed-data BIC among the models and graph scales below. All graph candidates use actual multinomial sampling, with graph seed 1 fixed across the grid. The selected graph has seven edges, 131 free parameters and BIC 37367.26.

| Latent law | Requested ratio c | Latent samples | Edges | Log likelihood | Parameters | BIC |
|---|---:|---:|---:|---:|---:|---:|
| Independent | — | — | — | -18685.08 | 123 | 38161.01 |
| Higher-order: one factor | — | — | — | -18343.92 | 130 | 37523.71 |
| Higher-order: two subscales | — | — | — | -18281.21 | 131 | 37404.71 |
| Saturated / original Stage I | — | — | — | -18223.45 | 243 | 38009.32 |
| DAG | 0.05 | 31 | 7 | -18303.60 | 134 | 37468.78 |
| DAG | 0.1 | 62 | 8 | -18258.14 | 133 | 37371.42 |
| DAG | 0.2 | 124 | 7 | -18262.48 | 131 | **37367.26** |
| DAG | 0.3 | 186 | 9 | -18253.12 | 137 | 37387.12 |
| DAG | 0.5 | 310 | 10 | -18242.30 | 146 | 37423.34 |
| DAG | 0.75 | 465 | 9 | -18250.76 | 137 | 37382.40 |
| DAG | 1.0 | 620 | 9 | -18248.86 | 137 | 37378.59 |
| DAG | 2.0 | 1240 | 14 | -18233.43 | 175 | 37592.07 |
| DAG | logN | 3986 | 18 | -18225.97 | 211 | 37808.60 |

The DAG and the two-subscale higher-order model both have 131 parameters. Their BIC difference of 37.45 therefore reflects improved fit at the same parameter count. The saturated model fits better in likelihood but uses 243 parameters, yielding a BIC 642.06 higher than the selected DAG. These results favor the DAG among the models compared on these TIMSS data.

## Common fitting protocol

All fits use the original 620 students, 29 items and supplied Q matrix. Times are clipped to [e, e^6], transformed by log, and missing cells omitted from the likelihood. The reported likelihood includes the Gaussian normalizing constant on log times. The fixed-Q measurement model has 29 intercepts, 58 active coefficients and 29 variances, for 116 parameters. Intercepts are bounded to [-2,4], active coefficients to [-3,4], and variances to [1e-12,2]. BIC = -2 log likelihood + d log(620), including both measurement and latent parameters. The original sample size remains 620 for every row.

The higher-order laws adapt the Gaussian/probit layer of [Liu, Lee and Gu (2025)](https://doi.org/10.1017/psy.2025.15) to this same fixed-Q lognormal measurement model. The one-factor law has seven intercepts and seven loadings. The two-subscale law assigns the four content attributes and three cognitive attributes to separate standard Gaussian factors with an estimated correlation, for 15 latent parameters. These fits use direct observed-likelihood optimization and numerical integration; they do not implement the paper's MCEM algorithm. Independent, higher-order and saturated fits have no graph-resampling step and were retained when the graph-scale comparison was rerun.

For each requested ratio c, m=max(1,floor(c*620)) latent observations are sampled from the same saturated fit in `data/TIMSS/timss_refit.json`. GES uses causal-learn 0.1.4.8 with its default BDeu score. A DAG extension of each CPDAG is then fixed while all measurement and conditional-probability parameters are re-estimated on the original students. Each graph was fitted from the saturated, independent, two higher-order fits, and four random initializations; the highest likelihood was retained and polished if needed. Optimization uses analytic-gradient L-BFGS-B. The logN ratio uses m=floor(N log N); it is a diagnostic outside the theorem's rate condition.

The saved first-stage fit uses initialization seed 48, selected by likelihood from the earlier 101-start fitting comparison. Graph seed 1 is separate and was held fixed across every ratio. Both supplied TIMSS entry points produce identical latent samples and the same selected CPDAG with their defaults. The raw-data solver can differ slightly in its final fitted probabilities at the stopping tolerance; this did not change the samples or graph in validation.

## Audit the reported scores

`data/TIMSS/latent_law_comparison.json` stores all 13 selected fits, their latent probabilities, item parameters, graph extensions, fitting diagnostics and source-file hashes. The command below recalculates the observed likelihood and BIC from the original data, checks DAG factorization, and regenerates every multinomial CPDAG. It verifies saved estimates; it does not repeat parameter optimization.

```bash
python experiments/verify_timss_comparison.py
```

The main TIMSS commands fit the saturated first stage and discover its graph. A graph attached to that fit does not reduce its parameter count; the BIC values in this table use the separately fitted constrained models stored in the comparison artifact. The table is a descriptive model comparison after graph selection, with boundary estimates possible.

For the nondecreasing resample-size amendment and its assumptions, see [resampling-consistency.md](resampling-consistency.md).
