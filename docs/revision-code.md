# Revision code: options, diagnostics, and reproducibility

This extends commit `06da9c9`. The default `profile="legacy"` retains the
original initialization, Gibbs-SAEM, thresholding and resampling procedure.
`profile="revision"` changes only the default Bernoulli epsilon to 0.001;
an explicit epsilon takes precedence.

The legacy sampled-reference helper now uses the requested K and bounds for
Tree/Markov runs, and supports Model-16. The usual K=10 defaults are unchanged.
`requirements-reproducible.txt` pins the Python 3.12 validation environment.
Manifests record package versions; resume checks reject environment changes.

## Graph estimation

```python
from dcrl.graph_scoring import learn_latent_graph

record=learn_latent_graph(p_hat, K, effective_n=N,
                         method="plugin", ess=1.0)
cpdag=record["graph"]
```

For a node i and parents A, the plug-in counts are

    n[a,b]=m * sum_z p_hat(z) 1{z_A=a, z_i=b}.

The local BDeu score uses the log-gamma formula at these fractional
counts, with parent prior mass ESS/2^|A| and cell prior mass ESS/2^(|A|+1).
These binary cardinalities assume both levels occur; for a degenerate variable
the scorer follows causal-learn's observed-support cardinality convention.
Zero-count parent configurations contribute zero likelihood score. Sparse
masses and the uniform component are marginalized directly, without 2^K
enumeration, rounding or pseudo-observations. GES remains a greedy search.

`method="resampling"` draws multinomial counts first. With default options,
the evaluation helper preserves its sampling and RNG path for K>=3. An
explicit `seed` isolates graph randomness from fitting; plug-in mode accepts
no seed. For K=1 the structure prior is zero; for K=2 the default edge
probability is 0.5. Results include score parameters, effective sample size
and implementation metadata.

`effective_n=m` scales the counts; `ess` sets the prior concentration. Both
paths pass ESS to the scorer through a private function binding because the
pinned causal-learn GES branch discards BDeu `parameters`. No global function
is modified. Integer-count scores are tested against causal-learn.

The two methods can return different graphs. Plug-in removes resampling
noise, but lower SHD is not guaranteed and the resampling theorem does not
establish consistency of plug-in GES.

## Observed-data fitting and multiple starts

```python
from dcrl.config import ExperimentConfig
from dcrl.fitting import fit_observed

cfg=ExperimentConfig(N=len(X), J=X.shape[1], K=13,
                     distribution="Bernoulli", profile="revision",
                     init_cutoff_c=2.5, epsilon=0.001).finalize()
fit=fit_observed(X, cfg, start_specs=[
    {"kind": "spectral", "seed": 1001},
    {"kind": "spectral_jitter", "seed": 1002, "jitter": 0.1},
    {"kind": "random", "seed": 1003},
], orientation="column_sum")
```

The full-model fitter uses X and the configuration, with no simulation truth.
Each start has a SAEM seed; `init_seed` controls random initialization.
Spectral initialization is deterministic, so repeating it with different seeds
changes only the SAEM path. The default is one spectral start. Optional
spectral-jitter and random starts begin inside the coefficient boxes.

All starts and failures are saved. Selection takes the largest attained
observed log likelihood minus `pen * sum(min(abs(beta), tau))`; intercepts
are unpenalized and likelihoods include their distribution constants. This
compares the supplied starts and does not certify a global optimum. If none
can be scored, the result is marked failed and retains all start records.

SAEM accumulates complete-data likelihood from zero, so its finite-iteration
likelihood weight is generally below one. Iteration records save that weight
and the optimizer's surrogate objective before hard-thresholding, separately
from the observed-data selection score. Stopping uses the original squared
parameter change, including variance change for Lognormal. Optimizer failure,
max_iter, parameter-change tolerance and finite output are recorded separately.

Exact observed likelihood is evaluated in batches over stored latent states.
A nonzero uniform component requires the full state space. Exceeding
`likelihood_max_states` raises a scoring failure. Use process parallelism
for SAEM: it uses NumPy's global RNG, whose state the wrapper restores on return.

## Identifying orientation and the numerical parameter space

Optimization retains the raw boxes: Poisson intercept [0,2], slopes [0,3];
Lognormal intercept [-2,4], slopes [0,4], variance [0,2]; Bernoulli intercept
[-5,5], slopes [-2.5,5]. The coefficient scale option scales coefficient
bounds only, not latent CPT bounds or the Lognormal variance bound. Finite
Lognormal scoring requires strictly positive fitted variances.

Bernoulli optimization allows signed slopes. Final fitted outputs always
apply the following column-sum orientation after fitting:

    s_k=1{sum_j beta[j,k] < 0}
    beta_new[j,k]=(-1)^s_k beta[j,k]
    intercept_new[j]=intercept[j] + sum_k s_k beta[j,k]
    p_new(z)=p_old(z XOR s), gamma_new=gamma_old.

This preserves the observable distribution and Q support. Raw B/p and flip
metadata are retained; zero column sums are marked unresolved. Oriented
coefficients can lie outside the optimization box. The change sets an output
orientation, not a constraint during fitting. The low-level PSAEM return
remains raw for optimizer-trajectory tests; public fit outputs are oriented.

The supplied study plans set `evaluation_alignment="latent_sample"`. For
evaluation only, let Ahat be the final retained Gibbs sample and Atrue the
generating sample, both N-by-K. Match true coordinate k to fitted coordinate l
with cost `min(sum(Atrue[:,k] != Ahat[:,l]), N - sum(...))`; Hungarian matching
minimizes the total classification error, and each matched column takes its
better bit orientation. The same permutation is applied to Q, p and both
CPDAG axes; complements also transform p. Ties are recorded. This evaluates
the final Gibbs draw, not a posterior-MAP classification.
Selected `final_sample_A` uses the same orientation as selected B/p; raw
samples and raw per-start outputs remain available. Known-Q fits retain their
specified coordinate labels. Across-start variability aligns to a fitted
reference without simulation truth.

Plans omitting this option retain `evaluation_alignment="Q_support"` for
compatibility. With latent-sample alignment, `Q_support_reference` additionally
retains the old optimally Q-matched measurement metrics. Historical
Q-support-matched results use a different alignment; changing the permutation
can change graph error with identical fitted parameters.

## Reviewer 2: diagnostic stages and measurements

`run_study.py` records these distinct graph inputs when `oracles=true`:

1. The actual generating latent sample A: graph search with no second sample.
2. The exact generating population p: either resampling or plug-in scoring.
3. An estimated p from fitting X with known Q: a fixed-support nuisance fit.
4. The fully estimated model fitted from X without Q or other simulation truth.

Known-Q fitting fixes excluded slopes to zero, with no selection penalty or
threshold on allowed slopes. It is a separate oracle fit and does not compete
for full-model start selection. Oracle stages run even if full-model fitting
fails; unavailable fitted stages are recorded as failures.

The wrapper captures the generator's CPT draws by replaying its pre-generation
Python RNG state locally. Generated A/X and RNG streams are unchanged. These
CPTs give the exact population p and the generating DAG gives its CPDAG.
Population enumeration is capped at K<=16 by default; larger runs require
raising the cap. Graph-only plug-in scoring does not enumerate 2^K states.

Outputs include raw/aligned Q errors, FP/FN counts and rates, exact Q recovery;
latent CPDAG SHD/exact recovery, skeleton and v-structure metrics; raw/aligned
composite graph SHD; p errors; across-start p/Q/CPDAG variability; objectives,
optimizer status and timing. Full, Q-only and latent-only errors are reported
separately. Across-start comparisons use common graph seeds, although changing
coordinates can change the realized resample. Plug-in comparisons have no
resampling variation.

Skeleton, v-structure and compelled directed-edge metrics include precision,
recall and F1. A reversed compelled arrow contributes one false-positive and
one false-negative directed edge; CPDAG SHD still counts one changed unordered
pair. Empty precision/recall denominators are reported as undefined, with FP
counts available; F1 is one when both sets are empty.

The Monte Carlo unit is one independently generated dataset. Graph draws are
averaged within each dataset before computing standard errors. Summaries count
failed fits, failed graphs and missing attempts, and report pointwise Monte
Carlo intervals. Raw per-start records are also saved.

## Sensitivity configuration and output

JSON `model` supplies fixed ExperimentConfig fields; `grid` takes their
Cartesian product. Additional generation settings are `Q_matrix` (binary
J-by-K), `signal_scheme` (`original`/`restore_minimum`), `signal_multiplier`,
and optional `intercept`. They modify the population, not the estimator.
The built-in Q designs have J=3K; other item counts require an explicit
`Q_matrix`, installed before coefficient generation.
The unchanged spectral initializer still requires J>=3K. Smaller custom
designs can be generated and fitted with an explicitly requested random start.
Restore-minimum uses total*q/min(row_degree,3), with total 3 for Bernoulli/
Lognormal and 2 for Poisson. Multiplication applies to every active slope.

Useful existing-procedure grid fields are `epsilon`, `init_cutoff_c`, `pen`
or `pen_multiplier`, `tau` or `tau_multiplier`, `max_iter`, `inner_max_iter`,
`min_outer_iter`, `tol`, `gain_numerator`, `gain_offset`, and
`coefficient_bound_scale`. Graph methods, ratios m/N, ESS values, structure
priors, maximum parent count, and number of resampling draws are configured
in the separate `graph` object. Plug-in m may be fractional; the resampling
size uses max(1,floor(ratio*N)). Graph draws do not rerun the measurement fit.

Each output directory contains a frozen `manifest.json` with source hashes,
configuration and software versions; dataset directories save X/A/Q/B/truth,
every start's parameter and sparse-law artifacts, and graph counts/results.
`summary.json` and `summary.txt` aggregate the study. Existing output is not
overwritten. `--resume` requires matching source, plan and environment and
skips completed records, including failures. Incomplete directories from
interrupted runs are reported for inspection.

The supplied configurations are experiment plans; the smoke plan checks
integration. Completed checks are listed in `validation-2026-10-08.md`.
The strong-signal settings differ from the original manuscript's generating
parameters and do not reproduce its old Q2 table.

## Release scope

This release covers general simulation fitting, graph options and the
Reviewer 2 diagnostics. Seesaw and TIMSS retain their fitting iterations, but fitted outputs now
receive final column-sum orientation. `run_timss.py` uses one graph draw and its saved
comparison covers the earlier graph-scale experiment. The newer TIMSS
200-candidate graph-constrained BIC search, bootstrap, EBIC/K-selection,
Gaussian grayscale and additional baseline experiments are not included.

The code retains a finite numerical M-step and post-M-step hard threshold;
final fitted-output orientation is mandatory. For L>0, the public implementation adds its
uniform pseudocount contribution to all states. This differs from the
response-letter formula; L=0 experiments are unaffected.
