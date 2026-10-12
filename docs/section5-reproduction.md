# Section 5 reproduction

Run `python experiments/run_section5.py` from the repository root in the
Python 3.12 environment specified by `requirements-reproducible.txt`.
The entry point sets numerical libraries to one thread per worker. Use
`--jobs` for process-level parallelism. The full plan contains 45,000 fits;
`--plan-only` exports the 90 JSON plans without starting them.

## Configuration

Tasks are ordered by graph (Chain-10, Tree-10, Model-7, Model-8, Model-13),
then Q (1, 2), distribution (Bernoulli, Poisson, Lognormal), and sample size
(3000, 5000, 7000). Each task has 500 repetitions. Chain-10 uses `Markov`
in the data generator. J=3K for all graphs.

The base plan reproduces the original Section 5 cluster plans from source
`61cd9635be6a451761fd81e2506c2db055029905`. It uses spectral initialization,
`init_cutoff_c=2.5`, coefficient-bound scale 1, at most 20 outer iterations,
and the original distribution-specific gain and stopping defaults. The
Bernoulli epsilon is 0.001, Poisson epsilon is 1e-50, and Lognormal retains
1e-5 (unused by its initializer). Restored minimum measurement coefficients
are multiplied by 1.25. No other data-generation parameters are changed.

Only tasks 39–41, 69–71 and 78–80 use the validation overrides. Their full
recipe is listed in the README and in `configs/section5.json`; it applies
at all three sample sizes. In particular, `kappa=0.01` is smoothing **per
latent state**, C=3 is the Gibbs-sweep count, and the doubled threshold is
`tau=4*N**(-0.275)`. The penalty remains `N**0.25`.
`min_outer_iter=max_iter` makes the 100/400-round budget fixed. The probe
records convergence diagnostics without an additional stopping rule.

The three groups were chosen after inspecting the original results and
were subsequently evaluated on fresh independent datasets with the recipe
fixed. The remaining 27 groups retain their original settings. Thus the
reproduction plan contains explicit group-specific overrides; it is not a
claim that one common new parameter setting was tested on all 30 groups.

## Random seeds

For zero-based task index t and replicate index r=0,...,499:

| Tasks | Data seed s | SAEM seed | Initialization seed | Graph seed |
| --- | --- | --- | --- | --- |
| Original 81 tasks | `20261008 + 1000000*t + r` | `s+100000` | `s+200000` | `s+300000` |
| Validation tasks 39–41, 69–71, 78–80 | `910000000 + 1000000*t + r` | `s+100000+N` | `s+200000+N` | `s+300000+N` |

Every dataset is generated directly at its own N with its own data seed.
There is no nesting, warm start or fitting reuse between sample sizes.
Changing worker count does not change these seeds.

## Fitting and evaluation

The original 81 tasks use `dcrl.estimator.DAGEstimator`. The nine validation
tasks use `dcrl.section5_backend`: accumulated sufficient-statistic objectives,
analytic likelihood derivatives with bound-aware finite differences for the
penalty, vectorized Gibbs sampling, and convergence diagnostics. The Gibbs
implementation consumes random numbers differently from the original
scalar loop, so replacing it with the scalar estimator would not reproduce
the validation results even with the same parameters and seeds.

The four backend modules are from the validation bundle
`52d32aa6def677f92ea3ffa877b636f4283ad35b71e7cbdd1b718f4f8cf711a1`
(SHA-256 of `validate3_500_independent_20261011.tar.gz`). Packaging changes
relative imports and the audited estimator-file hash. The current public
estimator adds final orientation in its legacy `run()` method; its PSAEM
and optimizer methods used here are unchanged from the validation source.
The hash guard prevents silently applying the backend to a different
estimator implementation. Backend selection is per fit, not a global patch.

All final fits use column-sum orientation. Evaluation matches latent
coordinates using the final Gibbs sample and simulated latent sample; truth
is not used to fit the model or choose a start. The same coordinate matching
is used for Q, latent graph and composite graph errors. Graph recovery uses
multinomial latent resampling with m=N and ESS=1. The structure prior is the
original default (`null`) for 81 tasks and 0.01 for the nine validation tasks.
The reference latent graph is the exact generator CPDAG.

Overall SHD is `matched_full_shd`: the SHD of the combined measurement and
latent CPDAG after matching. Composite exact recovery means this SHD is zero.
The other exact-recovery columns refer to Q, the latent CPDAG, its skeleton,
v-structures, and compelled directed edges. CPDAG edge recovery does not
claim identification of every direction in a particular DAG.

## Independent validation

The cluster validation recorded 4,500/4,500 successful datasets, with 500
per setting and no recorded optimizer failures. The reported values are:

| Group | Overall SHD, N=3000 / 5000 / 7000 | Composite exact recovery, N=3000 / 5000 / 7000 |
| --- | --- | --- |
| Model-7 / Q1 / Poisson | 0.684 / 0.212 / 0.184 | 77.8% / 90.8% / 92.8% |
| Model-8 / Q2 / Lognormal | 0.104 / 0.058 / 0.014 | 97.0% / 98.6% / 99.4% |
| Model-13 / Q1 / Lognormal | 0.522 / 0.118 / 0.028 | 88.2% / 98.0% / 99.2% |

All three observed SHD sequences decrease. Four of the six adjacent-N
difference intervals include zero; these trends do not establish a strictly
decreasing population error at every step. The combined summary saves
independent-N mean differences and approximate, unadjusted 95% normal
intervals in `section5_summary.json`.

Per-setting manifests record the resolved plan, source hashes and package
versions. Use the pinned packages and fixed seeds when comparing numerical
results. Platform and numerical-library differences can still affect
optimization trajectories. Resume refuses changed provenance and does not
replace failed attempts. Keep original and validation runs in separate
folders from any new reproduction run.
