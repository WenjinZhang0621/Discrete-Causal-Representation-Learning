# Validation completed on 2026-10-08

Base version: `06da9c9`. All changes are isolated in the revision branch;
the original checkout remains unchanged. These are implementation and
integration checks, not completed 500-repetition revision experiments.

* Full test suite after the final release review: **148 passed**, with four warnings from the preserved
  Poisson boundary behavior in the reference/parity test. No new numerical
  truncation was inserted to suppress these warnings.
* Six legacy trajectory comparisons: all three distributions crossed with
  Q1/Q2 reproduce original fitted arrays and RNG evolution.
* Integer BDeu score and graph parity, real ESS/prior forwarding, unrounded
  fractional counts, sparse/uniform support, deterministic plug-in scoring,
  concurrent graph calls, and K=1/2 are tested.
* True-CPT capture preserves the original generator's samples and Python/
  NumPy RNG streams for all seven supported latent graph families.
* Signed output orientation preserves observed likelihood and support;
  permutation/level alignment, fixed-Q constraints and multi-start selection
  are independently checked.
* Failure-path tests retain failed/missing starts and graphs. True-p and
  true-latent-sample oracle stages still run after fitted-stage failure.
  Summaries use datasets as the MC unit and expose failure denominators.
* Three-family end-to-end smoke, rerun after the release fixes: six generated datasets, 36 full/known-Q
  start fits, and 228 graph fits completed with no failed outputs. Both
  graph methods, two ESS values, repeated graph draws, orientation and all
  oracle stages were exercised. Final-sample classification alignment was
  enabled in the release rerun. All 264 existing data/fitting arrays and all
  228 graph arrays matched the previous smoke run exactly; evaluation labels
  can change under the newly explicit alignment criterion. Source hashes
  matched functional release `a4b0d8a`. Resume with identical metadata succeeded on the
  earlier smoke run.
* An additional Model-13 Bernoulli run at N=3000 exercised two full-model
  starts, two known-Q starts, exact population truth, both graph methods,
  all oracle stages and graph variability; the full run completed.
* Two previous strong-signal Model-13 datasets (N=3000/7000) reproduced the
  saved epsilon=.001/c=2.5 initial/final arrays, latent laws and graph arrays
  exactly when using the same graph-law representation, seed and no output
  orientation. Graph identity is not asserted across different samplers.
* The legacy simulation CLI's plug-in option and the graph-only CLI on the
  saved TIMSS law completed. The legacy reference helper now honors the
  requested K, avoiding its former fixed-K dimension mismatch.
* Final review also tested classification matching with complements, a shared
  Q/p/graph permutation, directed-edge recovery, late artifact-write failure
  accounting, and custom Q matrices with item counts above/below 3K. The
  original spectral initializer still requires J>=3K; other valid custom
  matrices can be used with the explicit random-start option.
* An isolated wheel build/install verified package imports, the TIMSS/seesaw
  subpackages, and plug-in graph scoring without relying on a source-path hack.
* An explicit CLI run with nondefault c=3.0 and epsilon=0.002 completed, and
  its output manifest recorded both requested values. Both parameters are
  also exposed through ExperimentConfig and frozen JSON model/grid settings.
* The later compact-style pass changed whitespace, comments and docstrings.
  All 26 reviewed Python files have the same executable AST as `a4b0d8a`,
  and the full test suite again passed all 148 tests. Source hashes change
  after formatting, so an old frozen study still requires its original checkout
  for `--resume`.

Saved full study configurations are templates for subsequent experiments.
Larger runs, Monte Carlo performance claims and final manuscript tables must
be based on those actual outputs. Passing tests or attaining the best score
among starts does not establish global optimization or the assumptions of
the theoretical consistency result.
