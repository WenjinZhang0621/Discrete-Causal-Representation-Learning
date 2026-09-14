# Discrete Causal Representation Learning

## Installation

After unzipping the archive, open a terminal in the extracted folder and run

```bash
conda env create -f environment.yml
conda activate dcrl
```

## Run

For the main simulation pipeline,

```bash
python experiments/run_parallel.py --start 1 --end 11 --n 4000 --k 10 --dag_type Tree --distribution Lognormal
```

For the seesaw experiment,

```bash
python experiments/run_seesaw.py --N 10000 --K 4 --seed 123 --bank_size 2000 --jitter 0.001 --noisy_z3 --p_up_if11 0.8 --p_up_else 0.2 --p_ball4_if_up 0.99 --p_ball4_if_down 0
```

For the seesaw experiment using the pre-generated dataset used in the paper, which reproduces the exact seesaw results reported in the paper,

```bash
python experiments/run_seesaw_from_dataset.py --dataset_path data/seesaw/seesaw_dataset.npz
```

For TIMSS, fit the original student responses and then use multinomial latent
resampling. The fitting seed is 48 (selected by likelihood in the prior
initialization comparison); the separate graph-sampling seed is 1. The default
ratio is `f(N)/N = 0.2`, selected by the lowest BIC in the complete graph-scale
comparison using multinomial sampling.

```bash
python experiments/run_timss.py --fn_ratio 0.2 --seed 48 --graph_seed 1
```

To reproduce the graph from the saved **fresh** latent distribution used in
the revised comparison:

```bash
python experiments/run_timss_from_saved_p.py --fn_ratio 0.2 --seed 1
```

Both commands use actual multinomial samples, with
`f(N) = max(1, floor(fn_ratio * N))`. With `N=620` and ratio `0.2`, they draw
124 latent observations and return a seven-edge CPDAG for the supplied fit
and seed. Ratio `0.5` gives ten edges and a higher BIC. See the
[complete latent-law and graph-scale comparison](docs/timss-comparison.md)
for all examined settings, saved fitted parameters, and score verification.

`--no_plot` skips PNG generation. Plotting uses Matplotlib. The JSON
summary and numeric graph are always written. `--N` in the saved-fit command
means original students, not the number of latent resamples. The new default
checkpoint is `data/TIMSS/timss_refit.json`, which includes its fit provenance
and input hashes. The historical `TIMSS_estimate_p.npz` is retained for explicit
legacy comparisons; it is not the revised default.

The fast TIMSS refit solves the same bounded fixed-Q lognormal item model by
weighted least squares and profiled variance updates, using observed cells.
Its reported likelihood includes the Gaussian normalizing constant on clipped
log times. It is a saturated latent-law fit; attaching a graph afterward does
not turn that likelihood into a graph-constrained likelihood or reduce its
parameter count. Graph-constrained BIC comparisons require separate refitting.

The default stays fixed for ball images and the simulation pipeline. Sparse
initialization changes storage, not their statistical settings. The sampling
condition needed for ratios below one is explained in
[the consistency note](docs/resampling-consistency.md).

## Output

Main simulation results are written automatically to `results/`, for example `results/results_4000_Lognormal_Tree_2.txt` with columns `iter,shd_val,shd2_val,shd3_val`.

The seesaw script writes outputs to `results/seesaw/`, including `p_hat.npy`, `B_hat.npy`, `latent_graph_estimated.png`, `latent_graph_estimated.npy`, `summary.json`, and heatmaps under `results/seesaw/heatmaps/`.

The TIMSS script writes outputs to `results/timss/`, including `p_hat.npy`, `B_hat.npy`, `gamma_hat.npy`, `latent_samples_from_phat.npy`, `latent_counts.npy`, `latent_graph_estimated.npy`,
`latent_graph_estimated.png`, and `summary.json`. Summaries record the actual
resample size, seeds, input/sample hashes, and numerical package versions.

## File map

* `src/dcrl/utils.py`

  * generic helpers such as `binary`, `TLP`, `thres`, `sigmoid`
* `src/dcrl/config.py`

  * experiment configuration and default hyperparameter construction
* `src/dcrl/latent_models.py`

  * latent DAG / latent-state generators
  * `Markov`, `Tree`, `DiverseTree`, `Model-7`, `Model-8`, `Model-13`, `Model-16`
* `src/dcrl/data_generator.py`

  * `GenerateData` class
  * creates `Q`, `B`, `gamma`, latent `A`, and observed `X`
* `src/dcrl/initialization.py`

  * spectral / varimax initialization of `p`, `B`, `gamma`, `A`
* `src/dcrl/evaluation.py`

  * graph stitching and SHD code
  * resampling from estimated `p_hat` and evaluating recovered graphs
* `src/dcrl/estimator.py`

  * `DAGEstimator`
  * `PSAEM` and the main `estimate` method
* `src/dcrl/runner.py`

  * `ParallelDAGEstimator`
  * parallel streaming experiment runner with file locking
* `src/dcrl/seesaw/dataset.py`

  * seesaw image generation, masks, pooled observations, and latent variables
* `src/dcrl/seesaw/init.py`

  * custom initialization for the seesaw Bernoulli setting
* `src/dcrl/seesaw/psaem.py`

  * Bernoulli PSAEM for the seesaw experiment
* `src/dcrl/seesaw/plotting.py`

  * heatmap visualization for the estimated coefficient matrix `\\hat B`
* `src/dcrl/timss/em.py`

  * lognormal EM algorithm with missing data for the TIMSS experiment
* `experiments/truth_graph.py`

  * builds truth graphs for each named DAG family
* `experiments/run_parallel.py`

  * the executable CLI script for the main simulation pipeline
* `experiments/run_seesaw.py`

  * the executable CLI script for the seesaw Bernoulli experiment
* `experiments/run_seesaw_from_dataset.py`

  * the executable CLI script that uses the pre-generated seesaw dataset used in the paper and reproduces the exact seesaw results reported in the paper
* `experiments/run_timss.py`

  * the executable CLI script for the TIMSS experiment
* `experiments/run_timss_from_saved_p.py`

  * the saved-fit TIMSS CLI, using multinomial resampling from the fresh comparison fit
* `data/TIMSS/Q.csv`

  * Q-matrix for the TIMSS experiment
* `data/TIMSS/time.csv`

  * observed TIMSS data matrix used by `run_timss.py`
* `data/TIMSS/TIMSS_estimate_p.npz`

  * historical latent distribution, retained for explicit legacy comparisons

## Sparse probabilities in the main Gibbs–SAEM pipeline

`DAGEstimator.PSAEM()` now returns `p_hat` as a `SparseLatentLaw`. Initialization,
Gibbs updates, finalization and graph-evaluation resampling no longer allocate
arrays over all `2**K` states. The measurement estimates and Gibbs count updates
are unchanged. Integer state keys can exceed 64 bits.

```python
from dcrl import SparseLatentLaw

# After fitting an estimator:
p_hat = estimator.p_hat
p_hat.save_npz("p_hat_sparse.npz")
p_hat = SparseLatentLaw.load_npz("p_hat_sparse.npz")
latent_samples = p_hat.sample(1000)
probability = p_hat.probability(5)  # encoded binary configuration
```

The law stores encountered-state masses and, when needed, one aggregate uniform
mixture weight. Positive `L_final` preserves the previous public implementation's
rule of adding `L_final / U` to every state, where `U` is the number of unvisited
states. A zero-count law retains the previous uniform fallback. Sampling the
uniform component draws independent fair bits, without listing configurations.

For `S` stored states and `M` requested samples, finalization takes `O(S)` work
and storage, and resampling takes `O(S + M*K)` work and storage under machine-word
key accounting. Larger keys require `O(K*S)` bits and additional polynomial bit
operations. This removes mandatory state-space enumeration; it does not guarantee
polynomial convergence or polynomial worst-case unrestricted GES search.

Legacy consumers can explicitly request `DAGEstimator(..., dense_p_output=True)`
or call `p_hat.to_dense()`; these optional conversions cost `O(2**K)`. The separate
seesaw and TIMSS estimators retain their existing interfaces. The main simulation
runner and its GES evaluation accept the sparse result directly.


The ball-image initializer also counts encountered states without enumerating
all configurations when called by its PSAEM routine. Its final dense output
and the existing ball-image sampling settings are preserved.

## Verification

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest tests -q
```

The regression tests compare the initialization and main Gibbs–SAEM estimates
with public commit `190f198f9134655cca7c517905cdf61a994da6a8`, test sparse state
keys beyond 64 bits, and verify multinomial resampling and repeated TIMSS runs.
Use a normal clone with that commit in its history (CI fetches full history).
