# File-by-file split

## `src/dcrl/utils.py`
Put only generic helper functions here:
- `binary`
- `TLP`
- `thres`
- `nchoosek_prac`
- `initialize_function`
- `sigmoid`

## `src/dcrl/config.py`
Put experiment-level configuration here:
- `BoundsConfig`
- `ExperimentConfig`
- default formulas for `tau`, `pen`, `tol`, `epsilon`, `Q_N`

## `src/dcrl/latent_models.py`
Put all latent-state generators here:
- `DiverseBTree`
- `generate_prob_markov`
- `generate_markov_chain`
- `generate_tree`
- `diverse_tree`
- `model7`
- `model8`
- `model13`
- `model16`
- `sample_latent_matrix`

## `src/dcrl/data_generator.py`
Put the data-generation class here:
- `GenerateData`
- alias `Generate_Data`

Responsibilities:
- build `Q`
- build `B`
- build `gamma`
- call latent-model sampler to get `A`
- generate observed `X`

## `src/dcrl/initialization.py`
Put only the spectral/varimax initializer here:
- `initialize_parameters`

This file contains:
- SVD-based inverse transform
- varimax rotation
- sign flip
- column permutation
- initial `A_est`, `B_ini`, `gamma_in`, `p_init`

## `src/dcrl/evaluation.py`
Put graph post-processing and metrics here:
- `stitch_full_A`
- `shd_cpdag`
- `sample_latent_states_from_p`
- `recover_full_graph_from_estimates`
- `compute_shd_triplet`

## `src/dcrl/estimator.py`
Put the estimation class here:
- `DAGEstimator`
- alias `DAG_Estimator`

Methods here:
- `ftn_T`
- `ftn_h`
- `ftn_A`
- `objective`
- `ftn_pen`
- `F_1_SAEM`
- `PEM`
- `PSAEM`
- `init`
- `estimate`

## `src/dcrl/runner.py`
Put only parallel experiment-running code here:
- `ParallelDAGEstimator`
- `run_and_log`
- `parallel_estimate_streaming`

## `experiments/truth_graphs.py`
Put truth-graph construction here:
- `build_truth_graph`
- `build_all_truth_graphs`

## `experiments/run_parallel.py`
Put only CLI / executable experiment code here:
- `argparse`
- config construction
- truth graph lookup
- `ParallelDAGEstimator(...)`
- `.parallel_estimate_streaming(...)`
