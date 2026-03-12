# Discrete Causal Representation Learning

Code for the paper **Discrete Causal Representation Learning**: [PDF](https://yuqigu.github.io/assets/pdf/DCRL_Mar07_2026.pdf)

## Installation

```bash
git clone https://github.com/WenjinZhang0621/Discrete-Causal-Representation-Learning.git
cd Discrete-Causal-Representation-Learning
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

For the TIMSS experiment,

```bash
python experiments/run_timss.py --x_path data/TIMSS/time.csv --q_path data/TIMSS/Q.csv --K 7 --max_iter 100 --tol 0.9
```

## Output

Main simulation results are written automatically to `results/`, for example `results/results_4000_Lognormal_Tree_2.txt` with columns `iter,shd_val,shd2_val,shd3_val`.

The seesaw script writes outputs to `results/seesaw/`, including `p_hat.npy`, `B_hat.npy`, `latent_graph_estimated.png`, `latent_graph_estimated.npy`, `summary.json`, and heatmaps under `results/seesaw/heatmaps/`.

The TIMSS script writes outputs to `results/timss/`, including `p_hat.npy`, `B_hat.npy`, `gamma_hat.npy`, `latent_samples_from_phat.npy`, `latent_graph_estimated.png`, `latent_graph_estimated.npy`, and `summary.json`.

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
* `experiments/run_timss.py`

  * the executable CLI script for the TIMSS experiment
* `data/TIMSS/Q.csv`

  * Q-matrix for the TIMSS experiment
* `data/TIMSS/time.csv`

  * observed TIMSS data matrix used by `run_timss.py`
