# Discrete Causal Representation Learning

Code for the paper **Discrete Causal Representation Learning**: [PDF](https://yuqigu.github.io/assets/pdf/DCRL_Mar07_2026.pdf)

## Installation

```bash
git clone https://github.com/WenjinZhang0621/Discrete-Causal-Representation-Learning.git
cd Discrete-Causal-Representation-Learning
conda env create -f environment.yml
conda activate dcrl
```

Or install dependencies with

```bash
pip install -r requirements.txt
```

## Run

```bash
python experiments/run_parallel.py --start 1 --end 11 --n 4000 --k 10 --dag_type Tree --distribution Lognormal
```

Note that the script runs iterations in `range(start, end)`, so the right endpoint is excluded. For example,

```bash
python experiments/run_parallel.py --start 1 --end 2 --n 4000 --k 10 --dag_type Tree --distribution Lognormal
```

runs exactly one iteration.

## Output

Results are written automatically to `results/`, for example `"results/results_4000_Lognormal_Tree_2.txt"` with columns `"iter,shd_val,shd2_val,shd3_val"`.

## File map

- `src/dcrl/utils.py`
  - generic helpers such as `binary`, `TLP`, `thres`, `sigmoid`
- `src/dcrl/config.py`
  - experiment configuration and default hyperparameter construction
- `src/dcrl/latent_models.py`
  - latent DAG / latent-state generators
  - `Markov`, `Tree`, `DiverseTree`, `Model-7`, `Model-8`, `Model-13`, `Model-16`
- `src/dcrl/data_generator.py`
  - `GenerateData` class
  - creates `Q`, `B`, `gamma`, latent `A`, and observed `X`
- `src/dcrl/initialization.py`
  - spectral / varimax initialization of `p`, `B`, `gamma`, `A`
- `src/dcrl/evaluation.py`
  - graph stitching and SHD code
  - resampling from estimated `p_hat` and evaluating recovered graphs
- `src/dcrl/estimator.py`
  - `DAGEstimator`
  - `PEM`, `PSAEM`, and the main `estimate` method
- `src/dcrl/runner.py`
  - `ParallelDAGEstimator`
  - parallel streaming experiment runner with file locking
- `experiments/truth_graph.py`
  - builds truth graphs for each named DAG family
- `experiments/run_parallel.py`
  - the executable CLI script
