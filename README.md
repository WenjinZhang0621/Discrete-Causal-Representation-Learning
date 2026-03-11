# Discrete Causal Representation Learning

Code for the paper **Discrete Causal Representation Learning**  
Wenjin Zhang, Yixin Wang, and Yuqi Gu

**Manuscript:** [PDF](https://yuqigu.github.io/assets/pdf/DCRL_Mar07_2026.pdf)

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
- `experiments/truth_graphs.py`
  - builds truth graphs for each named DAG family
- `experiments/run_parallel.py`
  - the executable CLI script

## Example

```bash
python experiments/run_parallel.py --start 1 --end 11 --n 4000 --k 10 --dag_type Tree --distribution Lognormal
```
