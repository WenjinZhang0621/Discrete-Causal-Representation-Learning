from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BoundsConfig:
    lob: float = 0.60
    upb: float = 0.65
    lob2: float = 0.35
    upb2: float = 0.40
    lob3: float = 0.77
    upb3: float = 0.82
    lob4: float = 0.20
    upb4: float = 0.25
    lob5: float = 0.65
    upb5: float = 0.70


@dataclass
class ExperimentConfig:
    N: int
    K: int
    J: Optional[int] = None
    Q_type: str = "2"
    DAG_type: str = "Tree"
    distribution: str = "Lognormal"
    algorithm: str = "PSAEM"
    bounds: BoundsConfig = field(default_factory=BoundsConfig)
    tau: Optional[float] = None
    pen: Optional[float] = None
    max_iter: int = 20
    tol: Optional[float] = None
    C: int = 1
    epsilon: Optional[float] = None
    Q_N: Optional[int] = None
    kappa: float = 1e-200
    L_final: float = 0.0
    n_jobs: int = 5

    def finalize(self):
        self.Q_type = str(self.Q_type)
        if self.J is None:
            self.J = 3 * self.K
        lambda_vec = [self.N ** (1 / 8), self.N ** (2 / 8), self.N ** (3 / 8)]
        const = 0.9
        tau_vec = 2 * self.N ** ((const * (1 / 8)) - 1 / 2), 2 * self.N ** ((const * (2 / 8)) - 1 / 2), 2 * self.N ** ((const * (3 / 8)) - 1 / 2)
        if self.pen is None:
            self.pen = lambda_vec[1]
        if self.tau is None:
            self.tau = tau_vec[1]
        if self.tol is None:
            self.tol = 0.5 if self.distribution == "Bernoulli" else 0.05
        if self.epsilon is None:
            self.epsilon = 1e-50 if self.distribution == "Poisson" else 1e-5
        if self.Q_N is None:
            self.Q_N = self.N
        return self
