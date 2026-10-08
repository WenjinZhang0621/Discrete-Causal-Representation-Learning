"""Custom designs must be used before coefficients or observations are built."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dcrl.config import ExperimentConfig
from dcrl.data_generator import GenerateData


@pytest.mark.parametrize("family",["Bernoulli","Poisson","Lognormal"])
@pytest.mark.parametrize("J",[5,8])
def test_custom_q_is_used_for_coefficients_and_data(family,J):
    q=np.array([[1,0],[0,1],[1,1],[1,0],[0,1],[1,1],[1,0],[1,1]])[:J]
    cfg=ExperimentConfig(N=25,J=J,K=2,distribution=family).finalize()
    generator=GenerateData.from_config(cfg,Q_matrix=q)
    np.testing.assert_array_equal(generator.Q,q)
    expected_total=2 if family=="Poisson" else 3
    np.testing.assert_array_equal(generator.B[:,1:]!=0,q)
    np.testing.assert_allclose(generator.B[:,1:].sum(axis=1),expected_total)
    np.testing.assert_array_equal(generator.B[:,0],1 if family=="Poisson" else -1)
    # Keep generation deterministic without depending on a particular latent DAG.
    generator.A=np.random.RandomState(12).binomial(1,.5,size=(cfg.N,cfg.K))
    state=np.random.get_state()
    try:
        np.random.seed(17)
        X=generator.generate_data()
    finally:
        np.random.set_state(state)
    assert X.shape==(cfg.N,J)
    assert np.all(np.isfinite(X))
    q[:]=0
    assert np.all(generator.Q.sum(axis=1)>0),"Caller mutation must not change the stored design"


@pytest.mark.parametrize("q",[np.ones((5,2)),np.full((8,2),.5),np.zeros((8,2)),np.full((8,2),np.nan)])
def test_invalid_custom_q_is_rejected(q):
    cfg=ExperimentConfig(N=25,J=8,K=2,distribution="Bernoulli")
    with pytest.raises(ValueError,match="Q_matrix"):
        GenerateData.from_config(cfg,Q_matrix=q)


@pytest.mark.parametrize("J",[5,8])
def test_nonstandard_item_count_requires_explicit_design(J):
    with pytest.raises(ValueError,match="default Q design requires J=3\\*K"):
        GenerateData.from_config(ExperimentConfig(N=25,J=J,K=2,distribution="Bernoulli"))
