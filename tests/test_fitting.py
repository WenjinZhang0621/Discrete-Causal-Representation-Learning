"""Regression and statistical invariants for the optional fitting API."""
import contextlib
import io
from pathlib import Path
import random
import subprocess
import sys
import types

import numpy as np
import pytest
from scipy.special import expit,logsumexp
from scipy.stats import bernoulli,lognorm,poisson
from threadpoolctl import threadpool_limits

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/"src"))
from dcrl.config import ExperimentConfig
from dcrl.estimator import DAGEstimator
from dcrl.fitting import fit_observed,observed_loglikelihood,LikelihoodLimitError
from dcrl.initialization import initialize_parameters
from dcrl.latent_law import SparseLatentLaw
from dcrl.orientation import orient_columns

PUBLIC_COMMIT="06da9c9"


def public_module(filename):
    source=subprocess.check_output(["git","-C",str(REPO),"show",f"{PUBLIC_COMMIT}:src/dcrl/{filename}.py"],text=True)
    module=types.ModuleType(f"dcrl.public_{filename}")
    module.__package__="dcrl"
    exec(compile(source,f"public_{filename}.py","exec"),module.__dict__)
    return module


@pytest.mark.parametrize("family",["Bernoulli","Poisson","Lognormal"])
@pytest.mark.parametrize("q_type",["1","2"])
def test_legacy_trajectory_matches_public_commit(family,q_type):
    config=ExperimentConfig(N=120,K=3,distribution=family,Q_type=q_type,max_iter=4).finalize()
    names=("N","J","K","Q_type","DAG_type","distribution","algorithm","tau","pen",
             "max_iter","tol","C","epsilon","Q_N","kappa","L_final")
    kwargs={key: getattr(config,key) for key in names}
    kwargs.update(vars(config.bounds))
    reference=public_module("estimator")
    reference.initialize_parameters=public_module("initialization").initialize_parameters
    results,states=[],[]
    for cls in (reference.DAGEstimator,DAGEstimator):
        random.seed(9311)
        np.random.seed(9311)
        est=cls(**kwargs)
        with threadpool_limits(limits=1),contextlib.redirect_stdout(io.StringIO()):
            results.append(est.PSAEM(0))
        states.append(np.random.get_state())
    np.testing.assert_array_equal(results[0][0].to_dense(),results[1][0].to_dense())
    for a,b in zip(results[0][1:],results[1][1:]):
        np.testing.assert_array_equal(a,b)
    for a,b in zip(states[0],states[1]):
        np.testing.assert_array_equal(a,b)
    assert len(est.iteration_diagnostics)==results[-1][-2]
    assert len(est.optimizer_diagnostics)==config.J*results[-1][-2]
    assert est.stopping_reason in {"tolerance","max_iter"}


@pytest.mark.parametrize("family",["Bernoulli","Poisson","Lognormal"])
def test_exact_likelihood_and_orientation_match_direct_mixture(family):
    B=np.array([[3.,-.7,.4],[2.,-.3,-.2]])
    gamma=np.array([[.5],[1.4]])
    p=SparseLatentLaw(2,{0: .2,1: .3,3: .4},.1)
    A=np.array([[0,0],[0,1],[1,0],[1,1]])
    mean=A@B[:,1:].T+B[:,0]
    X=np.array([[0.,1.],[1.,0.],[1.,1.]])
    if family=="Poisson":
        X=np.array([[1.,4.],[3.,0.],[2.,1.]])
        conditional=poisson.logpmf(X[:,None,:],mean[None,:,:]).sum(axis=2)
    elif family=="Lognormal":
        X=np.array([[.4,1.3],[2.,3.],[1.,1.]])
        conditional=lognorm.logpdf(X[:,None,:],s=np.sqrt(gamma.ravel()),scale=np.exp(mean)[None,:,:]).sum(axis=2)
    else:
        conditional=bernoulli.logpmf(X[:,None,:],expit(mean)[None,:,:]).sum(axis=2)
    expected=logsumexp(conditional+np.log([p.probability(s) for s in range(4)]),axis=1).sum()
    actual=observed_loglikelihood(X,B,p,gamma,family,batch_size=2)
    np.testing.assert_allclose(actual,expected,rtol=1e-13)
    changed=orient_columns(B,p,gamma)
    assert changed["flipped_columns"]==[0]
    assert changed["xor_mask_hex"]=="2"  # MSB-first coordinate encoding.
    np.testing.assert_allclose(observed_loglikelihood(X,changed["B"],changed["p"],gamma,family),actual,rtol=1e-13)
    for state in range(4):
        assert changed["p"].probability(state^2)==p.probability(state)
    np.testing.assert_array_equal(B,np.array([[3.,-.7,.4],[2.,-.3,-.2]]))


def test_zero_columns_and_large_sparse_law():
    B=np.zeros((2,81))
    p=SparseLatentLaw(80,{0: .7,1<<79: .3})
    oriented=orient_columns(B,p)
    assert oriented["unresolved_columns"]==list(range(80))
    actual=observed_loglikelihood(np.ones((3,2)),B,p,None,"Bernoulli",max_states=2)
    np.testing.assert_allclose(actual,-6*np.log(2))
    with pytest.raises(LikelihoodLimitError,match="nonzero uniform mass"):
        observed_loglikelihood(np.ones((3,2)),B,SparseLatentLaw(80,{},1),None,"Bernoulli")


def test_known_support_is_fixed_and_not_thresholded():
    rng=np.random.RandomState(27)
    X=rng.binomial(1,.5,size=(40,6))
    Q=np.tile(np.eye(2,dtype=int),(3,1))
    cfg=ExperimentConfig(N=40,K=2,distribution="Bernoulli",max_iter=1,
                           inner_max_iter=2,tau=100.,pen=100.)
    with threadpool_limits(limits=1),contextlib.redirect_stdout(io.StringIO()):
        result=fit_observed(X,cfg,[{"kind": "random","seed": 901}],known_Q=Q)
    assert result["status"]=="ok",result
    assert result["fit_kind"]=="fixed_support_nuisance"
    assert result["effective_penalty"]==result["penalty"]==0
    np.testing.assert_array_equal(result["B"][:,1:][Q==0],0)
    np.testing.assert_array_equal(result["initial_B"][:,1:][Q==0],0)
    assert np.any(result["B"][:,1:]!=0)  # tau=100 would delete all slopes.
    assert result["diagnostics"]["optimizer_failures"]>0
    assert result["starts"][0]["fit_valid"]  # Inner failure is not fit failure.


def test_multistart_selection_failure_records_and_rng_restoration():
    X=np.random.RandomState(5).binomial(1,.5,size=(40,6))
    cfg=ExperimentConfig(N=40,K=2,distribution="Bernoulli",max_iter=1,inner_max_iter=3)
    specs=[{"kind": "unknown","seed": 20},{"kind": "random","seed": 21},{"kind": "random","seed": 22}]
    state=np.random.get_state()
    with threadpool_limits(limits=1),contextlib.redirect_stdout(io.StringIO()):
        result=fit_observed(X,cfg,specs,orientation="column_sum")
        repeat=fit_observed(X,cfg,specs,orientation="column_sum")
    for a,b in zip(state,np.random.get_state()):
        np.testing.assert_array_equal(a,b)
    assert len(result["starts"])==3
    assert result["starts"][0]["status"]=="failed"
    eligible=[r for r in result["starts"] if r["scorable"]]
    assert result["selected_start"]==max(eligible,key=lambda r: r["penalized_objective"])["id"]
    assert not result["global_optimum_certified"]
    np.testing.assert_array_equal(result["B"],repeat["B"])
    selected=next(s for s in result['starts'] if s['id']==result['selected_start'])
    expected_sample=selected['final_sample_A'].copy()
    flips=result['orientation']['flipped_columns']
    expected_sample[:,flips]=1-expected_sample[:,flips]
    np.testing.assert_array_equal(result['final_sample_A'],expected_sample)
    np.testing.assert_array_equal(result['raw_final_sample_A'],selected['final_sample_A'])
    assert result["observed_loglik"]==repeat["observed_loglik"]
    np.testing.assert_allclose(observed_loglikelihood(X,result["B"],result["p"],result["gamma"],"Bernoulli"),result["observed_loglik"],rtol=1e-13)
    failed=fit_observed(X,cfg,[{"kind": "unknown","seed": 1}])
    assert failed["status"]=="failed" and len(failed["starts"])==1
    with pytest.raises(ValueError,match="Start ids must be unique"):
        fit_observed(X,cfg,[{"id": "duplicate","seed": 1},{"id": "duplicate","seed": 2}])


def test_revision_profile_changes_only_default_bernoulli_epsilon():
    legacy=ExperimentConfig(N=100,K=3,distribution="Bernoulli").finalize()
    revision=ExperimentConfig(N=100,K=3,distribution="Bernoulli",profile="revision").finalize()
    differences={key for key in vars(legacy) if getattr(legacy,key)!=getattr(revision,key)}
    assert differences=={"profile","epsilon"}
    assert revision.epsilon==.001 and legacy.epsilon==1e-5
    assert ExperimentConfig(N=100,K=3,distribution="Bernoulli",profile="revision",epsilon=.02).finalize().epsilon==.02


def test_low_level_observed_data_never_generates_truth(monkeypatch):
    cfg=ExperimentConfig(N=120,K=3,distribution="Bernoulli",max_iter=2).finalize()
    names=("N","J","K","Q_type","DAG_type","distribution","algorithm","tau","pen",
             "max_iter","tol","C","epsilon","Q_N","kappa","L_final")
    kwargs={name: getattr(cfg,name) for name in names}
    kwargs.update(vars(cfg.bounds))
    X=np.random.RandomState(441).binomial(1,.5,(cfg.N,cfg.J))
    def forbidden(*args,**kwargs):
        raise AssertionError("Observed-data fitting must never generate data or truth")
    for method in ("generate_Q","generate_B","generate_gamma","generate_latent_data","generate_data"):
        monkeypatch.setattr(DAGEstimator,method,forbidden)
    est=DAGEstimator(**kwargs,observed_X=X)
    with threadpool_limits(limits=1),contextlib.redirect_stdout(io.StringIO()):
        expected_initial=initialize_parameters(X,cfg.J,cfg.K,cfg.distribution,cfg.epsilon,sparse=True)
        actual_initial=est.init(0)
        np.random.seed(432)
        direct=est.PSAEM(0)
        fitted=fit_observed(X,cfg,[{"kind": "spectral","seed": 432}])
    assert fitted["status"]=="ok",fitted
    for a,b in zip(expected_initial[1:],actual_initial[1:]):
        np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(est.X,X)
    np.testing.assert_array_equal(direct[1],fitted["B"])
    np.testing.assert_array_equal(direct[0].to_dense(),fitted["p"].to_dense())
