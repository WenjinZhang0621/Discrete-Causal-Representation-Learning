"""BDeu parity, sparse sufficient statistics, and legacy-path preservation."""
from concurrent.futures import ThreadPoolExecutor
import importlib
from itertools import combinations
import math
import random

import numpy as np
import pytest
from scipy.special import gammaln

from dcrl.evaluation import (recover_full_graph_from_estimates,
                             sample_latent_states_from_p,stitch_full_A)
from dcrl.graph_scoring import (expected_count_local_bdeu,
                                learn_graph_from_samples,learn_latent_graph)
from dcrl.latent_law import SparseLatentLaw


GES=importlib.import_module("causallearn.search.ScoreBased.GES")


def binary_sample(counts,K):
    return np.repeat([[state>>shift&1 for shift in range(K-1,-1,-1)]
                      for state in counts],list(counts.values()),axis=0)


def assert_rng_equal(left,right):
    assert left[0]==right[0]
    np.testing.assert_array_equal(left[1],right[1])
    assert left[2:]==right[2:]


@pytest.mark.parametrize("counts",[
    {0: 21,1: 4,2: 6,3: 17,4: 8,5: 31,6: 25,7: 3},
    {0: 21,1: 4,2: 6,3: 17},  # one constant variable
])
@pytest.mark.parametrize("ess",[.3,1.,7.5])
def test_integer_scores_match_pinned_causal_learn(counts,ess):
    sample=binary_sample(counts,3)
    parameters=dict(sample_prior=ess,structure_prior=.8,
                      r_i_map={i: len(np.unique(sample[:,i])) for i in range(3)})
    for node in range(3):
        other=[i for i in range(3) if i!=node]
        for size in range(3):
            for parents in combinations(other,size):
                expected=GES.local_score_BDeu(sample,node,list(parents),parameters)
                actual=expected_count_local_bdeu(counts,3,len(sample),node,parents,
                                                   ess=ess,structure_prior=.8)
                assert actual==pytest.approx(expected,abs=1e-10)


def test_fractional_counts_are_not_rounded():
    p=np.array([.13,.17,.23,.47])
    n,ess=13.7,2.3
    cells=n*p.reshape(2,2)
    expected=math.log(.5)+np.sum(
        gammaln(ess/2)-gammaln(cells.sum(axis=1)+ess/2)
        +np.sum(gammaln(cells+ess/4)-gammaln(ess/4),axis=1))
    assert expected_count_local_bdeu(p,2,n,1,[0],ess=ess)==pytest.approx(expected)
    result=learn_latent_graph(p,2,n,method="plugin",ess=ess)
    assert result["sample_counts"] is None
    np.testing.assert_allclose(list(result["sufficient_statistics"]["stored_state_expected_counts"].values()),n*p)


def test_sparse_uniform_matches_dense_without_enumeration(monkeypatch):
    law=SparseLatentLaw(4,{0: .15,9: .23,15: .12},.5)
    dense=law.to_dense()
    monkeypatch.setattr(SparseLatentLaw,"to_dense",lambda *_: pytest.fail("Dense enumeration"))
    for node in range(4):
        other=[i for i in range(4) if i!=node]
        for size in range(4):
            for parents in combinations(other,size):
                sparse=expected_count_local_bdeu(law,4,211.3,node,parents,ess=2.1)
                full=expected_count_local_bdeu(dense,4,211.3,node,parents,ess=2.1)
                assert sparse==pytest.approx(full,abs=1e-10)
    large=SparseLatentLaw(80,{1<<79: .25,1: .25},.5)
    assert math.isfinite(expected_count_local_bdeu(large,80,100,79,[0,1,2]))


def test_uniform_component_missing_parent_cells_and_uniform_only():
    for law in (SparseLatentLaw(3,{0: .6},.4),SparseLatentLaw(3,{},1.)):
        for node in range(3):
            parents=[i for i in range(3) if i!=node]
            assert expected_count_local_bdeu(law,3,99,node,parents)==pytest.approx(
                expected_count_local_bdeu(law.to_dense(),3,99,node,parents))
    result=learn_latent_graph(SparseLatentLaw(4,{},1.),4,100,method="plugin")
    assert not np.any(result["graph"])


def test_plugin_graph_matches_expanded_integer_counts():
    counts={0: 131,1: 11,2: 17,3: 71,4: 13,5: 97,6: 61,7: 5}
    sample=binary_sample(counts,3)
    original=GES.ges(sample,score_func="local_score_BDeu")
    plugin=learn_latent_graph(counts,3,len(sample),method="plugin")
    np.testing.assert_array_equal(plugin["graph"],original["G"].graph)
    assert plugin["score"]==pytest.approx(float(original["score"]),abs=1e-9)


@pytest.mark.parametrize("p",[
    np.array([.2,0.,.3,0.,.1,.05,0.,.35]),
    SparseLatentLaw(3,{7: .3,0: .2,5: .1},.4),
])
def test_default_resampling_preserves_original_graph_counts_and_rng(p):
    np.random.seed(91)
    sample=sample_latent_states_from_p(p,3,200)
    expected=GES.ges(sample,score_func="local_score_BDeu")
    expected_state=np.random.get_state()
    np.random.seed(91)
    actual=learn_latent_graph(p,3,200)
    np.testing.assert_array_equal(actual["graph"],expected["G"].graph)
    assert actual["score"]==float(expected["score"])
    assert_rng_equal(np.random.get_state(),expected_state)
    sample_from_counts=binary_sample(actual["sample_counts"],3)
    np.testing.assert_array_equal(sample_from_counts[np.lexsort(sample_from_counts.T)],
                                  sample[np.lexsort(sample.T)])
    B=np.ones((6,4))
    np.random.seed(91)
    full=recover_full_graph_from_estimates(p,B,3,6,200)
    np.testing.assert_array_equal(full,stitch_full_A(expected["G"].graph,np.ones((3,6)),6))
    assert_rng_equal(np.random.get_state(),expected_state)


def test_explicit_seed_is_local_and_matches_legacy_seed():
    for law in (SparseLatentLaw(3,{7: .3,0: .2,5: .1},.4),
                np.array([.2,0.,.3,0.,.1,.05,0.,.35])):
        np.random.seed(501)
        old=learn_latent_graph(law,3,100)
        before=np.random.get_state()
        new=learn_latent_graph(law,3,100,seed=501)
        assert_rng_equal(np.random.get_state(),before)
        assert new["sample_counts"]==old["sample_counts"]
        np.testing.assert_array_equal(new["graph"],old["graph"])
        assert new["score"]==old["score"]


def test_plugin_consumes_no_rng_and_does_not_monkeypatch():
    law=SparseLatentLaw(3,{0: .2,7: .5},.3)
    original=GES.local_score_BDeu
    numpy_before,python_before=np.random.get_state(),random.getstate()
    def fit(ess):
        return learn_latent_graph(law,3,211,method="plugin",ess=ess)
    serial=[fit(ess) for ess in [.5,1.,8.]]
    with ThreadPoolExecutor(max_workers=3) as pool:
        parallel=list(pool.map(fit,[.5,1.,8.]))
    for left,right in zip(serial,parallel):
        np.testing.assert_array_equal(left["graph"],right["graph"])
        assert left["score"]==right["score"]
    assert GES.local_score_BDeu is original
    assert_rng_equal(np.random.get_state(),numpy_before)
    assert random.getstate()==python_before


@pytest.mark.parametrize("ess",[.2,1.,9.])
def test_custom_ess_is_used_in_resampling_and_plugin(ess):
    counts={0: 31,1: 5,2: 7,3: 23,4: 11,5: 37,6: 29,7: 2}
    p=SparseLatentLaw.from_counts(3,counts)
    for method in ("plugin","resampling"):
        result=learn_latent_graph(p,3,300,method=method,ess=ess,
                                    structure_prior=.7,max_parents=0)
        score_counts=counts if method=="plugin" else result["sample_counts"]
        sample=binary_sample(score_counts,3)
        params=dict(sample_prior=ess,structure_prior=.7,
                      r_i_map={i: len(np.unique(sample[:,i])) for i in range(3)})
        if method=="resampling":
            expected=sum(GES.local_score_BDeu(sample,node,[],params) for node in range(3))
        else:
            expected=sum(expected_count_local_bdeu(p,3,300,node,[],ess,.7) for node in range(3))
        assert not np.any(result["graph"])
        assert result["score"]==pytest.approx(expected,abs=1e-9)
        assert result["ess"]==ess
        assert result["structure_prior"]==.7


@pytest.mark.parametrize("K",[1,2])
@pytest.mark.parametrize("method",["plugin","resampling"])
def test_small_dimensions_have_finite_scores(K,method):
    result=learn_latent_graph(np.ones(2**K)/2**K,K,100,method=method)
    assert result["graph"].shape==(K,K)
    assert math.isfinite(result["score"])
    assert result["structure_prior"]==(0 if K==1 else .5)


@pytest.mark.parametrize("kwargs",[
    dict(ess=0),dict(ess=float("nan")),dict(structure_prior=0),
    dict(structure_prior=2),dict(method="unknown"),
    dict(method="plugin",seed=4),dict(max_parents=3),
])
def test_invalid_score_options_are_rejected(kwargs):
    with pytest.raises(ValueError):
        learn_latent_graph(np.ones(8)/8,3,100,**kwargs)


def test_empty_invalid_and_unnormalized_laws():
    for p in ({},{0:-1,1: 2},np.zeros(8),[1.,2.]):
        with pytest.raises(ValueError):
            learn_latent_graph(p,3,100,method="plugin")
    with pytest.raises(ValueError):
        learn_latent_graph(np.ones(8),3,100)
    a=learn_latent_graph(np.ones(8),3,100,method="plugin")
    b=learn_latent_graph(np.ones(8)/8,3,100,method="plugin")
    assert a["score"]==b["score"]
