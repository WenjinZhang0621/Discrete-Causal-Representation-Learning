"""Oracle CPT truth, exact graph structure, and unchanged generation checks."""
import random
import numpy as np
import pytest

from dcrl.config import ExperimentConfig
from dcrl.data_generator import GenerateData
from dcrl.metrics import vstructures
from dcrl.oracle import (empirical_law,exact_cpdag,generate_latent_with_truth,
                        latent_dag,true_model)


def data_object(kind,K,N=400):
    return GenerateData.from_config(ExperimentConfig(N=N,K=K,DAG_type=kind,distribution="Bernoulli").finalize())


@pytest.mark.parametrize("kind,K,edge_count",[("Tree",5,4),("Markov",5,4),("DiverseTree",10,9),
                                             ("Model-7",7,8),("Model-8",8,8),("Model-13",13,18),("Model-16",16,22)])
def test_capture_preserves_actual_samples_and_global_rng(kind,K,edge_count):
    captured,reference=data_object(kind,K),data_object(kind,K)
    random.seed(812); np.random.seed(917)
    generate_latent_with_truth(captured)
    captured.generate_data()
    after=(random.random(),np.random.random())
    random.seed(812); np.random.seed(917)
    reference.generate_latent_data()
    reference.generate_data()
    expected_after=(random.random(),np.random.random())
    np.testing.assert_array_equal(captured.A,reference.A)
    np.testing.assert_array_equal(captured.X,reference.X)
    assert after==expected_after
    truth=true_model(captured)
    assert truth["dag_true"].sum()==edge_count
    assert truth["p_true"].shape==(1<<K,)
    assert truth["p_true"].sum()==pytest.approx(1,abs=2e-14)


def test_known_cpdag_and_model13_colliders():
    chain=np.array([[0,1,0],[0,0,1],[0,0,0]])
    np.testing.assert_array_equal(exact_cpdag(chain),np.array([[0,-1,0],[-1,0,-1],[0,-1,0]]))
    collider=np.array([[0,1,0],[0,0,0],[0,1,0]])
    assert vstructures(exact_cpdag(collider))=={(0,1,2)}
    graph=exact_cpdag(latent_dag("Model-13",13))
    assert np.sum(graph==-1)==24  # 6 undirected + 12 directed tails.
    assert np.sum(graph==1)==12
    assert vstructures(graph)=={(3,k,4) for k in (7,8,9)}|{(5,k,6) for k in (10,11,12)}
    with pytest.raises(ValueError,match="cycle"):
        exact_cpdag(np.array([[0,1],[1,0]]))


@pytest.mark.parametrize("kind,K",[("Tree",4),("Markov",4),("DiverseTree",4),("Model-7",7),
                                   ("Model-8",8),("Model-13",13),("Model-16",16)])
def test_exact_cpts_agree_with_actual_small_monte_carlo_conditionals(kind,K):
    data=data_object(kind,K,N=18000)
    random.seed(246); np.random.seed(813)
    generate_latent_with_truth(data)
    truth=true_model(data)
    A=data.A.astype(int)
    for child,parents in enumerate(truth["parent_sets"]):
        indices=A[:,parents]@(1<<np.arange(len(parents)-1,-1,-1)) if parents else np.zeros(len(A),dtype=int)
        for parent_state,chance in enumerate(truth["cpts"][child]):
            selected=A[indices==parent_state,child]
            assert len(selected)>300
            assert abs(selected.mean()-chance)<.045


def test_known_cpt_formula_population_truth_is_not_empirical():
    data=data_object("Tree",2,N=5)
    random.seed(172); np.random.seed(817)
    generate_latent_with_truth(data)
    truth=true_model(data)
    p0,p1=truth["cpts"][1]
    expected=[.5*(1-p0),.5*p0,.5*(1-p1),.5*p1]
    np.testing.assert_allclose(truth["p_true"],expected,rtol=0,atol=1e-15)
    empirical=empirical_law(data.A).to_dense().ravel()
    assert not np.allclose(empirical,truth["p_true"])
    assert np.array_equal(empirical*5,np.round(empirical*5))


def test_tree_non_alternating_probabilities_are_captured():
    data=data_object("Tree",7,N=15000)
    random.seed(718); np.random.seed(893)
    generate_latent_with_truth(data,alternate_prob=False)
    truth=true_model(data)
    for child in range(1,7):
        parent=(child-1)//2
        p0,p1=truth["cpts"][child]
        assert .60<=p1<=.65 and .35<=p0<=.40
        assert abs(data.A[data.A[:,parent]==1,child].mean()-p1)<.03


def test_oracle_refuses_uncaptured_stale_or_oversized_truth():
    data=data_object("Tree",3,N=10)
    data.generate_latent_data()
    with pytest.raises(ValueError,match="not captured"):
        true_model(data)
    generate_latent_with_truth(data)
    with pytest.raises(ValueError,match="exceeds max_k"):
        true_model(data,max_k=2)
    data.generate_latent_data()
    with pytest.raises(ValueError,match="replaced"):
        true_model(data)
