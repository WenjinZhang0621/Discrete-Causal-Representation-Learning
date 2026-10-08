"""Independent small-graph and latent coordinate transformation checks."""
import numpy as np
import pytest

from dcrl.latent_law import SparseLatentLaw
from dcrl.metrics import (align_latent_graph,align_latent_law,fit_variability,
                         graph_metrics,latent_law_distance,mc_summary,
                         measurement_metrics,latent_sample_alignment,
                         vstructures,wilson_rate)


def test_support_metrics_matching_and_rate_denominators():
    Q=np.array([[1,0],[0,1],[1,1]])
    B=np.column_stack((np.zeros(3),Q[:,::-1]))
    result=measurement_metrics(B,Q)
    assert result["q_error"]==0 and result["q_error_raw"]==4
    assert result["permutation"]==[1,0] and result["exact_Q"]
    B[2,0+1]=0
    result=measurement_metrics(B,Q)
    assert result["fn"]==1 and result["fp"]==0
    assert result["fnr"]==.25 and result["fpr"]==0


def test_supplied_measurement_alignment_is_not_optimized_again():
    Q=np.array([[1,0],[0,1],[1,1]])
    B=np.column_stack((np.zeros(3),Q[:,::-1]))
    default=measurement_metrics(B,Q)
    fixed=measurement_metrics(B,Q,permutation=[0,1])
    assert default['q_error']==0 and default['alignment_criterion']=='Q_support_Hungarian'
    assert fixed['q_error']==fixed['q_error_raw']==4
    assert fixed['permutation']==[0,1] and fixed['alignment_criterion']=='supplied_permutation'
    for permutation in [[0,0],[0],[0,2]]:
        with pytest.raises(ValueError,match='permutation'):
            measurement_metrics(B,Q,permutation=permutation)


def test_latent_sample_alignment_recovers_labels_and_complements():
    truth=np.array([[int(bit) for bit in format(s,'03b')] for s in range(8)])
    estimated=truth[:,[2,0,1]]^[1,0,1]
    result=latent_sample_alignment(estimated,truth)
    assert result['permutation']==[1,2,0]
    assert result['bit_flips']==[False,True,True]
    assert result['total_errors']==0 and result['joint_accuracy']==1
    assert result['mean_error']==0 and not result['has_ties']
    estimated[0,0]^=1
    noisy=latent_sample_alignment(estimated,truth)
    assert noisy['per_column_errors']==[0,0,1]
    assert noisy['total_errors']==1 and noisy['mean_error']==pytest.approx(1/24)
    assert noisy['joint_accuracy']==7/8


def test_latent_sample_alignment_matches_exhaustive_small_assignment():
    from itertools import permutations,product
    rng=np.random.RandomState(994)
    truth,estimated=rng.randint(0,2,size=(2,7,3))
    best=min(np.count_nonzero((estimated[:,perm]^flips)!=truth)
               for perm in permutations(range(3)) for flips in product([0,1],repeat=3))
    result=latent_sample_alignment(estimated,truth)
    assert result['total_errors']==best
    actual=estimated[:,result['permutation']]^result['bit_flips']
    assert np.count_nonzero(actual!=truth)==best
    fixed=latent_sample_alignment(estimated,truth,fixed_labels=True)
    fixed_best=min(np.count_nonzero((estimated^flips)!=truth)
                     for flips in product([0,1],repeat=3))
    assert fixed['permutation']==[0,1,2] and fixed['total_errors']==fixed_best
    assert fixed['assignment_ties']==[False,False,False]


def test_latent_sample_ties_are_explicit():
    duplicate=np.array([[0,0],[1,1],[0,0],[1,1]])
    result=latent_sample_alignment(duplicate,duplicate)
    assert result['assignment_ties']==[True,True]
    assert result['level_ties']==[False,False] and result['has_ties']
    result=latent_sample_alignment([[0],[1],[0],[1]],[[0],[0],[1],[1]])
    assert result['bit_flips']==[False]  # Prefer no complement on a tie.
    assert result['assignment_ties']==[False] and result['level_ties']==[True]
    assert result['total_errors']==2 and result['joint_accuracy']==.5


@pytest.mark.parametrize('estimated,truth',[
    ([],[]),([0,1],[0,1]),(np.empty((0,2)),np.empty((0,2))),
    (np.empty((2,0)),np.empty((2,0))),([[0]],[[0,1]]),
    ([[np.nan]],[[0]]),([[0]],[[2]]),([[.5]],[[1]])])
def test_latent_sample_alignment_rejects_invalid_samples(estimated,truth):
    with pytest.raises(ValueError,match='nonempty binary'):
        latent_sample_alignment(estimated,truth)


def test_known_chain_and_collider_metrics():
    chain=np.array([[0,-1,0],[-1,0,-1],[0,-1,0]])
    collider=np.array([[0,-1,0],[1,0,1],[0,-1,0]])
    result=graph_metrics(collider,chain)
    assert result["cpdag_shd"]==2 and result["skeleton_shd"]==0
    assert result["vstructure_fp"]==1 and result["vstructure_fn"]==0
    assert vstructures(collider)=={(0,1,2)}
    assert result["vstructure_fnr"] is None  # No true colliders.
    assert graph_metrics(collider,collider)["exact_cpdag"]
    missing=collider.copy()
    missing[1,2]=missing[2,1]=0
    result=graph_metrics(missing,collider)
    assert result["skeleton_fn"]==1 and result["skeleton_fnr"]==.5
    assert result["vstructure_fn"]==1
    assert result['skeleton_precision']==1 and result['skeleton_recall']==.5
    assert result['skeleton_f1']==pytest.approx(2/3)
    assert result['vstructure_f1']==0
    assert result['directed_tp']==1 and result['directed_fn']==1
    assert result['directed_precision']==1 and result['directed_recall']==.5
    assert result['directed_f1']==pytest.approx(2/3)
    with pytest.raises(ValueError,match="no stitched"):
        graph_metrics(np.array([[0,1],[0,0]]),np.zeros((2,2)))


def test_empty_graph_precision_recall_and_f1_conventions():
    empty=np.zeros((3,3),dtype=int)
    result=graph_metrics(empty,empty)
    for kind in ['skeleton','vstructure','directed']:
        assert result[f'{kind}_precision'] is None
        assert result[f'{kind}_recall'] is None
        assert result[f'{kind}_f1']==1
    assert result['exact_directed_edges']
    assert result['directed_tp']==result['directed_fp']==result['directed_fn']==0
    assert result['true_directed_edges']==result['estimated_directed_edges']==0


def test_reversed_arrow_is_one_cpdag_change_and_two_set_errors():
    truth=np.array([[0,-1],[1,0]])
    result=graph_metrics(truth.T,truth)
    assert result['cpdag_shd']==1 and result['skeleton_shd']==0
    assert result['directed_fp']==result['directed_fn']==1
    assert result['directed_tp']==result['directed_f1']==0
    assert result['directed_precision']==result['directed_recall']==0
    assert not result['exact_directed_edges']
    # Undirected skeleton edges are not counted as compelled arrows.
    undirected=np.array([[0,-1],[-1,0]])
    result=graph_metrics(undirected,truth)
    assert result['directed_fp']==0 and result['directed_fn']==1
    assert result['directed_precision'] is None and result['directed_recall']==0


def test_permutation_and_flip_probability_mass_and_predictor_invariance():
    p=np.arange(1,9,dtype=float)/36
    permutation,flips=[2,0,1],[1,0,1]
    transformed=align_latent_law(p,permutation,flips)
    expected=np.zeros(8)
    for state in range(8):
        bits=np.array([int(x) for x in format(state,"03b")])
        newbits=bits[permutation]^flips
        destination=int("".join(map(str,newbits)),2)
        expected[destination]+=p[state]
    np.testing.assert_array_equal(transformed,expected)
    law=SparseLatentLaw(3,{0:.2,5:.3},.5)
    sparse=align_latent_law(law,permutation,flips)
    np.testing.assert_allclose(sparse.to_dense().ravel(),align_latent_law(law.to_dense(),permutation,flips))
    assert sparse.uniform_mass==.5
    assert latent_law_distance(sparse,transformed,3)["tv"]==pytest.approx(.5*np.abs(sparse.to_dense().ravel()-transformed).sum())


def test_graph_alignment_permutates_both_axes():
    graph=np.array([[0,-1,0],[1,0,1],[0,-1,0]])
    perm=[1,2,0]
    aligned=align_latent_graph(graph,perm)
    assert vstructures(aligned)=={(1,0,2)}
    np.testing.assert_array_equal(aligned,graph[np.ix_(perm,perm)])


def test_truth_free_variability_recovers_permuted_flipped_equivalent_fit():
    B=np.array([[-1,2,0],[-.5,0,3],[.3,1,1]])
    p=np.array([.1,.2,.3,.4])
    graph=np.array([[0,-1],[-1,0]])
    # new latent coordinates w = (1-z1, z0)
    alternate=np.column_stack((B[:,0]+B[:,2],-B[:,2],B[:,1]))
    alternative_p=align_latent_law(p,[1,0],[1,0])
    out=fit_variability([dict(B=B,p=p,graph=graph),dict(B=alternate,p=alternative_p,graph=graph)])
    pair=out["pairwise"][0]
    assert pair["q_disagreement"]==pair["cpdag_shd"]==0
    assert pair["coefficient_frobenius"]==pytest.approx(0,abs=1e-14)
    assert pair["p_tv"]==pytest.approx(0)
    assert pair["p_tv_raw"]>0
    assert out["alignments"][1]==dict(permutation=[1,0],bit_flips=[0,1])


def test_mc_uncertainty_and_exact_rate_boundary_cases():
    result=mc_summary([1,2,3,4])
    assert result["mean"]==2.5
    assert result["mc_se"]==pytest.approx(np.std([1,2,3,4],ddof=1)/2)
    assert result["ci"][0]<2.5<result["ci"][1]
    assert mc_summary([1])["ci"] is None
    assert wilson_rate(0,20)["ci"][1]>.15
    assert wilson_rate(20,20)["ci"][0]<.85
    with pytest.raises(ValueError,match="finite"):
        mc_summary([1,float("nan")])


def test_sparse_latent_alignment_does_not_enumerate_large_states(monkeypatch):
    law=SparseLatentLaw(80,{1<<79:.7},.3)
    monkeypatch.setattr(SparseLatentLaw,"to_dense",lambda self: pytest.fail("dense export forbidden"))
    aligned=align_latent_law(law,list(reversed(range(80))))
    assert aligned.masses=={1:.7}
    assert latent_law_distance(aligned,law,80)["tv"]==pytest.approx(.7)
