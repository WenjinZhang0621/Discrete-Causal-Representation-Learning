from types import SimpleNamespace

import numpy as np
import pytest

from dcrl.study import run_replicate,fit_metrics,expanded_settings
from dcrl.latent_law import SparseLatentLaw


def test_oracle_graphs_survive_all_fit_failures(tmp_path,monkeypatch):
    import dcrl.fitting
    def fail(*args,**kwargs):
        raise ValueError('deliberate fitting setup failure')
    monkeypatch.setattr(dcrl.fitting,'fit_observed',fail)
    plan=dict(model=dict(N=80,K=3,DAG_type='Tree',distribution='Bernoulli'),
                starts=[dict(kind='spectral'),dict(kind='random')],oracles=True,
                graph_all_starts=True,graph=dict(methods=['plugin'],ratios=[1.],ess=[1.]))
    key,spec=expanded_settings(plan)[0]
    result=run_replicate(plan,key,spec,19,tmp_path)
    assert result['status']=='partial_fit_failure'
    graphs={g['stage']:g for g in result['graphs']}
    assert graphs['true_p']['status']=='ok'
    assert graphs['true_latent_sample']['status']=='ok'
    assert graphs['fully_estimated']['status']=='failed'
    assert graphs['known_Q']['status']=='failed'
    assert len(result['fit']['starts'])==2
    assert all(s['status']=='not_attempted' for s in result['fit']['starts'])
    assert len([g for g in result['graphs'] if g['stage']=='fully_estimated_start'])==2


def test_known_q_probability_error_retains_known_labels(monkeypatch):
    import dcrl.metrics
    monkeypatch.setattr(dcrl.metrics,'measurement_metrics',lambda *args,**kw:dict(permutation=[1,0]))
    B=np.array([[0.,1.,1.],[0.,2.,2.]])
    data=SimpleNamespace(K=2,Q=np.ones((2,2)),B=B)
    p=np.array([.1,.2,.3,.4])
    fit=dict(B=B,p=SparseLatentLaw.from_counts(2,dict(enumerate(p))))
    fixed=fit_metrics(fit,data,p,fixed_labels=True)
    flexible=fit_metrics(fit,data,p)
    assert fixed['latent_p_l1']==pytest.approx(0)
    assert fixed['latent_p_permutation']==[0,1]
    assert flexible['latent_p_l1']==pytest.approx(.2)


def test_duplicate_grid_settings_rejected():
    with pytest.raises(ValueError,match='duplicate'):
        expanded_settings(dict(model={'N':100,'K':3},grid={'epsilon':[.001,.001]}))


def test_graph_artifact_failures_are_not_counted_as_success(tmp_path,monkeypatch):
    import dcrl.graph_scoring
    from dcrl.study import graph_experiments
    graph=np.zeros((2,2),dtype=int)
    monkeypatch.setattr(dcrl.graph_scoring,'learn_latent_graph',lambda *a,**kw:dict(graph=graph))
    monkeypatch.setattr(dcrl.graph_scoring,'learn_graph_from_samples',lambda *a,**kw:dict(graph=graph))
    def fail_save(*args,**kwargs):
        raise OSError('deliberate disk write failure')
    monkeypatch.setattr(np,'savez_compressed',fail_save)
    B=np.array([[0.,1.,0.],[0.,0.,1.]])
    data=SimpleNamespace(N=4,J=2,K=2,B=B,Q=np.eye(2),A=np.array([[0,0],[0,1],[1,0],[1,1]]))
    p=np.ones(4)/4
    fits=dict(full=dict(B=B,p=p),known_Q=dict(B=B,p=p))
    plan=dict(oracles=True,graph_all_starts=False,graph=dict(methods=['plugin']))
    records=graph_experiments(plan,tmp_path/'graphs',data,dict(p_true=p,cpdag_true=graph),fits,1)
    assert len(records)==4
    assert all(r['status']=='failed' and 'disk write' in r['error'] for r in records)


def test_classification_alignment_is_shared_by_q_p_and_graph(tmp_path,monkeypatch):
    import dcrl.graph_scoring
    from dcrl.study import graph_experiments
    from dcrl.metrics import align_latent_law
    A=np.array([[0,0],[0,1],[1,0],[1,1]])
    B=np.array([[0.,1.,0.],[0.,1.,1.]])
    data=SimpleNamespace(N=4,J=2,K=2,A=A,B=B,Q=(B[:,1:]!=0).astype(int))
    p=np.array([.1,.2,.3,.4])
    B_est=np.column_stack((B[:,0]+B[:,2],-B[:,2],B[:,1]))
    fit=dict(B=B_est,p=align_latent_law(p,[1,0],[1,0]),
               final_sample_A=A[:,[1,0]]^np.array([1,0]))
    report=fit_metrics(fit,data,p,alignment='latent_sample')
    assert report['permutation']==[1,0]
    assert report['latent_level_flips']==[False,True]
    assert report['q_error']==0 and report['q_error_raw']==2
    assert report['latent_p_l1']==pytest.approx(0)
    assert report['evaluation_alignment']['joint_accuracy']==1
    truth=np.array([[0,-1],[1,0]])
    raw=truth[np.ix_([1,0],[1,0])]
    monkeypatch.setattr(dcrl.graph_scoring,'learn_latent_graph',lambda *a,**kw:dict(graph=raw))
    plan=dict(evaluation_alignment='latent_sample',graph_all_starts=False,graph=dict(methods=['plugin']))
    records=graph_experiments(plan,tmp_path/'graphs',data,dict(cpdag_true=truth),dict(full=fit),1)
    assert records[0]['status']=='ok'
    assert records[0]['metrics']['cpdag_shd']==1
    assert records[0]['matched_metrics']['cpdag_shd']==0
    assert records[0]['matched_full_shd']==0
    with np.load(tmp_path/'graphs'/records[0]['artifact']) as artifact:
        np.testing.assert_array_equal(artifact['permutation'],[1,0])


def test_custom_q_study_generation_above_default_item_count():
    from dcrl.study import make_config,generate_observations
    q=np.vstack([np.eye(2)]*4).astype(int)
    cfg,extras=make_config(dict(N=20,J=8,K=2,Q_matrix=q.tolist(),distribution='Bernoulli'))
    data=generate_observations(cfg,extras,7)
    np.testing.assert_array_equal(data.Q,q)
    assert data.X.shape==(20,8)
