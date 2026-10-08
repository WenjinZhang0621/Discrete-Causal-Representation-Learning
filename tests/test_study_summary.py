"""Independent synthetic checks of Monte Carlo units and failure denominators."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dcrl.study import expanded_settings
from dcrl.study_summary import summarize_study,metric_summary


def fixture(tmp_path,*,repetitions=3,draws=2,oracles=False):
    plan=dict(model=dict(N=100,K=3,distribution='Bernoulli'),seed_start=1,
                repetitions=repetitions,graph_all_starts=False,oracles=oracles,
                graph=dict(methods=['resampling'],ratios=[1.],ess=[1.],draws=draws))
    setting,spec=expanded_settings(plan)[0]
    (tmp_path/'manifest.json').write_text(json.dumps(dict(plan=plan,expected_replicates=repetitions)))
    def save(seed,graphs,**extra):
        folder=tmp_path/f'{setting}__s{seed}'
        folder.mkdir(exist_ok=True)
        row=dict(setting_id=setting,setting=spec,seed=seed,graphs=graphs,
                   status='ok',wall_seconds=1.,**extra)
        (folder/'result.json').write_text(json.dumps(row))
        return row
    return save,setting


def graph(draw,*,shd=0,status='ok',stage='fully_estimated'):
    result=dict(stage=stage,start_id=None,method='resampling',ratio=1.,ess=1.,
                  structure_prior=None,draw=draw,status=status)
    if status=='ok':
        metric=dict(cpdag_shd=shd,exact_cpdag=shd==0,
                      exact_skeleton=shd==0,exact_vstructures=shd==0)
        result.update(metrics=metric,matched_metrics=metric.copy(),full_shd=shd)
    return result


def selected_graph(summary,stage='fully_estimated'):
    return next(x for x in summary['graphs'] if x['stage']==stage)


def test_failures_and_missing_draws_never_disappear(tmp_path):
    save,_=fixture(tmp_path,oracles=True)
    save(1,[graph(0),graph(1,shd=2)],measurement=dict(q_error=0,exact_Q=True,exact_Q_raw=True),
         known_Q_measurement=dict(q_error=0,exact_Q=True,exact_Q_raw=True))
    save(2,[graph(0),graph(1,status='failed')])
    save(3,[])
    summary=summarize_study(tmp_path)
    full=selected_graph(summary)
    assert summary['is_complete']
    assert full['complete_valid_datasets']==1
    assert full['partial_valid_datasets']==1
    assert full['datasets_without_valid_draws']==1
    assert full['attempted_draws']==4
    assert full['valid_draws']==3
    assert full['failed_draws']==1
    assert full['missing_draws']==2
    assert full['unavailable_or_failed_draw_fraction']==.5
    assert full['metrics']['raw_latent.cpdag_shd']['mean']==1
    assert full['partial_draw_conditional_metrics']['raw_latent.cpdag_shd']['mean']==0
    # Per-dataset exact fractions are .5, .5, 0, including unavailable outputs.
    exact=full['exact_recovery_all_attempts']['raw_latent.exact_cpdag']
    assert exact['mean']==pytest.approx(1/3)
    assert exact['n']==3
    assert 'wilson' not in exact
    # All-failed / entirely absent oracle groups are retained.
    oracle=selected_graph(summary,'true_p')
    assert oracle['datasets']==0 and oracle['missing_draws']==6
    assert oracle['metrics']=={}
    assert oracle['exact_recovery_all_attempts']['raw_latent.exact_cpdag']['mean']==0
    setting=summary['settings'][0]
    assert setting['full_fit_failure_rate']==pytest.approx(2/3)
    assert setting['measurement']['exact_Q']['mean']==1
    assert setting['measurement_exact_recovery_all_attempts']['exact_Q']['mean']==pytest.approx(1/3)
    assert setting['known_Q_measurement']['q_error']['mean']==0


def test_draws_are_averaged_before_mc_uncertainty(tmp_path):
    save,_=fixture(tmp_path,repetitions=2)
    save(1,[graph(0,shd=0),graph(1,shd=2)])
    save(2,[graph(0,shd=4),graph(1,shd=6)])
    item=selected_graph(summarize_study(tmp_path))['metrics']['raw_latent.cpdag_shd']
    assert item['n']==2
    assert item['mean']==3
    assert item['mc_se']==2  # Dataset means 1,5, not four independent draws.
    assert item['ci'][0]<0<item['ci'][1]


def test_wilson_is_based_on_design_even_for_all_zero_or_one_means():
    outcomes=[{'exact_cpdag': 1.},{'exact_cpdag': 0.}]
    assert 'wilson' not in metric_summary(outcomes)['exact_cpdag']
    assert metric_summary(outcomes,binary_outcomes=True)['exact_cpdag']['wilson']['n']==2
    with pytest.raises(ValueError,match='declared binary'):
        metric_summary([{'exact_cpdag': .5}],binary_outcomes=True)


def test_partial_study_keeps_unattempted_plan_and_single_draw_wilson(tmp_path):
    save,_=fixture(tmp_path,repetitions=3,draws=1)
    save(1,[graph(0)])
    summary=summarize_study(tmp_path)
    assert not summary['is_complete']
    assert len(summary['missing_replicates'])==2
    group=selected_graph(summary)
    assert group['completed_datasets']==1
    assert group['missing_datasets']==2
    assert group['planned_draws']==3
    assert group['exact_recovery_all_attempts']['raw_latent.exact_cpdag']['wilson']['n']==1


def test_duplicate_or_unplanned_graph_draw_is_rejected(tmp_path):
    save,_=fixture(tmp_path)
    save(1,[graph(0),graph(0)])
    with pytest.raises(ValueError,match='Duplicate graph'):
        summarize_study(tmp_path)
    save(1,[graph(2)])
    with pytest.raises(ValueError,match='Unplanned graph'):
        summarize_study(tmp_path)


def test_graph_variability_uses_dataset_units(tmp_path):
    save,_=fixture(tmp_path,repetitions=2)
    def variation(draw,value):
        return dict(method='resampling',ratio=1.,ess=1.,structure_prior=None,draw=draw,
                    variability=dict(n_starts=3,summary=dict(cpdag_shd=dict(n_pairs=3,mean=value))))
    save(1,[graph(0),graph(1)],graph_variability=[variation(0,1),variation(1,3)])
    save(2,[graph(0),graph(1)],graph_variability=[variation(0,4),variation(1,8)])
    output=summarize_study(tmp_path)['settings'][0]['graph_variability'][0]
    metric=output['metrics']['summary.cpdag_shd.mean']
    assert metric['n']==2 and metric['mean']==4 and metric['mc_se']==2
