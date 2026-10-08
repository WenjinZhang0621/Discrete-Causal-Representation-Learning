"""Dataset-level Monte Carlo summaries with explicit unsuccessful attempts.

The primary graph means condition on all planned draws succeeding for a
dataset. Partial-draw means are separately labelled. Exact-recovery summaries
over all attempted datasets count failed or unavailable outputs as not exact;
this convention does not impute numerical SHD values to failures.
"""
from collections import Counter,defaultdict
import json
from pathlib import Path

import numpy as np

from .study import expanded_settings,graph_grid,start_specs,write_json
from .metrics import mc_summary,wilson_rate


def numeric_fields(record,prefix=''):
    result={}
    for name,value in record.items():
        key=prefix+name
        if isinstance(value,dict):
            result.update(numeric_fields(value,key+'.'))
        elif isinstance(value,(int,float,bool)) and np.isfinite(value):
            result[key]=float(value)
    return result


def dataset_aggregate(records):
    fields=defaultdict(list)
    for record in records:
        for key,value in numeric_fields(record).items():
            fields[key].append(value)
    return {key: float(np.mean(values)) for key,values in fields.items()}


def metric_summary(records,*,binary_outcomes=False):
    """Binary status is a design property, not inferred from realized values."""
    records=list(records)
    values=defaultdict(list)
    for record in records:
        for key,value in numeric_fields(record).items():
            values[key].append(value)
    result={key: mc_summary(x) for key,x in values.items()}
    for key,x in values.items():
        result[key]['eligible_datasets']=len(records)
        result[key]['undefined_or_missing_datasets']=len(records)-len(x)
        if binary_outcomes and 'exact' in key.lower():
            if not set(x).issubset({0.,1.}):
                raise ValueError(f'{key} was declared binary but contains fractions')
            result[key]['wilson']=wilson_rate(int(sum(x)),len(x))
    return result


def planned_graph_groups(plan):
    """Return group keys (without setting) and planned draws per dataset."""
    draws=plan.get('graph',{}).get('draws',1)
    if not isinstance(draws,int) or isinstance(draws,bool) or draws<1:
        raise ValueError('graph.draws must be a positive integer')
    stages=[('fully_estimated',None)]
    if plan.get('oracles',False):
        stages+=[('true_p',None),('known_Q',None)]
    if plan.get('graph_all_starts',True):
        stages+=[('fully_estimated_start',s['id'])
                   for s in start_specs(plan,int(plan.get('seed_start',1)))]
    result={}
    for method,ratio,ess,prior in graph_grid(plan):
        for stage,start_id in stages:
            key=(stage,start_id,method,ratio,ess,prior)
            if key in result:
                raise ValueError('Duplicate planned graph group')
            result[key]=draws if method=='resampling' else 1
        if plan.get('oracles',False):
            result[('true_latent_sample',None,'empirical',1.,ess,prior)]=1
    return result


def graph_key(graph):
    return (graph['stage'],graph.get('start_id'),graph['method'],
            graph['ratio'],graph['ess'],graph.get('structure_prior'))


def graph_payload(graph):
    result={'raw_latent': graph['metrics'],'aligned_latent': graph['matched_metrics']}
    for name in ['full_shd','matched_full_shd']:
        if name in graph:
            result[name]=graph[name]
    if 'elapsed_seconds' in graph.get('diagnostics',{}):
        result['elapsed_seconds']=graph['diagnostics']['elapsed_seconds']
    return result


def _exact_measurement_all_attempts(rows,field):
    result={}
    for metric in ['exact_Q','exact_Q_raw']:
        outcomes=[{metric: bool(row.get(field,{}).get(metric,False))} for row in rows]
        result.update(metric_summary(outcomes,binary_outcomes=True))
    return result


def _graph_summary(setting,key,expected_draws,rows,records):
    complete_means,partial_means,all_exact=[],[],[]
    counts=Counter()
    exact_names=[f'{kind}.{metric}' for kind in ['raw_latent','aligned_latent']
                   for metric in ['exact_cpdag','exact_skeleton','exact_vstructures','exact_directed_edges']]
    for row in rows:
        draws=records.get((setting,row['seed'],key),{})
        valid=[graph_payload(r) for r in draws.values() if r['status']=='ok']
        counts['attempted_draws']+=len(draws)
        counts['valid_draws']+=len(valid)
        counts['failed_draws']+=len(draws)-len(valid)
        counts['missing_draws']+=expected_draws-len(draws)
        counts['datasets_with_all_draw_records']+=len(draws)==expected_draws
        if len(valid)==expected_draws:
            counts['complete_valid_datasets']+=1
            complete_means.append(dataset_aggregate(valid))
        elif valid:
            counts['partial_valid_datasets']+=1
            partial_means.append(dataset_aggregate(valid))
        else:
            counts['datasets_without_valid_draws']+=1
        # Dataset weights are equal. Repeated-draw rates are clustered
        # proportions, not independent Bernoulli outcomes.
        flat=[numeric_fields(v) for v in valid]
        all_exact.append({name: sum(v.get(name,0.) for v in flat)/expected_draws
                          for name in exact_names})
    stage,start_id,method,ratio,ess,prior=key
    complete=counts['complete_valid_datasets']
    return dict(setting_id=setting,stage=stage,start_id=start_id,method=method,
                ratio=ratio,ess=ess,structure_prior=prior,
                planned_draws_per_dataset=expected_draws,
                completed_datasets=len(rows),datasets=complete,
                complete_valid_datasets=complete,
                partial_valid_datasets=counts['partial_valid_datasets'],
                datasets_without_valid_draws=counts['datasets_without_valid_draws'],
                datasets_with_all_draw_records=counts['datasets_with_all_draw_records'],
                expected_draws_for_completed_datasets=len(rows)*expected_draws,
                attempted_draws=counts['attempted_draws'],valid_draws=counts['valid_draws'],
                failed_draws=counts['failed_draws'],missing_draws=counts['missing_draws'],
                graph_draws=counts['valid_draws'],
                unavailable_or_failed_draw_fraction=(
                    (counts['failed_draws']+counts['missing_draws'])/(len(rows)*expected_draws)
                    if rows else None),
                metrics=metric_summary(complete_means,binary_outcomes=expected_draws==1),
                metrics_conditioning='All planned graph draws succeeded for the dataset.',
                partial_draw_conditional_metrics=metric_summary(partial_means),
                partial_draw_conditioning='Only datasets with some but not all planned draws successful; average successful draws only.',
                exact_recovery_all_attempts=metric_summary(all_exact,binary_outcomes=expected_draws==1),
                exact_recovery_convention='Failed or unavailable draws count as not exact; average planned draws within each completed dataset.')


def _variability_summaries(rows):
    grouped=defaultdict(lambda: defaultdict(list))
    for row in rows:
        for record in row.get('graph_variability',[]):
            key=(record['method'],record['ratio'],record['ess'],record.get('structure_prior'))
            grouped[key][row['seed']].append(record['variability'])
    result=[]
    for key,per_dataset in sorted(grouped.items(),key=lambda item: repr(item[0])):
        method,ratio,ess,prior=key
        result.append(dict(method=method,ratio=ratio,ess=ess,structure_prior=prior,
                           datasets=len(per_dataset),completed_datasets=len(rows),
                           metrics=metric_summary([dataset_aggregate(values) for values in per_dataset.values()]),
                           conditioning='Available valid starts/graphs; repeated draws averaged within dataset. Pairwise start comparisons are not independent replications.'))
    return result


def summarize_study(output):
    output=Path(output)
    manifest=json.loads((output/'manifest.json').read_text())
    plan=manifest['plan']
    settings_plan=dict(expanded_settings(plan))
    repetitions,seed_start=int(plan.get('repetitions',1)),int(plan.get('seed_start',1))
    planned_keys={(key,seed) for key in settings_plan
                    for seed in range(seed_start,seed_start+repetitions)}
    if len(planned_keys)!=manifest['expected_replicates']:
        raise ValueError('Manifest expected_replicates disagrees with its plan')
    groups=planned_graph_groups(plan)
    rows=[json.loads(p.read_text()) for p in sorted(output.glob('*/result.json'))]
    seen,by_setting,records,graph_failures=set(),defaultdict(list),defaultdict(dict),[]
    for row in rows:
        identity=(row['setting_id'],row['seed'])
        if identity in seen or identity not in planned_keys:
            raise ValueError('Duplicate or unplanned setting/seed result')
        if row['setting']!=settings_plan[row['setting_id']]:
            raise ValueError('Result setting does not match frozen plan')
        seen.add(identity)
        by_setting[row['setting_id']].append(row)
        for graph in row.get('graphs',[]):
            key,draw=graph_key(graph),graph['draw']
            if key not in groups or not isinstance(draw,int) or not 0<=draw<groups[key]:
                raise ValueError('Unplanned graph group/draw in results')
            saved=records[(row['setting_id'],row['seed'],key)]
            if draw in saved:
                raise ValueError('Duplicate graph draw in one dataset')
            saved[draw]=graph
            if graph['status']!='ok':
                graph_failures.append(dict(setting=row['setting_id'],seed=row['seed'],**graph))
    settings,graphs=[],[]
    for key,spec in sorted(settings_plan.items()):
        group=by_setting[key]
        measurements=[r['measurement'] for r in group if 'measurement' in r]
        known_q=[r['known_Q_measurement'] for r in group if 'known_Q_measurement' in r]
        starts=[s for r in group for s in r.get('fit',{}).get('starts',[])]
        settings.append(dict(setting_id=key,setting=spec,completed=len(group),planned=repetitions,
                             missing_datasets=repetitions-len(group),
                             statuses=dict(Counter(r['status'] for r in group)),
                             valid_full_fits=len(measurements),failed_or_unavailable_full_fits=len(group)-len(measurements),
                             full_fit_failure_rate=(len(group)-len(measurements))/len(group) if group else None,
                             measurement=metric_summary(measurements,binary_outcomes=True),
                             measurement_conditioning='Selected fit and measurement metrics are available.',
                             measurement_exact_recovery_all_attempts=_exact_measurement_all_attempts(group,'measurement'),
                             valid_known_Q_fits=len(known_q),
                             known_Q_measurement=metric_summary(known_q,binary_outcomes=True),
                             known_Q_measurement_conditioning='Known-Q selected fit and measurement metrics are available.',
                             known_Q_exact_recovery_all_attempts=(_exact_measurement_all_attempts(group,'known_Q_measurement')
                                                                if plan.get('oracles',False) else {}),
                             wall_seconds=mc_summary([r['wall_seconds'] for r in group]),
                             start_statuses=dict(Counter(s.get('status','unknown') for s in starts)),
                             fit_variability=metric_summary([r['fit_variability'] for r in group if 'fit_variability' in r]),
                             graph_variability=_variability_summaries(group)))
        for graph_group,expected_draws in sorted(groups.items(),key=lambda pair: repr(pair[0])):
            result=_graph_summary(key,graph_group,expected_draws,group,records)
            result.update(planned_datasets=repetitions,planned_draws=repetitions*expected_draws,
                          missing_datasets=repetitions-len(group))
            graphs.append(result)
    failures=[dict(setting_id=r['setting_id'],seed=r['seed'],status=r['status'],
                     error=r.get('error'),failure_phase=r.get('failure_phase'))
                for r in rows if r['status']!='ok']
    summary=dict(completed=len(rows),planned=len(planned_keys),is_complete=seen==planned_keys,
                   missing_replicates=[dict(setting_id=key,seed=seed) for key,seed in sorted(planned_keys-seen)],
                   settings=settings,graphs=graphs,failures=failures,graph_failures=graph_failures,
                   inference_unit='One independently generated observed dataset; repeated graph draws are averaged within dataset before MC uncertainty.',
                   conditioning='Primary graph means use datasets with every planned draw valid. All-attempt exact recovery treats missing/failed outputs as not exact; no SHD is imputed.',
                   selection='Starts selected using observed penalized likelihood; oracle truth is used only for labeled diagnostics.',
                   warning='Finite fits and a stopped SAEM iteration do not certify a global penalized maximum. Confidence intervals are pointwise Monte Carlo summaries, not simultaneous inference.')
    write_json(output/'summary.json',summary)
    lines=['DCRL diagnostic study',f"Completed datasets: {len(rows)}/{len(planned_keys)}",
             f'Dataset failures/partial results: {len(failures)}; recorded graph failures: {len(graph_failures)}',
             summary['inference_unit'],summary['conditioning'],summary['warning'],'',
             'Setting | stage | start | graph method | m/N | ESS | complete datasets / completed | valid / expected draws | raw latent CPDAG SHD | aligned latent CPDAG SHD | full raw SHD']
    def mean(item,name):
        value=item['metrics'].get(name,{}).get('mean')
        return 'NA' if value is None else f'{value:.4f}'
    for group in graphs:
        lines.append(' | '.join(map(str,[group['setting_id'],group['stage'],group['start_id'],group['method'],
                     group['ratio'],group['ess'],f"{group['datasets']}/{group['completed_datasets']}",
                     f"{group['valid_draws']}/{group['expected_draws_for_completed_datasets']}",
                     mean(group,'raw_latent.cpdag_shd'),mean(group,'aligned_latent.cpdag_shd'),mean(group,'full_shd')])))
    lines+=['','Failure denominators, uncertainty, all-attempt exact recovery, conditional partial-draw summaries, known-Q metrics and variability are in summary.json.',
              'All individual objectives, stopping diagnostics, optimizer statuses and initial/final estimates remain in each dataset directory.']
    (output/'summary.txt').write_text('\n'.join(lines)+'\n')
    return summary
