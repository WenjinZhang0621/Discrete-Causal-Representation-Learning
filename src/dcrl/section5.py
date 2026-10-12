"""Frozen Section 5 plans and their thirty sample-size comparisons."""
from copy import deepcopy
import csv
import json
import math
from pathlib import Path

from .study import expanded_settings, source_hashes, versions, write_json


def task_plans(config):
    """Expand in the original HPC task order; override only named groups."""
    if config['schema_version'] != 1:
        raise ValueError('Unsupported Section 5 config schema')
    overrides = {}
    for item in config['overrides']:
        key = (item['graph'], item['Q_type'], item['distribution'])
        if key in overrides:
            raise ValueError('Duplicate Section 5 override')
        overrides[key] = item
    tasks, used = [], set()
    for graph in config['graphs']:
        for q in config['Q_types']:
            for family in config['distributions']:
                key = (graph['label'], q, family['name'])
                for n in config['sample_sizes']:
                    tid = len(tasks)
                    plan = deepcopy(config['base_plan'])
                    plan['model'].update(N=n, K=graph['K'], J=3*graph['K'],
                        DAG_type=graph['DAG_type'], Q_type=q,
                        distribution=family['name'], epsilon=family['epsilon'])
                    plan['seed_start'] += tid*config['seed_stride']
                    if key in overrides:
                        used.add(key)
                        rounds = overrides[key]['rounds']
                        plan['model'].update(config['validation_recipe'],
                            max_iter=rounds, min_outer_iter=rounds)
                        plan['graph']['structure_prior'] = [0.01]
                        plan['fitting_backend'] = 'section5_validated'
                        plan['fit_graph_seed_offset'] = n
                        plan['seed_start'] = config['validation_seed_base']+tid*config['seed_stride']
                    setting_id, _ = expanded_settings(plan)[0]
                    tasks.append(dict(task_id=tid, graph_label=graph['label'],
                                      setting_id=setting_id, plan=plan))
    if used != set(overrides):
        raise ValueError('Section 5 override does not match any group')
    identities = [(t['graph_label'], t['plan']['model']['Q_type'],
                   t['plan']['model']['distribution'], t['plan']['model']['N']) for t in tasks]
    if len(set(identities)) != len(tasks):
        raise ValueError('Duplicate Section 5 setting')
    return tasks


def export_plans(tasks, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        path = folder/f"task_{task['task_id']:03d}.json"
        if path.exists():
            if json.loads(path.read_text()) != task['plan']:
                raise ValueError(f'Existing plan differs: {path}')
        else:
            write_json(path, task['plan'])


def summarize(tasks, output):
    """Read only small summaries, checking plans and provenance before combining."""
    output = Path(output)
    rows, groups = [], {}
    hashes, runtime = source_hashes(), versions()
    exact_fields = {'Q_exact': ('measurement_exact_recovery_all_attempts', 'exact_Q'),
                    'composite_exact': ('exact_recovery_all_attempts', 'exact_composite'),
                    'CPDAG_exact': ('exact_recovery_all_attempts', 'aligned_latent.exact_cpdag'),
                    'skeleton_exact': ('exact_recovery_all_attempts', 'aligned_latent.exact_skeleton'),
                    'vstructures_exact': ('exact_recovery_all_attempts', 'aligned_latent.exact_vstructures'),
                    'directed_edges_exact': ('exact_recovery_all_attempts', 'aligned_latent.exact_directed_edges')}
    for task in tasks:
        plan, tid = task['plan'], task['task_id']
        cfg = plan['model']
        row = dict(task_id=tid, model=task['graph_label'], Q=cfg['Q_type'],
                   distribution=cfg['distribution'], N=cfg['N'],
                   backend=plan.get('fitting_backend', 'standard'), max_iter=cfg['max_iter'],
                   planned=plan['repetitions'], recorded=0, valid=0, failures=0,
                   full_SHD=None, full_SHD_SE=None, latent_SHD=None, complete=False)
        row.update({name: None for name in exact_fields})
        folder = output/f'task_{tid:03d}'
        if (folder/'summary.json').exists():
            meta = json.loads((folder/'manifest.json').read_text())
            if meta['plan'] != plan or meta['source_hashes'] != hashes or meta['versions'] != runtime:
                raise ValueError(f'Plan, source or environment differs: {folder}')
            summary = json.loads((folder/'summary.json').read_text())
            if summary['planned'] != row['planned'] or len(summary['settings']) != 1:
                raise ValueError(f'Unexpected summary design: {folder}')
            setting = summary['settings'][0]
            graphs = [g for g in summary['graphs'] if g['stage'] == 'fully_estimated'
                      and g['setting_id'] == task['setting_id']]
            if setting['setting_id'] != task['setting_id'] or len(graphs) != 1:
                raise ValueError(f'Unexpected graph summary: {folder}')
            graph = graphs[0]
            row.update(recorded=summary['completed'], valid=graph['complete_valid_datasets'],
                       failures=len(summary['failures']))
            full = graph['metrics'].get('matched_full_shd', {})
            row.update(full_SHD=full.get('mean'), full_SHD_SE=full.get('se'),
                       latent_SHD=graph['metrics'].get('aligned_latent.cpdag_shd', {}).get('mean'))
            for name, (section, metric) in exact_fields.items():
                source = setting if name == 'Q_exact' else graph
                item = source.get(section, {}).get(metric, {})
                # All planned attempts form the denominator, including missing ones.
                if item.get('mean') is not None:
                    row[name] = item['mean']*item['n']/row['planned']
            row['complete'] = (summary['is_complete'] and row['valid'] == row['planned']
                               and row['failures'] == 0 and row['full_SHD'] is not None)
        rows.append(row)
        groups.setdefault((row['model'], row['Q'], row['distribution']), []).append(row)
    lines = ['Section 5: matched_full_shd; independent data at each N',
             'SHD means use valid datasets. Exact recovery uses all planned attempts.',
             'Trend labels describe observed means; they are not significance tests.', '']
    comparisons = []
    for key, items in groups.items():
        items.sort(key=lambda x: x['N'])
        complete = len(items) == 3 and all(x['complete'] for x in items)
        means = [x['full_SHD'] for x in items]
        strict = all(b < a for a, b in zip(means, means[1:])) if complete else None
        nonincrease = all(b <= a for a, b in zip(means, means[1:])) if complete else None
        label = ('INCOMPLETE' if not complete else 'STRICT DECREASE' if strict
                 else 'NONINCREASING' if nonincrease else 'HAS INCREASE')
        lines.append(f'{key[0]} / Q{key[1]} / {key[2]}')
        for row in items:
            mean = 'NA' if row['full_SHD'] is None else f"{row['full_SHD']:.4f}"
            se = 'NA' if row['full_SHD_SE'] is None else f"{row['full_SHD_SE']:.4f}"
            exact = 'NA' if row['composite_exact'] is None else f"{100*row['composite_exact']:.1f}%"
            lines.append(f"  N={row['N']}: full SHD {mean} (SE {se}); valid {row['valid']}/{row['planned']}; composite exact {exact}")
        differences = []
        for low, high in zip(items, items[1:]):
            delta = se = interval = None
            if low['full_SHD'] is not None and high['full_SHD'] is not None:
                delta = high['full_SHD']-low['full_SHD']
                if low['full_SHD_SE'] is not None and high['full_SHD_SE'] is not None:
                    se = math.hypot(low['full_SHD_SE'], high['full_SHD_SE'])
                    interval = [delta-1.96*se, delta+1.96*se]
            differences.append(dict(N_low=low['N'], N_high=high['N'], mean_difference=delta,
                                    se=se, ci95_normal=interval))
        lines.append('  Trend: '+label)
        comparisons.append(dict(model=key[0], Q=key[1], distribution=key[2], complete=complete,
                                strict_decrease=strict, nonincreasing=nonincrease,
                                independent_differences=differences))
    counts = dict(complete=sum(g['complete'] for g in comparisons),
                  strict_decrease=sum(g['strict_decrease'] is True for g in comparisons),
                  nonincreasing=sum(g['nonincreasing'] is True for g in comparisons),
                  total=len(comparisons))
    lines.extend(['', f"Complete groups: {counts['complete']}/{counts['total']}; strictly decreasing: {counts['strict_decrease']}; nonincreasing: {counts['nonincreasing']}"])
    report = dict(settings=rows, groups=comparisons, counts=counts,
                  interval_note='Approximate, unadjusted independent-N normal intervals.')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'section5_summary.json', report)
    with (output/'section5_metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (output/'overview.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
    return report
