"""Run simulations, fit diagnostics and stage-wise oracle comparisons."""
from __future__ import annotations

from dataclasses import asdict,fields
import copy
import hashlib
import importlib.metadata
import itertools
import json
import math
from pathlib import Path
import platform
import random
import time
import traceback

import numpy as np
from joblib import Parallel,delayed

from .config import ExperimentConfig,BoundsConfig
from .data_generator import GenerateData
from .evaluation import stitch_full_A,shd_cpdag
from .latent_law import SparseLatentLaw


def jsonable(value):
    if isinstance(value,dict):
        return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value,np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value,np.generic):
        return jsonable(value.item())
    if isinstance(value,float) and not math.isfinite(value):
        return None
    if isinstance(value,Path):
        return str(value)
    return value


def write_json(path,value):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(jsonable(value),indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def source_hashes():
    root=Path(__file__).resolve().parent
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*.py'))}


def versions():
    result={'python':platform.python_version(),'platform':platform.platform()}
    for name in ['numpy','scipy','factor_analyzer','causal-learn','joblib']:
        try:
            result[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name]=None
    return result


def dense_law(law,K):
    if isinstance(law,SparseLatentLaw):
        return law.to_dense().reshape(-1)
    if isinstance(law,dict):
        p=np.zeros(1<<K)
        for state,mass in law.items():
            p[int(state)]=mass
        return p
    return np.asarray(law).reshape(-1)


def save_law(path,law,K):
    if not isinstance(law,SparseLatentLaw):
        if isinstance(law,dict):
            law=SparseLatentLaw.from_counts(K,law)
        else:
            p=np.asarray(law).reshape(-1)
            law=SparseLatentLaw.from_counts(K,{int(i):float(p[i]) for i in np.flatnonzero(p)})
    law.save_npz(path)


def expanded_settings(plan):
    model=dict(plan.get('model',{}))
    grid=plan.get('grid',{})
    if any(not isinstance(v,list) or not v for v in grid.values()):
        raise ValueError('Each grid value must be a nonempty list')
    names=sorted(grid)
    settings=[]
    for values in itertools.product(*(grid[name] for name in names)):
        spec=dict(model,**dict(zip(names,values)))
        spec.setdefault('profile',plan.get('profile','legacy'))
        raw=json.dumps(spec,sort_keys=True,separators=(',',':'))
        settings.append((hashlib.sha256(raw.encode()).hexdigest()[:12],spec))
    if len({key for key,_ in settings})!=len(settings):
        raise ValueError('Grid contains duplicate settings')
    return settings


def make_config(spec):
    spec=copy.deepcopy(spec)
    extras={name:spec.pop(name) for name in ['signal_scheme','signal_multiplier',
              'pen_multiplier','tau_multiplier','intercept','Q_matrix'] if name in spec}
    valid={f.name for f in fields(ExperimentConfig)}
    unknown=set(spec)-valid
    if unknown:
        raise ValueError(f'Unknown fitting settings: {sorted(unknown)}')
    if isinstance(spec.get('bounds'),dict):
        spec['bounds']=BoundsConfig(**spec['bounds'])
    cfg=ExperimentConfig(**spec).finalize()
    for name,field in [('pen_multiplier','pen'),('tau_multiplier','tau')]:
        value=float(extras.get(name,1.0))
        if not np.isfinite(value) or value<=0:
            raise ValueError(f'{name} must be finite and positive')
        setattr(cfg,field,getattr(cfg,field)*value)
    return cfg,extras


def generate_observations(cfg,extras,seed):
    from .oracle import generate_latent_with_truth
    data=GenerateData.from_config(cfg,Q_matrix=extras.get('Q_matrix'))
    scheme=extras.get('signal_scheme','original')
    multiplier=float(extras.get('signal_multiplier',1.0))
    if not np.isfinite(multiplier) or multiplier<=0:
        raise ValueError('signal_multiplier must be finite and positive')
    if scheme=='restore_minimum':
        total=2.0 if cfg.distribution=='Poisson' else 3.0
        degree=data.Q.sum(axis=1)
        data.B[:,1:]=total*data.Q/np.minimum(degree[:,None],3)
    elif scheme!='original':
        raise ValueError('signal_scheme must be original or restore_minimum')
    data.B[:,1:]*=multiplier
    if 'intercept' in extras:
        intercept=float(extras['intercept'])
        if not np.isfinite(intercept):
            raise ValueError('intercept must be finite')
        data.B[:,0]=intercept
    random.seed(seed); np.random.seed(seed)
    generate_latent_with_truth(data)
    data.generate_data()
    return data


def start_specs(plan,seed):
    templates=plan.get('starts',[{'kind':'spectral'}])
    if not templates:
        raise ValueError('At least one start is required')
    result=[]
    for index,template in enumerate(templates):
        item=dict(template)
        item.setdefault('id',f"{item.get('kind','spectral')}_{index}")
        item.setdefault('seed',seed+100000+1009*index)
        item.setdefault('init_seed',seed+200000+1009*index)
        result.append(item)
    if len({x['id'] for x in result})!=len(result):
        raise ValueError('Start ids must be unique')
    return result


def persist_fit(folder,fit,K):
    """Persist every successful and failed start; keep arrays out of JSON logs."""
    folder.mkdir(parents=True,exist_ok=True)
    array_fields=['B','gamma','raw_B','initial_B','initial_A','initial_gamma',
                    'final_sample_A','raw_final_sample_A']
    law_fields=['p','raw_p','initial_p']
    def save_record(record,destination):
        destination.mkdir(exist_ok=True)
        arrays={name:np.asarray(record[name]) for name in array_fields
                  if record.get(name) is not None}
        if arrays:
            np.savez_compressed(destination/'parameters.npz',**arrays)
        for name in law_fields:
            if record.get(name) is not None:
                save_law(destination/(name+'.npz'),record[name],K)
        result={key:value for key,value in record.items()
                  if key not in set(array_fields+law_fields+['starts'])}
        result['artifacts']=str(destination.relative_to(folder))
        write_json(destination/'diagnostics.json',result)
        return result
    starts=[save_record(start,folder/f'start_{index:03d}')
              for index,start in enumerate(fit.get('starts',[]))]
    selected=save_record(fit,folder/'selected')
    selected['starts']=starts
    write_json(folder/'fit.json',selected)
    return selected


def fit_has_output(fit):
    return fit.get('B') is not None and fit.get('p') is not None


def simulation_alignment(fit,data,mode='Q_support',fixed_labels=False):
    """Evaluation-only matching; neither fitting nor start selection sees truth."""
    from .metrics import measurement_metrics,latent_sample_alignment
    if mode not in {'Q_support','latent_sample'}:
        raise ValueError('evaluation_alignment must be Q_support or latent_sample')
    if mode=='latent_sample' and not fixed_labels:
        result=latent_sample_alignment(fit['final_sample_A'],data.A)
        result['sample_source']='final_Gibbs_sample'
        return result
    result=measurement_metrics(fit['B'],data.Q)
    permutation=np.arange(data.K) if fixed_labels else np.asarray(result['permutation'],dtype=int)
    sums=np.asarray(fit['B'])[:,1:][:,permutation].sum(axis=0)
    truth_sums=data.B[:,1:].sum(axis=0)
    flips=sums*truth_sums<0
    return dict(permutation=permutation.tolist(),bit_flips=flips.tolist(),
                alignment_criterion='known_Q_fixed_labels' if fixed_labels else 'Q_support_matching',
                unresolved_orientation=np.flatnonzero((sums==0)|(truth_sums==0)).tolist())


def fit_metrics(fit,data,p_true,fixed_labels=False,alignment='Q_support'):
    from .metrics import measurement_metrics,align_latent_law
    match=simulation_alignment(fit,data,alignment,fixed_labels)
    permutation=np.asarray(match['permutation'],dtype=int)
    flips=np.asarray(match['bit_flips'],dtype=int)
    result=measurement_metrics(fit['B'],data.Q,permutation=permutation)
    p=dense_law(align_latent_law(fit['p'],permutation,flips),data.K)
    difference=p-np.asarray(p_true).reshape(-1)
    result.update(latent_p_l1=float(np.abs(difference).sum()),
                  latent_p_tv=float(np.abs(difference).sum()/2),
                  latent_p_linf=float(np.abs(difference).max()),
                  latent_p_permutation=permutation.tolist(),
                  latent_p_alignment=match['alignment_criterion'],
                  latent_level_flips=flips.tolist(),
                  evaluation_alignment=match)
    if alignment=='latent_sample' and not fixed_labels:
        result['Q_support_reference']=measurement_metrics(fit['B'],data.Q)
    return result


def graph_grid(plan):
    graph=plan.get('graph',{})
    methods=graph.get('methods',['resampling'])
    ratios=graph.get('ratios',[1.0])
    ess_values=graph.get('ess',[1.0])
    priors=graph.get('structure_prior',[None])
    for method,ratio,ess,prior in itertools.product(methods,ratios,ess_values,priors):
        if method not in ['resampling','plugin']:
            raise ValueError('Graph method must be resampling or plugin')
        if not np.isfinite(ratio) or ratio<=0:
            raise ValueError('Graph ratios must be finite and positive')
        yield method,float(ratio),ess,prior


def graph_experiments(plan,folder,data,truth,fits,seed):
    from .graph_scoring import learn_latent_graph,learn_graph_from_samples
    from .metrics import graph_metrics
    folder.mkdir(exist_ok=True)
    graph_cfg=plan.get('graph',{})
    draws=int(graph_cfg.get('draws',1))
    if draws<1:
        raise ValueError('graph.draws must be positive')
    full=fits.get('full',{})
    stages=[('fully_estimated',full.get('p'),full.get('B'),None,full)]
    if plan.get('oracles',False):
        stages.insert(0,('true_p',truth['p_true'],data.B,None,None))
        stages.insert(1,('known_Q',fits.get('known_Q',{}).get('p'),data.B,None,None))
    if plan.get('graph_all_starts',True):
        available={s['id']:s for s in full.get('starts',[])}
        for spec in start_specs(plan,seed):
            start=available.get(spec['id'],{})
            usable=start.get('fit_valid',False) and fit_has_output(start)
            stages.append(('fully_estimated_start',start.get('p') if usable else None,
                           start.get('B') if usable else None,spec['id'],start))
    alignments={}
    for stage,p,B,start_id,fit in stages:
        if p is not None and stage.startswith('fully_estimated'):
            try:
                alignments[(stage,start_id)]=simulation_alignment(
                    fit,data,plan.get('evaluation_alignment','Q_support'))
            except Exception as exc:
                alignments[(stage,start_id)]=exc
    records=[]
    index=0
    for method,ratio,ess,prior in graph_grid(plan):
        size=max(1,int(np.floor(ratio*data.N))) if method=='resampling' else ratio*data.N
        for stage,p,B,start_id,fit in stages:
            for draw in range(draws if method=='resampling' else 1):
                index+=1
                record=dict(stage=stage,start_id=start_id,method=method,ratio=ratio,
                              effective_n=size,ess=ess,structure_prior=prior,draw=draw,
                              seed=seed+300000+draw if method=='resampling' else None,status='failed')
                try:
                    if p is None:
                        raise RuntimeError('This graph stage is unavailable because its fitted latent law failed; other oracle stages are still evaluated')
                    result=learn_latent_graph(p,data.K,size,method=method,ess=ess,
                              structure_prior=prior,seed=record['seed'],
                              max_parents=graph_cfg.get('max_parents'))
                    graph=np.asarray(result['graph'])
                    if stage in ['true_p','known_Q']:
                        perm=np.arange(data.K)
                    else:
                        match=alignments[(stage,start_id)]
                        if isinstance(match,Exception):
                            raise match
                        perm=np.asarray(match['permutation'],int)
                        record['evaluation_alignment']=match
                    matched=graph[np.ix_(perm,perm)]
                    record.update(status='ok',metrics=graph_metrics(graph,truth['cpdag_true']),
                                  matched_metrics=graph_metrics(matched,truth['cpdag_true']))
                    if stage.startswith('fully_estimated'):
                        q=(np.asarray(B)[:,1:]!=0).astype(int)
                        full_graph=stitch_full_A(graph,q.T,data.J)
                        target=stitch_full_A(truth['cpdag_true'],data.Q.T,data.J)
                        order=np.r_[perm,np.arange(data.K,data.K+data.J)]
                        record['full_shd']=int(shd_cpdag(target,full_graph))
                        record['matched_full_shd']=int(shd_cpdag(target,full_graph[np.ix_(order,order)]))
                    path=folder/f'graph_{index:05d}.npz'
                    np.savez_compressed(path,graph=graph,truth=truth['cpdag_true'],permutation=perm)
                    record['artifact']=path.name
                    record['diagnostics']={k:v for k,v in result.items()
                                             if k not in ['graph','samples','sample_counts','counts','sufficient_statistics']}
                    if result.get('sample_counts') is not None:
                        write_json(folder/f'counts_{index:05d}.json',result['sample_counts'])
                        record['counts_artifact']=f'counts_{index:05d}.json'
                    elif result.get('sufficient_statistics') is not None:
                        write_json(folder/f'counts_{index:05d}.json',result['sufficient_statistics'])
                        record['counts_artifact']=f'counts_{index:05d}.json'
                except Exception as exc:
                    record.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
                records.append(record)
        # The actual latent sample is a distinct oracle: no estimated p or
        # secondary resampling. Evaluate once per ESS/prior, not once per m.
    if plan.get('oracles',False):
        seen=set()
        for _,_,ess,prior in graph_grid(plan):
            if (ess,prior) in seen:
                continue
            seen.add((ess,prior)); index+=1
            record=dict(stage='true_latent_sample',start_id=None,method='empirical',ratio=1.0,
                          effective_n=data.N,ess=ess,structure_prior=prior,draw=0,seed=None,status='failed')
            try:
                result=learn_graph_from_samples(data.A,ess=ess,structure_prior=prior,
                                                   max_parents=graph_cfg.get('max_parents'))
                graph=np.asarray(result['graph'])
                record.update(status='ok',metrics=graph_metrics(graph,truth['cpdag_true']),
                              matched_metrics=graph_metrics(graph,truth['cpdag_true']))
                path=folder/f'graph_{index:05d}.npz'
                np.savez_compressed(path,graph=graph,truth=truth['cpdag_true'],permutation=np.arange(data.K))
                record['artifact']=path.name
                record['diagnostics']={k:v for k,v in result.items()
                                         if k not in ['graph','samples','sample_counts','counts']}
            except Exception as exc:
                record.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
            records.append(record)
    write_json(folder/'results.json',records)
    return records


def run_replicate(plan,setting_id,spec,seed,output):
    from .fitting import fit_observed
    from .oracle import true_model
    from .metrics import fit_variability
    folder=Path(output)/f'{setting_id}__s{seed}'
    folder.mkdir(exist_ok=False)
    began=time.perf_counter()
    row=dict(setting_id=setting_id,setting=spec,seed=seed,status='failed',artifacts=folder.name)
    phase='generation'
    try:
        cfg,extra=make_config(spec)
        data=generate_observations(cfg,extra,seed)
        truth=true_model(data,max_k=int(plan.get('truth_max_k',16)))
        np.savez_compressed(folder/'data.npz',X=data.X,A_true=data.A,Q_true=data.Q,
                            B_true=data.B,gamma_true=data.gamma,p_true=truth['p_true'],
                            cpdag_true=truth['cpdag_true'],dag_true=truth['dag_true'])
        write_json(folder/'truth.json',{k:v for k,v in truth.items()
                                        if k not in ['p_true','cpdag_true','dag_true']})
        row.update(config=asdict(cfg),data_sha256=hashlib.sha256(data.X.tobytes()).hexdigest(),
                   start_specs=start_specs(plan,seed),orientation='column_sum')
        phase='fit'
        def attempt_fit(known_Q=None):
            try:
                return fit_observed(data.X,cfg,start_specs=row['start_specs'],known_Q=known_Q,
                                    orientation=row['orientation'])
            except Exception as exc:
                starts=[dict(id=s['id'],spec=s,status='not_attempted',fit_valid=False,
                               scorable=False,error=repr(exc),failure_stage='fit_setup') for s in row['start_specs']]
                return dict(status='failed',starts=starts,error=repr(exc),traceback=traceback.format_exc())
        fits={'full':attempt_fit()}
        row['fit']=persist_fit(folder/'fit',fits['full'],cfg.K)
        if fit_has_output(fits['full']):
            row['measurement']=fit_metrics(fits['full'],data,truth['p_true'],
                                            alignment=plan.get('evaluation_alignment','Q_support'))
        valid=[s for s in fits['full'].get('starts',[]) if s.get('fit_valid',False) and fit_has_output(s)]
        row['fit_variability']=fit_variability(valid)
        if plan.get('oracles',False):
            phase='known_Q_fit'
            fits['known_Q']=attempt_fit(known_Q=data.Q)
            row['known_Q_fit']=persist_fit(folder/'known_Q_fit',fits['known_Q'],cfg.K)
            if fit_has_output(fits['known_Q']):
                row['known_Q_measurement']=fit_metrics(fits['known_Q'],data,truth['p_true'],fixed_labels=True,
                                                        alignment=plan.get('evaluation_alignment','Q_support'))
        phase='graphs'
        row['graphs']=graph_experiments(plan,folder/'graphs',data,truth,fits,seed)
        # Hold graph settings and random draw fixed when comparing starts.
        # Alignment here uses the first fitted start, never simulation truth.
        graph_groups={}
        lookup={s['id']:s for s in valid}
        for record in row['graphs']:
            if record['stage']!='fully_estimated_start' or record['status']!='ok':
                continue
            key=(record['method'],record['ratio'],record['ess'],record['structure_prior'],record['draw'])
            with np.load(folder/'graphs'/record['artifact']) as artifact:
                item=dict(lookup[record['start_id']],graph=artifact['graph'].copy())
            graph_groups.setdefault(key,[]).append(item)
        row['graph_variability']=[dict(method=key[0],ratio=key[1],ess=key[2],structure_prior=key[3],draw=key[4],
                                         variability=fit_variability(group)) for key,group in graph_groups.items()]
        row['status']='ok' if all(r['status']=='ok' for r in row['graphs']) else 'partial_graph_failure'
        row['fit_stage_failures']=[name for name,fit in fits.items() if not fit_has_output(fit)]
        if row['fit_stage_failures']:
            row['status']='partial_fit_failure'
    except Exception as exc:
        row.update(error=repr(exc),failure_phase=phase,traceback=traceback.format_exc())
    row['wall_seconds']=time.perf_counter()-began
    write_json(folder/'result.json',row)
    return row


def run_study(plan,output,jobs=1,resume=False):
    output=Path(output).resolve()
    repeats=int(plan.get('repetitions',1)); start=int(plan.get('seed_start',1))
    if repeats<1 or jobs==0:
        raise ValueError('repetitions must be positive and jobs must be nonzero')
    if plan.get('evaluation_alignment','Q_support') not in {'Q_support','latent_sample'}:
        raise ValueError('evaluation_alignment must be Q_support or latent_sample')
    if plan.get('orientation','column_sum')!='column_sum':
        raise ValueError('Study fits must use column_sum orientation')
    settings=expanded_settings(plan)
    # Validate configuration before creating output or launching workers.
    for _,spec in settings:
        make_config(spec)
    list(graph_grid(plan)); start_specs(plan,start)
    hashes=source_hashes()
    if output.exists():
        if not resume:
            raise FileExistsError(f'{output} exists; choose a new output or --resume')
        metadata=json.loads((output/'manifest.json').read_text())
        if metadata['plan']!=jsonable(plan) or metadata['source_hashes']!=hashes:
            raise ValueError('Resume requires the same frozen plan and source hashes')
        if metadata['versions']!=versions():
            raise ValueError('Resume requires the same numerical environment')
    else:
        output.mkdir(parents=True)
        metadata=dict(plan=plan,source_hashes=hashes,versions=versions(),expected_replicates=repeats*len(settings),
                        schema_version=1,created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                        note='Known-Q and true-p paths are explicitly labeled oracle diagnostics; they never select full-model tuning or starts.')
        write_json(output/'manifest.json',metadata)
    tasks=[]
    for key,spec in settings:
        for seed in range(start,start+repeats):
            folder=output/f'{key}__s{seed}'
            if folder.exists():
                if not (folder/'result.json').exists():
                    raise RuntimeError(f'Incomplete artifact directory {folder}; preserve it and choose a new output')
                continue
            tasks.append((key,spec,seed))
    completed=metadata['expected_replicates']-len(tasks)
    work=(delayed(run_replicate)(plan,key,spec,seed,output) for key,spec,seed in tasks)
    for row in Parallel(n_jobs=jobs,return_as='generator_unordered')(work):
        completed+=1
        print(json.dumps(dict(completed=completed,total=metadata['expected_replicates'],
                             setting=row['setting_id'],seed=row['seed'],status=row['status'])),flush=True)
    if source_hashes()!=hashes:
        raise RuntimeError('Source changed while experiments were running')
    from .study_summary import summarize_study
    return summarize_study(output)
