"""Evaluation and across-start variability, with explicit latent alignment.

Matching is an evaluation operation. It never modifies a fitted estimator.
Permutations map new coordinates to old coordinates; flips apply afterwards.
"""
from collections.abc import Mapping
from itertools import combinations
import math

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import norm,t

from .latent_law import SparseLatentLaw


def _ratio(numerator,denominator):
    return float(numerator/denominator) if denominator else None


def _support(B,threshold):
    B=np.asarray(B,dtype=float)
    if B.ndim!=2 or B.shape[1]<2 or not np.all(np.isfinite(B)):
        raise ValueError("B must be a finite J by (K+1) coefficient matrix")
    if not np.isfinite(threshold) or threshold<0:
        raise ValueError("threshold must be finite and nonnegative")
    return np.abs(B[:,1:])>threshold


def measurement_metrics(B,Q_true,threshold=0,*,permutation=None):
    """Count support errors; rates use true zeros/ones as denominators.

    Support means abs(slope)>threshold. At zero this is the public nonzero
    convention. Undefined rates (no eligible true entries) are None. By
    default Hungarian matching minimizes Q support errors. A supplied
    permutation is applied directly, allowing one externally chosen latent
    alignment to be used consistently for Q, the graph and the latent law.
    """
    estimated=_support(B,threshold)
    Q=np.asarray(Q_true)
    if Q.shape!=estimated.shape or not np.isin(Q,[0,1]).all():
        raise ValueError("Q_true must be a binary J by K matrix matching B")
    truth=Q.astype(bool)
    if permutation is None:
        costs=np.count_nonzero(truth[:,:,None]!=estimated[:,None,:],axis=0)
        rr,cc=linear_sum_assignment(costs)
        permutation=cc[np.argsort(rr)]
        criterion="Q_support_Hungarian"
    else:
        permutation,_=_alignment(permutation)
        if len(permutation)!=estimated.shape[1]:
            raise ValueError("permutation dimension must match Q_true")
        criterion="supplied_permutation"
    aligned=estimated[:,permutation]
    fp=int(np.sum(aligned&~truth))
    fn=int(np.sum(~aligned&truth))
    tp=int(np.sum(aligned&truth))
    tn=int(np.sum(~aligned&~truth))
    raw_fp,raw_fn=int(np.sum(estimated&~truth)),int(np.sum(~estimated&truth))
    return dict(q_error=fp+fn,q_error_raw=raw_fp+raw_fn,fp=fp,fn=fn,tp=tp,tn=tn,
                fp_raw=raw_fp,fn_raw=raw_fn,fpr=_ratio(fp,fp+tn),fnr=_ratio(fn,fn+tp),
                precision=_ratio(tp,tp+fp),recall=_ratio(tp,tp+fn),
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,
                exact_Q=bool(fp+fn==0),exact_Q_raw=bool(raw_fp+raw_fn==0),
                support_size=int(estimated.sum()),true_edges=int(truth.sum()),
                possible_edges=int(truth.size),permutation=permutation.tolist(),threshold=float(threshold),
                alignment_criterion=criterion)


def latent_sample_alignment(A_est,A_true,fixed_labels=False):
    """Align paired binary samples by minimum total classification errors.

    The returned coordinates are ``A_est[:, permutation] XOR bit_flips``.
    Hungarian costs minimize errors after choosing the better of a column
    and its complement. Fixed labels allow complements but no permutation.
    Exact ties choose the deterministic SciPy assignment and prefer no flip;
    tie flags identify ambiguity rather than implying unique recovery.
    This uses simulation truth for evaluation, never fitting or selection.
    """
    estimated,truth=np.asarray(A_est),np.asarray(A_true)
    if (estimated.ndim!=2 or truth.shape!=estimated.shape or
            not estimated.shape[0] or not estimated.shape[1] or
            not np.isin(estimated,[0,1]).all() or not np.isin(truth,[0,1]).all()):
        raise ValueError("A_est and A_true must be nonempty binary matrices with equal N and K")
    if not isinstance(fixed_labels,(bool,np.bool_)):
        raise ValueError("fixed_labels must be boolean")
    estimated,truth=estimated.astype(np.int64),truth.astype(np.int64)
    N,K=truth.shape
    errors=truth.sum(axis=0)[:,None]+estimated.sum(axis=0)[None,:]-2*truth.T@estimated
    costs=np.minimum(errors,N-errors)
    assignment_ties=np.zeros(K,dtype=bool)
    if fixed_labels:
        permutation=np.arange(K)
        criterion="paired_latent_sample_fixed_labels"
    else:
        rows,columns=linear_sum_assignment(costs)
        permutation=columns[np.argsort(rows)]
        criterion="paired_latent_sample_Hungarian_with_complements"
        optimum=int(costs[np.arange(K),permutation].sum())
        # A chosen pair is ambiguous iff forbidding it leaves the optimum
        # unchanged. Rowwise equal costs alone do not establish global ties.
        if K>1:
            for column in range(K):
                alternative=costs.copy()
                alternative[column,permutation[column]]=N*K+1
                rr,cc=linear_sum_assignment(alternative)
                assignment_ties[column]=int(alternative[rr,cc].sum())==optimum
    selected_errors=errors[np.arange(K),permutation]
    flips=selected_errors>N-selected_errors
    level_ties=selected_errors==N-selected_errors
    aligned=estimated[:,permutation]^flips.astype(np.int64)
    per_column=np.count_nonzero(aligned!=truth,axis=0)
    total=int(per_column.sum())
    return dict(permutation=permutation.tolist(),bit_flips=flips.tolist(),
                per_column_errors=per_column.tolist(),total_errors=total,
                mean_error=float(total/(N*K)),accuracy=float(1-total/(N*K)),
                per_column_accuracy=(1-per_column/N).tolist(),
                joint_accuracy=float(np.mean(np.all(aligned==truth,axis=1))),
                assignment_ties=assignment_ties.tolist(),level_ties=level_ties.tolist(),
                has_ties=bool(assignment_ties.any() or level_ties.any()),
                N=N,K=K,alignment_criterion=criterion)


def _validated_cpdag(graph):
    graph=np.asarray(graph)
    if graph.ndim!=2 or graph.shape[0]!=graph.shape[1] or not np.isfinite(graph).all():
        raise ValueError("A CPDAG must be a finite square endpoint matrix")
    if np.any(np.diag(graph)):
        raise ValueError("A CPDAG cannot have self edges")
    ix=np.triu_indices(len(graph),1)
    a,b=graph[ix],graph.T[ix]
    valid=((a==0)&(b==0))|((a==-1)&(b==-1))|((a==-1)&(b==1))|((a==1)&(b==-1))
    if not valid.all():
        raise ValueError("Use CPDAG endpoints: (0,0),(-1,-1),(-1,1),(1,-1); no stitched Q block")
    return graph


def _edge_states(graph):
    ix=np.triu_indices(len(graph),1)
    a,b=graph[ix],graph.T[ix]
    result=np.zeros(len(a),dtype=np.int8)
    result[(a==-1)&(b==-1)]=2
    result[(a==-1)&(b==1)]=1
    result[(a==1)&(b==-1)]=-1
    return result


def vstructures(graph):
    """Unshielded collider triples (smaller parent, child, larger parent)."""
    graph=_validated_cpdag(graph)
    result=set()
    for child in range(len(graph)):
        parents=np.flatnonzero((graph[:,child]==-1)&(graph[child,:]==1))
        for first,second in combinations(parents.tolist(),2):
            if graph[first,second]==graph[second,first]==0:
                result.add((min(first,second),child,max(first,second)))
    return result


def graph_metrics(estimated,true):
    """CPDAG, skeleton, and unshielded-collider recovery in fixed labels.

    No matching occurs here. Align both latent axes before calling if needed.
    V-structure FPR is over all K*choose(K-1,2) possible labeled colliders.
    Directed-edge metrics compare the sets of compelled arrows in the two
    supplied CPDAGs; a reversal counts as one FP and one FN, while CPDAG SHD
    continues to count a changed pair once. Precision/recall on empty sets
    are None; F1 is one when both sets are empty.
    """
    estimated,true=_validated_cpdag(estimated),_validated_cpdag(true)
    if estimated.shape!=true.shape:
        raise ValueError("Graphs must have the same dimensions")
    es,ts=_edge_states(estimated),_edge_states(true)
    skeleton_e,skeleton_t=es!=0,ts!=0
    fp,fn=int(np.sum(skeleton_e&~skeleton_t)),int(np.sum(~skeleton_e&skeleton_t))
    tp=int(np.sum(skeleton_e&skeleton_t))
    ve,vt=vstructures(estimated),vstructures(true)
    vfp,vfn=len(ve-vt),len(vt-ve)
    vtp=len(ve&vt)
    directed_e=(estimated==-1)&(estimated.T==1)
    directed_t=(true==-1)&(true.T==1)
    dtp=int(np.sum(directed_e&directed_t))
    dfp,dfn=int(np.sum(directed_e&~directed_t)),int(np.sum(~directed_e&directed_t))
    K=len(true)
    possible_v=K*(K-1)*(K-2)//2 if K>=3 else 0
    distance=int(np.count_nonzero(es!=ts))
    return dict(cpdag_shd=distance,exact_cpdag=bool(distance==0),
                skeleton_shd=fp+fn,skeleton_fp=fp,skeleton_fn=fn,
                skeleton_fpr=_ratio(fp,int((~skeleton_t).sum())),
                skeleton_fnr=_ratio(fn,int(skeleton_t.sum())),
                skeleton_precision=_ratio(tp,tp+fp),skeleton_recall=_ratio(tp,tp+fn),
                skeleton_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,
                exact_skeleton=bool(fp+fn==0),true_skeleton_edges=int(skeleton_t.sum()),
                estimated_skeleton_edges=int(skeleton_e.sum()),
                vstructure_fp=vfp,vstructure_fn=vfn,vstructure_shd=vfp+vfn,
                vstructure_fpr=_ratio(vfp,possible_v-len(vt)),vstructure_fnr=_ratio(vfn,len(vt)),
                vstructure_precision=_ratio(vtp,len(ve)),
                vstructure_recall=_ratio(vtp,len(vt)),
                vstructure_f1=2*vtp/(2*vtp+vfp+vfn) if 2*vtp+vfp+vfn else 1.,
                exact_vstructures=bool(ve==vt),true_vstructures=len(vt),estimated_vstructures=len(ve),
                directed_tp=dtp,directed_fp=dfp,directed_fn=dfn,
                directed_precision=_ratio(dtp,dtp+dfp),directed_recall=_ratio(dtp,dtp+dfn),
                directed_f1=2*dtp/(2*dtp+dfp+dfn) if 2*dtp+dfp+dfn else 1.,
                true_directed_edges=int(directed_t.sum()),estimated_directed_edges=int(directed_e.sum()),
                exact_directed_edges=bool(dfp+dfn==0))


def _alignment(permutation,bit_flips=None):
    perm=np.asarray(permutation)
    K=len(perm)
    if perm.ndim!=1 or K<1 or not np.array_equal(np.sort(perm),np.arange(K)):
        raise ValueError("permutation must be a bijection of 0,...,K-1")
    flips=np.zeros(K,dtype=int) if bit_flips is None else np.asarray(bit_flips)
    if flips.shape!=(K,) or not np.isin(flips,[0,1]).all():
        raise ValueError("bit_flips must be K binary values in new-coordinate order")
    return perm.astype(int),flips.astype(int)


def align_latent_law(p,permutation,bit_flips=None):
    """Law of z_new=z_old[permutation] XOR bit_flips; no dense sparse export.

    Returns SparseLatentLaw, dict, or 1-D ndarray according to input type.
    Uniform smoothing mass is invariant under this bijective transformation.
    """
    perm,flips=_alignment(permutation,bit_flips)
    K=len(perm)
    def transform(state):
        result=0
        for old_coordinate,flip in zip(perm,flips):
            result=(result<<1)|(((int(state)>>(K-1-int(old_coordinate)))&1)^int(flip))
        return result
    if isinstance(p,SparseLatentLaw):
        if p.K!=K:
            raise ValueError("Latent law dimension does not match permutation")
        return SparseLatentLaw(K,{transform(s): w for s,w in p.masses.items()},p.uniform_mass)
    if isinstance(p,Mapping):
        law=SparseLatentLaw(K,dict(p))
        return {transform(s): w for s,w in law.masses.items()}
    p=np.asarray(p,dtype=float).reshape(-1)
    if len(p)!=1<<K or not np.isfinite(p).all() or np.any(p<0) or not np.isclose(p.sum(),1):
        raise ValueError("p must be a normalized probability vector of length 2**K")
    result=np.empty_like(p)
    for state,mass in enumerate(p):
        result[transform(state)]=mass
    return result


def align_latent_graph(graph,permutation):
    """Apply one coordinate permutation to both graph axes."""
    perm,_=_alignment(permutation)
    graph=_validated_cpdag(graph)
    if graph.shape!=(len(perm),len(perm)):
        raise ValueError("Graph and permutation dimensions disagree")
    return graph[np.ix_(perm,perm)].copy()


def _as_sparse(p,K):
    if isinstance(p,SparseLatentLaw):
        if p.K!=K:
            raise ValueError("Inconsistent latent dimensions")
        return p
    if isinstance(p,Mapping):
        return SparseLatentLaw(K,dict(p))
    values=np.asarray(p,dtype=float).reshape(-1)
    if len(values)!=1<<K:
        raise ValueError("Latent probability vector has wrong length")
    return SparseLatentLaw(K,{int(s): float(values[s]) for s in np.flatnonzero(values)})


def latent_law_distance(first,second,K):
    """L1, TV, and L2 distances, including uniform background without enumeration."""
    first,second=_as_sparse(first,K),_as_sparse(second,K)
    keys=set(first.masses)|set(second.masses)
    background_total=first.uniform_mass-second.uniform_mass
    per_state=math.ldexp(background_total,-K)
    differences=[first.masses.get(s,0.)-second.masses.get(s,0.)+per_state for s in keys]
    unvisited_fraction=1-len(keys)/(1<<K)
    l1=math.fsum(abs(v) for v in differences)+abs(background_total)*unvisited_fraction
    l2sq=math.fsum(v*v for v in differences)+abs(background_total)*abs(per_state)*unvisited_fraction
    return dict(l1=l1,tv=.5*l1,l2=math.sqrt(max(0.,l2sq)))


def mc_summary(values,confidence=.95):
    """Monte Carlo uncertainty across independent replications; reject nonfinite values."""
    values=np.asarray(list(values),dtype=float)
    if values.ndim!=1 or not np.isfinite(values).all() or not 0<confidence<1:
        raise ValueError("Use finite replicate values and confidence in (0,1); account for failures separately")
    n=len(values)
    if not n:
        return dict(n=0,mean=None,mc_se=None,se=None,ci=None,confidence=confidence)
    mean=float(values.mean())
    if n==1:
        return dict(n=n,mean=mean,mc_se=None,se=None,ci=None,confidence=confidence)
    se=float(values.std(ddof=1)/np.sqrt(n))
    radius=float(t.ppf((1+confidence)/2,n-1)*se)
    return dict(n=n,mean=mean,mc_se=se,se=se,ci=[mean-radius,mean+radius],confidence=confidence)


def wilson_rate(successes,total,confidence=.95):
    if not isinstance(successes,(int,np.integer)) or not isinstance(total,(int,np.integer)) or not 0<=successes<=total or not 0<confidence<1:
        raise ValueError("Require integer 0 <= successes <= total and confidence in (0,1)")
    if not total:
        return dict(successes=int(successes),n=0,rate=None,ci=None,confidence=confidence)
    rate,z=successes/total,float(norm.ppf((1+confidence)/2))
    denominator=1+z*z/total
    center=(rate+z*z/(2*total))/denominator
    radius=z*math.sqrt(rate*(1-rate)/total+z*z/(4*total*total))/denominator
    return dict(successes=int(successes),n=int(total),rate=rate,
                ci=[max(0.,center-radius),min(1.,center+radius)],confidence=confidence)


def fit_variability(fits,threshold=0):
    """Truth-free variability after common alignment to the first supplied fit.

    Each fit is a mapping with B,p and optional cpdag (or graph). Orient columns by
    positive sum (largest-magnitude entry breaks a zero-sum tie), then match
    support to fit 0 with coefficient distance only breaking support ties.
    Graphs receive the same permutation on both axes; p receives flips and
    permutation. Pairwise observations are dependent: no MC CI is claimed.
    """
    fits=list(fits)
    if not fits:
        return dict(n_starts=0,alignments=[],pairwise=[],summary={})
    oriented=[]
    shape=np.asarray(fits[0]["B"]).shape
    for fit in fits:
        B=np.asarray(fit["B"],dtype=float).copy()
        _support(B,threshold)
        if B.shape!=shape:
            raise ValueError("All starts must use the same item and latent dimensions")
        slopes=B[:,1:]
        orientation=slopes.sum(axis=0)
        tied=orientation==0
        orientation[tied]=slopes[np.argmax(np.abs(slopes[:,tied]),axis=0),np.flatnonzero(tied)] if tied.any() else []
        flips=(orientation<0).astype(int)
        B[:,0]+=slopes[:,flips.astype(bool)].sum(axis=1)
        B[:,1:]*=1-2*flips
        K=B.shape[1]-1
        oriented.append(dict(B=B,p=align_latent_law(fit["p"],np.arange(K),flips),
                             cpdag=fit.get("cpdag",fit.get("graph")),flips=flips))
    reference=oriented[0]["B"]
    aligned,alignments=[],[]
    for fit in oriented:
        current=fit["B"]
        costs=np.count_nonzero(_support(reference,threshold)[:,:,None]!=_support(current,threshold)[:,None,:],axis=0).astype(float)
        coefficient_cost=np.sum((reference[:,1:,None]-current[:,None,1:])**2,axis=0)
        costs+=coefficient_cost/max(1.,float(coefficient_cost.max()))/(2*K)
        rr,cc=linear_sum_assignment(costs)
        perm=cc[np.argsort(rr)]
        graph=None if fit["cpdag"] is None else align_latent_graph(fit["cpdag"],perm)
        aligned.append(dict(B=np.column_stack((current[:,0],current[:,1:][:,perm])),
                            p=align_latent_law(fit["p"],perm),cpdag=graph))
        alignments.append(dict(permutation=perm.tolist(),bit_flips=fit["flips"][perm].tolist()))
    pairs=[]
    for i,j in combinations(range(len(fits)),2):
        first,second=aligned[i],aligned[j]
        distance=latent_law_distance(first["p"],second["p"],K)
        raw_distance=latent_law_distance(fits[i]["p"],fits[j]["p"],K)
        row=dict(first=i,second=j,q_disagreement=int(np.count_nonzero(_support(first["B"],threshold)!=_support(second["B"],threshold))),
                   q_disagreement_raw=int(np.count_nonzero(_support(fits[i]["B"],threshold)!=_support(fits[j]["B"],threshold))),
                   p_tv=distance["tv"],p_l2=distance["l2"],p_tv_raw=raw_distance["tv"],p_l2_raw=raw_distance["l2"],
                   coefficient_frobenius=float(np.linalg.norm(first["B"]-second["B"])))
        if first["cpdag"] is not None and second["cpdag"] is not None:
            row["cpdag_shd"]=graph_metrics(first["cpdag"],second["cpdag"])["cpdag_shd"]
            raw_first=fits[i].get("cpdag",fits[i].get("graph"))
            raw_second=fits[j].get("cpdag",fits[j].get("graph"))
            row["cpdag_shd_raw"]=graph_metrics(raw_first,raw_second)["cpdag_shd"]
        pairs.append(row)
    names=("q_disagreement","q_disagreement_raw","p_tv","p_l2","p_tv_raw","p_l2_raw",
             "coefficient_frobenius","cpdag_shd","cpdag_shd_raw")
    summary={name: dict(n_pairs=len(values),mean=float(np.mean(values)),maximum=float(np.max(values)))
               for name in names if (values:=[row[name] for row in pairs if name in row])}
    return dict(n_starts=len(fits),reference_start=0,alignments=alignments,pairwise=pairs,summary=summary,
                note="Common observed-fit alignment, no truth; ambiguous/identical loading columns may leave label ambiguity. "
                     "Pairwise distances are dependent; no Monte Carlo CI. Initial and final stages must be passed separately.")
