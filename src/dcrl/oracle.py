"""Exact generator truth for stagewise simulation diagnostics only.

The public sampler does not retain its CPT draws. The capture helper calls
it unchanged, then reconstructs those actual draws with a private replay of
the preceding Python RNG state. Population p is never inferred from A.
"""
from collections import Counter
import copy
import random

import numpy as np
from causallearn.graph.Dag import Dag
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.DAG2CPDAG import dag2cpdag

from .latent_law import SparseLatentLaw


def latent_dag(dag_type,K):
    """Known directed adjacency of an existing generator; no sampled-data GES."""
    if not isinstance(K,(int,np.integer)) or K<1:
        raise ValueError("K must be a positive integer")
    edges=[]
    if dag_type in ("Markov","Chain"):
        edges=[(k-1,k) for k in range(1,K)]
    elif dag_type=="Tree":
        edges=[((k-1)//2,k) for k in range(1,K)]
    elif dag_type=="DiverseTree":
        complete=2**int(np.log2(K+1))-1
        size=(complete+1)//2
        start=0
        while size>1:
            next_size=size//2
            edges+=[(start+2*j+offset,start+size+j) for j in range(next_size) for offset in (0,1)]
            start+=size
            size=next_size
        edges+=[(k-1,k) for k in range(complete,K)]
    else:
        expected={"Model-7":7,"Model-8":8,"Model-13":13,"Model-16":16}
        if dag_type not in expected or K!=expected[dag_type]:
            raise ValueError(f"Unsupported generator/dimension: {dag_type}, K={K}")
        edges=[(0,1),(0,2)]
        if dag_type=="Model-7":
            edges+=[(a,b) for a in (1,2) for b in (3,4)]+[(3,5),(4,6)]
        elif dag_type in ("Model-8","Model-13"):
            edges+=[(1,3),(1,4),(2,5),(2,6)]
            if dag_type=="Model-8":
                edges+=[(4,7),(6,7)]
            else:
                edges+=[(a,b) for a in (3,4) for b in (7,8,9)]
                edges+=[(a,b) for a in (5,6) for b in (10,11,12)]
        else:
            edges+=[(1,3),(2,3),(3,4),(3,5),(4,6),(4,7),(5,8),(5,9)]
            edges+=[(a,b) for a in (6,7) for b in (10,11,12)]
            edges+=[(a,b) for a in (8,9) for b in (13,14,15)]
    adjacency=np.zeros((K,K),dtype=int)
    for parent,child in edges:
        adjacency[parent,child]=1
    return adjacency


def exact_cpdag(adjacency):
    """Convert an explicitly supplied DAG, not observations, to its CPDAG."""
    adjacency=np.asarray(adjacency)
    if adjacency.ndim!=2 or adjacency.shape[0]!=adjacency.shape[1] or not np.isin(adjacency,[0,1]).all() or np.diag(adjacency).any():
        raise ValueError("adjacency must be a binary square DAG adjacency without self edges")
    remaining=set(range(len(adjacency)))
    while remaining:
        roots={k for k in remaining if not any(adjacency[j,k] for j in remaining)}
        if not roots:
            raise ValueError("adjacency contains a directed cycle")
        remaining-=roots
    nodes=[GraphNode(f"Z{k}") for k in range(len(adjacency))]
    graph=Dag(nodes)
    for parent,child in zip(*np.nonzero(adjacency)):
        graph.add_directed_edge(nodes[parent],nodes[child])
    return dag2cpdag(graph).graph.copy()


def _capture_spec(data,state,alternate_prob):
    K,kind,bounds=data.K,data.DAG_type,data.bounds
    adjacency=latent_dag(kind,K)
    parents=[np.flatnonzero(adjacency[:,k]).tolist() for k in range(K)]
    cpts=[[.5] for _ in range(K)]
    rng=random.Random()
    rng.setstate(state)
    if kind=="Markov":
        for k in range(1,K):
            sign=rng.random()
            if sign>.5:
                p1,p0=rng.uniform(bounds.lob,bounds.upb),rng.uniform(1-bounds.upb,1-bounds.lob)
            else:
                p0,p1=rng.uniform(bounds.lob,bounds.upb),rng.uniform(1-bounds.upb,1-bounds.lob)
            cpts[k]=[p0,p1]
    elif kind=="Tree":
        for parent in range(K):
            children=np.flatnonzero(adjacency[parent])
            if not len(children):
                continue
            level=int(np.log2(parent+1))+1
            if not alternate_prob or level%2==1:
                p1,p0=rng.uniform(bounds.lob,bounds.upb),rng.uniform(1-bounds.upb,1-bounds.lob)
            else:
                p1,p0=rng.uniform(1-bounds.upb,1-bounds.lob),rng.uniform(bounds.lob,bounds.upb)
            for child in children:
                cpts[child]=[p0,p1]
    else:
        p0=rng.uniform(bounds.lob,bounds.upb)
        p1=rng.uniform(bounds.lob2,bounds.upb2)
        p2=rng.uniform(bounds.lob3,bounds.upb3)
        p3=rng.uniform(bounds.lob4,bounds.upb4)
        p4=rng.uniform(bounds.lob5,bounds.upb5)
        negative_children={"Model-7":{2,6},"Model-8":{2,4,6},
                             "Model-13":{2,4,6},"Model-16":{2,5,7,9}}
        for child,par in enumerate(parents):
            if len(par)==2:
                cpts[child]=[p2,p1,p0,p4] if kind=="DiverseTree" else [p3,p2,p1,p0]
            elif len(par)==1:
                if kind=="DiverseTree":
                    cpts[child]=[p3,1-p3]
                else:
                    cpts[child]=[p4,1-p4] if child in negative_children[kind] else [1-p4,p4]
    return dict(K=int(K),DAG_type=kind,dag_true=adjacency,parent_sets=parents,cpts=cpts,
                provenance="Actual public generator CPT draws replayed from captured pre-generation Python RNG state")


def generate_latent_with_truth(data,alternate_prob=True):
    """Generate once with the original sampler and attach its exact population specification.

    No additional random draws are consumed. Call before generate_data().
    Existing data.A is replaced just as with generate_latent_data(); the new
    attached specification always describes the newly returned A.
    """
    state=random.getstate()
    spec=_capture_spec(data,state,alternate_prob)
    A=data.generate_latent_data(alternate_prob=alternate_prob)
    data._oracle_model_spec=spec
    data._oracle_latent_rows=A
    return A


def true_model(data,max_k=16):
    """Exact p, DAG, CPDAG, and actual CPTs; enumeration is capped at max_k.

    Refuses uncaptured objects: empirical frequencies are not population p.
    The CPT indexing uses binary parent states in increasing parent-index
    order, with the first parent as the most significant bit.
    """
    if not isinstance(max_k,(int,np.integer)) or max_k<1:
        raise ValueError("max_k must be a positive integer")
    if data.K>max_k:
        raise ValueError(f"Exact oracle enumeration requires 2**K states; K={data.K} exceeds max_k={max_k}")
    if not hasattr(data,"_oracle_model_spec"):
        raise ValueError("Actual CPT draws were not captured; first use generate_latent_with_truth(data). Empirical A is not population truth.")
    if data.A is not getattr(data,"_oracle_latent_rows",None):
        raise ValueError("Latent rows were replaced after CPT capture; recapture with generate_latent_with_truth(data)")
    spec=copy.deepcopy(data._oracle_model_spec)
    K=spec["K"]
    if K!=data.K or spec["DAG_type"]!=data.DAG_type:
        raise ValueError("Captured generator specification does not match current data dimensions/type")
    states=((np.arange(1<<K)[:,None]>>np.arange(K-1,-1,-1))&1)
    probabilities=np.ones(1<<K)
    for child,parents in enumerate(spec["parent_sets"]):
        parent_index=states[:,parents]@(1<<np.arange(len(parents)-1,-1,-1)) if parents else np.zeros(len(states),dtype=int)
        chance=np.asarray(spec["cpts"][child])[parent_index]
        probabilities*=np.where(states[:,child]==1,chance,1-chance)
    if not np.isfinite(probabilities).all() or np.any(probabilities<0) or not np.isclose(probabilities.sum(),1,rtol=1e-12,atol=1e-12):
        raise ValueError("Captured CPT specification does not define a normalized probability distribution")
    return {**spec,"p_true":probabilities,"cpdag_true":exact_cpdag(spec["dag_true"])}


def empirical_law(A):
    """Empirical law of the supplied actual latent rows; explicitly not p_true."""
    A=np.asarray(A)
    if A.ndim!=2 or not A.shape[0] or not A.shape[1] or not np.isin(A,[0,1]).all():
        raise ValueError("A must be a nonempty binary N by K matrix")
    counts=Counter()
    for row in A:
        state=0
        for value in row:
            state=(state<<1)|int(value)
        counts[state]+=1
    return SparseLatentLaw.from_counts(A.shape[1],counts)
