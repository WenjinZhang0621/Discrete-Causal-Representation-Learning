"""Reproducible graph artifacts and input provenance for both TIMSS CLIs."""
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from .graph import learn_graph

LABELS = ['Number', 'Algebra', 'Geometry', 'Data and Probability', 'Knowing', 'Applying', 'Reasoning']


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plot_graph(graph, path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.patches import FancyArrowPatch

    K = len(graph)
    skeleton = nx.from_numpy_array((graph != 0).astype(int))
    positions = nx.spring_layout(skeleton, seed=17)
    labels = LABELS if K == 7 else [f'Z{i+1}' for i in range(K)]
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(K):
        for j in range(i+1, K):
            if not graph[i, j]:
                continue
            directed = graph[i, j] != graph[j, i]
            a, b = (i, j) if not directed or graph[i, j] == -1 else (j, i)
            ax.add_patch(FancyArrowPatch(positions[a], positions[b],
                         arrowstyle='-|>' if directed else '-', mutation_scale=18,
                         shrinkA=24, shrinkB=24, color='#41636c', linewidth=1.8))
    for i in range(K):
        x, y = positions[i]
        ax.text(x, y, labels[i].replace(' and ', ' and\n'), ha='center', va='center',
                bbox=dict(boxstyle='round,pad=.4', facecolor='#e4eee8', edgecolor='#829a93'))
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.2, 1.2)
    ax.set_title(title, pad=18)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_graph(p, N, K, fn_ratio, seed, results_dir, metadata=None, plot=True):
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    record, samples, counts = learn_graph(p, N, K, fn_ratio, seed)
    graph = record['G'].graph.astype(int)
    np.save(out/'latent_samples_from_phat.npy', samples)
    np.save(out/'latent_counts.npy', counts)
    np.save(out/'latent_graph_estimated.npy', graph)
    if plot:
        plot_graph(graph, out/'latent_graph_estimated.png',
                   f'TIMSS CPDAG: multinomial resampling, f(N)/N={fn_ratio:g}, seed={seed}')
    summary = dict(metadata or {})
    summary.update(N=int(N), K=int(K), fn_ratio=float(fn_ratio), f_N=int(len(samples)),
                   graph_seed=int(seed), sampling='multinomial', rng='numpy.random.RandomState',
                   rounding='max(1, floor(fn_ratio*N))', score='local_score_BDeu',
                   graph_edges=int(np.count_nonzero(graph)//2), cpdag=graph.tolist(),
                   sample_sha256=hashlib.sha256(samples.tobytes()).hexdigest(),
                   versions={name: importlib.metadata.version(name) for name in ['numpy', 'scipy', 'causal-learn']})
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary
