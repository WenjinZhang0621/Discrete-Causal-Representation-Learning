from causallearn.search.ScoreBased.GES import ges

from dcrl.config import BoundsConfig
from dcrl.data_generator import GenerateData


DEFAULT_BOUNDS=BoundsConfig()


def build_truth_graph(dag_type,distribution="Lognormal",N_truth=4000,*,K=None,bounds=None):
    dag_specs={
        "DiverseTree": {"J": 45,"K": 15},
        "Tree": {"J": 30,"K": 10},
        "Markov": {"J": 30,"K": 10},
        "Model-8": {"J": 24,"K": 8},
        "Model-7": {"J": 21,"K": 7},
        "Model-13": {"J": 39,"K": 13},
        "Model-16": {"J": 48,"K": 16},
    }

    if dag_type not in dag_specs:
        raise ValueError(f"Unknown DAG_type: {dag_type}")

    spec=dict(dag_specs[dag_type])
    if K is not None:
        if not isinstance(K,int) or K<1:
            raise ValueError("K must be a positive integer")
        if dag_type.startswith("Model-") and K!=spec["K"]:
            raise ValueError(f"{dag_type} requires K={spec['K']}")
        spec={"K": K,"J": 3*K}
    b=DEFAULT_BOUNDS if bounds is None else bounds

    dag_model=GenerateData(
        N=N_truth,
        J=spec["J"],
        K=spec["K"],
        Q_type="2",
        DAG_type=dag_type,
        distribution=distribution,
        upb=b.upb,
        lob=b.lob,
        upb2=b.upb2,
        lob2=b.lob2,
        upb3=b.upb3,
        lob3=b.lob3,
        upb4=b.upb4,
        lob4=b.lob4,
        upb5=b.upb5,
        lob5=b.lob5,
    )
    A1=dag_model.generate_latent_data()
    record=ges(A1,score_func="local_score_BDeu")
    return record["G"]


def build_all_truth_graphs(distribution="Lognormal",N_truth=4000):
    out={}
    for dag_type in ["DiverseTree","Tree","Markov","Model-8","Model-7","Model-13"]:
        out[dag_type]=build_truth_graph(dag_type=dag_type,distribution=distribution,N_truth=N_truth)
    return out
