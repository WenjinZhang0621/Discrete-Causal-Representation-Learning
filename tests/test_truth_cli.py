import importlib.util
from pathlib import Path

import pytest


def load_helper():
    path=Path(__file__).parents[1]/'experiments/truth_graph.py'
    spec=importlib.util.spec_from_file_location('_cli_truth_test',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_requested_tree_dimension_is_used():
    helper=load_helper()
    graph=helper.build_truth_graph('Tree',N_truth=500,K=3)
    assert graph.graph.shape==(3,3)


def test_fixed_model_dimension_rejected():
    with pytest.raises(ValueError,match='requires K=13'):
        load_helper().build_truth_graph('Model-13',K=10)
