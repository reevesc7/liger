# liger - Helper functions for the Likert General Regressor project
# Copyright (C) 2024  Chris Reeves
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


from typing import Any
from tpot import search_spaces as tpss
from tpot import config as tpcfg
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters.hyperparameter import Hyperparameter
from liger import sklearn as lsk


def create_search_space(
    param_search_space: Any,
    n_samples: int,
    n_features: int,
    random_state: int | None = None
) -> tpss.SearchSpace:
    search_space = items_to_search_space(
        param_search_space["node_type"],
        {key: value for key, value in param_search_space.items() if key != "node_type"},
        n_samples,
        n_features,
        random_state,
    )
    return search_space


def create_search_spaces(
    param_search_spaces: Any,
    n_samples: int,
    n_features: int,
    random_state: int | None = None
) -> list[tpss.SearchSpace]:
    search_spaces = []
    for param_search_space in param_search_spaces:
        search_spaces.append(create_search_space(
            param_search_space,
            n_samples,
            n_features,
            random_state,
        ))
    return search_spaces


def _append_to_hyperparameter_name(hyperparameter: Hyperparameter, add: str) -> Hyperparameter:
    hyperparameter.name = add + hyperparameter.name
    return hyperparameter


def _make_lg_transformed_target_regressor(
    node_parameters: dict[str, str],
    random_state: int | None = None,
) -> tpss.SearchSpace:
    rg_node = tpcfg.get_search_space(node_parameters["regressor"], random_state=random_state)
    tf_node = tpcfg.get_search_space(node_parameters["transformer"], random_state=random_state)
    if not isinstance(rg_node, tpss.nodes.EstimatorNode):
        raise ValueError(f"{node_parameters["regressor"]} could not be converted into an EstimatorNode")
    if not isinstance(tf_node, tpss.nodes.EstimatorNode):
        raise ValueError(f"{node_parameters["transformer"]} could not be converted into an EstimatorNode")
    models = {"regressor": rg_node.method, "transformer": tf_node.method}
    rg_params = {
        "regressor_" + key: _append_to_hyperparameter_name(value, "regressor_")
        for key, value in rg_node.space.items()
    }
    tf_params = {
        "transformer_" + key: _append_to_hyperparameter_name(value, "transformer_")
        for key, value in tf_node.space.items()
    }
    return tpss.nodes.EstimatorNode(
        method=lsk.LgTransformedTargetRegressor,
        space=ConfigurationSpace(space=models | rg_params | tf_params),
    )


def _make_lg_passthrough(
) -> tpss.SearchSpace:
    return tpss.nodes.EstimatorNode(
        method=lsk.LgPassthrough,
        space=ConfigurationSpace(),
    )


def items_to_search_space(
    node_type: str,
    node_parameters: Any,
    n_samples: int,
    n_features: int,
    random_state: int | None = None
) -> tpss.SearchSpace:
    node_kwargs = {}
    for key, value in node_parameters.items():
        if key == "search_spaces":
            node_kwargs[key] = create_search_spaces(value, n_samples, n_features, random_state)
        elif "search_space" in key:
            node_kwargs[key] = create_search_space(value, n_samples, n_features, random_state)
    node_kwargs.update({
        key: value
        for key, value in node_parameters.items()
        if "search_space" not in key
    })
    match node_type:
        case "ChoicePipeline":
            search_space = tpss.pipelines.ChoicePipeline(**node_kwargs)
        case "SequentialPipeline":
            search_space = tpss.pipelines.SequentialPipeline(**node_kwargs)
        case "DynamicLinearPipeline":
            search_space = tpss.pipelines.DynamicLinearPipeline(**node_kwargs)
        case "UnionPipeline":
            search_space = tpss.pipelines.UnionPipeline(**node_kwargs)
        case "DynamicUnionPipeline":
            search_space = tpss.pipelines.DynamicUnionPipeline(**node_kwargs)
        case "TreePipeline":
            search_space = tpss.pipelines.TreePipeline(**node_kwargs)
        case "GraphSearchPipeline":
            search_space = tpss.pipelines.GraphSearchPipeline(**node_kwargs)
        case "EstimatorNode":
            search_space = tpcfg.get_search_space(
                name=node_parameters["method"],
                n_samples=n_samples,
                n_features=n_features,
                random_state=random_state,
            )
            if not isinstance(search_space, (
                tpss.nodes.EstimatorNode,
                tpss.pipelines.ChoicePipeline,
            )):
                raise ValueError(f"{node_parameters["method"]} could not be converted into an EstimatorNode or ChoicePipeline")
        case "GeneticFeatureSelectorNode":
            search_space = tpss.nodes.GeneticFeatureSelectorNode(n_features, **node_kwargs)
        case "LgTransformedTargetRegressor":
            search_space = _make_lg_transformed_target_regressor(node_kwargs, random_state)
        case "LgPassthrough":
            search_space = _make_lg_passthrough()
        case _:
            raise ValueError(f"{node_type} does not match a TPOT pipeline or node type (WrapperPipeline not included)")
    return search_space

