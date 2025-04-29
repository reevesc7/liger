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
import tpot


def create_search_space(param_search_space: Any, n_features: int, random_state: int | None = None) -> tpot.search_spaces.SearchSpace:
    search_space = items_to_search_space(
        param_search_space["node_type"],
        {key: value for key, value in param_search_space.items() if key != "node_type"},
        n_features,
        random_state,
    )
    return search_space


def create_search_spaces(param_search_spaces: Any, n_features: int, random_state: int | None = None) -> list[tpot.search_spaces.SearchSpace]:
    search_spaces = []
    for param_search_space in param_search_spaces:
        search_spaces.append(create_search_space(param_search_space, n_features, random_state))
    return search_spaces


def items_to_search_space(node_type: str, node_parameters: Any, n_features: int, random_state: int | None = None) -> tpot.search_spaces.SearchSpace:
    node_kwargs = {}
    for key, value in node_parameters.items():
        if key == "search_spaces":
            node_kwargs[key] = create_search_spaces(value, n_features, random_state)
        elif "search_space" in key:
            node_kwargs[key] = create_search_space(value, n_features, random_state)
    node_kwargs.update({key: value for key, value in node_parameters.items() if "search_space" not in key})
    match node_type:
        case "ChoicePipeline":
            search_space = tpot.search_spaces.pipelines.ChoicePipeline(**node_kwargs)
        case "SequentialPipeline":
            search_space = tpot.search_spaces.pipelines.SequentialPipeline(**node_kwargs)
        case "DynamicLinearPipeline":
            search_space = tpot.search_spaces.pipelines.DynamicLinearPipeline(**node_kwargs)
        case "UnionPipeline":
            search_space = tpot.search_spaces.pipelines.UnionPipeline(**node_kwargs)
        case "DynamicUnionPipeline":
            search_space = tpot.search_spaces.pipelines.DynamicUnionPipeline(**node_kwargs)
        case "TreePipeline":
            search_space = tpot.search_spaces.pipelines.TreePipeline(**node_kwargs)
        case "GraphSearchPipeline":
            search_space = tpot.search_spaces.pipelines.GraphSearchPipeline(**node_kwargs)
        case "EstimatorNode":
            search_space = tpot.config.get_search_space(node_parameters["class_name"], random_state=random_state)
            if not isinstance(search_space, (tpot.search_spaces.nodes.EstimatorNode, tpot.search_spaces.pipelines.ChoicePipeline)):
                raise ValueError(f"{node_parameters['class_name']} could not be converted into an EstimatorNode or ChoicePipeline")
        case "GeneticFeatureSelectorNode":
            search_space = tpot.search_spaces.nodes.GeneticFeatureSelectorNode(n_features, **node_kwargs)
        case _:
            raise ValueError(f"{node_type} does not match a TPOT pipeline or node type (WrapperPipeline not included)")
    return search_space

