from typing import Any
import tpot
import numpy as np


def create_search_space(param_search_space: Any, random_state: int | None = None) -> tpot.search_spaces.SearchSpace:
    search_space = items_to_search_space(
        param_search_space.pop("node_type"),
        param_search_space,
        random_state,
    )
    #keys = param_search_space.keys()
    #if len(keys) > 1:
    #    raise ValueError(f"More that one search space ({len(keys)}) is defined by the \"search_space\"")
    #else:
    #    key = list(keys)[0]
    #search_space = items_to_search_space(key, param_search_space[key], random_state)
    return search_space


def create_search_spaces(param_search_spaces: Any, random_state: int | None = None) -> list[tpot.search_spaces.SearchSpace]:
    search_spaces = []
    for param_search_space in param_search_spaces:
        search_spaces.append(create_search_space(param_search_space, random_state))
    #for node_type, node_parameters in param_search_spaces.items():
    #    search_spaces.append(items_to_search_space(node_type, node_parameters, random_state))
    return search_spaces


def items_to_search_space(node_type: str, node_parameters: Any, random_state: int | None = None) -> tpot.search_spaces.SearchSpace:
    match node_type:
        case "ChoicePipeline":
            search_space = construct_choice_pipeline(node_parameters, random_state)
        case "SequentialPipeline":
            search_space = construct_sequential_pipeline(node_parameters, random_state)
        case "DynamicLinearPipeline":
            search_space = construct_dynamic_linear_pipeline(node_parameters, random_state)
        case "UnionPipeline":
            search_space = construct_union_pipeline(node_parameters, random_state)
        case "DynamicUnionPipeline":
            search_space = construct_dynamic_union_pipeline(node_parameters, random_state)
        case "TreePipeline":
            search_space = construct_tree_pipeline(node_parameters, random_state)
        case "GraphSearchPipeline":
            search_space = construct_graph_search_pipeline(node_parameters, random_state)
        case "EstimatorNode":
            search_space = construct_estimator_node(node_parameters, random_state)
        case _:
            raise ValueError(f"{node_type} does not match a TPOT pipeline type (WrapperPipeline not included)")
    return search_space


def construct_choice_pipeline(params: dict, random_state: int | None = None) -> tpot.search_spaces.pipelines.ChoicePipeline:
    search_spaces = create_search_spaces(params.get("search_spaces"), random_state)
    return tpot.search_spaces.pipelines.ChoicePipeline(
        search_spaces=search_spaces,
    )


def construct_sequential_pipeline(params: dict, random_state: int | None = None) -> tpot.search_spaces.pipelines.SequentialPipeline:
    search_spaces = create_search_spaces(params.get("search_spaces"), random_state)
    return tpot.search_spaces.pipelines.SequentialPipeline(
        search_spaces=search_spaces,
    )


def construct_dynamic_linear_pipeline(params: dict, random_state: int | None = None) -> tpot.search_spaces.pipelines.DynamicLinearPipeline:
    search_space = create_search_space(params.get("search_space"), random_state)
    return tpot.search_spaces.pipelines.DynamicLinearPipeline(
        search_space=search_space,
        max_length=params["max_length"],
    )


def construct_union_pipeline(params: dict, random_state: int | None = None) -> tpot.search_spaces.pipelines.UnionPipeline:
    search_spaces = create_search_spaces(params.get("search_spaces"), random_state)
    return tpot.search_spaces.pipelines.UnionPipeline(
        search_spaces=search_spaces,
    )


def construct_dynamic_union_pipeline(params: dict, random_state: int | None = None) -> tpot.search_spaces.pipelines.DynamicUnionPipeline:
    search_space = create_search_space(params.get("search_space"), random_state)
    return tpot.search_spaces.pipelines.DynamicUnionPipeline(
        search_space=search_space,
        max_estimators=params.get("max_estimators"),
        allow_repeats=params.get("allow_repeats", False),
    )


def construct_tree_pipeline(params: dict, random_state: int | None = None) -> tpot.search_spaces.pipelines.TreePipeline:
    root_search_space = create_search_space(params.get("root_search_space"), random_state)
    leaf_search_space = create_search_space(params.get("leaf_search_space"), random_state)
    inner_search_space = create_search_space(params.get("inner_search_space"), random_state)
    return tpot.search_spaces.pipelines.TreePipeline(
        root_search_space=root_search_space,
        leaf_search_space=leaf_search_space,
        inner_search_space=inner_search_space,
        min_size=params.get("min_size", 2),
        max_size=params.get("max_size", 10),
        crossover_same_depth=params.get("crossover_same_depth", False),
    )


def construct_graph_search_pipeline(params: dict, random_state: int | None = None) -> tpot.search_spaces.pipelines.GraphSearchPipeline:
    root_search_space = create_search_space(params.get("root_search_space"), random_state)
    leaf_search_space = create_search_space(params.get("leaf_search_space"), random_state)
    inner_search_space = create_search_space(params.get("inner_search_space"), random_state)
    return tpot.search_spaces.pipelines.GraphSearchPipeline(
        root_search_space=root_search_space,
        leaf_search_space=leaf_search_space,
        inner_search_space=inner_search_space,
        max_size=params.get("max_size", np.inf),
        crossover_same_depth=params.get("crossover_same_depth", False),
        cross_val_predict_cv=params.get("cross_val_predict_cv", 0),
        method=params.get("method", "auto"),
        use_label_encoder=params.get("use_label_encoder", False),
    )


def construct_estimator_node(params: dict, random_state: int | None = None) -> tpot.search_spaces.nodes.EstimatorNode:
    search_space = tpot.config.get_search_space(params["class_name"], random_state=random_state)
    if not isinstance(search_space, tpot.search_spaces.nodes.EstimatorNode):
        raise ValueError(f"{params} does not match a TPOT EstimatorNode type")
    return search_space

