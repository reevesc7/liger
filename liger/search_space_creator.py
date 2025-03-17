from typing import Any
import tpot
#import numpy as np


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
            #search_space = construct_choice_pipeline(node_parameters, n_features, random_state)
        case "SequentialPipeline":
            search_space = tpot.search_spaces.pipelines.SequentialPipeline(**node_kwargs)
            #search_space = construct_sequential_pipeline(node_parameters, n_features, random_state)
        case "DynamicLinearPipeline":
            search_space = tpot.search_spaces.pipelines.DynamicLinearPipeline(**node_kwargs)
            #search_space = construct_dynamic_linear_pipeline(node_parameters, n_features, random_state)
        case "UnionPipeline":
            search_space = tpot.search_spaces.pipelines.UnionPipeline(**node_kwargs)
            #search_space = construct_union_pipeline(node_parameters, n_features, random_state)
        case "DynamicUnionPipeline":
            search_space = tpot.search_spaces.pipelines.DynamicUnionPipeline(**node_kwargs)
            #search_space = construct_dynamic_union_pipeline(node_parameters, n_features, random_state)
        case "TreePipeline":
            search_space = tpot.search_spaces.pipelines.TreePipeline(**node_kwargs)
            #search_space = construct_tree_pipeline(node_parameters, n_features, random_state)
        case "GraphSearchPipeline":
            search_space = tpot.search_spaces.pipelines.GraphSearchPipeline(**node_kwargs)
            #search_space = construct_graph_search_pipeline(node_parameters, n_features, random_state)
        case "EstimatorNode":
            search_space = tpot.config.get_search_space(node_parameters["class_name"], random_state=random_state)
            #search_space = construct_estimator_node(node_parameters, random_state)
        case "GeneticFeatureSelectorNode":
            search_space = tpot.search_spaces.nodes.GeneticFeatureSelectorNode(n_features, **node_kwargs)
            #search_space = construct_genetic_feature_selector_node(node_parameters, n_features)
        case _:
            raise ValueError(f"{node_type} does not match a TPOT pipeline type (WrapperPipeline not included)")
    return search_space


#def construct_choice_pipeline(params: dict, n_features: int, random_state: int | None = None) -> tpot.search_spaces.pipelines.ChoicePipeline:
#    search_spaces = create_search_spaces(params.get("search_spaces"), n_features, random_state)
#    return tpot.search_spaces.pipelines.ChoicePipeline(
#        search_spaces=search_spaces,
#    )
#
#
#def construct_sequential_pipeline(params: dict, n_features: int, random_state: int | None = None) -> tpot.search_spaces.pipelines.SequentialPipeline:
#    search_spaces = create_search_spaces(params.get("search_spaces"), n_features, random_state)
#    return tpot.search_spaces.pipelines.SequentialPipeline(
#        search_spaces=search_spaces,
#    )
#
#
#def construct_dynamic_linear_pipeline(params: dict, n_features: int, random_state: int | None = None) -> tpot.search_spaces.pipelines.DynamicLinearPipeline:
#    search_space = create_search_space(params.get("search_space"), n_features, random_state)
#    return tpot.search_spaces.pipelines.DynamicLinearPipeline(
#        search_space=search_space,
#        max_length=params["max_length"],
#    )
#
#
#def construct_union_pipeline(params: dict, n_features: int, random_state: int | None = None) -> tpot.search_spaces.pipelines.UnionPipeline:
#    search_spaces = create_search_spaces(params.get("search_spaces"), n_features, random_state)
#    return tpot.search_spaces.pipelines.UnionPipeline(
#        search_spaces=search_spaces,
#    )
#
#
#def construct_dynamic_union_pipeline(params: dict, n_features: int, random_state: int | None = None) -> tpot.search_spaces.pipelines.DynamicUnionPipeline:
#    search_space = create_search_space(params.get("search_space"), n_features, random_state)
#    return tpot.search_spaces.pipelines.DynamicUnionPipeline(
#        search_space=search_space,
#        #max_estimators=params.get("max_estimators"),
#        #allow_repeats=params.get("allow_repeats", False),
#        **{key: value for key, value in params.items() if "search_space" not in key}
#    )
#
#
#def construct_tree_pipeline(params: dict, n_features: int, random_state: int | None = None) -> tpot.search_spaces.pipelines.TreePipeline:
#    root_search_space = create_search_space(params.get("root_search_space"), n_features, random_state)
#    leaf_search_space = create_search_space(params.get("leaf_search_space"), n_features, random_state)
#    inner_search_space = create_search_space(params.get("inner_search_space"), n_features, random_state)
#    return tpot.search_spaces.pipelines.TreePipeline(
#        root_search_space=root_search_space,
#        leaf_search_space=leaf_search_space,
#        inner_search_space=inner_search_space,
#        #min_size=params.get("min_size", 2),
#        #max_size=params.get("max_size", 10),
#        #crossover_same_depth=params.get("crossover_same_depth", False),
#        **{key: value for key, value in params.items() if "search_space" not in key}
#    )
#
#
#def construct_graph_search_pipeline(params: dict, n_features: int, random_state: int | None = None) -> tpot.search_spaces.pipelines.GraphSearchPipeline:
#    root_search_space = create_search_space(params.get("root_search_space"), n_features, random_state)
#    leaf_search_space = create_search_space(params.get("leaf_search_space"), n_features, random_state)
#    inner_search_space = create_search_space(params.get("inner_search_space"), n_features, random_state)
#    return tpot.search_spaces.pipelines.GraphSearchPipeline(
#        root_search_space=root_search_space,
#        leaf_search_space=leaf_search_space,
#        inner_search_space=inner_search_space,
#        #max_size=params.get("max_size", np.inf),
#        #crossover_same_depth=params.get("crossover_same_depth", False),
#        #cross_val_predict_cv=params.get("cross_val_predict_cv", 0),
#        #method=params.get("method", "auto"),
#        #use_label_encoder=params.get("use_label_encoder", False),
#        **{key: value for key, value in params.items() if "search_space" not in key}
#    )
#
#
#def construct_estimator_node(params: dict, random_state: int | None = None) -> tpot.search_spaces.nodes.EstimatorNode | tpot.search_spaces.pipelines.ChoicePipeline:
#    search_space = tpot.config.get_search_space(params["class_name"], random_state=random_state)
#    if not isinstance(search_space, (tpot.search_spaces.nodes.EstimatorNode, tpot.search_spaces.pipelines.ChoicePipeline)):
#        raise ValueError(f"{params} does not match a TPOT EstimatorNode type")
#    return search_space
#
#
#def construct_genetic_feature_selector_node(params: dict, n_features: int) -> tpot.search_spaces.nodes.GeneticFeatureSelectorNode:
#    search_space = GeneticFeatureSelectorNode(
#        n_features=n_features,
#        **params,
#    )
#    return search_space

