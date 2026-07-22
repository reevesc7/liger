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
from sklearn.decomposition import TruncatedSVD
from tpot import search_spaces as tpss
from tpot import config as tpcfg
from tpot.builtin_modules import EstimatorTransformer
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters.hyperparameter import Hyperparameter
from liger import sklearn as lsk


class SearchSpaceParser:
    def __init__(
        self,
        n_samples: int,
        n_features: int,
        random_state: int | None = None,
    ) -> None:
        self.n_samples = n_samples
        self.n_features = n_features
        self.random_state = random_state

    def _parse_subspaces(
        self,
        node_parameters: dict[str, Any],
    ) -> dict[str, Any]:
        node_kwargs: dict[str, Any] = {}
        for key, value in node_parameters.items():
            if key == "search_spaces":
                node_kwargs[key] = [self.parse(config) for config in value]
            elif "search_space" in key:
                node_kwargs[key] = self.parse(value)
        node_kwargs.update({
            key: value
            for key, value in node_parameters.items()
            if "search_space" not in key
        })
        return node_kwargs

    def _configure_estimator_node(
        self,
        method_name: str,
        search_space: tpss.nodes.EstimatorNode,
    ) -> tpss.nodes.EstimatorNode:
        match method_name:
            case "BaggingRegressor":
                search_space.space = ConfigurationSpace({
                    key: value
                    for key, value in search_space.space.items()
                    if key != "oob_score"
                })
            case "Nystroem":
                search_space.space = ConfigurationSpace({
                    key: value if key != "kernel" else [
                        "rbf",
                        "cosine",
                        "laplacian",
                        "polynomial",
                        "poly",
                        "linear",
                        "sigmoid",
                    ]
                    for key, value in search_space.space.items()
                })
        return search_space

    def _make_estimator_node(
        self,
        method_name: str,
    ) -> tpss.nodes.EstimatorNode | tpss.pipelines.ChoicePipeline:
        search_space = tpcfg.get_search_space(
            name=method_name,
            n_samples=self.n_samples,
            n_features=self.n_features,
            random_state=self.random_state,
        )
        if isinstance(search_space, tpss.pipelines.ChoicePipeline):
            return search_space
        if isinstance(search_space, tpss.nodes.EstimatorNode):
            search_space = self._configure_estimator_node(method_name, search_space)
            return search_space
        raise ValueError(f"{method_name} "
            "could not be converted into an EstimatorNode or ChoicePipeline")

    def _make_wrapper_pipeline(
        self,
        node_parameters: dict[str, Any],
    ) -> tpss.pipelines.WrapperPipeline:
        method_name = node_parameters.pop("method")
        if method_name == "EstimatorTransformer":
            method = EstimatorTransformer
            space = ConfigurationSpace({"passthrough": [False, True]})
        else:
            estimator_config_space = self._make_estimator_node(method_name)
            if not isinstance(estimator_config_space, tpss.nodes.EstimatorNode):
                raise ValueError(f"{method_name} "
                    "must represent a single estimator class")
            method = estimator_config_space.method
            space = estimator_config_space.space
        return tpss.pipelines.WrapperPipeline(method, space, **node_parameters)

    @staticmethod
    def _append_to_hyperparameter_name(
        hyperparameter: Hyperparameter,
        add: str,
    ) -> Hyperparameter:
        hyperparameter.name = add + hyperparameter.name
        return hyperparameter

    def _make_lg_transformed_target_regressor(
        self,
        node_parameters: dict[str, Any],
    ) -> tpss.nodes.EstimatorNode:
        rg_node = tpcfg.get_search_space(
            node_parameters["regressor"],
            n_samples=self.n_samples,
            n_features=self.n_features,
            random_state=self.random_state,
        )
        tf_node = tpcfg.get_search_space(
            node_parameters["transformer"],
            n_samples=self.n_samples,
            n_features=self.n_features,
            random_state=self.random_state,
        )
        if not isinstance(rg_node, tpss.nodes.EstimatorNode):
            raise ValueError(f"{node_parameters['regressor']} "
                "could not be converted to an EstimatorNode")
        if not isinstance(tf_node, tpss.nodes.EstimatorNode):
            raise ValueError(f"{node_parameters['transformer']} "
                "could not be converted to an EstimatorNode")
        models = {"regressor": rg_node.method, "transformer": tf_node.method}
        rg_params = {
            "regressor_" + key: self._append_to_hyperparameter_name(
                value,
                "regressor_",
            )
            for key, value in rg_node.space.items()
        }
        tf_params = {
            "transformer_" + key: self._append_to_hyperparameter_name(
                value,
                "transformer_",
            )
            for key, value in tf_node.space.items()
        }
        return tpss.nodes.EstimatorNode(
            method=lsk.LgTransformedTargetRegressor,
            space=ConfigurationSpace(space=models | rg_params | tf_params),
        )

    def _make_lg_truncated_svd(
        self,
    ) -> tpss.nodes.EstimatorNode:
        max_components = min(self.n_samples, self.n_features)
        space = {
            "n_components": (1, min(max_components - 1, 200)),
            "algorithm": ["arpack", "randomized"],
            "n_iter": (2, 8),
            "n_oversamples": (1, 30),
        }
        if self.random_state is not None:
            space["random_state"] = self.random_state
        return tpss.nodes.EstimatorNode(
            method=TruncatedSVD,
            space=ConfigurationSpace(space=space),
        )

    @staticmethod
    def _make_lg_passthrough() -> tpss.nodes.EstimatorNode:
        return tpss.nodes.EstimatorNode(
            method=lsk.LgPassthrough,
            space=ConfigurationSpace(),
        )

    def _make_lg_estimator_node(
        self,
        method_name: str,
        node_parameters: dict[str, Any],
    ) -> tpss.nodes.EstimatorNode:
        match method_name:
            case "TransformedTargetRegressor":
                return self._make_lg_transformed_target_regressor(
                    node_parameters,
                )
            case "TruncatedSVD":
                return self._make_lg_truncated_svd()
            case "Passthrough":
                return self._make_lg_passthrough()
            case _:
                raise ValueError(f"{method_name} does not match a liger estimator type")

    def parse(self, config: dict[str, Any]) -> tpss.SearchSpace:
        node_type = config["node_type"]
        node_parameters = {
            key: value
            for key, value in config.items()
            if key != "node_type"
        }
        node_kwargs = self._parse_subspaces(node_parameters)
        match node_type:
            case "ChoicePipeline":
                return tpss.pipelines.ChoicePipeline(**node_kwargs)
            case "SequentialPipeline":
                return tpss.pipelines.SequentialPipeline(**node_kwargs)
            case "DynamicLinearPipeline":
                return tpss.pipelines.DynamicLinearPipeline(**node_kwargs)
            case "UnionPipeline":
                return tpss.pipelines.UnionPipeline(**node_kwargs)
            case "DynamicUnionPipeline":
                return tpss.pipelines.DynamicUnionPipeline(**node_kwargs)
            case "TreePipeline":
                return tpss.pipelines.TreePipeline(**node_kwargs)
            case "GraphSearchPipeline":
                return tpss.pipelines.GraphSearchPipeline(**node_kwargs)
            case "EstimatorNode":
                return self._make_estimator_node(node_parameters["method"])
            case "WrapperPipeline":
                return self._make_wrapper_pipeline(node_kwargs)
            case "GeneticFeatureSelectorNode":
                return tpss.nodes.GeneticFeatureSelectorNode(
                    self.n_features,
                    **node_kwargs,
                )
            case "LgEstimatorNode":
                return self._make_lg_estimator_node(
                    node_parameters["method"],
                    node_kwargs,
                )
        raise ValueError(f"{node_type} does not match a TPOT pipeline or node type")
